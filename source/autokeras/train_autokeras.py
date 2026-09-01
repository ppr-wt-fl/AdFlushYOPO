"""
- tf.keras.regularizers.L2 serializes as class name "L2", which tfjs-layers' deserializer doesn't recognize. 

- AutoKeras's default StructuredDataBlock normalizes numeric columns with a tf.keras.layers.Normalization (adapt-based) preprocessing layer baked into
the exported model graph. -> tfjs.loadLayersModel() fails on
    - tfjs-layers (checked against the @tensorflow/tfjs-layers v4.22.0 source: no `Normalization` class exists in tfjs-layers/src/layers/normalization.ts only has BatchNormalization and LayerNormalization;
    - nothing under layers/preprocessing/ covers it either) does not implement that layer
 
 tf.keras.layers.Normalization layer swapped for a tf.keras.layers.Rescaling built from the Normalization
layer's own adapted mean/variance (mathematically identical)

- Every other layer instance (with its trained weights) is reused unchanged.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import autokeras as ak
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_autokeras")

SOURCE_DIR = Path(__file__).resolve().parent
REPO_DIR = SOURCE_DIR.parent
DATADIR = REPO_DIR / "dataset"
MODELDIR = REPO_DIR / "model"
OUTDIR = REPO_DIR / "output"

TRAINSET = DATADIR / "AdFlush_train.csv"
TESTSET = DATADIR / "AdFlush_test.csv"

SEED = 42
VALIDATION_SAMPLE_SIZE = 1024

# ---------------------------------------------------------------------------
# Full in-depth search. AutoKeras has no wall-clock budget knob, so the
# ~9-10 day ceiling is bounded by MAX_TRIALS * (worst-case per-trial time):
#
#   * SEARCH_EPOCHS hard-caps each trial; the stock Keras EarlyStopping
#     (val_loss, passed to clf.fit below, restore_best_weights) ends most
#     trials earlier. Worst case per trial = SEARCH_EPOCHS full epochs.
#   * On the full train set (~664k rows) one epoch of a small dense model
#     is roughly 10-40 s on a GPU, so a trial is <= ~15-40 min and
#     400 trials <= ~5-10 days. If the logs show trials are much
#     faster/slower, raise/lower MAX_TRIALS -- it is the one knob to turn.
# ---------------------------------------------------------------------------
MAX_TRIALS = 700
# SEARCH_EPOCHS = 60
# EARLY_STOPPING_PATIENCE = 8


def load_dataset(path):
    df = pd.read_csv(path, index_col=0)
    df.reset_index(inplace=True, drop=True)
    feature_columns = [c for c in df.columns if c != "label"]
    X = df[feature_columns].astype("float32").values
    y = df["label"].astype("float32").values
    return X, y, feature_columns


def replace_normalization_with_rescaling(model):
    """Rebuild the AutoKeras export, swapping the preprocessing layers that
    tfjs-layers can't load. The exported StructuredDataClassifier model is a
    plain linear stack, so we just re-apply each layer in order.
    """
    eps = tf.keras.backend.epsilon()  # 1e-7; matches Normalization.call()'s own clamp exactly.
    inputs = model.inputs
    x = inputs[0]
    replaced = False

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if isinstance(layer, tf.keras.layers.Normalization):
            replaced = True
            # Normalization stores [mean, variance, count]; call() computes
            # (x - mean) / max(sqrt(variance), epsilon), which is exactly a Rescaling.
            mean, variance = layer.get_weights()[:2]
            std_safe = np.maximum(np.sqrt(variance), eps)
            new_layer = tf.keras.layers.Rescaling(
                scale=(1.0 / std_safe).astype("float32").reshape(-1).tolist(),
                offset=(-mean / std_safe).astype("float32").reshape(-1).tolist(),
                name=layer.name + "_rescaling",
            )
            x = new_layer(x)
            logger.info(f"  Replaced Normalization '{layer.name}' with Rescaling '{new_layer.name}'")
        elif type(layer).__name__ == "MultiCategoryEncoding":
            replaced = True
            # Not a linear op, so no faithful tfjs-layers replacement. Safe to
            # drop only because every column is declared "numerical" (see main()),
            # making it a pass-through -- the identity check after this confirms it.
            logger.info(f"  Dropped MultiCategoryEncoding layer '{layer.name}' (asserted no-op below)")
        else:
            x = layer(x)

    new_model = tf.keras.Model(inputs=inputs, outputs=x, name=model.name + "_tfjs_safe")
    return new_model, replaced


def metrics(true, pred, _is_mutated=False):
    """Same performance stats source/main.py's metrics() prints for the
    ONNX / MOJO models, so the AutoKeras model is reported the same way.
    """
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred).astype(int)

    print(f"Accuracy : {accuracy_score(true, pred)} ")
    print(f"Precision : {precision_score(true, pred)} ")
    print(f"Recall : {recall_score(true, pred)} ")
    print(f"F1 : {f1_score(true, pred)} ")

    # Number of attacks
    total_attacks = len(true)
    # Number of successful attacks (misclassifications)
    successful_attacks = sum(true != pred)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    # Calculate FNR
    fnr = fn / (tp + fn)
    print('False Negative Rate:', fnr)

    # Calculate FPR
    fpr = fp / (fp + tn)
    print('False Positive Rate:', fpr)

    print("AUROC: ", roc_auc_score(true, pred))
    fprlist, tprlist, thresholds = roc_curve(true, pred)
    cutoff = np.argmax(tprlist - fprlist)
    print("TPR ", tprlist[cutoff], "at FPR ", fprlist[cutoff])

    # ASR
    if _is_mutated:
        asr = successful_attacks / total_attacks
        print("Attack Success Rate: ", asr)


def evaluate_model(model, X, y):
    logger.info("Evaluating model on the test set (same stats as source/main.py) ...")
    start_time = time.time()
    prob = np.asarray(model.predict(X, batch_size=512, verbose=0)).reshape(-1)
    print("Inference time elapsed: ", time.time() - start_time, "seconds for ", len(y), " samples.")
    pred = (prob >= 0.5).astype(int)
    metrics(y, pred, _is_mutated=False)


def run_tfjs_validation(model_json_path, payload_path):
    validate_js = SOURCE_DIR / "tfjs_validate" / "validate.js"
    node_modules = SOURCE_DIR / "tfjs_validate" / "node_modules"
    if not node_modules.exists():
        raise RuntimeError(f"{node_modules} not found. Run `npm install` in {validate_js.parent} first.")
    logger.info("Validating exported tf.js model with a real tfjs-node round-trip (predict + one fit step) ...")
    result = subprocess.run(
        ["node", str(validate_js), str(model_json_path), str(payload_path)],
        cwd=str(validate_js.parent),
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("tf.js round-trip validation FAILED -- see node output above.")
    logger.info("tf.js round-trip validation PASSED.")


def main():
    MODELDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {TRAINSET} / {TESTSET}")
    X_train, y_train, feature_columns = load_dataset(TRAINSET)
    X_test, y_test, test_feature_columns = load_dataset(TESTSET)
    assert feature_columns == test_feature_columns, "train/test feature columns must match"
    input_dim = len(feature_columns)
    logger.info(f"{input_dim} features, {len(X_train)} train rows, {len(X_test)} test rows")


    tf.keras.utils.set_random_seed(SEED)
    logger.info(f"TensorFlow {tf.__version__}, tf.keras {getattr(tf.keras, '__version__', 'unknown')}")
    logger.info(f"GPUs visible: {tf.config.list_physical_devices('GPU')}")
    logger.info(f"AutoKeras {ak.__version__}")

    # Categorical layers (e.g. is_third_party, num_get_storage -- are 0/1/2-valued)
    # -> autokeras.keras_layers.MultiCategoryEncoding preprocessing layer
    # -> not supported by tfjs-layets
    # -> no clean replacement
    # == force everything to be numerical
    column_types = {c: "numerical" for c in feature_columns}

    ak_dir = OUTDIR / "autokeras_search"
    clf = ak.StructuredDataClassifier(
        max_trials=MAX_TRIALS,
        tuner="bayesian",   # Switches from the default task-specific tuner to Bayesian Optimization
        overwrite=True,
        directory=str(ak_dir),
        project_name="adflush_ak",
        seed=SEED,
        column_names=feature_columns,
        column_types=column_types,
    )

    logger.info(
        f"Running StructuredDataClassifier.fit (max_trials={MAX_TRIALS}, "
        f"per-trial epoch cap={SEARCH_EPOCHS}, {len(X_train)} rows, "
        f"all columns forced 'numerical') ..."
    )
    X_train_df = pd.DataFrame(X_train, columns=feature_columns)

    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    # )

    clf.fit(
        X_train_df,
        y_train, 
        # epochs=SEARCH_EPOCHS,
        # callbacks=[early_stopping],
        verbose=2,
    )

    model = clf.export_model()
    logger.info(f"Exported model type: {type(model)}")

    model.summary(print_fn=logger.info)

    evaluate_model(model, X_test, y_test)

    logger.info("Layer types found in the AutoKeras-exported model:")
    for layer in model.layers:
        logger.info(f"  {layer.name}: {type(layer).__module__}.{type(layer).__name__}")

    tfjs_model, replaced = replace_normalization_with_rescaling(model)
    if replaced:
        logger.info("Rebuilt model with Normalization -> Rescaling swap for tfjs-layers compatibility.")
    else:
        logger.info("No Normalization layer found; exporting AutoKeras's model unchanged.")
    tfjs_model.summary(print_fn=logger.info)

    # Sanity-check the rebuilt model is numerically identical to the
    # original AutoKeras export before trusting its predictions below.
    probe = X_test[:32].astype("float32")
    orig_probe_pred = np.asarray(model.predict(probe, verbose=0)).reshape(-1)
    new_probe_pred = np.asarray(tfjs_model.predict(probe, verbose=0)).reshape(-1)
    max_rebuild_diff = float(np.max(np.abs(orig_probe_pred - new_probe_pred)))
    logger.info(f"Max |original_pred - rebuilt_pred| on a 32-row probe: {max_rebuild_diff}")
    if max_rebuild_diff > 1e-4:
        raise RuntimeError(
            f"Rebuilt model diverges from AutoKeras's original export (max diff {max_rebuild_diff}); "
            "the Normalization->Rescaling swap is not equivalent -- aborting before export."
        )

    keras_path = MODELDIR / "AdFlush_autokeras.keras"
    tfjs_model.save(str(keras_path))
    logger.info(f"Saved native Keras model to {keras_path}")

    tfjs_dir = MODELDIR / "AdFlush_autokeras_tfjs"
    tfjs_dir.mkdir(parents=True, exist_ok=True)

    tfjs.converters.save_keras_model(tfjs_model, str(tfjs_dir))
    logger.info(f"Exported tf.js LayersModel to {tfjs_dir}")

    with open(tfjs_dir / "feature_order.json", "w") as f:
        json.dump({"feature_order": feature_columns}, f, indent=2)

    rng = np.random.RandomState(SEED)
    sample_size = min(VALIDATION_SAMPLE_SIZE, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]
    sample_preds = np.asarray(tfjs_model.predict(X_sample, batch_size=512, verbose=0)).reshape(-1)

    payload_path = tfjs_dir / "validation_samples.json"
    with open(payload_path, "w") as f:
        json.dump(
            {
                "feature_order": feature_columns,
                "samples": X_sample.tolist(),
                "labels": y_sample.astype(int).tolist(),
                "expected_predictions": sample_preds.tolist(),
            },
            f,
        )
    logger.info(f"Wrote validation payload to {payload_path}")

    run_tfjs_validation(tfjs_dir / "model.json", payload_path)

    logger.info("Done.")
    logger.info(f"  Keras model:  {keras_path}")
    logger.info(f"  tf.js model:  {tfjs_dir}/model.json (+ weight shard(s))")


if __name__ == "__main__":
    main()
