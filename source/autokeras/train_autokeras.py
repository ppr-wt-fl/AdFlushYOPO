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
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import autokeras as ak

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_autokeras")

SOURCE_DIR = Path(__file__).resolve().parent
REPO_DIR = SOURCE_DIR.parent
DATADIR = REPO_DIR / "dataset"
MODELDIR = REPO_DIR / "model"
OUTDIR = REPO_DIR / "output"

TRAINSET = DATADIR / "AdFlush_train_sample.csv"
TESTSET = DATADIR / "AdFlush_test_sample.csv"

SEED = 42
VALIDATION_SAMPLE_SIZE = 256

# Feasibility check, not a real search: keep this small.
MAX_TRIALS = 3
SEARCH_EPOCHS = 8


def load_dataset(path):
    df = pd.read_csv(path, index_col=0)
    df.reset_index(inplace=True, drop=True)
    feature_columns = [c for c in df.columns if c != "label"]
    X = df[feature_columns].astype("float32").values
    y = df["label"].astype("float32").values
    return X, y, feature_columns


def replace_normalization_with_rescaling(model):

    replaced = False
    new_layers_by_old_name = {}
    inputs = model.inputs
    tensor_map = {}
    for inp in inputs:
        tensor_map[inp.ref()] = inp

    def get_input_tensors(layer):
        node = layer._inbound_nodes[0]
        in_tensors = node.input_tensors
        if not isinstance(in_tensors, list):
            in_tensors = [in_tensors]
        return [tensor_map[t.ref()] for t in in_tensors]

    keras_epsilon = tf.keras.backend.epsilon()  # default 1e-7; matches Normalization.call()'s own clamp exactly.

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        in_tensors = get_input_tensors(layer)
        call_arg = in_tensors[0] if len(in_tensors) == 1 else in_tensors

        if isinstance(layer, tf.keras.layers.Normalization):
            replaced = True
            weights = layer.get_weights()
            # tf.keras Normalization stores [mean, variance, count].
            mean, variance = weights[0], weights[1]
            # Exact match to Normalization.call(): (x - mean) / max(sqrt(variance), epsilon).
            std_safe = np.maximum(np.sqrt(variance), keras_epsilon)
            scale = (1.0 / std_safe).astype("float32").reshape(-1)
            offset = (-mean / std_safe).astype("float32").reshape(-1)
            new_layer = tf.keras.layers.Rescaling(
                scale=scale.tolist(), offset=offset.tolist(), name=layer.name + "_rescaling"
            )
            out = new_layer(call_arg)
            logger.info(f"  Replaced Normalization layer '{layer.name}' with Rescaling '{new_layer.name}'")
        elif type(layer).__name__ == "MultiCategoryEncoding":
            replaced = True
            # Not a linear op -- there's no faithful drop-in tfjs-layers
            # replacement for an arbitrary learned lookup table. Only safe
            # to drop because every column was declared "numerical" (see
            # main()), which the identity check right after this loop
            # confirms actually made it a pass-through on real data.
            out = call_arg
            logger.info(f"  Dropped MultiCategoryEncoding layer '{layer.name}' (asserted no-op below)")
        else:
            out = layer(call_arg)

        out_tensor = out if not isinstance(out, list) else out[0]
        old_out = layer.output if not isinstance(layer.output, list) else layer.output[0]
        tensor_map[old_out.ref()] = out_tensor

    old_outputs = model.outputs
    new_outputs = [tensor_map[t.ref()] for t in old_outputs]
    new_model = tf.keras.Model(inputs=inputs, outputs=new_outputs, name=model.name + "_tfjs_safe")
    return new_model, replaced


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
    keras_version = getattr(tf.keras, "__version__", None)
    if keras_version is None:
        try:
            import keras as _keras_pkg
            keras_version = _keras_pkg.__version__
        except Exception:
            keras_version = "unknown"
    logger.info(f"TensorFlow {tf.__version__}, tf.keras {keras_version}")
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
        max_trials=MAX_TRIALS,    # Push this as high as you can (default is 100)
        tuner="bayesian",   # Switches from the default task-specific tuner to Bayesian Optimization
        overwrite=True,
        directory=str(ak_dir),
        project_name="adflush_ak",
        seed=SEED,
        column_names=feature_columns,
        column_types=column_types,
    )
    logger.info(f"Running StructuredDataClassifier.fit (max_trials={MAX_TRIALS}, epochs={SEARCH_EPOCHS}, all columns forced 'numerical') ...")
    X_train_df = pd.DataFrame(X_train, columns=feature_columns)
    clf.fit(X_train_df, y_train, epochs=SEARCH_EPOCHS,  verbose=2)

    model = clf.export_model()
    logger.info(f"Exported model type: {type(model)}")
    model.summary(print_fn=logger.info)
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
