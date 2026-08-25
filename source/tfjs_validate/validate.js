// Round-trip validation for the tf.js model exported by train_archai.py.
//
// This is the actual proof behind "the conversion will go over smoothly":
// it loads the exported model.json with a real tfjs runtime (tfjs-node,
// same JS LayersModel loader the browser extension will use), checks that
// predictions on real validation rows match the Python-side predictions
// within a small numeric tolerance, and performs one model.fit() step to
// confirm the exported model is actually trainable client-side -- not just
// loadable. If either check fails, this process exits non-zero and the
// calling Python script treats that as a hard failure, not a warning.
//
// Usage: node validate.js <model.json path> <validation_samples.json path>

const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs-node");

const PREDICTION_TOLERANCE = 1e-3;

async function main() {
  const [, , modelJsonPath, payloadPath] = process.argv;
  if (!modelJsonPath || !payloadPath) {
    console.error(
      "Usage: node validate.js <model.json path> <validation_samples.json path>",
    );
    process.exit(2);
  }

  const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
  const {
    feature_order: featureOrder,
    samples,
    labels,
    expected_predictions: expectedPredictions,
  } = payload;

  console.log(`Loading model from ${modelJsonPath} ...`);
  const modelUrl = "file://" + path.resolve(modelJsonPath);
  const model = await tf.loadLayersModel(modelUrl);
  model.summary();

  const numFeatures = samples[0].length;
  if (numFeatures !== featureOrder.length) {
    throw new Error(
      `Sample width ${numFeatures} != feature_order length ${featureOrder.length}`,
    );
  }
  const inputShape = model.inputs[0].shape;
  if (inputShape[inputShape.length - 1] !== numFeatures) {
    throw new Error(
      `Model expects ${inputShape[inputShape.length - 1]} input features, validation samples have ${numFeatures}.`,
    );
  }

  // --- 1. Forward-pass parity with the Python-side predictions ---
  console.log(
    `Checking predictions on ${samples.length} real validation rows ...`,
  );
  const xs = tf.tensor2d(samples);
  const predTensor = model.predict(xs);
  const predictions = Array.from(await predTensor.data());

  let maxAbsDiff = 0;
  for (let i = 0; i < predictions.length; i++) {
    const diff = Math.abs(predictions[i] - expectedPredictions[i]);
    if (diff > maxAbsDiff) maxAbsDiff = diff;
  }
  console.log(
    `Max |tfjs_pred - python_pred| over ${predictions.length} samples: ${maxAbsDiff}`,
  );
  if (maxAbsDiff > PREDICTION_TOLERANCE) {
    throw new Error(
      `Prediction mismatch too large: ${maxAbsDiff} > tolerance ${PREDICTION_TOLERANCE}. ` +
        "The exported tf.js model is not numerically equivalent to the Python model.",
    );
  }
  console.log("Prediction parity check: PASS");

  // --- 2. Trainability: the whole point of shipping a LayersModel ---
  console.log(
    "Checking the exported model can be compiled and trained (one fit() step) ...",
  );
  model.compile({
    optimizer: tf.train.adam(1e-3),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  const ys = tf.tensor2d(labels, [labels.length, 1]);
  const history = await model.fit(xs, ys, {
    epochs: 1,
    batchSize: Math.min(64, samples.length),
    verbose: 0,
  });
  const finalLoss = history.history.loss[history.history.loss.length - 1];
  if (!Number.isFinite(finalLoss)) {
    throw new Error(`Training step produced a non-finite loss (${finalLoss}).`);
  }
  console.log(`Trainability check: PASS (one-step fit loss=${finalLoss})`);

  tf.dispose([xs, ys, predTensor]);
  console.log(
    "ALL CHECKS PASSED: exported tf.js model is loadable, numerically correct, and trainable.",
  );
}

main().catch((err) => {
  console.error("VALIDATION FAILED:", err);
  process.exit(1);
});
