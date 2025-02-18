const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  target:['web'],
  entry: {
    background:'./background.js',
    popup: './popup.js'
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'dist_firefox'),
    library: {
      type: 'umd'
    }
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: 'babel-loader',
      },
    ],
  },
  plugins: [new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
  })],
  mode: 'production'
};