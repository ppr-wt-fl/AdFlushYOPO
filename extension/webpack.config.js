const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const { javascript } = require('webpack');

module.exports = {
  target: 'web',
  entry: {
    background: './background.js',
    popup: './popup.js',
    model: './model.js'
  },
  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, 'dist_chrome'),
    library: {
      type: 'module'  // ✅ Ensures Webpack outputs ES modules
    },
    chunkFormat: 'module'  // ✅ Ensures correct ES module format
  },
  experiments: {
    outputModule: true  // ✅ Allows Webpack to emit ES modules
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        type: 'javascript/esm',  // ✅ Ensures Webpack treats JS files as ES modules
      }
    ],
    parser:{
      javascript:{
        importMeta:false
      }
    }
  },
  plugins: [
    new CopyPlugin({
      patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
    })
  ],
  mode: 'production',
  resolve: {
    fallback: {
      fs: false,
      path: false
    }
  }
};
