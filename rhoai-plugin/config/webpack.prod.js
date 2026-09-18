const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const path = require('path');

module.exports = merge(common, {
  mode: 'production',
  devtool: 'source-map',
  output: {
    path: path.resolve(__dirname, '../dist'),
    clean: true,
  },
  plugins: [
    new MiniCssExtractPlugin(),
  ],
  optimization: {
    runtimeChunk: false,
    splitChunks: false,
  },
});
