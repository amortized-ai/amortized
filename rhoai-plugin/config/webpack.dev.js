const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');
const path = require('path');

module.exports = merge(common, {
  mode: 'development',
  devtool: 'eval-source-map',
  devServer: {
    port: parseInt(process.env.PORT, 10) || 9500, // [PLUGIN-SPECIFIC] dev port
    historyApiFallback: true,
    hot: true,
    proxy: [
      {
        context: ['/amortized-studio/api'], // [PLUGIN-SPECIFIC] BFF proxy — must come before the general proxy
        target: 'http://localhost:3000',
        pathRewrite: { '^/amortized-studio/api': '/api' },
      },
      {
        context: ['/amortized-studio'], // [PLUGIN-SPECIFIC] must match route prefix
        target: 'http://localhost:8443',
        pathRewrite: { '^/amortized-studio': '/amortized-studio' },
      },
    ],
  },
  optimization: {
    runtimeChunk: false,
    splitChunks: false,
  },
});
