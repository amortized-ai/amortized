const { ModuleFederationPlugin } = require('webpack').container;
const path = require('path');

// Load environment variables
require('dotenv').config({ path: path.resolve(__dirname, '../.env.development') });

module.exports = {
  name: 'amortizedStudio', // [PLUGIN-SPECIFIC] must match package.json and plugin.yaml
  filename: 'remoteEntry.js',
  exposes: {
    './extensions': './src/rhoai/extensions.ts',
    './Icon': './src/app/components/AmortizedStudioNavIcon.tsx',
  },
  optimization: {
    runtimeChunk: false,
  },
  shared: {
    react: {
      singleton: true,
      requiredVersion: '^18',
    },
    'react-dom': {
      singleton: true,
      requiredVersion: '^18',
    },
    'react-router-dom': {
      singleton: true,
      requiredVersion: '^7',
    },
    '@patternfly/react-core': {
      singleton: true,
      requiredVersion: '^6',
    },
    '@openshift/dynamic-plugin-sdk': {
      singleton: true,
      requiredVersion: '^5',
    },
  },
};
