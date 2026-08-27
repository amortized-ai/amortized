// [SHARED] Common section for all community plugins — never changes across plugins.
// Do not change the id or name: all community plugins share this section
// so they appear grouped together in the dashboard sidebar.
export const communityPluginsSectionExtension = {
  type: 'app.navigation/section' as const,
  properties: {
    id: 'community-plugins', // [SHARED] common section for all community plugins
    title: 'Community plugins', // [SHARED]
    group: '9_plugins', // [SHARED]
    iconRef: () => import(/* webpackMode: "eager" */ './CommunityNavIcon'),
  },
};

// [PLUGIN-SPECIFIC] Everything below is specific to this plugin

export const amortizedStudioAreaExtension = {
  type: 'app.area' as const,
  properties: {
    id: 'amortized-studio', // [PLUGIN-SPECIFIC] unique area ID
    featureFlags: [] as string[],
  },
};

export const amortizedStudioSectionExtension = {
  type: 'app.navigation/section' as const,
  properties: {
    id: 'amortized-studio', // [PLUGIN-SPECIFIC] unique nav section ID
    title: 'Amortized Studio', // [PLUGIN-SPECIFIC] display name in sidebar
    group: '1_amortized_studio', // [PLUGIN-SPECIFIC] sort key within community-plugins
    section: 'community-plugins', // [SHARED] must match communityPluginsSectionExtension.id — do not change
    iconRef: () => import(/* webpackMode: "eager" */ '~/app/components/AmortizedStudioNavIcon'),
  },
};

// Single entry — the embedded studio owns all navigation below this route.
export const studioNavExtension = {
  type: 'app.navigation/href' as const,
  properties: {
    id: 'amortized-studio-app', // [PLUGIN-SPECIFIC] unique nav item ID
    title: 'Studio',
    href: '/amortized-studio', // [PLUGIN-SPECIFIC] must match route prefix
    section: 'amortized-studio', // [PLUGIN-SPECIFIC] references this plugin's section ID
    path: '/amortized-studio/*', // [PLUGIN-SPECIFIC] route-matching pattern
  },
};

export const amortizedStudioRouteExtension = {
  type: 'app.route' as const,
  properties: {
    path: '/amortized-studio/*', // [PLUGIN-SPECIFIC] top-level route prefix
    component: () => import(/* webpackMode: "eager" */ '~/app/App'),
  },
};

export const extensions = [
  communityPluginsSectionExtension,
  amortizedStudioAreaExtension,
  amortizedStudioSectionExtension,
  studioNavExtension,
  amortizedStudioRouteExtension,
];

export default extensions;
