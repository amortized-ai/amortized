import {
  amortizedStudioAreaExtension,
  communityPluginsSectionExtension,
  amortizedStudioSectionExtension,
  userInfoNavExtension,
  clusterResourcesNavExtension,
  namespaceSummaryNavExtension,
  amortizedStudioRouteExtension,
  extensions,
} from '../extensions';

describe('RHOAI Plugin Extensions', () => {
  describe('amortizedStudioAreaExtension', () => {
    it('should have the correct type and id', () => {
      expect(amortizedStudioAreaExtension.type).toBe('app.area');
      expect(amortizedStudioAreaExtension.properties.id).toBe('amortized-studio');
    });

    it('should have an empty featureFlags array', () => {
      expect(amortizedStudioAreaExtension.properties.featureFlags).toEqual([]);
    });
  });

  describe('communityPluginsSectionExtension', () => {
    it('should define the community-plugins section', () => {
      expect(communityPluginsSectionExtension.type).toBe('app.navigation/section');
      expect(communityPluginsSectionExtension.properties.id).toBe('community-plugins');
      expect(communityPluginsSectionExtension.properties.title).toBe('Community plugins');
      expect(communityPluginsSectionExtension.properties.group).toBe('9_plugins');
    });

    it('should have an iconRef function', () => {
      expect(typeof communityPluginsSectionExtension.properties.iconRef).toBe('function');
    });
  });

  describe('amortizedStudioSectionExtension', () => {
    it('should define a subsection nested under community-plugins', () => {
      expect(amortizedStudioSectionExtension.type).toBe('app.navigation/section');
      expect(amortizedStudioSectionExtension.properties.id).toBe('amortized-studio');
      expect(amortizedStudioSectionExtension.properties.title).toBe('Amortized Studio');
      expect(amortizedStudioSectionExtension.properties.group).toBe('1_amortized_studio');
      expect(amortizedStudioSectionExtension.properties.section).toBe('community-plugins');
      expect(typeof amortizedStudioSectionExtension.properties.iconRef).toBe('function');
    });
  });

  describe('navigation extensions', () => {
    it('should define User Info nav item under amortized-studio section', () => {
      expect(userInfoNavExtension.type).toBe('app.navigation/href');
      expect(userInfoNavExtension.properties.id).toBe('amortized-studio-user-info');
      expect(userInfoNavExtension.properties.title).toBe('User Info');
      expect(userInfoNavExtension.properties.href).toBe('/amortized-studio/user-info');
      expect(userInfoNavExtension.properties.section).toBe('amortized-studio');
      expect(userInfoNavExtension.properties.path).toBe('/amortized-studio/user-info/*');
    });

    it('should define Cluster Resources nav item under amortized-studio section', () => {
      expect(clusterResourcesNavExtension.type).toBe('app.navigation/href');
      expect(clusterResourcesNavExtension.properties.id).toBe('amortized-studio-cluster-resources');
      expect(clusterResourcesNavExtension.properties.title).toBe('Cluster Resources');
      expect(clusterResourcesNavExtension.properties.href).toBe('/amortized-studio/cluster-resources');
      expect(clusterResourcesNavExtension.properties.section).toBe('amortized-studio');
      expect(clusterResourcesNavExtension.properties.path).toBe('/amortized-studio/cluster-resources/*');
    });

    it('should define Namespace Summary nav item under amortized-studio section', () => {
      expect(namespaceSummaryNavExtension.type).toBe('app.navigation/href');
      expect(namespaceSummaryNavExtension.properties.id).toBe('amortized-studio-namespace-summary');
      expect(namespaceSummaryNavExtension.properties.title).toBe('Namespace Summary');
      expect(namespaceSummaryNavExtension.properties.href).toBe('/amortized-studio/namespace-summary');
      expect(namespaceSummaryNavExtension.properties.section).toBe('amortized-studio');
      expect(namespaceSummaryNavExtension.properties.path).toBe('/amortized-studio/namespace-summary/*');
    });
  });

  describe('route extension', () => {
    it('should define a single wildcard route with lazy component', () => {
      expect(amortizedStudioRouteExtension.type).toBe('app.route');
      expect(amortizedStudioRouteExtension.properties.path).toBe('/amortized-studio/*');
      expect(typeof amortizedStudioRouteExtension.properties.component).toBe('function');
      expect(amortizedStudioRouteExtension.properties.component()).toBeInstanceOf(Promise);
    });
  });

  describe('extensions array', () => {
    it('should contain all seven extensions', () => {
      expect(extensions).toHaveLength(7);
    });

    it('should include all extensions in the correct order', () => {
      expect(extensions).toEqual([
        communityPluginsSectionExtension,
        amortizedStudioAreaExtension,
        amortizedStudioSectionExtension,
        userInfoNavExtension,
        clusterResourcesNavExtension,
        namespaceSummaryNavExtension,
        amortizedStudioRouteExtension,
      ]);
    });
  });
});
