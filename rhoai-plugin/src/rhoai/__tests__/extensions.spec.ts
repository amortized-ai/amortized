import {
  communityPluginsSectionExtension,
  amortizedStudioAreaExtension,
  amortizedStudioSectionExtension,
  studioNavExtension,
  amortizedStudioRouteExtension,
  extensions,
} from '../extensions';

describe('RHOAI Plugin Extensions', () => {
  it('defines the shared community-plugins section', () => {
    expect(communityPluginsSectionExtension.type).toBe('app.navigation/section');
    expect(communityPluginsSectionExtension.properties.id).toBe('community-plugins');
    expect(communityPluginsSectionExtension.properties.title).toBe('Community plugins');
    expect(communityPluginsSectionExtension.properties.group).toBe('9_plugins');
    expect(typeof communityPluginsSectionExtension.properties.iconRef).toBe('function');
  });

  it('registers the amortized-studio area', () => {
    expect(amortizedStudioAreaExtension.type).toBe('app.area');
    expect(amortizedStudioAreaExtension.properties.id).toBe('amortized-studio');
    expect(amortizedStudioAreaExtension.properties.featureFlags).toEqual([]);
  });

  it('nests the Amortized Studio section under community-plugins', () => {
    expect(amortizedStudioSectionExtension.type).toBe('app.navigation/section');
    expect(amortizedStudioSectionExtension.properties.id).toBe('amortized-studio');
    expect(amortizedStudioSectionExtension.properties.title).toBe('Amortized Studio');
    expect(amortizedStudioSectionExtension.properties.section).toBe('community-plugins');
    expect(typeof amortizedStudioSectionExtension.properties.iconRef).toBe('function');
  });

  it('exposes a single Studio nav item under the section', () => {
    expect(studioNavExtension.type).toBe('app.navigation/href');
    expect(studioNavExtension.properties.href).toBe('/amortized-studio');
    expect(studioNavExtension.properties.section).toBe('amortized-studio');
  });

  it('mounts the app at the /amortized-studio route', () => {
    expect(amortizedStudioRouteExtension.type).toBe('app.route');
    expect(amortizedStudioRouteExtension.properties.path).toBe('/amortized-studio/*');
    expect(typeof amortizedStudioRouteExtension.properties.component).toBe('function');
  });

  it('exports all extensions in the array', () => {
    expect(extensions).toEqual([
      communityPluginsSectionExtension,
      amortizedStudioAreaExtension,
      amortizedStudioSectionExtension,
      studioNavExtension,
      amortizedStudioRouteExtension,
    ]);
  });
});
