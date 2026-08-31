#!/usr/bin/env node
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..');

// --- Name conversion helpers ---

function toKebab(displayName) {
  return displayName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function toCamel(kebab) {
  return kebab.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());
}

function toPascal(kebab) {
  const camel = toCamel(kebab);
  return camel.charAt(0).toUpperCase() + camel.slice(1);
}

function toSnake(kebab) {
  return kebab.replace(/-/g, '_');
}

function toTitle(kebab) {
  return kebab
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function toInitials(displayName) {
  const words = displayName.trim().split(/\s+/);
  if (words.length === 1) {
    return words[0].charAt(0).toUpperCase();
  }
  return (words[0].charAt(0) + words[1].charAt(0)).toUpperCase();
}

const ICON_COLORS = [
  '#0066cc', // blue
  '#3e8635', // green
  '#c9190b', // red
  '#6b21a8', // purple (original)
  '#ec7a08', // orange
  '#009596', // teal
  '#7d1007', // dark red
  '#1f0066', // indigo
];

function pickIconColor(kebab) {
  let hash = 0;
  for (let i = 0; i < kebab.length; i++) {
    hash = (hash * 31 + kebab.charCodeAt(i)) | 0;
  }
  return ICON_COLORS[Math.abs(hash) % ICON_COLORS.length];
}

// --- CLI argument parsing ---

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dry-run') {
      args.dryRun = true;
    } else if (arg === '--yes' || arg === '-y') {
      args.yes = true;
    } else if (arg.startsWith('--') && i + 1 < argv.length) {
      const key = arg.slice(2);
      args[key] = argv[++i];
    }
  }
  return args;
}

// --- Interactive prompting ---

function createPrompt() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr,
    terminal: process.stdin.isTTY !== false,
  });
  return {
    ask(question, defaultValue) {
      const suffix = defaultValue ? ` [${defaultValue}]` : '';
      return new Promise((resolve) => {
        rl.question(`${question}${suffix}: `, (answer) => {
          resolve(answer.trim() || defaultValue || '');
        });
      });
    },
    confirm(question) {
      return new Promise((resolve) => {
        rl.question(`${question} (y/N): `, (answer) => {
          resolve(answer.trim().toLowerCase() === 'y');
        });
      });
    },
    close() {
      rl.close();
    },
  };
}

// --- File operations ---

function replaceInFile(filePath, replacements, dryRun) {
  const absPath = path.resolve(ROOT, filePath);
  if (!fs.existsSync(absPath)) {
    return null;
  }
  const original = fs.readFileSync(absPath, 'utf8');
  let content = original;
  for (const [from, to] of replacements) {
    content = content.split(from).join(to);
  }
  if (content === original) {
    return null;
  }
  if (!dryRun) {
    fs.writeFileSync(absPath, content);
  }
  return filePath;
}

function renameFile(fromPath, toPath, dryRun) {
  const absFrom = path.resolve(ROOT, fromPath);
  const absTo = path.resolve(ROOT, toPath);
  if (!fs.existsSync(absFrom)) {
    return null;
  }
  if (!dryRun) {
    fs.renameSync(absFrom, absTo);
  }
  return { from: fromPath, to: toPath };
}

// --- Build the full change plan ---

function buildChangePlan(names) {
  const { kebab, camel, pascal, snake, display, port, initials, iconColor } = names;

  // Base replacements for identifiers (no package/chart-name patterns).
  // Order matters: longer/more-specific patterns first to avoid partial matches.
  const identifierReplacements = [
    ['AmortizedStudio', pascal],
    ['amortizedStudio', camel],
    ['amortized-studio', kebab],
    ['amortized_studio', snake],
    ['Amortized Studio', display],
  ];

  // package.json: match the JSON name field specifically so route paths get the plain kebab
  const npmReplacements = [
    ['"name": "amortized-studio"', `"name": "${kebab}"`],
    ...identifierReplacements,
  ];

  // BFF package.json uses "amortized-studio-bff" as name
  const bffNpmReplacements = [
    ['amortized-studio-bff', `${kebab}-bff`],
    ...identifierReplacements,
  ];

  const imageReplacements = [
    ['quay.io/rh-ai-community-plugins/amortized-studio-bff', `quay.io/OWNER/${kebab}-bff`],
    ['quay.io/rh-ai-community-plugins/amortized-studio', `quay.io/OWNER/${kebab}`],
    ['nameOverride: "amortized-studio"', `nameOverride: "${kebab}"`],
    ['cp-amortized-studio', `cp-${kebab}`],
    ['amortized-studio-bff', `${kebab}-bff`],
    ['amortized-studio', kebab],
  ];

  const portReplacements =
    port !== '9500' ? [['9500', port]] : [];

  // Icon-specific replacements (initials + color + component name)
  const iconReplacements = [
    ['HelloIcon', `${pascal}Icon`],
    ['      HW\n', `      ${initials}\n`],
    ['the HW text', `the ${initials} text`],
    ["'HW'", `'${initials}'`],
    ['#6b21a8', iconColor],
  ];

  // HTML title replacement
  const titleReplacements = [
    ['<title>Amortized Studio! - RHOAI</title>', `<title>${display} - RHOAI</title>`],
  ];

  // Welcome message replacement
  const welcomeReplacements = [
    ['Hello, welcome to your plugin example', `Welcome to ${display}`],
  ];

  // Prose de-seeding: remove seed/scaffold/demo framing from docs
  const proseReplacements = {
    'AGENTS.md': [
      [', each demonstrating a different integration pattern', ''],
      ['that demonstrates the BFF pattern', 'that implements the BFF pattern'],
      ['- `scripts/rename-plugin.js` — Interactive script to rename all plugin identifiers when forking this seed project into a new plugin. Prompts for a display name and updates all files.\n', ''],
      ['docs/archives/       — Project plan and historical documents\n', ''],
      ['- Plugin-specific identifiers are annotated with `[PLUGIN-SPECIFIC]` comments; shared conventions use `[SHARED]`. See `docs/development/CUSTOMIZATION.md` for the full reference.\n', ''],
    ],
    'README.md': [
      [' that serves as both a **reference implementation** and a **scaffold** for building your own plugins. It', '. It'],
      ['The plugin provides three pages, each demonstrating a different way to integrate with the dashboard and the cluster. These are the three patterns you will use when building your own plugin:', 'The plugin provides three pages, each using a different integration pattern:'],
      ['### Developing a New Plugin\n\nThis repository is designed as a **seed project**. To start developing your own plugin, **duplicate** the repo — do not fork it. Forking creates a link back to this upstream repository, which isn\'t what you want for an independent plugin with its own identity and lifecycle.\n\n```bash\ngit clone https://github.com/rh-ai-community-plugins/amortized-studio.git my-plugin\ncd my-plugin\nrm -rf .git\ngit init\n```\n\nThen follow the [Customization Guide](docs/development/CUSTOMIZATION.md) to rename identifiers, update routes, and make the plugin your own.\n\n', ''],
    ],
    'docs/architecture/BFF_PATTERN.md': [
      ['this reference plugin', 'this plugin'],
    ],
    'docs/development/DASHBOARD_APIS.md': [
      ['This reference plugin demonstrates three', 'This plugin uses three'],
      ['**Example page:**', '**See:**'],
    ],
    'docs/development/CUSTOMIZATION.md': [
      ["See the seed project's", 'See'],
      ['The seed project includes a ready-to-use GitHub Actions workflow', 'This project includes a GitHub Actions workflow'],
      ['The seed project includes one triggered via', 'This project includes one triggered via'],
    ],
    'docs/development/PROJECT_LAYOUT.md': [
      ['This is the directory structure every RHOAI community plugin should follow. Use it as a map when starting from this seed project.', 'Directory structure of the plugin.'],
      ['#     Example: dashboard API pattern', '#     Uses dashboard API pattern'],
      ['# Example: K8s API pass-through', '# Uses K8s API pass-through'],
      ['# Example: BFF pattern', '# Uses BFF pattern'],
      ['#   Data-fetching hooks for the example pages', '#   Data-fetching hooks'],
      ['## Where to start\n\n1. **Rename identifiers** — follow [CUSTOMIZATION.md](CUSTOMIZATION.md) to replace `amortized-studio` with your plugin name. Do this first.\n2. **Read** `src/rhoai/extensions.ts` — this is what the dashboard loads. It defines your nav items and routes.\n3. **Add pages** under `src/app/pages/` and corresponding nav entries in `extensions.ts`.\n4. **Add hooks** under `src/app/hooks/` for data fetching.', '## Codebase orientation\n\n1. **Read** `src/rhoai/extensions.ts` — this is what the dashboard loads. It defines your nav items and routes.\n2. **Add pages** under `src/app/pages/` and corresponding nav entries in `extensions.ts`.\n3. **Add hooks** under `src/app/hooks/` for data fetching.'],
    ],
    'CONTRIBUTING.md': [
      ['amortized-studio community plugin for Red Hat OpenShift AI Dashboard', `${display} community plugin for Red Hat OpenShift AI Dashboard`],
    ],
  };

  // Files grouped by category with their replacement sets
  const sourceFiles = [
    'config/webpack.common.js',
    'config/moduleFederation.js',
    'config/webpack.dev.js',
    '.env.development',
    'src/bootstrap.tsx',
    'src/rhoai/extensions.ts',
    'src/app/hooks/useNamespaceSummary.ts',
  ];

  const iconFiles = [
    'src/app/components/AmortizedStudioNavIcon.tsx',
    'src/app/components/__tests__/AmortizedStudioNavIcon.spec.tsx',
  ];

  const testFiles = [
    'src/rhoai/__tests__/extensions.spec.ts',
    'src/app/hooks/__tests__/useNamespaceSummary.spec.ts',
  ];

  const chartFiles = [
    'chart/Chart.yaml',
    'chart/values.yaml',
    'chart/templates/_helpers.tpl',
    'chart/templates/deployment.yaml',
    'chart/templates/service.yaml',
    'chart/templates/namespace.yaml',
    'chart/templates/bff-deployment.yaml',
    'chart/templates/bff-service.yaml',
    'chart/templates/serviceaccount.yaml',
  ];

  const ciFiles = [
    '.github/workflows/build-push.yml',
    'scripts/build-push.sh',
  ];

  const buildFiles = [
    'Makefile',
    'scripts/scan-image.sh',
  ];

  const docFiles = [
    'README.md',
    'AGENTS.md',
    'docs/architecture/BFF_PATTERN.md',
    'docs/development/CUSTOMIZATION.md',
    'docs/development/LOCAL_SETUP.md',
    'docs/development/DASHBOARD_APIS.md',
    'docs/development/PROJECT_LAYOUT.md',
    'docs/development/BUILD_AND_PUSH.md',
    'docs/deployment/OPENSHIFT_DEPLOY.md',
    'docs/archives/PROJECT_PLAN.md',
    'docs/archives/BUILD_PLAN.md',
    'CONTRIBUTING.md',
  ];

  const fileReplacements = [];

  // package.json: name field "amortized-studio" → "{kebab}"
  fileReplacements.push({
    file: 'package.json',
    replacements: [...npmReplacements, ...portReplacements],
  });

  // bff/package.json: "amortized-studio-bff" → "{kebab}-bff"
  fileReplacements.push({
    file: 'bff/package.json',
    replacements: [...bffNpmReplacements],
  });

  // plugin.yaml: image URL replacement (specific, not catch-all) + identifiers
  fileReplacements.push({
    file: 'plugin.yaml',
    replacements: [
      ['quay.io/rh-ai-community-plugins/amortized-studio', `quay.io/OWNER/${kebab}`],
      ['cp-amortized-studio', `cp-${kebab}`],
      ...identifierReplacements,
    ],
  });

  // Source + test files: identifier replacements + port
  for (const f of [...sourceFiles, ...testFiles]) {
    fileReplacements.push({
      file: f,
      replacements: [...identifierReplacements, ...portReplacements],
    });
  }

  // Icon files: identifier replacements + icon-specific (initials, color)
  for (const f of iconFiles) {
    fileReplacements.push({
      file: f,
      replacements: [...identifierReplacements, ...iconReplacements],
    });
  }

  // Chart files: image replacements + identifiers
  for (const f of chartFiles) {
    fileReplacements.push({
      file: f,
      replacements: [...imageReplacements, ...identifierReplacements],
    });
  }

  // CI files: image replacements first, then identifiers
  for (const f of ciFiles) {
    fileReplacements.push({
      file: f,
      replacements: [...imageReplacements, ...identifierReplacements],
    });
  }

  // Build files (Makefile, scan-image.sh): image + identifier replacements
  for (const f of buildFiles) {
    fileReplacements.push({
      file: f,
      replacements: [...imageReplacements, ...identifierReplacements],
    });
  }

  // HTML title in index.html
  fileReplacements.push({
    file: 'src/index.html',
    replacements: [...titleReplacements],
  });

  // Welcome message in UserInfoPage
  fileReplacements.push({
    file: 'src/app/pages/UserInfoPage.tsx',
    replacements: [...welcomeReplacements],
  });

  // Doc files: all replacements (images + npm name + identifiers + per-file prose)
  for (const f of docFiles) {
    const prose = proseReplacements[f] || [];
    fileReplacements.push({
      file: f,
      replacements: [...prose, ...imageReplacements, ...npmReplacements, ...identifierReplacements],
    });
  }

  // File renames (after content replacements)
  const fileRenames = [
    {
      from: `src/app/components/AmortizedStudioNavIcon.tsx`,
      to: `src/app/components/${pascal}NavIcon.tsx`,
    },
    {
      from: `src/app/components/__tests__/AmortizedStudioNavIcon.spec.tsx`,
      to: `src/app/components/__tests__/${pascal}NavIcon.spec.tsx`,
    },
  ];

  return { fileReplacements, fileRenames };
}

// --- Execute the plan ---

function executePlan(plan, dryRun) {
  const modified = [];
  const renamed = [];

  for (const { file, replacements } of plan.fileReplacements) {
    const result = replaceInFile(file, replacements, dryRun);
    if (result) {
      modified.push(result);
    }
  }

  for (const { from, to } of plan.fileRenames) {
    if (from === to) continue;
    const result = renameFile(from, to, dryRun);
    if (result) {
      renamed.push(result);
    }
  }

  return { modified, renamed };
}

// --- Seed artifact cleanup ---

function removeFileOrDir(filePath, dryRun) {
  const absPath = path.resolve(ROOT, filePath);
  if (!fs.existsSync(absPath)) return null;
  if (!dryRun) {
    const stat = fs.statSync(absPath);
    if (stat.isDirectory()) {
      fs.rmSync(absPath, { recursive: true });
    } else {
      fs.unlinkSync(absPath);
    }
  }
  return filePath;
}

function stripLineFromFile(filePath, lineContent, dryRun) {
  const absPath = path.resolve(ROOT, filePath);
  if (!fs.existsSync(absPath)) return false;
  const content = fs.readFileSync(absPath, 'utf8');
  const idx = content.indexOf(lineContent);
  if (idx === -1) return false;
  if (!dryRun) {
    fs.writeFileSync(absPath, content.split(lineContent).join(''));
  }
  return true;
}

function stripSectionFromFile(filePath, startMarker, endMarker, dryRun) {
  const absPath = path.resolve(ROOT, filePath);
  if (!fs.existsSync(absPath)) return false;
  const content = fs.readFileSync(absPath, 'utf8');
  const startIdx = content.indexOf(startMarker);
  if (startIdx === -1) return false;
  let endIdx;
  if (endMarker) {
    endIdx = content.indexOf(endMarker, startIdx + startMarker.length);
    if (endIdx === -1) endIdx = content.length;
  } else {
    endIdx = content.length;
  }
  if (!dryRun) {
    const before = content.slice(0, startIdx);
    const after = content.slice(endIdx);
    fs.writeFileSync(absPath, before + after);
  }
  return true;
}

function cleanupSeedArtifacts(dryRun) {
  const deleted = [];
  const stripped = [];

  // Delete seed-only files and directories
  const toDelete = [
    'scripts/rename-plugin.js',
    'docs/archives',
    'docs/architecture/COMMUNITY_PLUGINS.md',
    '.claude/skills/rename-plugin',
  ];
  for (const f of toDelete) {
    if (removeFileOrDir(f, dryRun)) deleted.push(f);
  }

  // Remove rename-plugin script from package.json
  if (stripLineFromFile('package.json', '    "rename-plugin": "node scripts/rename-plugin.js",\n', dryRun)) {
    stripped.push('package.json: removed rename-plugin script');
  }

  // Remove Archives section from docs/README.md
  if (stripSectionFromFile('docs/README.md', '\n### [Archives]', null, dryRun)) {
    stripped.push('docs/README.md: removed Archives section');
  }

  // Remove COMMUNITY_PLUGINS.md bullet from docs/architecture/README.md
  if (stripLineFromFile('docs/architecture/README.md', '- [COMMUNITY_PLUGINS.md](COMMUNITY_PLUGINS.md) -- Overview of existing community plugins (amortized-studio, kueue-visualizer) and key reference files in the dashboard codebase.\n', dryRun)) {
    stripped.push('docs/architecture/README.md: removed COMMUNITY_PLUGINS.md bullet');
  }

  // Strip "Automated Rename" section from CUSTOMIZATION.md
  if (stripSectionFromFile('docs/development/CUSTOMIZATION.md', '\n## Automated Rename', '\n## Manual Reference', dryRun)) {
    stripped.push('docs/development/CUSTOMIZATION.md: removed Automated Rename section');
  }

  // Remove "Start here when creating a new plugin" from dev/README.md
  const devReadmePath = path.resolve(ROOT, 'docs/development/README.md');
  if (fs.existsSync(devReadmePath)) {
    const content = fs.readFileSync(devReadmePath, 'utf8');
    const oldText = 'Start here when creating a new plugin. Lists';
    const newText = 'Lists';
    if (content.includes(oldText)) {
      if (!dryRun) {
        fs.writeFileSync(devReadmePath, content.split(oldText).join(newText));
      }
      stripped.push('docs/development/README.md: removed seed-framing text');
    }
  }

  // Rewrite CUSTOMIZATION.md intro
  const customizePath = path.resolve(ROOT, 'docs/development/CUSTOMIZATION.md');
  if (fs.existsSync(customizePath)) {
    const content = fs.readFileSync(customizePath, 'utf8');
    const oldIntro = 'When forking this repository to create your own community plugin, you need to produce a set of artifacts and replace all plugin-specific identifiers with values unique to your plugin.';
    const newIntro = "This guide documents the plugin's deliverables, naming conventions, and identifiers.";
    if (content.includes(oldIntro) && !dryRun) {
      fs.writeFileSync(customizePath, content.split(oldIntro).join(newIntro));
    }
    if (content.includes(oldIntro)) stripped.push('docs/development/CUSTOMIZATION.md: rewrote intro');
  }

  // Reset CHANGELOG.md
  const changelogPath = path.resolve(ROOT, 'CHANGELOG.md');
  if (fs.existsSync(changelogPath)) {
    const freshChangelog = `# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- Initial plugin created from amortized-studio seed project.
`;
    if (!dryRun) {
      fs.writeFileSync(changelogPath, freshChangelog);
    }
    stripped.push('CHANGELOG.md: reset to fresh template');
  }

  // Reset version to 0.1.0 — npm's "version" lifecycle hook runs
  // sync-chart-version.js which propagates to chart/Chart.yaml,
  // bff/package.json, plugin.yaml, and doc version flags.
  const RESET_VERSION = '0.1.0';
  const rootPkg = JSON.parse(fs.readFileSync(path.resolve(ROOT, 'package.json'), 'utf8'));
  if (rootPkg.version !== RESET_VERSION) {
    if (!dryRun) {
      execSync(`npm version ${RESET_VERSION} --no-git-tag-version`, { cwd: ROOT, stdio: 'ignore' });
    }
    stripped.push(`version reset to ${RESET_VERSION} (package.json + chart + plugin.yaml + docs)`);
  }

  return { deleted, stripped };
}

// --- Display helpers ---

function printSummary(names, plan, prefix = '') {
  console.log(`${prefix}Plugin identifiers:`);
  console.log(`${prefix}  Display name : ${names.display}`);
  console.log(`${prefix}  kebab-case   : ${names.kebab}`);
  console.log(`${prefix}  camelCase    : ${names.camel}`);
  console.log(`${prefix}  PascalCase   : ${names.pascal}`);
  console.log(`${prefix}  snake_case   : ${names.snake}`);
  console.log(`${prefix}  Icon initials: ${names.initials}`);
  console.log(`${prefix}  Icon color   : ${names.iconColor}`);
  console.log(`${prefix}  Dev port     : ${names.port}`);
  console.log('');

  const allFiles = plan.fileReplacements.map((r) => r.file);
  console.log(`${prefix}Files to modify (${allFiles.length}):`);
  for (const f of allFiles) {
    const exists = fs.existsSync(path.resolve(ROOT, f));
    console.log(`${prefix}  ${exists ? '  ' : '? '}${f}`);
  }

  if (plan.fileRenames.some((r) => r.from !== r.to)) {
    console.log('');
    console.log(`${prefix}Files to rename:`);
    for (const { from, to } of plan.fileRenames) {
      if (from !== to) {
        console.log(`${prefix}  ${from}`);
        console.log(`${prefix}    → ${to}`);
      }
    }
  }

  console.log('');
  console.log(`${prefix}Seed artifacts to remove:`);
  console.log(`${prefix}  scripts/rename-plugin.js`);
  console.log(`${prefix}  docs/archives/ (directory)`);
  console.log(`${prefix}  docs/architecture/COMMUNITY_PLUGINS.md`);
  console.log(`${prefix}  .claude/skills/rename-plugin/ (directory)`);
  console.log(`${prefix}  + strip seed sections from package.json, CHANGELOG.md,`);
  console.log(`${prefix}    CUSTOMIZATION.md, docs/README.md, docs/architecture/README.md`);
}

function printResults(results, dryRun) {
  const label = dryRun ? 'Would modify' : 'Modified';
  if (results.modified.length) {
    console.log(`\n${label} ${results.modified.length} file(s):`);
    for (const f of results.modified) {
      console.log(`  ${f}`);
    }
  }

  const renameLabel = dryRun ? 'Would rename' : 'Renamed';
  if (results.renamed.length) {
    console.log(`\n${renameLabel} ${results.renamed.length} file(s):`);
    for (const { from, to } of results.renamed) {
      console.log(`  ${from} → ${to}`);
    }
  }

  if (results.cleanup) {
    const deleteLabel = dryRun ? 'Would delete' : 'Deleted';
    if (results.cleanup.deleted.length) {
      console.log(`\n${deleteLabel} seed artifacts:`);
      for (const f of results.cleanup.deleted) {
        console.log(`  ${f}`);
      }
    }
    const stripLabel = dryRun ? 'Would strip' : 'Stripped';
    if (results.cleanup.stripped.length) {
      console.log(`\n${stripLabel} seed sections:`);
      for (const s of results.cleanup.stripped) {
        console.log(`  ${s}`);
      }
    }
  }

  if (!results.modified.length && !results.renamed.length) {
    console.log('\nNo changes needed.');
  }
}

function printNextSteps(names, dryRun) {
  const verb = dryRun ? 'will need to' : 'need to';
  console.log('');
  console.log('╔══════════════════════════════════════════════════════════════════╗');
  console.log('║  IMPORTANT — Manual steps required before running the plugin    ║');
  console.log('╚══════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  You ${verb} update the dashboard configuration to match the`);
  console.log('  new plugin identifiers. The plugin will NOT load without this.');
  console.log('');
  console.log('  1. Update MODULE_FEDERATION_CONFIG (env var or .env.local):');
  console.log('');
  console.log(`     "name": "${names.camel}",`);
  console.log('     ...');
  console.log('     "proxyService": [{');
  console.log(`       "path": "/${names.kebab}/api",`);
  console.log('       ...');
  console.log('     }]');
  console.log('');
  console.log('  2. Restart ALL services (dashboard, plugin dev server, BFF).');
  console.log('     A running dashboard caches the old federation config;');
  console.log('     it must be fully restarted to pick up the new names.');
  console.log('');
  console.log('  3. Update container image repositories:');
  console.log('     - Replace quay.io/OWNER/ placeholders with your registry org');
  console.log('       in chart/values.yaml and .github/workflows/build-push.yml');
  console.log('');
  console.log('  4. Update plugin.yaml remoteEntry URL to your deployed image URL.');
}

// --- Main ---

async function main() {
  const args = parseArgs(process.argv);
  const isInteractive = !args.name;

  let names;

  if (isInteractive) {
    const prompt = createPrompt();
    try {
      console.log('Rename plugin — generate new identifiers from a display name.\n');

      const displayName = await prompt.ask('Plugin display name (e.g. "My Cool Plugin")');
      if (!displayName) {
        console.error('Error: display name is required.');
        process.exit(1);
      }

      const defaultKebab = toKebab(displayName);
      const kebab = await prompt.ask('  kebab-case', defaultKebab);
      const camel = await prompt.ask('  camelCase', toCamel(kebab));
      const pascal = await prompt.ask('  PascalCase', toPascal(kebab));
      const snake = await prompt.ask('  snake_case', toSnake(kebab));
      const display = await prompt.ask('  Display name', toTitle(kebab));
      const initials = await prompt.ask('  Icon initials', toInitials(display));
      const iconColor = await prompt.ask('  Icon color', pickIconColor(kebab));
      const port = await prompt.ask('  Dev port', '9500');

      names = { kebab, camel, pascal, snake, display, initials, iconColor, port };
      const plan = buildChangePlan(names);

      console.log('');
      printSummary(names, plan);
      console.log('');

      const confirmed = await prompt.confirm('Apply these changes?');
      if (!confirmed) {
        console.log('Aborted.');
        process.exit(0);
      }

      const results = executePlan(plan, false);
      results.cleanup = cleanupSeedArtifacts(false);
      printResults(results, false);
      printNextSteps(names, false);
      console.log('\nRun `npm run validate` to verify the changes.');
    } finally {
      prompt.close();
    }
  } else {
    // Non-interactive mode
    const displayName = args.name;
    const kebab = args.kebab || toKebab(displayName);
    const camel = args.camel || toCamel(kebab);
    const pascal = args.pascal || toPascal(kebab);
    const snake = args.snake || toSnake(kebab);
    const display = args.display || toTitle(kebab);
    const initials = args.initials || toInitials(display);
    const iconColor = args['icon-color'] || pickIconColor(kebab);
    const port = args.port || '9500';

    names = { kebab, camel, pascal, snake, display, initials, iconColor, port };
    const plan = buildChangePlan(names);
    const dryRun = !!args.dryRun;

    if (!args.yes && !dryRun) {
      const prompt = createPrompt();
      try {
        printSummary(names, plan);
        console.log('');
        const confirmed = await prompt.confirm('Apply these changes?');
        if (!confirmed) {
          console.log('Aborted.');
          process.exit(0);
        }
      } finally {
        prompt.close();
      }
    } else if (dryRun) {
      printSummary(names, plan);
    }

    const results = executePlan(plan, dryRun);
    results.cleanup = cleanupSeedArtifacts(dryRun);
    printResults(results, dryRun);

    printNextSteps(names, dryRun);

    if (!dryRun) {
      console.log('\nRun `npm run validate` to verify the changes.');
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
