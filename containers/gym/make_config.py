"""Write the runtime config for one Gym verifier: its resources_servers block from Gym's own
config (resources_servers/<name>/configs/*.yaml), bound to 0.0.0.0:8000, plus the global
runtime defaults — the config `gym env start` would pass the server via NEMO_GYM_CONFIG_DICT.

Usage: python make_config.py <gym_root> <name> <defaults.yaml> <out_dir>
"""

import sys
from pathlib import Path

import yaml

gym_root, name, defaults_path, out_dir = sys.argv[1:5]
block = None
for cfg in sorted((Path(gym_root) / "resources_servers" / name / "configs").glob("*.yaml")):
    doc = yaml.safe_load(cfg.read_text()) or {}
    for top in doc.values():
        servers = top.get("resources_servers", {}) if isinstance(top, dict) else {}
        if name in servers:
            block = dict(servers[name])
            break
    if block is not None:
        break
if block is None:
    raise SystemExit(f"make_config: no resources_servers.{name} block in Gym's configs")
block.update(host="0.0.0.0", port=8000)
config = {name: {"resources_servers": {name: block}}, **yaml.safe_load(open(defaults_path))}
Path(out_dir, f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
