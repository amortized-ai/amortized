#!/bin/sh
# Start one NeMo Gym resources server the way `gym env start` would, minus the orchestrator:
# no head server, no Ray (v0.7.0's orchestrator always boots a Ray cluster, which OOMs a pod).
# GYM_SERVER names both configs/<name>.yaml and resources_servers/<name>/.
set -eu
: "${GYM_SERVER:=mcqa}"
cfg="/opt/amortized-gym/configs/${GYM_SERVER}.yaml"
dir="/opt/gym/resources_servers/${GYM_SERVER}"
[ -f "$cfg" ] || { echo "no verifier config: $cfg" >&2; exit 1; }
[ -x "$dir/.venv/bin/python" ] || { echo "no prebuilt venv for ${GYM_SERVER}: $dir/.venv" >&2; exit 1; }
cd "$dir"
NEMO_GYM_CONFIG_PATH="$GYM_SERVER" NEMO_GYM_CONFIG_DICT="$(cat "$cfg")" exec .venv/bin/python app.py
