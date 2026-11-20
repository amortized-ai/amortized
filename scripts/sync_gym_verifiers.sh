#!/bin/sh
# Copy the verifier manifest out of a built gym image into amortized's registry
# (src/amortized/gym_verifiers.json). Run after changing containers/gym/verifiers.txt:
#   docker build -t ghcr.io/amortized-ai/gym:dev containers/gym && scripts/sync_gym_verifiers.sh
set -eu
image="${1:-ghcr.io/amortized-ai/gym:dev}"
docker run --rm --entrypoint cat "$image" /opt/amortized-gym/verifiers.json \
    | python3 -c 'import json,sys; print(json.dumps(sorted(json.load(sys.stdin), key=lambda v: v["name"]), indent=1))' \
    > "$(dirname "$0")/../src/amortized/gym_verifiers.json"
echo "wrote src/amortized/gym_verifiers.json from $image"
