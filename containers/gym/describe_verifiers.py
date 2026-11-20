"""Describe one Gym verifier for the manifest: its description and the row fields its /verify
request declares beyond Gym's base request (the prompt and the model's response).

Run with the verifier's own venv python, from its server directory (the build does this for
each verifier in verifiers.txt). Prints one JSON object.

Usage: python describe_verifiers.py <name> <runtime_config.yaml>
"""

import importlib.util
import inspect
import json
import os
import sys

import yaml
from pydantic import BaseModel

name, config_path = sys.argv[1:3]
block = yaml.safe_load(open(config_path))[name]["resources_servers"][name]

# `python app.py` puts the server dir on sys.path; many servers import sibling modules.
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("verifier_app", "app.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The most-derived *VerifyRequest model defined or used by the server.
requests = [
    obj
    for _, obj in inspect.getmembers(module, inspect.isclass)
    if issubclass(obj, BaseModel) and obj.__name__.endswith("VerifyRequest")
]
requests.sort(key=lambda cls: len(cls.__mro__), reverse=True)
# Fields every Gym request has (prompt, response, rollout bookkeeping) — not row data.
from nemo_gym.base_resources_server import BaseVerifyRequest  # noqa: E402

base = set(BaseVerifyRequest.model_fields)
fields: list[str] = []
required: list[str] = []
if requests:
    model = requests[0]
    for field_name, info in model.model_fields.items():
        if field_name in base:
            continue
        fields.append(field_name)
        if info.is_required():
            required.append(field_name)

print(
    json.dumps(
        {
            "name": name,
            "description": str(block.get("description") or ""),
            "domain": str(block.get("domain") or ""),
            "fields": fields,
            "required": required,
        }
    )
)
