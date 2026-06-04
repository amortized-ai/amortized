"""Export the current OpenAPI spec to openapi/v1.json."""

import json
import sys
from pathlib import Path

from amortized.main import app


def main() -> None:
    spec = app.openapi()
    out_dir = Path(__file__).resolve().parent.parent.parent / "openapi"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v1.json"
    out_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
