"""Explicit network acquisition; never runs during import or fixture checks."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import socket
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="New snapshot directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    status.write_text('{"state":"incomplete"}\n')
    try:
        from ucimlrepo import fetch_ucirepo

        socket.setdefaulttimeout(60)
        data = fetch_ucirepo(id=555)
        raw = pd.concat([data.data.features, data.data.targets], axis=1)
        raw.to_csv(args.out / "input.csv", index=False)
        metadata = {
            "uci_id": 555,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified",
            "rows": len(raw),
            "sha256": hashlib.sha256((args.out / "input.csv").read_bytes()).hexdigest(),
            "note": "Snapshot bytes, not a claim that upstream source will remain unchanged",
        }
        (args.out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status.write_text('{"state":"complete"}\n')
    except Exception:
        status.write_text('{"state":"failed"}\n')
        raise


if __name__ == "__main__":
    main()
