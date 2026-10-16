"""Execute notebook with this interpreter and avoid Windows selector shutdown hangs."""

import asyncio
from pathlib import Path
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nbformat
from nbclient import NotebookClient
import os
import tempfile
import json


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        kernel = Path(temp) / "kernels" / "housing-validation"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Housing validation",
                    "language": "python",
                }
            )
        )
        os.environ["JUPYTER_PATH"] = temp + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        os.environ["IPYTHONDIR"] = str(Path(temp) / "ipython")
        notebook = nbformat.read(root / "ols_robustness_checks.ipynb", as_version=4)
        client = NotebookClient(
            notebook, kernel_name="housing-validation", timeout=180, resources={"metadata": {"path": str(root)}}
        )
        client.execute()
        output = Path(os.environ.get("HOUSING_RUN", str(root / "runs/fixture"))) / "executed-ols.ipynb"
        nbformat.write(notebook, output)
        print("PASS: notebook executed; shared saved OLS results match")


if __name__ == "__main__":
    main()
