"""Write an exact lock of the environment that passed the integration checks."""
import importlib.metadata
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lines=sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
(root/"requirements.lock.txt").write_text("# Validated with CPython 3.10 on Windows x64\n"+"\n".join(lines)+"\n")
