"""MMCV 2.x moved collect_env to MMEngine. Idempotent compatibility fix."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "Metric3D/mono/utils/comm.py"
old = "from mmcv.utils import collect_env as collect_base_env"
new = "from mmengine.utils.dl_utils import collect_env as collect_base_env"
text = path.read_text(encoding="utf-8")
if old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patched {path}")
elif new in text:
    print("Metric3D compatibility fix already applied")
else:
    raise SystemExit("Unexpected upstream comm.py; inspect before patching")
