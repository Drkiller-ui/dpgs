"""Import checks, CUDA tensor operation, and CLI entry points. No model download."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
import torch

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.4.1"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
assert (torch.ones(4, device="cuda") * 2).sum().item() == 8
for name in ["torchvision", "numpy", "open3d", "mmengine", "mmcv", "faiss",
             "asmk.hamming", "pytorch3d.transforms", "diff_plane_rasterization", "simple_knn._C"]:
    importlib.import_module(name)
    print("OK", name)
for directory, script in [(root, "train.py"), (root, "render_and_mesh_extract.py"),
                           (root / "Metric3D", "test.py"), (root / "mast3r", "run_mast3r.py")]:
    subprocess.run([sys.executable, script, "--help"], cwd=directory, check=True,
                   stdout=subprocess.DEVNULL)
    print("OK CLI", directory.name, script)
print("Environment checks passed; no scene has been trained yet.")
