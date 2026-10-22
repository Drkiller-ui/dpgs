"""Minimal runtime/build guards for the pinned official MAtCha checkout."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1] / 'baselines/MAtCha'
for name in ['train.py', 'scripts/run_sfm.py', 'scripts/refine_free_gaussians.py', 'scripts/extract_tetra_mesh.py']:
    path = root / name
    code = path.read_text()
    if 'subprocess.run(' not in code or 'import subprocess' not in code:
        if 'import subprocess' not in code:
            code = 'import subprocess\n' + code
    # Preserve the official commands, but propagate child failures to the campaign.
    import re
    code = re.sub(r'(?m)^(\s*)os\.system\((\w+)\)$', r'\1subprocess.run(\2, shell=True, check=True)', code)
    path.write_text(code)

path = root / 'mast3r/dust3r/croco/models/curope/setup.py'
code = path.read_text()
old = "all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()"
new = "all_cuda_archs = []  # cpp_extension honors TORCH_CUDA_ARCH_LIST (8.9 on this host)"
if old in code:
    path.write_text(code.replace(old, new))

for name in ['2d-gaussian-splatting/train_with_charts.py', 'matcha/dm_scene/parallel_aligner.py']:
    path = root / name
    code = path.read_text()
    if 'Non-finite MAtCha training loss' not in code:
        code = code.replace('            loss.backward()', '            if not torch.isfinite(loss).all():\n                raise FloatingPointError("Non-finite MAtCha training loss")\n            loss.backward()')
        code = code.replace('        total_loss.backward()', '        if not torch.isfinite(total_loss).all():\n            raise FloatingPointError("Non-finite MAtCha training loss")\n        total_loss.backward()')
        path.write_text(code)

# Persist the same scheduled checkpoint before plotting training-report images.
path = root / '2d-gaussian-splatting/train_with_charts.py'
code = path.read_text()
report = '            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))'
save = '''            if (iteration in saving_iterations):
                print("\\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)'''
if report + '\n' + save in code:
    code = code.replace(report + '\n' + save, save + '\n' + report)
    path.write_text(code)

path=root/'matcha/dm_modules/matcher_3d.py'
code=path.read_text()
old='        self.reference_matches = depth_errors < matching_thr'
new='        self.reference_matches = (depth_errors < matching_thr) & (self.reference_depths[None] > 0)  # Missing COLMAP reference pixels are invalid'
if old in code:
    path.write_text(code.replace(old,new))

# COLMAP clouds may omit normals (DTU scan110). These placeholders are unused
# by GaussianModel.create_from_pcd; charts still supply the actual initialization.
path=root/'2d-gaussian-splatting/scene/dataset_readers.py'
code=path.read_text()
old="    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T"
new="    normals = (np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T\n               if all(name in vertices.data.dtype.names for name in ('nx', 'ny', 'nz'))\n               else np.zeros_like(positions))  # Optional COLMAP PLY normals"
if old in code:
    path.write_text(code.replace(old,new))

record = Path(__file__).resolve().parent / 'records/matcha_local.patch'
record.write_bytes(subprocess.check_output(['git', '-C', str(root), 'diff']))
