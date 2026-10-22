"""Regression for an empty propagated mask and unchanged nonempty-mask gradients."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from utils.loss_utils import binary_masked_l1_mean


def main():
    records = []
    for device in ['cpu', 'cuda']:
        for dtype in [torch.float32, torch.float64]:
            prediction = torch.tensor([1., 4., 7.], device=device, dtype=dtype, requires_grad=True)
            target = torch.tensor([3., 9., 3.], device=device, dtype=dtype)
            empty = torch.zeros_like(prediction)
            original = (torch.abs(prediction - target) * empty).sum() / empty.sum()
            assert not torch.isfinite(original)
            fixed = binary_masked_l1_mean(prediction, target, empty)
            gradient, = torch.autograd.grad(fixed, prediction)
            assert fixed.item() == 0 and torch.equal(gradient, torch.zeros_like(prediction))
            for mask in [torch.tensor([1., 0., 1.], device=device, dtype=dtype), torch.ones_like(prediction)]:
                original = (torch.abs(prediction - target) * mask).sum() / mask.sum()
                fixed = binary_masked_l1_mean(prediction, target, mask)
                original_grad, = torch.autograd.grad(original, prediction)
                fixed_grad, = torch.autograd.grad(fixed, prediction)
                assert torch.equal(original, fixed) and torch.equal(original_grad, fixed_grad)
            assert binary_masked_l1_mean(prediction, target, torch.tensor([1., 0., 1.], device=device, dtype=dtype)).item() == 3
            records.append({'device': device, 'dtype': str(dtype), 'empty_loss_and_gradient_zero': True,
                            'nonempty_loss_and_gradient_exactly_unchanged': True})
    report = {'checks': records, 'reason': 'Original scan83 total loss was NaN at iteration 4005; reject invalid model',
              'protocol': 'Binary mask count is an integer; clamp only changes the zero-support case'}
    (ROOT / 'repro/records/masked_depth_loss_regression.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
