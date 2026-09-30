#!/usr/bin/env python3
"""Numerically verify released loss invariants (section 3.2), without training."""
import json
from pathlib import Path

import torch

from fast3r.dust3r.losses import ConfLossMultiviewV2, L21Loss, Regr3DMultiviewV4


def inputs(scale=1.0, perturb=0.0, confidence=2.0, requires_grad=False):
    torch.manual_seed(42)
    gts, preds = [], []
    for i in range(3):
        points = torch.rand(1, 3, 4, 3) + torch.tensor([0.1, 0.2, 1.0])
        pred = points * scale + perturb * torch.randn_like(points)
        conf = torch.full((1, 3, 4), confidence)
        gts.append({'pts3d': points, 'camera_pose': torch.eye(4)[None],
                    'valid_mask': torch.ones(1, 3, 4, dtype=torch.bool)})
        preds.append({'pts3d_in_other_view': pred.clone().requires_grad_(requires_grad),
                      'pts3d_local': pred.clone().requires_grad_(requires_grad),
                      'conf': conf.clone().requires_grad_(requires_grad),
                      'conf_local': conf.clone().requires_grad_(requires_grad)})
    return gts, preds


def main():
    alpha = 0.2
    criterion = ConfLossMultiviewV2(Regr3DMultiviewV4(L21Loss(), norm_mode='avg_dis'), alpha=alpha)
    exact, _ = criterion(*inputs())
    scaled, _ = criterion(*inputs(scale=7.0))
    expected = -alpha * torch.log(torch.tensor(2.0))
    assert torch.allclose(exact, expected, atol=1e-6), 'Perfect points reduce to the confidence regularizer'
    assert torch.allclose(scaled, exact, atol=1e-6), 'Independent normalization should remove uniform scale'
    noisy, _ = criterion(*inputs(perturb=0.05))
    assert noisy > exact, 'Spatial perturbation must increase regression loss'
    gts, preds = inputs(perturb=0.05, requires_grad=True)
    loss, _ = criterion(gts, preds)
    loss.backward()
    gradients = [p[key].grad for p in preds for key in p]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    result = {'paper_section': '3.2, Equations 1-3', 'status': 'passed',
              'scope': 'Synthetic CPU loss/backprop invariants only; no model training or dataset benchmark.',
              'seed': 42, 'views': 3, 'confidence_alpha': alpha,
              'exact_prediction_loss': float(exact), 'uniformly_scaled_prediction_loss': float(scaled),
              'perturbed_prediction_loss': float(noisy), 'finite_gradients': True,
              'confidence_sign': 'released implementation: conf * regression - alpha * log(conf)'}
    path = Path('results/loss_checks.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
