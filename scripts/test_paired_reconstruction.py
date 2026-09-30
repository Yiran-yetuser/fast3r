"""CPU mocked regression: one inference and identical GT/predictions for both heads."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import fast3r_hf_dtu_eval as runner


class PairedRunnerTests(unittest.TestCase):
    def test_same_forward_and_aggregate_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['breakfast_room', 'kitchen']:
                path = root / 'neural_rgbd' / name
                (path / 'images').mkdir(parents=True)
                (path / 'depth').mkdir()
                (path / 'poses.txt').touch()
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            args = argparse.Namespace(seed=42, dataset='nrgbd', kf_every=40,
                head_chunk_size=2, data_root=root, device='cpu', resolution=512,
                scenes=None, max_scenes=0, dry_run=False, checkpoint_dir=checkpoint,
                head='both', alignment_confidence_percentile=85, metric_confidence_percentile=0)

            class Dataset:
                def __init__(self, **kwargs): self.scene_list = ['breakfast_room', 'kitchen']
                def __getitem__(self, index):
                    return [{'img': torch.zeros(3, 4, 4), 'true_shape': np.array([4, 4]),
                             'label': self.scene_list[index] + '/0', 'dataset': 'nrgbd'}]

            class Model:
                def to(self, device): return self
                def eval(self): return self
                def set_max_parallel_views_for_head(self, chunk): pass

            calls = []
            class Evaluator:
                def __init__(self): self.reconstruction_metrics_per_epoch = {}
                def evaluate_reconstruction(self, views, preds, **kwargs):
                    calls.append((views, preds))
                    name = views[0]['label'][0].split('/')[0]
                    value = 1 if kwargs['use_pts3d_from_local_head'] else 2
                    self.reconstruction_metrics_per_epoch['nrgbd'] = {name: {'accuracy': value, 'accuracy_median': value}}

            evaluator = Evaluator()
            with patch.object(runner, 'NRGBD', Dataset), \
                 patch.object(runner.Fast3R, 'from_pretrained', return_value=Model()), \
                 patch.object(runner.MultiViewDUSt3RLitModule, 'load_for_inference', return_value=evaluator), \
                 patch.object(runner, 'inference', side_effect=lambda *a, **k: {'preds': [{'point': torch.ones(1)}]}) as inference:
                report = runner.evaluate(args)
            self.assertEqual(inference.call_count, 2)  # once per scene, not once per head
            for offset in [0, 2]:
                self.assertIs(calls[offset][0], calls[offset+1][0])
                self.assertIs(calls[offset][1], calls[offset+1][1])
            self.assertTrue(report['paired_same_forward'])
            self.assertEqual(report['aggregate_by_head']['local']['accuracy'], 1)
            self.assertEqual(report['aggregate_by_head']['global']['accuracy'], 2)
            self.assertEqual(report['aggregate_mean'], report['aggregate_by_head']['local'])
            self.assertEqual(report['scene_count'], 2)
            self.assertEqual(report['paper_distance_multiplier'], 100)


if __name__ == '__main__': unittest.main()
