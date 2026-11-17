import unittest
import numpy as np
import torch
from crufnet.model import CRUFNet, blend_normals
from crufnet.losses import training_loss, utility_targets
from crufnet.metrics import angular_metrics, tiled_predict
from crufnet.settings import read_splits


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_native_shapes_and_joint_backward(self):
        torch.manual_seed(3)
        model = CRUFNet()
        batch = torch.rand(2, 12, 64, 96)
        target = torch.nn.functional.normalize(torch.randn(2, 3, 64, 96), dim=1)
        mask = torch.ones(2, 1, 64, 96)
        outputs = model(batch)
        loss, terms = training_loss(outputs, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('utility', terms)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad():
            outputs = model(torch.rand(1, 12, 65, 97))
        self.assertEqual(tuple(outputs['normal'].shape), (1, 3, 65, 97))
        self.assertTrue(torch.allclose(outputs['normal'].norm(dim=1), torch.ones(1, 65, 97), atol=1e-5))

    def test_fusion_endpoints_and_degenerate_blend(self):
        base = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
        candidate = -base
        self.assertTrue(torch.equal(blend_normals(base, candidate, torch.zeros(1, 1, 1, 1)), base))
        self.assertTrue(torch.equal(blend_normals(base, candidate, torch.ones(1, 1, 1, 1)), candidate))
        self.assertTrue(torch.equal(blend_normals(base, candidate, torch.full((1, 1, 1, 1), 0.5)), base))

    def test_utility_preference_sign_and_detachment(self):
        target = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1)
        base = torch.tensor([0.0, 1.0, 0.0]).reshape(1, 3, 1, 1).requires_grad_()
        candidate = target.clone().requires_grad_()
        preference, importance, gain = utility_targets(base, candidate, target)
        self.assertGreater(preference.item(), 0.99)
        self.assertEqual(importance.item(), 1.0)
        self.assertAlmostEqual(gain.item(), 90.0, places=4)
        self.assertFalse(preference.requires_grad)

    def test_masked_metric_and_tiled_prediction(self):
        class ConstantNormal(torch.nn.Module):
            def forward(self, x):
                normal = torch.zeros(x.shape[0], 3, *x.shape[-2:], device=x.device)
                normal[:, 2] = 1
                return {'normal': normal}
        result = tiled_predict(ConstantNormal().eval(), torch.zeros(12, 270, 380), torch.device('cpu'))
        self.assertEqual(tuple(result.shape), (3, 270, 380))
        target = np.zeros((3, 270, 380), dtype=np.float32)
        target[2] = 1
        mask = np.ones((270, 380), dtype=bool)
        metrics, angles = angular_metrics(result.numpy(), target, mask)
        self.assertAlmostEqual(metrics['mae_deg'], 0.0)
        self.assertAlmostEqual(metrics['acc_11_25_pct'], 100.0)
        self.assertEqual(len(angles), mask.size)

    def test_frozen_protocols(self):
        self.assertEqual(len(read_splits('mus')['train']), 570)
        splits = read_splits('deepsfp')
        self.assertEqual(len(splits['official_test21']), 21)
        self.assertEqual(set(splits['development_train']) | set(splits['development_val']), set(splits['official_train']))


if __name__ == '__main__':
    unittest.main()
