import torch
from torch import nn
from .model import CRUFNet, blend_normals
from .settings import VARIANTS


class IdentityReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()

    def forward(self, features, descriptors, normal):
        return features


class FixedFusion(nn.Module):
    def __init__(self, acceptance):
        super().__init__()
        self.acceptance = acceptance
        self.training_strength = 1.0

    def forward(self, features, descriptors, base, candidate):
        strength = self.training_strength if self.training else 1.0
        acceptance = torch.full_like(base[:, :1], 1 - strength + strength * self.acceptance)
        return blend_normals(base, candidate, acceptance), acceptance, torch.zeros_like(acceptance)


def build_model(variant='full'):
    if variant not in VARIANTS:
        raise ValueError('Unknown model variant')
    model = CRUFNet()
    if variant in ('no_cru', 'baseline'):
        model.decoder.quarter_condition = IdentityReconstruction()
        model.decoder.half_condition = IdentityReconstruction()
    if variant in ('no_ugf', 'baseline', 'fixed_blend'):
        model.decoder.utility = FixedFusion(0.5 if variant == 'fixed_blend' else 1.0)
    return model
