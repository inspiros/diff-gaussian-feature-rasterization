#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Sequence, NamedTuple, Tuple, Optional

import torch
import torch.nn as nn
from torch.types import Number

from . import _C

__all__ = [
    'rasterize_gaussians_features',
    'GaussianFeaturesRasterizationSettings',
    'GaussianFeaturesRasterizer',
]


def _cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def _get_background(bg, features: torch.Tensor) -> torch.Tensor:
    d = features.size(-1)
    if bg is None:
        return features.new_zeros([d])
    elif isinstance(bg, torch.Tensor):
        if bg.numel() == d:
            return bg.flatten()
        elif bg.numel() == 1:
            return features.new_full([d], bg.item())
        else:
            raise ValueError('background must have the same dimensions as features')
    elif isinstance(bg, Sequence):
        if len(bg) == d:
            return torch.tensor(bg, dtype=features.dtype, device=features.device)
        elif len(bg) == 1:
            return features.new_full([d], bg[0])
        else:
            raise ValueError('background must have the same dimensions as features')
    elif isinstance(bg, Number):
        return features.new_full([d], bg)
    else:
        raise ValueError('background must be a torch.Tensor, number, or sequence of numbers')


def rasterize_gaussians_features(
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3Ds_precomp: Optional[torch.Tensor],
        raster_settings: 'GaussianFeaturesRasterizationSettings') -> Tuple[torch.Tensor, torch.Tensor]:
    return _RasterizeGaussiansFeatures.apply(
        means3D,
        means2D,
        features,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussiansFeatures(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            means3D,
            means2D,
            features,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            _get_background(raster_settings.bg, features),
            means3D,
            features,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug,
        )
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = _cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, rendered_features, radii, geomBuffer, binningBuffer, imgBuffer = \
                    _C.rasterize_gaussians_features(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, rendered_features, radii, geomBuffer, binningBuffer, imgBuffer = \
                _C.rasterize_gaussians_features(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(features, means3D, scales, rotations, cov3Ds_precomp, radii,
                              geomBuffer, binningBuffer, imgBuffer)
        return rendered_features, radii

    @staticmethod
    def backward(ctx, grad_out, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (features, means3D, scales, rotations, cov3Ds_precomp, radii,
         geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            _get_background(raster_settings.bg, features),
            means3D,
            radii,
            features,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = _cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = \
                    _C.rasterize_gaussians_features_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_features, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = \
                _C.rasterize_gaussians_features_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_features,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )
        return grads


class GaussianFeaturesRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    prefiltered: bool
    debug: bool = False
    bg: Optional[torch.Tensor] = None


class GaussianFeaturesRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianFeaturesRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    @torch.no_grad()
    def mark_visible(self, positions: torch.Tensor):
        # Mark visible points (based on frustum culling for camera) with a boolean
        raster_settings = self.raster_settings
        visible = _C.mark_visible(
            positions,
            raster_settings.viewmatrix,
            raster_settings.projmatrix)
        return visible

    def forward(self,
                means3D: torch.Tensor,
                means2D: torch.Tensor,
                opacities: torch.Tensor,
                features: torch.Tensor,
                scales: Optional[torch.Tensor] = None,
                rotations: Optional[torch.Tensor] = None,
                cov3D_precomp: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raster_settings = self.raster_settings

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise ValueError('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if scales is None:
            scales = torch.tensor([])
        if rotations is None:
            rotations = torch.tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_features(
            means3D,
            means2D,
            features,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    @torch.no_grad()
    def visible_filter(self,
                       means3D: torch.Tensor,
                       scales: Optional[torch.Tensor] = None,
                       rotations: Optional[torch.Tensor] = None,
                       cov3D_precomp: Optional[torch.Tensor] = None) -> torch.Tensor:
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.tensor([])
        if rotations is None:
            rotations = torch.tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.tensor([])

        # Invoke C++/CUDA rasterization routine
        radii = _C.rasterize_gaussians_filter(means3D,
                                              scales,
                                              rotations,
                                              raster_settings.scale_modifier,
                                              cov3D_precomp,
                                              raster_settings.viewmatrix,
                                              raster_settings.projmatrix,
                                              raster_settings.tanfovx,
                                              raster_settings.tanfovy,
                                              raster_settings.image_height,
                                              raster_settings.image_width,
                                              raster_settings.prefiltered,
                                              raster_settings.debug)
        return radii
