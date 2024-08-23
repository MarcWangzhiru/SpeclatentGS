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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from typing import NamedTuple
# from diff_gaussian_rasterization import GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import compute_gaussian_normals, flip_align_view
import torch.nn.functional as F
import time


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    scale_factor = [1]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    rasterizers = []
    for scale in scale_factor:
        raster_setting = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height/scale),
        image_width=int(viewpoint_camera.image_width/scale),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        # num_sem_classes=pc.num_sem_classes,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_setting)
        rasterizers.append(rasterizer)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # xyz = pc.get_xyz
    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
    dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    color_features = pc.get_semantic

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    normals = pc.get_normal

    direction_features = pc.mlp_direction_head(
        torch.cat([pc.direction_encoding(dir_pp), normals], dim=-1)).float()

    Ln = 0 

    semantic = torch.cat([color_features, direction_features, normals], dim=-1)

    rendered_features = []
    rendered_depths = []
    visibility_filter = 0
    radii = 0

    for i in range(1):
        semantic_logits, rendered_depth, rendered_alpha, radii = rasterizers[i](
            means3D = means3D,
            means2D = means2D,
            # shs = shs,
            # colors_precomp = colors_precomp,
            semantic = semantic,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        rendered_features.append(semantic_logits.unsqueeze(0))
        rendered_depths.append(rendered_depth.unsqueeze(0))
        if i == 0:
            visibility_filter = radii > 0
            radii = radii
    # rendered_image: [3, 376, 1408]  [3, image_height, image_width]
    # radii: [10458]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rendered_features, screenspace_points, visibility_filter, radii, rendered_depths, Ln


