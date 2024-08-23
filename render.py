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
import numpy as np
from scene import Scene
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import imageio
from utils.general_utils import safe_state, calculate_reflection_direction, visualize_normal_map
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr, calculate_segmentation_metrics, evaluate
from mynet import MyNet, embedding_fn
from utils.graphics_utils import views_dir
import time


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    net_path = os.path.join(model_path, "model_ckpt{}pth".format(iteration))
    model = MyNet().to("cuda")
    net_weights = torch.load(net_path)
    model.load_state_dict(net_weights)

    model.eval()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask")
    f_path = os.path.join(model_path, name, "ours_{}".format(iteration), "f")
    highlight_path = os.path.join(model_path, name, "ours_{}".format(iteration), "highlight")
    color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "color")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    makedirs(mask_path, exist_ok=True)
    makedirs(f_path, exist_ok=True)
    makedirs(highlight_path, exist_ok=True)
    makedirs(color_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    all_time = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()

        render_pkg = render(view, gaussians, pipeline, background)

        rendered_features, viewspace_point_tensor, visibility_filter, radii, depths, _ = render_pkg

        views_emd = view.views_emd

        rendered_features[0] = torch.cat((rendered_features[0], views_emd), dim=1)
        image = model(*rendered_features)
        rendering = image['im_out'].squeeze(0)

        all_time.append(time.time() - start_time)

        gt = view.original_image[0:3, :, :]


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        #
        torchvision.utils.save_image(image['mask_out'].squeeze(0),
                                     os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(image['f_out'].squeeze(0), os.path.join(f_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(image['highlight_out'].squeeze(0),
                                     os.path.join(highlight_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(image['color_out'].squeeze(0),
                                     os.path.join(color_path, '{0:05d}'.format(idx) + ".png"))
    print("Average time per image: {}".format(sum(all_time) / len(all_time)))
    print("Render FPS: {}".format(1 / (sum(all_time) / len(all_time))))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.num_sem_classes)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for camera in scene.getTestCameras():
        #     print(camera.R)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                       background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
