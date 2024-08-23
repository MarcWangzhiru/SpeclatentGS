# roup, https://team.inria.fr/graphdeco
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
from model import UNet, SimpleUNet, SimpleNet
from mynet import MyNet, embedding_fn
from utils.graphics_utils import views_dir
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    net_path = os.path.join(model_path, "model_ckpt{}pth".format(iteration))
    model = MyNet().to("cuda")
    net_weights = torch.load(net_path)
    # print(net_weights)
    model.load_state_dict(net_weights)

    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize();
        t0 = time.time()
        # color
        render_pkg = render(view, gaussians, pipeline, background)

        rendered_features, viewspace_point_tensor, visibility_filter, radii, depths, _ = render_pkg

        # 加入视角信息
        rays_d = torch.from_numpy(views_dir(view.image_height, view.image_width, view.K, view.c2w)).cuda()

        views_emd = embedding_fn(rays_d).permute(2, 0, 1).unsqueeze(0)

        rendered_features[0] = torch.cat((rendered_features[0], views_emd), dim=1)
        image = model(*rendered_features)
        rendering = image['im_out'].squeeze(0)

        torch.cuda.synchronize();
        t1 = time.time()

        t_list.append(t1 - t0)

    t = np.array(t_list[3:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.num_sem_classes)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)



        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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