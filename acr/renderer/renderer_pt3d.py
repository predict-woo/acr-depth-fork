# -*- coding: utf-8 -*-
# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys, os
import json
import torch
from torch import nn
import pickle

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)
import numpy as np

import acr.config

from acr.config import args

# from models import smpl_model

colors = {
    "pink": [0.7, 0.7, 0.9],
    "neutral": [0.9, 0.9, 0.8],
    "capsule": [0.7, 0.75, 0.5],
    "yellow": [0.5, 0.7, 0.75],
}


class Renderer(nn.Module):
    def __init__(
        self,
        resolution=(512, 512),
        perps=True,
        R=None,
        T=None,
        use_gpu="-1" not in str(args().GPUS),
    ):
        super(Renderer, self).__init__()
        self.perps = perps
        if use_gpu:
            self.device = torch.device("cuda:{}".format(str(args().GPUS).split(",")[0]))
            print("visualize in gpu mode")
        else:
            self.device = torch.device("cpu")
            print("visualize in cpu mode")

        if R is None:
            R_init = torch.Tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        else:
            R_init = R
        if T is None:
            T_init = torch.Tensor([[0.0, 0.0, 0.0]])
        else:
            T_init = T

        if self.perps:
            self.cameras = FoVPerspectiveCameras(
                R=R_init, T=T_init, fov=args().FOV, device=self.device
            )
            self.lights = PointLights(
                ambient_color=((0.56, 0.56, 0.56),),
                location=torch.Tensor([[0.0, 0.0, -1]]),
                device=self.device,
            )
        else:
            self.cameras = FoVOrthographicCameras(
                R=R_init,
                T=T_init,
                znear=0.0,
                zfar=100.0,
                max_y=1.0,
                min_y=-1.0,
                max_x=1.0,
                min_x=-1.0,
                device=self.device,
            )
            self.lights = DirectionalLights(
                direction=torch.Tensor([[0.0, 1.0, 0.0]]), device=self.device
            )

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0.
        raster_settings = RasterizationSettings(
            image_size=resolution[0], blur_radius=0.0, faces_per_pixel=1
        )

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=self.cameras, lights=self.lights
            ),
        )

    def __call__(
        self,
        verts,
        faces,
        colors=torch.Tensor(colors["neutral"]),
        merge_meshes=True,
        cam_params=None,
        **kwargs,
    ):
        assert len(verts.shape) == 3, print(
            "The input verts of visualizer is bounded to be 3-dims (Nx6890 x3) tensor"
        )
        verts, faces = verts.to(self.device), faces.to(self.device)
        verts_rgb = torch.ones_like(verts)
        verts_rgb[:, :] = torch.from_numpy(colors).cuda().unsqueeze(1)

        textures = TexturesVertex(verts_features=verts_rgb)
        verts[:, :, :2] *= -1
        # print('meshes: ', verts.shape, faces.shape, verts_rgb.shape, colors.shape, colors)
        meshes = Meshes(verts, faces, textures)
        if merge_meshes:
            meshes = join_meshes_as_scene(meshes)

        current_cameras = self.cameras
        if cam_params is not None:
            # raise ValueError # Not raising to allow use, but be careful if self.cameras is modified elsewhere implicitly
            if self.perps:
                R_param, T_param, fov_param = cam_params
                current_cameras = FoVPerspectiveCameras(
                    R=R_param, T=T_param, fov=fov_param, device=self.device
                )
            else:
                R_param, T_param, xyz_ranges_param = cam_params
                current_cameras = FoVOrthographicCameras(
                    R=R_param, T=T_param, **xyz_ranges_param, device=self.device
                )

        fragments = self.renderer.rasterizer(
            meshes_world=meshes, cameras=current_cameras
        )
        depth_map = fragments.zbuf[
            ..., 0
        ]  # Shape: (batch_size, image_size, image_size)
        # Replace -1 (empty/background) with a large value (far away) or 0 if you prefer
        # Using a large value for actual depth, 0 for no-hand pixels might be better for some applications
        depth_map = torch.where(
            depth_map < 0, torch.full_like(depth_map, 0.0), depth_map
        )

        images = self.renderer.shader(fragments, meshes, cameras=current_cameras)

        images[:, :, :, :-1] *= 255  # RGBA to RGB and scale

        return images, depth_map


def get_renderer(test=False, **kwargs):
    renderer = Renderer(**kwargs)
    if test:
        import cv2

        dist = 1 / np.tan(np.radians(args().FOV / 2.0))
        print("dist:", dist)
        model = pickle.load(
            open(
                os.path.join(args().smpl_model_path, "smpl", "SMPL_NEUTRAL.pkl"), "rb"
            ),
            encoding="latin1",
        )
        np_v_template = (
            torch.from_numpy(np.array(model["v_template"])).cuda().float()[None]
        )
        face = torch.from_numpy(model["f"].astype(np.int32)).cuda()[None]
        np_v_template = np_v_template.repeat(2, 1, 1)
        np_v_template[1] += 0.3
        np_v_template[:, :, 2] += dist
        face = face.repeat(2, 1, 1)
        # Update for new return type
        rendered_images, rendered_depth = renderer(np_v_template, face)
        result = rendered_images.cpu().numpy()
        depth_result = rendered_depth.cpu().numpy()

        for ri in range(len(result)):
            cv2.imwrite(
                "test_color_{}.png".format(ri),
                (result[ri, :, :, :3] * 255).astype(
                    np.uint8
                ),  # Use first 3 channels for color
            )
            # Save depth map (normalized for visualization)
            depth_img = depth_result[ri]
            if depth_img.max() > 0:
                depth_img_normalized = (depth_img / depth_img.max() * 255).astype(
                    np.uint8
                )
                cv2.imwrite("test_depth_{}.png".format(ri), depth_img_normalized)
            else:
                cv2.imwrite(
                    "test_depth_{}.png".format(ri),
                    np.zeros_like(depth_img, dtype=np.uint8),
                )

    return renderer
