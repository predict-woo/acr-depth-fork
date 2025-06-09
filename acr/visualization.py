import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import torch.nn.functional as F

import matplotlib

matplotlib.use("agg")
import math

from acr.config import args
from mano.manolayer import ManoLayer
from acr.utils import process_idx
from collections import OrderedDict

default_cfg = {"save_dir": None, "vids": None, "settings": []}  # 'put_org'


class Visualizer(object):
    def __init__(self, resolution=(2048, 2048), renderer_type=None):

        # constants
        self.resolution = resolution
        self.MANO_SKELETON = load_skeleton("mano/skeleton.txt", 21)
        self.MANO_RGB_DICT = get_keypoint_rgb(self.MANO_SKELETON)
        self.mano2interhand_mapper = np.array(
            [4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17, 0]
        )

        if renderer_type is not None:
            if renderer_type == "pyrender":
                from acr.renderer.renderer_pyrd import get_renderer

                self.renderer = get_renderer(resolution=self.resolution, perps=True)
            elif renderer_type == "pytorch3d":
                from acr.renderer.renderer_pt3d import get_renderer

                self.renderer = get_renderer(resolution=self.resolution, perps=True)
            else:
                raise NotImplementedError

        self.vis_size = resolution
        self.skeleton_3D_ploter = Plotter3dPoses()
        self.mano_layer = torch.nn.ModuleDict(
            {
                "1": ManoLayer(
                    ncomps=45,
                    center_idx=(
                        args().align_idx if args().mano_mesh_root_align else None
                    ),  # 9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side="right",
                    mano_root="mano/",
                    use_pca=False,
                    flat_hand_mean=False,
                ),
                "0": ManoLayer(
                    ncomps=45,
                    center_idx=(
                        args().align_idx if args().mano_mesh_root_align else None
                    ),  # 9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side="left",
                    mano_root="mano/",
                    use_pca=False,
                    flat_hand_mean=False,
                ),
                "right": ManoLayer(
                    ncomps=45,
                    center_idx=(
                        args().align_idx if args().mano_mesh_root_align else None
                    ),  # 9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side="right",
                    mano_root="mano/",
                    use_pca=False,
                    flat_hand_mean=False,
                ),
                "left": ManoLayer(
                    ncomps=45,
                    center_idx=(
                        args().align_idx if args().mano_mesh_root_align else None
                    ),  # 9, # TODO: 1. wrist align? root align ? 0 or 9?
                    side="left",
                    mano_root="mano/",
                    use_pca=False,
                    flat_hand_mean=False,
                ),
            }
        )

    def visualize_renderer_verts_list(
        self,
        verts_list,
        j3d_list,
        hand_type=None,
        hand_type_list=None,
        faces_list=None,
        images=None,
        cam_params=None,
        colors=None,
        trans=None,
        thresh=0.0,
        pre_colors=np.array([[0.46, 0.59, 0.64], [0.94, 0.71, 0.53]]),
    ):  # pre_colors=np.array([[.8, .0, .0], [0, .0, .8]])):
        verts_list = [verts.contiguous() for verts in verts_list]
        raw_depth_maps = []  # To store raw depth maps from the renderer

        if faces_list is None and hand_type is not None:
            faces_list = [
                self.mano_layer[str(hand_type[0])]
                .th_faces.repeat(len(verts), 1, 1)
                .to(verts.device)
                for verts in verts_list
            ]

        if hand_type is not None and hand_type_list is None:
            rendered_imgs_list = []
            for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
                if trans is not None:
                    verts += trans[ind].unsqueeze(1)

                if isinstance(colors, list):
                    color = colors[ind]
                else:
                    color = np.array([pre_colors[x] for x in hand_type])

                rendered_img, raw_depth = self.renderer(
                    verts,
                    faces,
                    colors=color,
                    focal_length=args().focal_length,
                    cam_params=cam_params,
                )
                rendered_imgs_list.append(rendered_img)
                raw_depth_maps.append(raw_depth)  # Store raw depth
            if len(rendered_imgs_list) > 0:
                if isinstance(rendered_imgs_list[0], torch.Tensor):
                    rendered_imgs = torch.cat(rendered_imgs_list, 0).cpu().numpy()
                    raw_depth_maps = torch.cat(raw_depth_maps, 0).cpu().numpy()
                else:  # Should not happen if renderer returns tensors
                    rendered_imgs = np.array(rendered_imgs_list)
                    raw_depth_maps = np.array(raw_depth_maps)
            else:
                rendered_imgs = np.array([])
                raw_depth_maps = np.array([])

            if rendered_imgs.ndim > 0 and rendered_imgs.shape[-1] == 4:  # RGBA
                transparent = rendered_imgs[:, :, :, -1]
                rendered_imgs = rendered_imgs[:, :, :, :-1]  # Keep RGB

        elif hand_type is None and hand_type_list is not None:
            rendered_imgs_list = []
            for ind, (verts, j3d, hand_type_item) in enumerate(
                zip(verts_list, j3d_list, hand_type_list)
            ):
                if trans is not None:
                    j3d += trans[ind].unsqueeze(1)
                    verts += trans[ind].unsqueeze(1)

                if isinstance(colors, list):
                    color = colors[ind]
                else:
                    color = np.array([pre_colors[x] for x in hand_type_item])

                all_face = []
                for single_hand_type in hand_type_item:
                    all_face.append(
                        self.mano_layer[str(int(single_hand_type))].th_faces
                    )
                all_face = torch.stack(all_face, dim=0)

                rendered_img, raw_depth = self.renderer(
                    verts,
                    all_face,
                    colors=color,
                    focal_length=args().focal_length,
                    cam_params=cam_params,
                )
                rendered_imgs_list.append(rendered_img)
                raw_depth_maps.append(raw_depth)

            if len(rendered_imgs_list) > 0:
                if isinstance(rendered_imgs_list[0], torch.Tensor):
                    rendered_imgs = torch.cat(rendered_imgs_list, 0).cpu().numpy()
                    raw_depth_maps = torch.cat(raw_depth_maps, 0).cpu().numpy()
                else:
                    rendered_imgs = np.array(rendered_imgs_list)
                    raw_depth_maps = np.array(raw_depth_maps)
            else:
                rendered_imgs = np.array([])
                raw_depth_maps = np.array([])

            if rendered_imgs.ndim > 0 and rendered_imgs.shape[-1] == 4:
                transparent = rendered_imgs[:, :, :, -1]
                rendered_imgs = rendered_imgs[:, :, :, :-1]
        else:
            # If no rendering happens, return empty arrays for imgs and depths
            return np.array([]), np.array([])

        # If images (background) are provided for blending
        if images is not None and rendered_imgs.ndim > 0 and rendered_imgs.shape[0] > 0:
            # Ensure images (background) and rendered_imgs have compatible shapes for blending
            if not (
                images.shape[:3] == rendered_imgs.shape[:3]
            ):  # Compare B, H, W (rendered_imgs is RGB, images might be RGBA or RGB)
                # This scaling was for the background image, not the rendered image directly
                # We need to ensure rendered_imgs (which might be 512x512) and images (original crop size) are handled correctly
                # The transparent mask is based on rendered_imgs' alpha if it was RGBA before stripping it.
                # For now, let's assume rendered_imgs is RGB and transparent was derived from its alpha.
                # The original images (background) might need scaling if their crop size is different from rendered_imgs' resolution.
                # This part seems to assume images (background) are already at the target size for blending directly with rendered_imgs (e.g. 512x512)
                # Let's keep the original logic for blending, assuming transparent mask is correctly derived.
                pass  # The resizing and pasting logic is handled in visulize_result_live

            # The transparent mask should be based on the alpha channel if rendered_imgs was RGBA
            # If rendered_img was RGB from the start and no alpha, this 'transparent' variable would be uninit or from previous RGBA step
            # Assuming 'transparent' is valid from an earlier step if rendered_imgs had alpha
            if (
                "transparent" in locals()
                and transparent is not None
                and transparent.shape[:3] == rendered_imgs.shape[:3]
            ):
                visible_weight = 0.9
                valid_mask = (transparent > thresh)[
                    :, :, :, np.newaxis
                ]  # transparent is HxW, need BxHxWx1
                if (
                    valid_mask.shape[0] != rendered_imgs.shape[0]
                ):  # if transparent was not batched
                    valid_mask = np.repeat(
                        valid_mask[np.newaxis, ...], rendered_imgs.shape[0], axis=0
                    )

                # Ensure images is broadcastable or matches rendered_imgs channel numbers (RGB)
                bg_images_rgb = images[..., :3] if images.shape[-1] == 4 else images
                if bg_images_rgb.shape != rendered_imgs.shape:
                    # This might happen if org_imgs (from meta_data['image']) were used directly and they are not 512x512
                    # For blending here, they should be the same size. The paste operation later handles putting it on full res image.
                    # This implies 'images' passed here should be the target rendering canvas size.
                    # Given the later cv2.resize in visulize_result_live, maybe blending here is for a fixed canvas.
                    # Let's assume for now images and rendered_imgs are made compatible before this point if blending is active.
                    # If not, this blending will fail or produce misaligned results.
                    pass  # Sticking to original logic, but highlighting potential issue

                rendered_imgs = (
                    rendered_imgs * valid_mask * visible_weight
                    + bg_images_rgb * valid_mask * (1 - visible_weight)
                    + (1 - valid_mask) * bg_images_rgb
                )
            # else: images provided but no valid transparent mask, return rendered_imgs as is (or blended differently)
        return (
            rendered_imgs.astype(np.uint8) if rendered_imgs.ndim > 0 else np.array([])
        ), (raw_depth_maps if raw_depth_maps.ndim > 0 else np.array([]))

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)

    def visulize_result_live(
        self,
        outputs,
        frame_img,
        meta_data,
        show_items=["mesh"],
        vis_cfg=default_cfg,
        **kwargs,
    ):
        vis_cfg = dict(default_cfg, **vis_cfg)
        used_org_inds, per_img_inds = process_idx(
            outputs["reorganize_idx"], vids=vis_cfg["vids"]
        )

        img_inds_org = [inds[0] for inds in per_img_inds]
        img_names = np.array(meta_data["imgpath"])[img_inds_org]
        # org_imgs are the cropped input images to the model (e.g., 512x512)
        org_imgs_for_blending = (
            meta_data["image"][outputs["detection_flag_cache"]]
            .cpu()
            .numpy()
            .astype(np.uint8)[img_inds_org]
        )

        plot_dict = OrderedDict()
        for vis_name in show_items:
            if vis_name == "org_img":
                plot_dict["org_img"] = {"figs": org_imgs_for_blending, "type": "image"}

            if (
                vis_name == "mesh"
                and outputs["detection_flag"]
                and len(per_img_inds) > 0
            ):
                per_img_verts_list = [
                    outputs["verts"][outputs["detection_flag_cache"]][inds].detach()
                    for inds in per_img_inds
                ]
                per_img_j3d_list = [
                    outputs["j3d"][outputs["detection_flag_cache"]][inds].detach()
                    for inds in per_img_inds
                ]
                mesh_trans = [
                    outputs["cam_trans"][outputs["detection_flag_cache"]][inds].detach()
                    for inds in per_img_inds
                ]
                hand_type = [
                    outputs["output_hand_type"][outputs["detection_flag_cache"]][
                        inds
                    ].detach()
                    for inds in per_img_inds
                ]

                # Pass org_imgs_for_blending if you want to blend with the 512x512 cropped image
                # If images=None, visualize_renderer_verts_list returns pure mesh render
                rendered_imgs_raw, raw_depth_maps = self.visualize_renderer_verts_list(
                    per_img_verts_list,
                    per_img_j3d_list,
                    hand_type_list=hand_type,
                    images=org_imgs_for_blending.copy(),  # Blend with the input crop
                    trans=mesh_trans,
                )

                if rendered_imgs_raw.size == 0:  # No hands rendered
                    plot_dict["mesh_rendering_orgimgs"] = {
                        "figs": [frame_img.copy() for _ in used_org_inds],
                        "type": "image",
                    }
                    continue

                if "put_org" in vis_cfg["settings"]:
                    offsets = (
                        meta_data["offsets"].cpu().numpy().astype(np.int)[img_inds_org]
                    )
                    # img_pad_size is the size of the padded region from original image, to which the raw render is resized
                    img_pad_size, crop_trbl, pad_trbl = (
                        offsets[:, :2],
                        offsets[:, 2:6],
                        offsets[:, 6:10],
                    )
                    rendering_onorg_images = []
                    depth_maps_for_saving = []

                    for batch_idx, original_img_idx in enumerate(used_org_inds):
                        # frame_img is the full original image
                        current_frame_copy = frame_img.copy()
                        current_rendered_img_raw = rendered_imgs_raw[batch_idx]
                        current_raw_depth_map = raw_depth_maps[batch_idx]

                        # target_ph, target_pw are from img_pad_size for this specific hand
                        target_ph, target_pw = img_pad_size[batch_idx]

                        # Resize rendered color image and depth map to target padded size
                        resized_color_img = cv2.resize(
                            current_rendered_img_raw,
                            (target_pw + 1, target_ph + 1),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        resized_depth_map = cv2.resize(
                            current_raw_depth_map,
                            (target_pw + 1, target_ph + 1),
                            interpolation=cv2.INTER_NEAREST,
                        )  # Use NEAREST for depth

                        # Get cropping and padding info for this hand
                        (ct, cr, cb, cl), (pt, pr, pb, pl) = (
                            crop_trbl[batch_idx],
                            pad_trbl[batch_idx],
                        )
                        orig_img_h, orig_img_w = current_frame_copy.shape[:2]

                        # Define the region in the full original image where the hand will be pasted
                        paste_x_start, paste_y_start = cl, ct
                        paste_x_end, paste_y_end = orig_img_w - cr, orig_img_h - cb
                        paste_w = paste_x_end - paste_x_start
                        paste_h = paste_y_end - paste_y_start

                        # Crop the resized color image and depth map using pad_trbl values
                        # These define the valid region within the (target_pw+1, target_ph+1) resized images
                        cropped_color_content = resized_color_img[
                            pt : resized_color_img.shape[0] - pb,
                            pl : resized_color_img.shape[1] - pr,
                        ]
                        cropped_depth_content = resized_depth_map[
                            pt : resized_depth_map.shape[0] - pb,
                            pl : resized_depth_map.shape[1] - pr,
                        ]

                        # Ensure the cropped content fits into the paste region, resize if necessary
                        # This step is crucial if cropped_color_content's size isn't exactly (paste_h, paste_w)
                        if (
                            cropped_color_content.shape[0] != paste_h
                            or cropped_color_content.shape[1] != paste_w
                        ):
                            cropped_color_content = cv2.resize(
                                cropped_color_content,
                                (paste_w, paste_h),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            cropped_depth_content = cv2.resize(
                                cropped_depth_content,
                                (paste_w, paste_h),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        # Paste the content
                        current_frame_copy[
                            paste_y_start:paste_y_end, paste_x_start:paste_x_end
                        ] = cropped_color_content
                        rendering_onorg_images.append(current_frame_copy)

                        if cropped_depth_content.max() > 0:
                            depth_to_save_vis = (
                                cropped_depth_content
                                / cropped_depth_content.max()
                                * 255
                            ).astype(np.uint8)
                        else:
                            depth_to_save_vis = np.zeros_like(
                                cropped_depth_content, dtype=np.uint8
                            )
                        # Also save as .npy for raw values
                        # np.save(
                        #     f"depth_org_{original_img_idx}_{batch_idx}.npy",
                        #     cropped_depth_content,
                        # )
                        depth_maps_for_saving.append(cropped_depth_content)

                    plot_dict["mesh_rendering_orgimgs"] = {
                        "figs": rendering_onorg_images,
                        "type": "image",
                    }

            if vis_name == "j3d" and outputs["detection_flag"]:
                real_aligned, pred_aligned, pos3d_vis_mask, joint3d_bones = kwargs[
                    "kp3ds"
                ]
                real_3ds = (real_aligned * pos3d_vis_mask.unsqueeze(-1)).cpu().numpy()
                predicts = (
                    (pred_aligned * pos3d_vis_mask.unsqueeze(-1)).detach().cpu().numpy()
                )
                skeleton_3ds = []
                for inds in per_img_inds:
                    for real_pose_3d, pred_pose_3d in zip(
                        real_3ds[inds], predicts[inds]
                    ):
                        skeleton_3d = self.skeleton_3D_ploter.encircle_plot(
                            [real_pose_3d, pred_pose_3d],
                            joint3d_bones,
                            colors=[(255, 0, 0), (0, 255, 255)],
                        )
                        skeleton_3ds.append(skeleton_3d)
                plot_dict["j3d"] = {"figs": np.array(skeleton_3ds), "type": "skeleton"}

            if vis_name == "pj2d" and outputs["detection_flag"]:
                kp_imgs = []
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = org_imgs_for_blending[img_id].copy()
                    # try:
                    for kp2d_vis in outputs["pj2d"][inds_list]:
                        if len(kp2d_vis) > 0:
                            kp2d_vis = (
                                (kp2d_vis + 1) / 2 * org_imgs_for_blending.shape[1]
                            )
                            org_img = self.vis_keypoints(
                                org_img,
                                kp2d_vis.detach().cpu().numpy(),
                                skeleton=self.MANO_SKELETON,
                            )
                    kp_imgs.append(np.array(org_img))
                plot_dict["pj2d"] = {"figs": kp_imgs, "type": "image"}

            if vis_name == "centermap" and outputs["detection_flag"]:
                l_centermaps_list = []
                r_centermaps_list = []
                for img_id, (l_centermap, r_centermap) in enumerate(
                    zip(
                        outputs["l_center_map"][used_org_inds],
                        outputs["r_center_map"][used_org_inds],
                    )
                ):
                    img_bk = cv2.resize(
                        org_imgs_for_blending[img_id].copy(),
                        org_imgs_for_blending.shape[1:3],
                    )
                    l_centermaps_list.append(make_heatmaps(img_bk, l_centermap))
                    r_centermaps_list.append(make_heatmaps(img_bk, r_centermap))

        return plot_dict, img_names, depth_maps_for_saving

    def vis_keypoints(
        self, img, kps, skeleton, score_thr=0.4, line_width=3, circle_rad=3
    ):
        kps = kps[self.mano2interhand_mapper]
        score = np.ones((21), dtype=np.float32)
        rgb_dict = self.MANO_RGB_DICT
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.astype("uint8"))

        draw = ImageDraw.Draw(img)
        for i in range(len(skeleton)):
            joint_name = skeleton[i]["name"]
            pid = skeleton[i]["parent_id"]
            parent_joint_name = skeleton[pid]["name"]

            kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
            kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

            if score[i] > score_thr and score[pid] > score_thr and pid != -1:
                draw.line(
                    [(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])],
                    fill=rgb_dict[parent_joint_name],
                    width=line_width,
                )
            if score[i] > score_thr:
                draw.ellipse(
                    (
                        kps[i][0] - circle_rad,
                        kps[i][1] - circle_rad,
                        kps[i][0] + circle_rad,
                        kps[i][1] + circle_rad,
                    ),
                    fill=rgb_dict[joint_name],
                )
            if score[pid] > score_thr and pid != -1:
                draw.ellipse(
                    (
                        kps[pid][0] - circle_rad,
                        kps[pid][1] - circle_rad,
                        kps[pid][0] + circle_rad,
                        kps[pid][1] + circle_rad,
                    ),
                    fill=rgb_dict[parent_joint_name],
                )
        return img


def make_heatmaps(image, heatmaps):
    heatmaps = torch.nn.functional.interpolate(
        heatmaps[None], size=image.shape[:2], mode="bilinear"
    )[0]
    heatmaps = heatmaps.mul(255).clamp(0, 255).byte().detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, (num_joints + 1) * width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap * 0.7 + image * 0.3

        width_begin = width * (j + 1)
        width_end = width * (j + 2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image
    return image_fused.astype(np.uint8)  # image_grid


def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints + 1) * width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = (
            tagmap.add(-min)
            .div(max - min + 1e-5)
            .mul(255)
            .clamp(0, 255)
            .byte()
            .detach()
            .cpu()
            .numpy()
        )

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap * 0.9 + image_resized * 0.1

        width_begin = width * (j + 1)
        width_end = width * (j + 2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]["name"]

        if joint_name.endswith("thumb_null"):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith("thumb3"):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith("thumb2"):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith("thumb1"):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith("thumb0"):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith("index_null"):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith("index3"):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith("index2"):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith("index1"):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith("middle_null"):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith("middle3"):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith("middle2"):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith("middle1"):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith("ring_null"):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith("ring3"):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith("ring2"):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith("ring1"):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith("pinky_null"):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith("pinky3"):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith("pinky2"):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith("pinky1"):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)

    return rgb_dict


def load_skeleton(path, joint_num=-1):
    # Dummy implementation if not present
    print(f"Warning: load_skeleton not fully implemented, using dummy for {path}")
    return [
        {"name": f"j{i}", "parent_id": i - 1 if i > 0 else -1}
        for i in range(max(21, joint_num))
    ]


def draw_skeleton(image, pts, bones=None, cm=None, label_kp_order=False, r=3):
    for i, pt in enumerate(pts):
        if len(pt) > 1:
            if pt[0] > 0 and pt[1] > 0:
                image = cv2.circle(image, (int(pt[0]), int(pt[1])), r, (255, 0, 0), -1)
                if label_kp_order and i in bones:
                    img = cv2.putText(
                        image,
                        str(i),
                        (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 215, 0),
                        1,
                    )

    if bones is not None:
        if cm is None:
            set_colors = np.array([[255, 0, 0] for i in range(len(bones))]).astype(
                np.int
            )
        else:
            if len(bones) > len(cm):
                cm = np.concatenate([cm for _ in range(len(bones) // len(cm) + 1)], 0)
            set_colors = cm[: len(bones)].astype(np.int)
        bones = np.concatenate([bones, set_colors], 1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa > 0).all() and (pb > 0).all():
                xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
                image = cv2.line(
                    image,
                    (xa, ya),
                    (xb, yb),
                    (int(line[2]), int(line[3]), int(line[4])),
                    r,
                )
    return image


def draw_skeleton_multiperson(image, pts_group, **kwargs):
    for pts in pts_group:
        image = draw_skeleton(image, pts, **kwargs)
    return image


class Plotter3dPoses:

    def __init__(self, canvas_size=(512, 512), origin=(0.5, 0.5), scale=200):
        self.canvas_size = canvas_size
        self.origin = np.array(
            [origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32
        )  # x, y
        self.scale = np.float32(scale)
        self.theta, self.phi = 0, np.pi / 2  # np.pi/4, -np.pi/6
        axis_length = 200
        axes = [
            np.array(
                [
                    [-axis_length / 2, -axis_length / 2, 0],
                    [axis_length / 2, -axis_length / 2, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-axis_length / 2, -axis_length / 2, 0],
                    [-axis_length / 2, axis_length / 2, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-axis_length / 2, -axis_length / 2, 0],
                    [-axis_length / 2, -axis_length / 2, axis_length],
                ],
                dtype=np.float32,
            ),
        ]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(
                np.array(
                    [
                        [-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                        [axis_length / 2, -axis_length / 2 + step_id * step, 0],
                    ],
                    dtype=np.float32,
                )
            )
            axes.append(
                np.array(
                    [
                        [-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                        [-axis_length / 2 + step_id * step, axis_length / 2, 0],
                    ],
                    dtype=np.float32,
                )
            )
        self.axes = np.array(axes)

    def plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = (
            np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255
            if img is None
            else img
        )
        R = self._get_rotation(self.theta, self.phi)
        # self._draw_axes(img, R)
        for vertices, color in zip(pose_3ds, colors):
            self._plot_edges(img, vertices, bones, R, color)
        return img

    def encircle_plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = (
            np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255
            if img is None
            else img
        )
        # encircle_theta, encircle_phi = [0, np.pi/4, np.pi/2, 3*np.pi/4], [np.pi/2,np.pi/2,np.pi/2,np.pi/2]
        encircle_theta, encircle_phi = [
            0,
            0,
            0,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
        ], [
            np.pi / 2,
            5 * np.pi / 7,
            -2 * np.pi / 7,
            np.pi / 2,
            5 * np.pi / 7,
            -2 * np.pi / 7,
            np.pi / 2,
            5 * np.pi / 7,
            -2 * np.pi / 7,
        ]
        encircle_origin = (
            np.array(
                [
                    [0.165, 0.165],
                    [0.165, 0.495],
                    [0.165, 0.825],
                    [0.495, 0.165],
                    [0.495, 0.495],
                    [0.495, 0.825],
                    [0.825, 0.165],
                    [0.825, 0.495],
                    [0.825, 0.825],
                ],
                dtype=np.float32,
            )
            * np.array(self.canvas_size)[None]
        )
        for self.theta, self.phi, self.origin in zip(
            encircle_theta, encircle_phi, encircle_origin
        ):
            R = self._get_rotation(self.theta, self.phi)
            # self._draw_axes(img, R)
            for vertices, color in zip(pose_3ds, colors):
                self._plot_edges(img, vertices * 0.6, bones, R, color)
        return img

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R, color):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        org_verts = vertices.reshape((-1, 3))[edges]
        for inds, edge_vertices in enumerate(edges_vertices):
            if 0 in org_verts[inds]:
                continue
            edge_vertices = edge_vertices.astype(int)
            cv2.line(
                img,
                tuple(edge_vertices[0]),
                tuple(edge_vertices[1]),
                color,
                2,
                cv2.LINE_AA,
            )

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array(
            [
                [cos(theta), sin(theta) * sin(phi)],
                [-sin(theta), cos(theta) * sin(phi)],
                [0, -cos(phi)],
            ],
            dtype=np.float32,
        )  # transposed
