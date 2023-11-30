import json, csv
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np
import torch
import glob

from PIL import Image
from scipy.spatial.transform import Rotation

from utils.pose_utils import (
    average_poses,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
)

from utils.ray_utils import (
    get_ndc_rays_fx_fy,
    get_pixels_for_image,
    get_ray_directions_K,
    get_rays,
    sample_images_at_xy,
)

from .base import Base5DDataset, Base6DDataset

class EntVideoDatasetNew(Base6DDataset):
    def __init__(self, cfg, split="train", **kwargs):
        self.use_reference = (
            cfg.dataset.use_reference if "use_reference" in cfg.dataset else False
        )
        self.correct_poses = (
            cfg.dataset.correct_poses if "correct_poses" in cfg.dataset else False
        )
        self.use_ndc = cfg.dataset.use_ndc if "use_ndc" in cfg.dataset else False

        self.num_frames = cfg.dataset.num_frames if "num_frames" in cfg.dataset else 1
        self.start_frame = cfg.dataset.start_frame if "start_frame" in cfg.dataset else 1
        self.keyframe_step = cfg.dataset.keyframe_step if "keyframe_step" in cfg.dataset else 1
        self.num_keyframes = cfg.dataset.num_keyframes if "num_keyframes" in cfg.dataset else self.num_frames // self.keyframe_step

        self.load_full_step = 4
        

        super().__init__(cfg, split, **kwargs)
    
    def read_meta(self):
        W, H = self.img_wh

        # Poses, bounds
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'), 'rb'
        ) as f:
            poses_bounds = np.load(f)
        
        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'images/*.png'))
        )
        
        self.mask_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'new_masks/*.png'))
        )

        # self.importance_paths = sorted(
        #     glob.glob(os.path.join(self.root_dir, 'importance_masks/*.png'))
        # )

        self.images_per_frame = 1 
        self.total_images_per_frame = 1 

        # Get intrinsics & extrinsics
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        self.bounds = poses_bounds[:, -2:]

        H, W, self.focal = poses[0, :, -1]
        self.cx, self.cy = W / 2.0, H / 2.0

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Correct poses, bounds
        self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
            poses, self.bounds
        )

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

        # Ray directions for all pixels
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=True
        )

        # Repeat poses, times
        self.poses = self.poses.reshape(-1, 3, 4)
        self.times = np.linspace(0, 1, len(self.image_paths))[..., None]
        self.times = self.times.reshape(-1)
        self.camera_ids = np.tile(np.linspace(0, self.images_per_frame - 1, self.images_per_frame)[None, :], (len(self.image_paths), 1))
        self.camera_ids = self.camera_ids.reshape(-1)

        # Holdout validation images
        if len(self.val_set) > 0:
            val_indices = self.val_set
        elif self.val_skip != 'inf':
            self.val_skip = min(
                len(self.image_paths), self.val_skip
            )
            val_indices = list(range(0, len(self.image_paths), self.val_skip))
        else:
            val_indices = []

        train_indices = [
            i for i in range(len(self.poses))
        ]

        if self.val_all:
            val_indices = [i for i in train_indices]  # noqa

        if self.split == 'val' or self.split == 'test':
            self.image_paths = [self.image_paths[i] for i in val_indices]  

            self.poses = self.poses[val_indices]
            self.times = self.times[val_indices]
            self.camera_ids = self.camera_ids[val_indices]
        
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]  
            self.poses = self.poses[train_indices]
            self.times = self.times[train_indices]
            self.camera_ids = self.camera_ids[train_indices]
        
        self.num_images = len(self.poses)
    
    def mask_subsample(self, coords, rgb, idx):
        # Masks
        mask_path = self.mask_paths[idx]

        mask = Image.open(mask_path)
        mask = self.transform(mask)
        mask = mask.view(1, -1).squeeze()
        
        mask = mask == 0

        return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])

    def subsample(self, coords, rgb, idx):
        # if (idx % self.load_full_step) == 0:
        coords, rgb = self.mask_subsample(coords, rgb, idx)
        return coords, rgb
        
    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        # Collect training data
        self.all_coords = []
        self.all_rgb = []
        
        for idx in range(len(self.image_paths)):
            # coords
            cur_coords = self.get_coords(idx)

            # Color
            cur_rgb = self.get_rgb(idx)

            # subsample
            cur_coords, cur_rgb = self.subsample(cur_coords, cur_rgb, idx)

            self.all_coords += [cur_coords]
            self.all_rgb += [cur_rgb]
            
        self.all_coords = torch.cat(self.all_coords, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.update_all_data()

    def update_all_data(self):
        self.all_weights = self.get_weights()

        ## All inputs
        self.all_inputs = torch.cat(
            [
                self.all_coords,
                self.all_rgb,
                self.all_weights,
            ],
            -1,
        )

    def format_batch(self, batch):
        batch["coords"] = batch["inputs"][..., : self.all_coords.shape[-1]]
        batch["rgb"] = batch["inputs"][
            ..., self.all_coords.shape[-1] : self.all_coords.shape[-1] + 3
        ]
        batch["weight"] = batch["inputs"][..., -1:]
        del batch["inputs"]

        return batch

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
            c2w = torch.FloatTensor(self.poses[idx])
            time = self.times[idx]

            camera_id = self.camera_ids[idx]

            rays_o, rays_d = get_rays(self.directions, c2w)

            if self.use_ndc:
                rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
            else:
                rays = torch.cat([rays_o, rays_d], dim=-1)

            # Camera ID
            rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * camera_id], dim=-1)

            # Time stamp
            rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)
            return rays

    def get_rgb(self, idx):
        # Colors
        image_path = self.image_paths[idx]

        img = Image.open(image_path).convert('RGB')

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(idx),
                'pose': self.poses[idx],
                'time': self.times[idx],
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])

        elif self.split == 'val' or self.split == 'test':
            batch = {
                'coords': self.get_coords(idx),
                'rgb': self.get_rgb(idx),
                'idx': idx
            }

            batch['weight'] = torch.ones_like(batch['coords'][..., -1:])
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }


        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch
    




        
        


