import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset

import os
from PIL import Image
import pickle

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

from rasterization_q10.input_representation.static_layers import StaticLayerRasterizer
from rasterization_q10.input_representation.agents import AgentBoxesWithFadedHistory
from rasterization_q10.input_representation.interface import InputRepresentation
from rasterization_q10.input_representation.combinators import Rasterizer
from rasterization_q10 import PredictHelper

from pyquaternion import Quaternion
import numpy as np

from rasterization_q10.helper import convert_global_coords_to_local
import matplotlib.pyplot as plt


class NusLoaderQ10(Dataset):
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, agent_time=0, layer_names=None,
                 colors=None, resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 25, meters_behind: float = 25,
                 meters_left: float = 25, meters_right: float = 25, version='v1.0-mini'):
        if layer_names is None:
            layer_names = ['drivable_area', 'road_segment', 'road_block',
                           'lane', 'ped_crossing', 'walkway', 'stop_line',
                           'carpark_area', 'road_divider', 'lane_divider']
        if colors is None:
            colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), ]
        self.root = root
        self.nus = NuScenes(version, dataroot=self.root)
        self.scenes = self.nus.scene
        self.samples = self.nus.sample

        self.layer_names = layer_names
        self.colors = colors

        self.helper = PredictHelper(self.nus)

        self.seconds = sampling_time
        self.agent_seconds = agent_time

        self.static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors,
                                                  resolution=resolution, meters_ahead=meters_ahead,
                                                  meters_behind=meters_behind,
                                                  meters_left=meters_left, meters_right=meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.agent_seconds,
                                                      resolution=resolution, meters_ahead=meters_ahead,
                                                      meters_behind=meters_behind,
                                                      meters_left=meters_left, meters_right=meters_right)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id = idx
        sample = self.samples[idx]
        sample_token = sample['token']

        # 1. calculate ego pose
        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']

        # 2. Generate map mask
        map_masks, lanes, map_img = self.static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)

        # 3. Generate Agent Trajectory
        agent_mask, xy_global = self.agent_layer.generate_mask(
            ego_pose_xy, ego_pose_rotation, sample_token, self.seconds, show_agent=False)

        xy_local = []

        # past, future trajectory
        for path_global in xy_global[:2]:
            pose_xy = []
            for path_global_i in path_global:
                if len(path_global_i) == 0:
                    pose_xy.append(path_global_i)
                else:
                    pose_xy.append(convert_global_coords_to_local(path_global_i, ego_pose_xy, ego_pose_rotation))
            xy_local.append(pose_xy)
        # current pose
        if len(xy_global[2]) == 0:
            xy_local.append(xy_global[2])
        else:
            xy_local.append(convert_global_coords_to_local(xy_global[2], ego_pose_xy, ego_pose_rotation))

        # 4. Generate Virtual Agent Trajectory
        lane_tokens = list(lanes.keys())
        lanes_disc = [np.array(lanes[token])[:, :2] for token in lane_tokens]

        virtual_mask, virtual_xy = self.agent_layer.generate_virtual_mask(
            ego_pose_xy, ego_pose_rotation, lanes_disc, sample_token, show_agent=False,
            past_trj_len=4, future_trj_len=6, min_dist=6)

        virtual_xy_local = []

        # past, future trajectory
        for path_global in virtual_xy[:2]:
            pose_xy = []
            for path_global_i in path_global:
                if len(path_global_i) == 0:
                    pose_xy.append(path_global_i)
                else:
                    pose_xy.append(convert_global_coords_to_local(path_global_i, ego_pose_xy, ego_pose_rotation))
            virtual_xy_local.append(pose_xy)
        # current pose
        if len(virtual_xy[2]) == 0:
            virtual_xy_local.append(virtual_xy[2])
        else:
            virtual_xy_local.append(convert_global_coords_to_local(virtual_xy[2], ego_pose_xy, ego_pose_rotation))

        return map_masks, map_img, agent_mask, xy_local, virtual_mask, virtual_xy_local, scene_id

    def render_sample(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']
        self.nus.render_sample(sample_token)

    def render_scene(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        camera_channel = 'CAM_FRONT'
        nus_map.render_map_in_image(self.nus, sample_token, layer_names=layer_names, camera_channel=camera_channel)

    def render_map(self, idx, combined=True):
        sample = self.samples[idx]
        sample_token = sample['token']

        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']
        timestamp = ego_pose['timestamp']

        # 2. Generate Map & Agent Masks
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)

        static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors)
        agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.seconds)

        map_masks, lanes, map_img = static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)
        agent_mask = agent_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)

        if combined:
            plt.subplot(1, 2, 1)
            plt.title("combined map")
            plt.imshow(map_img)
            plt.subplot(1, 2, 2)
            plt.title("agent")
            plt.imshow(agent_mask)
            plt.show()
        else:
            num_labels = len(self.layer_names)
            num_rows = num_labels // 3
            fig, ax = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
            for row in range(num_rows):
                for col in range(3):
                    num = 3 * row + col
                    if num == num_labels - 1:
                        break
                    ax[row][col].set_title(self.layer_names[num])
                    ax[row][col].imshow(map_masks[num])
            plt.show()

