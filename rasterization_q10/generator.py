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
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, agent_time=0, layer_names=None, colors=None):
        if layer_names is None:
            layer_names = ['drivable_area', 'road_segment', 'road_block',
                           'lane', 'ped_crossing', 'walkway', 'stop_line',
                           'carpark_area', 'road_divider', 'lane_divider']
        if colors is None:
            colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                      (255, 255, 255), (255, 255, 255), (255, 255, 255),]
        self.root = root
        self.nus = NuScenes('v1.0-mini', dataroot=self.root)
        self.scenes = self.nus.scene
        self.samples = self.nus.sample

        self.layer_names = layer_names
        self.colors = colors

        self.helper = PredictHelper(self.nus)

        self.seconds = sampling_time
        self.agent_seconds = agent_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_token = sample['token']

        # 1. calculate ego pose
        sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose_xy = ego_pose['translation']
        ego_pose_rotation = ego_pose['rotation']
        # 타임스탬프
        timestamp = ego_pose['timestamp']

        # 2. Generate Map & Agent Masks
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)

        static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors)
        agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.agent_seconds)

        map_masks, lanes, map_img = static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)
        agent_mask = agent_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)

        lane_tokens = list(lanes.keys())
        num_lanes = len(lane_tokens)
        lanes_disc = [np.array(lanes[token])[:, :2] for token in lane_tokens]
        lanes_arc = []
        for seq in lanes_disc:
            seq_len = len(seq)
            lane_middle = seq[seq_len//2]
            opt_middle = (seq[0] + seq[-1])/2
            lane_h = np.linalg.norm(lane_middle - opt_middle)
            lane_w = np.linalg.norm(seq[-1] - seq[0])
            curve = lane_h/2 + lane_w**2/(8*lane_h)
            lanes_arc.append(curve)
        lanes_arc = np.array(lanes_arc)

        # 3. Generate Agent Trajectory
        annotation_tokens = sample['anns']
        num_agent = len(annotation_tokens)
        agents = []
        for ans_token in annotation_tokens:
            agent_states = []
            agent = self.nus.get('sample_annotation', ans_token)
            instance_token = agent['instance_token']

            # 에이전트 주행경로
            xy_global = agent['translation']
            past_xy_global = self.helper.get_past_for_agent(
                instance_token, sample_token, seconds=self.seconds, in_agent_frame=False)
            future_xy_global = self.helper.get_future_for_agent(
                instance_token, sample_token, seconds=self.seconds, in_agent_frame=False)

            # 경로 곡률
            agents_arc = []
            for seq in [past_xy_global, future_xy_global]:
                seq_len = len(seq)
                if seq_len < 2:
                    continue
                path_middle = seq[seq_len // 2]
                opt_middle = (seq[0] + seq[-1]) / 2
                path_h = np.linalg.norm(path_middle - opt_middle)
                path_w = np.linalg.norm(seq[-1] - seq[0])
                if path_h  == 0:
                    path_h = 0.001
                if path_w == 0:
                    path_w = 0.001
                curve = path_h / 2 + (path_w * path_w) / (8 * path_h)
                agents_arc.append(curve)
            agents_arc = np.array(agents_arc)

            # 로컬 주행경로
            xy_local = convert_global_coords_to_local(np.array([xy_global[:2]]), ego_pose_xy, ego_pose_rotation)
            if len(past_xy_global) < 1:
                past_xy_global = np.append(past_xy_global, [0.,0.])
            if len(future_xy_global) < 1:
                future_xy_global = np.append(future_xy_global, [0., 0.])
            past_xy_local = convert_global_coords_to_local(past_xy_global, ego_pose_xy, ego_pose_rotation)
            future_xy_local = convert_global_coords_to_local(future_xy_global, ego_pose_xy, ego_pose_rotation)

            # 에이전트 주행상태
            rot = agent['rotation']
            vel = self.helper.get_velocity_for_agent(instance_token, sample_token)
            accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
            yaw_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

            agent_states = {'present_pos': xy_global, 'past_pos': past_xy_global, 'future_pos': future_xy_global,
                            'rot': rot, 'vel': vel, 'accel': accel, 'yaw_rate': yaw_rate,
                            'present_local_xy': xy_local, 'past_local_xy': past_xy_local,
                            'future_local_xy': future_xy_local, 'curvature': agents_arc}

            agents.append(agent_states)

        return map_masks, agent_mask, agents, idx

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
            fig, ax = plt.subplots(num_rows, 3, figsize=(10, 3*num_rows))
            for row in range(num_rows):
                for col in range(3):
                    num = 3*row + col
                    if num == num_labels-1:
                        break
                    ax[row][col].set_title(self.layer_names[num])
                    ax[row][col].imshow(map_masks[num])
            plt.show()
