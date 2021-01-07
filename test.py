import torch
from torch.utils.data import DataLoader
from dataset.nuscenes import NuscenesDataset, nuscenes_collate

from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

import torch

from torch.utils.data import Dataset

import os
import pickle

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from rasterization_q10.input_representation.static_layers import StaticLayerRasterizer
from rasterization_q10.input_representation.agents import AgentBoxesWithFadedHistory
from rasterization_q10 import PredictHelper

from rasterization_q10.helper import convert_global_coords_to_local
import matplotlib.pyplot as plt


def calculateCurve(points):
    if len(points) < 3:
        return 0.0

    a = points[1] - points[0]
    b = points[-1] - points[0]

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) < 5.0:
        return 0.0

    au = a / np.linalg.norm(a)
    bu = b / np.linalg.norm(b)
    return np.arccos(np.clip(np.dot(au, bu), -1.0, 1.0))


# Data parser for NuScenes
class NusCustomParser(Dataset):
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
        self.show_agent = True

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
            ego_pose_xy, ego_pose_rotation, sample_token, self.seconds, show_agent=self.show_agent)

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
            ego_pose_xy, ego_pose_rotation, lanes_disc, sample_token, show_agent=self.show_agent,
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


class NusToolkit(torch.utils.data.dataset.Dataset):
    def __init__(self, root='/datasets/nuscene/v1.0-mini', version='v1.0-mini', load_dir='../nus_dataset'):
        self.DATAROOT = root
        self.version = version
        self.sampling_time = 3
        self.agent_time = 0  # zero for static mask, non-zero for overlap
        self.layer_names = ['drivable_area', 'lane']
        self.colors = [(255, 255, 255), (255, 255, 100)]
        self.dataset = NusCustomParser(
            root=self.DATAROOT,
            version=self.version,
            sampling_time=self.sampling_time,
            agent_time=self.agent_time,
            layer_names=self.layer_names,
            colors=self.colors,
            resolution=0.1,
            meters_ahead=25,
            meters_behind=25,
            meters_left=25,
            meters_right=25)
        print("num_samples: {}".format(len(self.dataset)))

        self.p_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([23.0582], [27.3226]),
            transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
        ])
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([23.0582], [27.3226])
        ])

        self.data_dir = os.path.join(load_dir, version)
        self.ids = np.arange(len(self.dataset))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if not (os.path.isdir(self.data_dir)):
            print('parse dataset first')
            return None
        else:
            with open('{}/map/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
                map_img = pickle.load(f)
            with open('{}/prior/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
                prior = pickle.load(f)
            with open('{}/fake/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
                episode_fake = pickle.load(f)
            with open('{}/real/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
                episode_real = pickle.load(f)

            episode_fake.extend([map_img, prior, idx])
            episode_real.extend([map_img, prior, idx])

            return episode_fake, episode_real

    def generateDistanceMaskFromColorMap(self, src, scene_size=(64, 64)):
        raw_image = cv2.cvtColor(cv2.resize(src, scene_size), cv2.COLOR_BGR2GRAY)
        raw_image[raw_image != 0] = 255

        raw_image = cv2.distanceTransform(raw_image.astype(np.uint8), cv2.DIST_L2, 5)

        raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
        raw_map_image[raw_map_image < 0] = 0  # Uniform on drivable area
        raw_map_image = raw_map_image.max() - raw_map_image

        image = self.img_transform(raw_image)
        prior = self.p_transform(raw_map_image)

        return image, prior

    def save_dataset(self):
        data_dir = self.data_dir
        map_dir = os.path.join(data_dir, 'map')
        prior_dir = os.path.join(data_dir, 'prior')
        fake_dir = os.path.join(data_dir, 'fake')
        real_dir = os.path.join(data_dir, 'real')

        if not (os.path.isdir(data_dir)):
            os.makedirs(data_dir)
            os.makedirs(map_dir)
            os.makedirs(prior_dir)
            os.makedirs(fake_dir)
            os.makedirs(real_dir)

            for idx in tqdm(range(len(self.dataset))):
                map_masks, map_img, agent_mask, xy_local, virtual_mask, virtual_xy_local, idx = self.dataset[idx]

                agent_past, agent_future, agent_translation = xy_local
                fake_past, fake_future, fake_translation = virtual_xy_local

                map_image, prior = self.generateDistanceMaskFromColorMap(map_masks[0], scene_size=(64, 64))
                with open('{}/{}.bin'.format(map_dir, idx), 'wb') as f:
                    pickle.dump(map_image, f, pickle.HIGHEST_PROTOCOL)
                with open('{}/{}.bin'.format(prior_dir, idx), 'wb') as f:
                    pickle.dump(prior, f, pickle.HIGHEST_PROTOCOL)

                # 1) fake agents
                episode_fake = self.get_episode(fake_past, fake_future, fake_translation)
                with open('{}/{}.bin'.format(fake_dir, idx), 'wb') as f:
                    pickle.dump(episode_fake, f, pickle.HIGHEST_PROTOCOL)

                # 2) real agents
                episode_real = self.get_episode(agent_past, agent_future, agent_translation)
                with open('{}/{}.bin'.format(real_dir, idx), 'wb') as f:
                    pickle.dump(episode_real, f, pickle.HIGHEST_PROTOCOL)

        else:
            print('directory exists')

    def get_episode(self, agent_past, agent_future, agent_translation, map_width=64):
        num_agents = len(agent_past)
        future_agent_masks = np.array([True] * num_agents)

        past_agents_traj = [[[0., 0.]] * 4] * num_agents
        future_agents_traj = [[[0., 0.]] * 6] * num_agents

        past_agents_traj = np.array(past_agents_traj)
        future_agents_traj = np.array(future_agents_traj)

        past_agents_traj_len = np.array([4] * num_agents)
        future_agents_traj_len = np.array([6] * num_agents)

        decode_start_vel = np.array([[0., 0.]] * num_agents)
        decode_start_pos = np.array([[0., 0.]] * num_agents)

        frame_masks = np.array([True] * num_agents)

        for idx, path in enumerate(zip(agent_past, agent_future)):
            past = path[0]
            future = path[1]
            pose = agent_translation[idx]

            # agent filtering
            side_length = map_width // 2
            if np.max(pose) > side_length or np.min(pose) < -side_length:
                frame_masks[idx] = False
                continue
            if len(past) < 4 or len(future) < 6:
                future_agent_masks[idx] = False

            # agent trajectory
            if len(past) < 4:
                past_agents_traj_len[idx] = len(past)
            for i, point in enumerate(past[:4]):
                past_agents_traj[idx, i] = point

            if len(future) < 6:
                future_agents_traj_len[idx] = len(future)
            for i, point in enumerate(future[:6]):
                future_agents_traj[idx, i] = point

            # vel, pose
            if len(future) != 0:
                decode_start_vel[idx] = (future[0] - agent_translation[idx]) / 0.5
            decode_start_pos[idx] = agent_translation[idx]

        if num_agents > 0 and np.sum(frame_masks) != 0:
            return [past_agents_traj[frame_masks], past_agents_traj_len[frame_masks],
                    future_agents_traj[frame_masks], future_agents_traj_len[frame_masks],
                    future_agent_masks[frame_masks], decode_start_vel[frame_masks], decode_start_pos[frame_masks]]
        else:
            return [past_agents_traj, past_agents_traj_len,
                    future_agents_traj, future_agents_traj_len,
                    future_agent_masks, decode_start_vel, decode_start_pos]


class DatasetQ10(torch.utils.data.dataset.Dataset):
    def __init__(self, version='v1.0-mini', load_dir='../nus_dataset', data_partition='train',
                 shuffle=False, val_ratio=0.3, data_type='real', min_angle=None, max_angle=None):

        self.data_dir = os.path.join(load_dir, version)
        self.data_type = data_type
        self.data_partition = data_partition

        n = len(os.listdir(os.path.join(self.data_dir, 'map')))

        self.ids = np.arange(n)
        if data_partition == 'train':
            self.ids = self.ids[: int(n * (1 - val_ratio))]
        elif data_partition == 'val':
            self.ids = self.ids[: int(n * val_ratio)]

        if shuffle:
            np.random.shuffle(self.ids)

        self.episodes = []
        self.total_agents = 0

        for idx in self.ids:
            with open('{}/{}/{}.bin'.format(self.data_dir, self.data_type, idx), 'rb') as f:
                episode = pickle.load(f)
                for i, points_i in enumerate(episode[2]):
                    if min_angle is not None and abs(calculateCurve(points_i)) < min_angle:
                        episode[4][i] = False
                    elif max_angle is not None and abs(calculateCurve(points_i)) > max_angle:
                        episode[4][i] = False

                if np.sum(episode[4]) != 0:
                    episode.append(idx)
                    self.episodes.append(episode)
                    self.total_agents += np.sum(episode[4])

        print('total agents: {}'.format(self.total_agents))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        if not (os.path.isdir(self.data_dir)):
            print('parse dataset first')
            return None
        else:
            episode = self.episodes[idx]
            past, past_len, future, future_len, agent_mask, vel, pos, sample_idx = episode

            with open('{}/map/{}.bin'.format(self.data_dir, sample_idx), 'rb') as f:
                map_img = pickle.load(f)
            with open('{}/prior/{}.bin'.format(self.data_dir, sample_idx), 'rb') as f:
                prior = pickle.load(f)

            data = (past, past_len, future[agent_mask], future_len[agent_mask], agent_mask,
                    vel, pos, map_img, prior, sample_idx)

            if 'test' in self.data_partition:
                data = (data[0], data[1], data[4], data[5], data[6], data[7], data[8], data[9])

            return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', type=str, default='/datasets/nuscene/v1.0-mini')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--min', type=float, default=None)
    parser.add_argument('--max', type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    print('min: {}, max: {}'.format(args.min, args.max))
    pkyLoader = CustomLoader(root=args.root, version=args.version)
    pkyLoader.saveData(min_angle=args.min, max_angle=args.max, name=args.version)
    print("finished...")
else:
    print("import:")
    print(__name__)
