import torch
from torch.utils.data import DataLoader

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

from rasterization_q10.input_representation.combinators import Rasterizer
from rasterization_q10.input_representation.static_layers import StaticLayerRasterizer
from rasterization_q10.input_representation.agents import AgentBoxesWithFadedHistory
from rasterization_q10 import PredictHelper

from rasterization_q10.helper import convert_global_coords_to_local
import matplotlib.pyplot as plt

import hydra


class NusTrajectoryExtractor:
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, agent_time=0, layer_names=None,
                 colors=None, resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 32, meters_behind: float = 32,
                 meters_left: float = 32, meters_right: float = 32, version='v1.0-mini'):
        self.layer_names = ['drivable_area', 'road_segment', 'road_block',
                            'lane', 'ped_crossing', 'walkway', 'stop_line',
                            'carpark_area', 'road_divider', 'lane_divider'] if layer_names is None else layer_names
        self.colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
                       (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                       (255, 255, 255), (255, 255, 255), (255, 255, 255), ] if colors is None else colors
        self.root = root
        self.nus = NuScenes(version, dataroot=self.root)
        self.helper = PredictHelper(self.nus)

        self.sampling_time = sampling_time
        self.agent_seconds = agent_time
        self.scene_size = (64, 64)
        self.past_len = int(sampling_time * 4/3)
        self.future_len = int(sampling_time * 2)

        self.static_layer = StaticLayerRasterizer(self.helper, layer_names=self.layer_names, colors=self.colors,
                                                  resolution=resolution, meters_ahead=meters_ahead,
                                                  meters_behind=meters_behind,
                                                  meters_left=meters_left, meters_right=meters_right)
        self.agent_layer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.agent_seconds,
                                                      resolution=resolution, meters_ahead=meters_ahead,
                                                      meters_behind=meters_behind,
                                                      meters_left=meters_left, meters_right=meters_right)
        self.combinator = Rasterizer()
        self.show_agent = True
        self.resized_ratio = 0.5  # resize ratio of image for visualizing

        self.drivable_area_idx = self.layer_names.index('drivable_area')
        self.road_divider_idx = self.layer_names.index('road_divider')
        self.lane_divider_idx = self.layer_names.index('lane_divider')

    def __len__(self):
        return len(self.nus.sample)

    def get_annotation(self, instance_token, sample_token):
        annotation = self.helper.get_sample_annotation(instance_token, sample_token)  # agent annotation
        # ego pose
        ego_pose_xy = annotation['translation']
        ego_pose_rotation = annotation['rotation']

        # agent attributes
        agent_attributes = {
            'category_name': annotation['category_name'],
            'translation': annotation['translation'],
            'size': annotation['size'],
            'rotation': annotation['rotation'],
            'visibility': int(annotation['visibility_token'])  # 1 ~ 4 (0~40%, 40~60%, 60~80%, 80~100%)
        }
        # agent trajectory annotation
        agent_mask, total_agents_annotation, agent_annotation = \
            self.agent_layer.generate_mask_with_path(
                instance_token, sample_token, seconds=self.sampling_time, vehicle_only=True)
        '''
        total_agents_annotation:
            'instance_tokens', category_names',
            'translations', 'pasts', 'futures', 'velocities', 'accelerations', 'yaw_rates'
        '''
        agent_trajectories = {
            'agent_mask': agent_mask,
            'total_agents_annotation': total_agents_annotation,
            'agent_annotation': agent_annotation
        }

        # map attributes
        map_masks, lanes, map_img, map_img_with_lanes = \
            self.static_layer.generate_mask(ego_pose_xy, ego_pose_rotation, sample_token)
        map_attributes = {
            'map_masks': map_masks,
            'lanes': lanes,
            'map_img': map_img,
            'map_img_with_lanes': map_img_with_lanes,
            'map_img_with_agents': self.combinator.combine([map_img, agent_mask])
        }

        scene_data = {
            'instance_token': instance_token,
            'sample_token': sample_token,

            'annotation': annotation,
            'ego_pose_xy': ego_pose_xy,
            'ego_pose_rotation': ego_pose_rotation,

            'agent_attributes': agent_attributes,
            'agent_trajectories': agent_trajectories,
            'map_attributes': map_attributes,
        }

        return scene_data

    def get_cmu_annotation(self, instance_tokens, sample_token):
        scene_data = self.get_annotation(instance_tokens[0], sample_token)
        ego_pose_xy = scene_data['ego_pose_xy']
        ego_pose_rotation = scene_data['ego_pose_rotation']

        agents_annotation = scene_data['agent_trajectories']['total_agents_annotation']

        agents_past = [convert_global_coords_to_local(path_global_i, ego_pose_xy, ego_pose_rotation).tolist()
                       if len(path_global_i) != 0 else [] for path_global_i in agents_annotation['pasts']]
        agents_future = [convert_global_coords_to_local(path_global_i, ego_pose_xy, ego_pose_rotation).tolist()
                         if len(path_global_i) != 0 else [] for path_global_i in agents_annotation['futures']]
        agents_translation = \
            convert_global_coords_to_local(agents_annotation['translations'], ego_pose_xy, ego_pose_rotation).tolist() \
                if len(agents_annotation['translations']) != 0 else []

        drivable_area = scene_data['map_attributes']['map_masks'][self.drivable_area_idx]
        road_divider = scene_data['map_attributes']['map_masks'][self.road_divider_idx]
        lane_divider = scene_data['map_attributes']['map_masks'][self.lane_divider_idx]

        drivable_area = cv2.cvtColor(cv2.resize(drivable_area, self.scene_size), cv2.COLOR_BGR2GRAY)
        road_divider = cv2.cvtColor(cv2.resize(road_divider, self.scene_size), cv2.COLOR_BGR2GRAY)
        lane_divider = cv2.cvtColor(cv2.resize(lane_divider, self.scene_size), cv2.COLOR_BGR2GRAY)

        # distance_map, prior = self.generateDistanceMaskFromColorMap(
        #     drivable_area, image_size=self.scene_size, prior_size=self.scene_size)
        map_image_show = cv2.resize(scene_data['map_attributes']['map_img_with_agents'], dsize=(0, 0),
                                    fx=self.resized_ratio, fy=self.resized_ratio, interpolation=cv2.INTER_LINEAR)

        num_agents = len(agents_past)
        agents_mask = [tk in instance_tokens for tk in agents_annotation['instance_tokens']]  # agents to decode
        future_agent_masks = []
        past_agents_traj = []
        future_agents_traj = []
        past_agents_traj_len = []
        future_agents_traj_len = []
        decode_start_vel = []
        decode_start_pos = []

        # For challenge submission
        predict_mask = []
        predict_instance_tokens = []

        for idx in range(num_agents):
            past = agents_past[idx][-self.past_len + 1:] + [agents_translation[idx]]
            future = agents_future[idx][:self.future_len]

            if len(past) != self.past_len:
                continue
            elif not (abs(past[-1][0]) <= self.scene_size[1] // 2) or not (abs(past[-1][1]) <= self.scene_size[0] // 2):
                continue

            if len(future) != self.future_len:
                agents_mask[idx] = False

            future_agent_masks.append(agents_mask[idx])
            past_agents_traj.append(past)
            past_agents_traj_len.append(len(past))
            future_agents_traj.append(future)
            future_agents_traj_len.append(len(future))
            decode_start_pos.append(past[-1])
            decode_start_vel.append(
                [(past[-1][0] - past[-2][0]) / self.sampling_time, (past[-1][1] - past[-2][1]) / self.sampling_time])

            if agents_annotation['instance_tokens'][idx] in instance_tokens:
                predict_mask.append(True)
                predict_instance_tokens.append(agents_annotation['instance_tokens'][idx])
            else:
                predict_mask.append(False)
                predict_instance_tokens.append('')

        episode = [past_agents_traj, past_agents_traj_len,
                   future_agents_traj, future_agents_traj_len,
                   future_agent_masks, decode_start_vel, decode_start_pos, predict_mask, predict_instance_tokens]
        episode_img = [drivable_area, road_divider, lane_divider]

        return {'episode': episode, 'episode_img': episode_img, 'img_show': map_image_show}

    def save_cmu_dataset(self, save_dir, partition='all'):
        from nuscenes.eval.prediction.splits import get_prediction_challenge_split

        split_types = ['mini_train', 'mini_val', 'train', 'train_val', 'val']
        if partition == 'mini':
            split_types = ['mini_train', 'mini_val']

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for split in tqdm(split_types, desc='split dataset'):
            partition_tokens = get_prediction_challenge_split(split, dataroot=self.root)
            tokens_dict = {}
            for token in partition_tokens:
                instance_token, sample_token = token.split('_')
                try:
                    tokens_dict[sample_token].append(instance_token)
                except KeyError:
                    tokens_dict[sample_token] = [instance_token]

            with open('{}/{}.tokens'.format(save_dir, split), 'wb') as f:
                pickle.dump(tokens_dict, f, pickle.HIGHEST_PROTOCOL)

            for sample_tk, instance_tks in tqdm(tokens_dict.items(), desc=split, total=len(tokens_dict)):
                sample_dir = os.path.join(save_dir, sample_tk)
                if not os.path.isdir(sample_dir):
                    os.mkdir(sample_dir)
                    scene_data = self.get_cmu_annotation(instance_tks, sample_tk)
                    with open('{}/map.bin'.format(sample_dir), 'wb') as f:
                        pickle.dump(scene_data['episode_img'], f, pickle.HIGHEST_PROTOCOL)
                    with open('{}/viz.bin'.format(sample_dir), 'wb') as f:
                        pickle.dump(scene_data['img_show'], f, pickle.HIGHEST_PROTOCOL)
                    with open('{}/episode.bin'.format(sample_dir), 'wb') as f:
                        pickle.dump(scene_data['episode'], f, pickle.HIGHEST_PROTOCOL)
                    with open('{}/instance_tks.bin'.format(sample_dir), 'wb') as f:
                        pickle.dump(instance_tks, f, pickle.HIGHEST_PROTOCOL)

    def render_sample(self, sample_token):
        self.nus.render_sample(sample_token)

    def render_scene(self, sample_token):
        sample = self.nus.get('sample', sample_token)
        scene = self.nus.get('scene', sample['scene_token'])
        log = self.nus.get('log', scene['log_token'])
        location = log['location']
        nus_map = NuScenesMap(dataroot=self.root, map_name=location)
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        camera_channel = 'CAM_FRONT'
        nus_map.render_map_in_image(self.nus, sample_token, layer_names=layer_names, camera_channel=camera_channel)


class NusCustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, load_dir='../nus_dataset', split='train', shuffle=False, min_angle=None, max_angle=None):
        self.load_dir = load_dir
        self.split = split
        self.shuffle = shuffle
        self.min_angle = min_angle
        self.max_angle = max_angle

        if split not in ['mini_train', 'mini_val', 'train', 'train_val', 'val']:
            msg = 'Unexpected split type: {}\nuse ["mini_train", "mini_val", "train", "train_val", "val"]'.format(split)
            raise Exception(msg)

        with open('{}/{}.tokens'.format(load_dir, split), 'rb') as f:
            self.tokens_dict = pickle.load(f)
            self.sample_tokens = list(self.tokens_dict.keys())
            if shuffle:
                np.random.shuffle(self.sample_tokens)

        sample_tks = []
        self.episodes = []
        self.total_agents = 0
        self.num_agents_list = []
        self.curvatures = []
        self.speeds = []
        self.distances = []

        for sample_tk in self.sample_tokens:
            sample_dir = os.path.join(load_dir, sample_tk)
            with open('{}/episode.bin'.format(sample_dir), 'rb') as f:
                episode = pickle.load(f)  # episode: past, past_len, future, future_len, agent_mask, vel, pos

            futures = episode[2]
            agent_mask = episode[4]
            vel = episode[5]

            for idx, future_i in enumerate(futures):
                if agent_mask[idx]:
                    curvature = np.rad2deg(self.calculateCurve(future_i))
                    if min_angle is not None and abs(curvature) < min_angle:
                        agent_mask[idx] = False
                    elif max_angle is not None and abs(curvature) > max_angle:
                        agent_mask[idx] = False
                    else:
                        self.curvatures.append(curvature)
                        self.speeds.append(np.linalg.norm(vel) / 0.5)
                        self.distances.append(np.linalg.norm(np.array(future_i[-1]) - np.array(future_i[0])))

            if np.sum(agent_mask) != 0:
                sample_tks.append(sample_tk)
                episode[4] = agent_mask
                self.episodes.append(episode)
                self.total_agents += np.sum(agent_mask)
                self.num_agents_list.append(np.sum(agent_mask))

        self.sample_tokens = sample_tks
        print('total samples: {}'.format(len(self.episodes)))
        print('total agents (to decode): {}'.format(self.total_agents))
        print('average curvature: {:.2f} deg.'.format(np.mean(self.curvatures)))
        print('average speed: {:.2f}m'.format(np.mean(self.speeds)))
        print('average future distance: {:.2f}m'.format(np.mean(self.distances)))
        print('average number of agents per scene: {:.2f}'.format(np.mean(self.num_agents_list)))

        # Note: mean, std values are calculated on total train dataset  (should be changed for another dataset)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([5.52345], [8.28154])
        ])
        self.p_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([20.20157], [7.17894]),
            transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
        ])

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        sample_tk = self.sample_tokens[idx]
        past, past_len, future, future_len, agent_mask, vel, pos, predict_mask, instance_tokens = self.episodes[idx]
        sample_dir = os.path.join(self.load_dir, sample_tk)
        with open('{}/map.bin'.format(sample_dir), 'rb') as f:
            map_img = pickle.load(f)
            drivable_area, road_divider, lane_divider = map_img
            # return map_img
            # distance_map, prior = map_img
        _, drivable_area = cv2.threshold(drivable_area, 0, 255, cv2.THRESH_BINARY)
        _, road_divider = cv2.threshold(road_divider, 0, 255, cv2.THRESH_BINARY)
        drivable_area = drivable_area - road_divider
        distance_map = cv2.distanceTransform(255 - drivable_area, cv2.DIST_L2, 5) - cv2.distanceTransform(drivable_area,
                                                                                                          cv2.DIST_L2,
                                                                                                          5)
        prior_map = distance_map.copy()
        prior_map[prior_map < 0] = 0
        prior_map = prior_map.max() - prior_map

        image = self.img_transform(distance_map)
        prior = self.p_transform(prior_map)

        data = (
            past, past_len,
            [future[i] for i in np.arange(len(agent_mask))[agent_mask]],
            [future_len[i] for i in np.arange(len(agent_mask))[agent_mask]],
            agent_mask, vel, pos, image, prior, sample_tk, predict_mask, instance_tokens
        )

        return data

    def get_scene_image_with_idx(self, idx: int):
        sample_tk = self.sample_tokens[idx]
        sample_dir = os.path.join(self.load_dir, sample_tk)
        with open('{}/viz.bin'.format(sample_dir), 'rb') as f:
            scene_img = pickle.load(f)
        return scene_img

    def get_scene_image_with_token(self, token: str):
        sample_dir = os.path.join(self.load_dir, token)
        with open('{}/viz.bin'.format(sample_dir), 'rb') as f:
            scene_img = pickle.load(f)
        return scene_img

    def draw_agents_with_token(self, sample_tk: str):
        sample_dir = os.path.join(self.load_dir, sample_tk)
        with open('{}/episode.bin'.format(sample_dir), 'rb') as f:
            episode = pickle.load(f)  # episode: past, past_len, future, future_len, agent_mask, vel, pos

        past, past_len, future, future_len, agent_mask, vel, pos = episode
        future = [future[i] for i in np.arange(len(agent_mask))[agent_mask]]
        future_len = [future_len[i] for i in np.arange(len(agent_mask))[agent_mask]]

        total_agent_pose = np.array(pos).reshape(-1, 2)
        decoded_agent_pose = (np.array(pos)[agent_mask]).reshape(-1, 2)

        plt.scatter(total_agent_pose[:, 0], total_agent_pose[:, 1], color='r')
        plt.scatter(decoded_agent_pose[:, 0], decoded_agent_pose[:, 1], color='b')
        for gt_past in np.array(past):
            plt.plot(gt_past[:, 0], gt_past[:, 1], color='salmon', alpha=0.5, linewidth=6)
        for gt_future, gt_pose in zip(np.array(future), decoded_agent_pose):
            plt.plot(np.append(gt_pose[0], gt_future[:, 0]), np.append(gt_pose[1], gt_future[:, 1]),
                     color='steelblue', alpha=0.3, linewidth=6)

    def calculate_dataset_distribution(self):
        img_mean = 0.0
        img_var = 0.0
        prior_mean = 0.0
        prior_var = 0.0

        n = len(self.sample_tokens)
        for idx in range(n):
            sample_tk = self.sample_tokens[idx]
            sample_dir = os.path.join(self.load_dir, sample_tk)
            with open('{}/map.bin'.format(sample_dir), 'rb') as f:
                map_img = pickle.load(f)
                drivable_area, road_divider, lane_divider = map_img
            _, drivable_area = cv2.threshold(drivable_area, 0, 255, cv2.THRESH_BINARY)
            _, road_divider = cv2.threshold(road_divider, 0, 255, cv2.THRESH_BINARY)
            drivable_area = drivable_area - road_divider
            distance_map = cv2.distanceTransform(255 - drivable_area, cv2.DIST_L2, 5) - cv2.distanceTransform(
                drivable_area, cv2.DIST_L2, 5)
            prior_map = distance_map.copy()
            prior_map[prior_map < 0] = 0
            prior_map = prior_map.max() - prior_map

            img_mean += np.mean(distance_map)
            img_var += np.var(distance_map)
            prior_mean += np.mean(prior_map)
            prior_var += np.var(prior_map)

        img_mean = img_mean / n
        img_var = np.sqrt(img_var / n)
        prior_mean = prior_mean / n
        prior_var = np.sqrt(prior_var / n)

        print('[{}] img: {} ({}), prior: {} ({})'.format(self.split, img_mean, img_var, prior_mean, prior_var))
        return img_mean, img_var, prior_mean, prior_var

    def show_distribution(self):
        print("Total episodes: {}, num agents: {}".format(len(self.episodes), self.total_agents))
        print("Average number of agents per episode: {:.2f}".format(np.mean(self.num_agents_list)))
        print("Average curvature (future path): {:.2f} (Deg)".format(np.mean(self.curvatures)))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.curvatures, bins=90, color='royalblue', range=(-90, 90))
        plt.xlabel('Path Curvature (Deg)')
        plt.ylabel('count')
        plt.xlim([-90, 90])
        plt.show()

    def show_speed_distribution(self):
        print("Total episodes: {}, num agents: {}".format(len(self.episodes), self.total_agents))
        print("Average number of agents per episode: {:.2f}".format(np.mean(self.num_agents_list)))
        print("Average agent speed: {:.2f} (m/s)".format(np.mean(self.speeds)))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.speeds, bins=90, color='royalblue', range=(0, 20))
        plt.xlabel('Agent speed (m/s)')
        plt.ylabel('count')
        plt.xlim([0, 20])
        plt.show()

    def show_distance_distribution(self):
        print("Total episodes: {}, num agents: {}".format(len(self.episodes), self.total_agents))
        print("Average number of agents per episode: {:.2f}".format(np.mean(self.num_agents_list)))
        print("Average driving distance (future path): {:.2f} (m)".format(np.mean(self.distances)))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.distances, bins=90, color='royalblue', range=(3, 40))
        plt.xlabel('Future Path Length (m)')
        plt.ylabel('count')
        plt.xlim([3, 40])
        plt.show()

    @staticmethod
    def calculateCurve(points_):
        if len(points_) < 3:
            raise Exception('number of points should be more than 3.')
        points = np.array(points_)
        a = points[1] - points[0]
        b = points[-1] - points[0]

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        try:
            angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
            if np.isnan(angle):
                angle = 0.0
            else:
                angle = 2 * np.pi - angle if angle > np.pi else angle
                angle = -angle if np.cross(np.append(a, 0), np.append(b, 0))[2] > 0 else angle
        except RuntimeWarning or ZeroDivisionError:
            angle = 0.0

        return angle


def nuscenes_collate(batch, test_set=False):
    # batch_i:
    # 1. past_agents_traj : (Num obv agents in batch_i X 20 X 2)
    # 2. past_agents_traj_len : (Num obv agents in batch_i, )
    # 3. future_agents_traj : (Num pred agents in batch_i X 20 X 2)
    # 4. future_agents_traj_len : (Num pred agents in batch_i, )
    # 5. future_agent_masks : (Num obv agents in batch_i)
    # 6. decode_rel_pos: (Num pred agents in batch_i X 2)
    # 7. decode_start_pos: (Num pred agents in batch_i X 2)
    # 8. map_image : (3 X 224 X 224)
    # 9. scene ID: (string)
    # Typically, Num obv agents in batch_i < Num pred agents in batch_i ##

    batch_size = len(batch)

    if test_set:
        past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id, predict_mask, instance_tokens  = list(
            zip(*batch))

    else:
        past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id, predict_mask, instance_tokens = list(
            zip(*batch))

        # Future agent trajectory
        num_future_agents = np.array([len(x) for x in future_agents_traj])
        future_agents_traj = np.concatenate(future_agents_traj, axis=0)
        future_agents_traj_len = np.concatenate(future_agents_traj_len, axis=0)

        future_agents_three_idx = future_agents_traj.shape[1]
        future_agents_two_idx = int(future_agents_three_idx * 2 // 3)

        future_agents_three_mask = future_agents_traj_len >= future_agents_three_idx
        future_agents_two_mask = future_agents_traj_len >= future_agents_two_idx

        future_agents_traj_len_idx = []
        for traj_len in future_agents_traj_len:
            future_agents_traj_len_idx.extend(list(range(traj_len)))

        # Convert to Tensor
        num_future_agents = torch.LongTensor(num_future_agents)
        future_agents_traj = torch.FloatTensor(future_agents_traj)
        future_agents_traj_len = torch.LongTensor(future_agents_traj_len)

        future_agents_three_mask = torch.BoolTensor(future_agents_three_mask)
        future_agents_two_mask = torch.BoolTensor(future_agents_two_mask)

        future_agents_traj_len_idx = torch.LongTensor(future_agents_traj_len_idx)

    # Past agent trajectory
    num_past_agents = np.array([len(x) for x in past_agents_traj])
    past_agents_traj = np.concatenate(past_agents_traj, axis=0)
    past_agents_traj_len = np.concatenate(past_agents_traj_len, axis=0)
    past_agents_traj_len_idx = []
    for traj_len in past_agents_traj_len:
        past_agents_traj_len_idx.extend(list(range(traj_len)))

    # Convert to Tensor
    num_past_agents = torch.LongTensor(num_past_agents)
    past_agents_traj = torch.FloatTensor(past_agents_traj)
    past_agents_traj_len = torch.LongTensor(past_agents_traj_len)
    past_agents_traj_len_idx = torch.LongTensor(past_agents_traj_len_idx)

    # Future agent mask
    future_agent_masks = np.concatenate(future_agent_masks, axis=0)
    future_agent_masks = torch.BoolTensor(future_agent_masks)

    # decode start vel & pos
    decode_start_vel = np.concatenate(decode_start_vel, axis=0)
    decode_start_pos = np.concatenate(decode_start_pos, axis=0)
    decode_start_vel = torch.FloatTensor(decode_start_vel)
    decode_start_pos = torch.FloatTensor(decode_start_pos)

    map_image = torch.stack(map_image, dim=0)
    prior = torch.stack(prior, dim=0)

    scene_id = np.array(scene_id)

    predict_mask = torch.BoolTensor(np.concatenate(predict_mask, 0))
    instance_tokens = np.concatenate(instance_tokens, 0)

    data = (
        map_image, prior,
        future_agent_masks,
        num_past_agents, past_agents_traj, past_agents_traj_len, past_agents_traj_len_idx,
        num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx,
        future_agents_two_mask, future_agents_three_mask,
        decode_start_vel, decode_start_pos,
        scene_id, predict_mask, instance_tokens
    )

    return data


@hydra.main(config_path="conf/config.yaml")
def save_dataset(cfg):
    root = cfg.dataset.dataset_path
    version = cfg.dataset.version
    sampling_time = cfg.dataset.sampling_time
    agent_time = cfg.dataset.agent_time
    layer_names = ['drivable_area', 'road_divider', 'lane_divider']
    colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0)]
    resolution = cfg.dataset.resolution
    meters_ahead = cfg.dataset.meters_ahead
    meters_behind = cfg.dataset.meters_behind
    meters_left = cfg.dataset.meters_left
    meters_right = cfg.dataset.meters_right

    save_dir = cfg.dataset.load_dir

    extractor = NusTrajectoryExtractor(
        root=root, version=version,
        sampling_time=sampling_time, agent_time=agent_time,
        layer_names=layer_names, colors=colors, resolution=resolution,  # meters / pixel
        meters_ahead=meters_ahead, meters_behind=meters_behind,
        meters_left=meters_left, meters_right=meters_right
    )

    extractor.save_cmu_dataset(save_dir=save_dir)


if __name__ == "__main__":
    save_dataset()
else:
    print("import:")
    print(__name__)
