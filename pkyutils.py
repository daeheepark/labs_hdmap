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
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

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

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) < 3.0:
        return 0.0

    angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    if angle > np.pi:
        angle = 2 * np.pi - angle
    if np.cross(np.append(a, 0), np.append(b, 0))[2] > 0:
        angle = -angle

    return angle


# Data parser for NuScenes
class NusCustomParser(Dataset):
    def __init__(self, root='/datasets/nuscene/v1.0-mini', sampling_time=3, agent_time=0, layer_names=None,
                 colors=None, resolution: float = 0.1,  # meters / pixel
                 meters_ahead: float = 25, meters_behind: float = 25,
                 meters_left: float = 25, meters_right: float = 25, version='v1.0-mini', split='train'):
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
        # self.scenes = self.nus.scene
        # self.samples = self.nus.sample
        self.dataset = get_prediction_challenge_split(split, dataroot=self.root)

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
        return len(self.dataset)

    def __getitem__(self, idx):
        scene_id = idx
        # sample = self.samples[idx]
        # sample_token = sample['token']
        instance_token, sample_token = self.dataset[idx].split('_')

        # 1. calculate ego pose
        # sample_data_lidar = self.nus.get('sample_data', sample['data']['LIDAR_TOP'])
        # ego_pose = self.nus.get('ego_pose', sample_data_lidar['ego_pose_token'])
        ego_pose = self.helper.get_sample_annotation(instance_token, sample_token)

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


        return map_masks, map_img, agent_mask, xy_local, scene_id

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
    def __init__(self, root='/datasets/nuscene/v1.0-mini', version='v1.0-mini', load_dir='../nus_dataset', split='val'):
        self.DATAROOT = root
        self.version = version
        self.sampling_time = 3
        self.agent_time = 0  # zero for static mask, non-zero for overlap
        self.layer_names = ['drivable_area', 'lane']
        self.colors = [(255, 255, 255), (255, 255, 100)]
        self.split = split
        self.dataset = NusCustomParser(
            root=self.DATAROOT,
            version=self.version,
            sampling_time=self.sampling_time,
            agent_time=self.agent_time,
            layer_names=self.layer_names,
            colors=self.colors,
            resolution=0.1,
            meters_ahead=32,
            meters_behind=32,
            meters_left=32,
            meters_right=32,
            split=self.split)
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
            # with open('{}/fake/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
            #     episode_fake = pickle.load(f)
            with open('{}/real/{}.bin'.format(self.data_dir, self.ids[idx]), 'rb') as f:
                episode_real = pickle.load(f)

            # episode_fake.extend([map_img, prior, idx])
            episode_fake = None
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
        data_dir = self.data_dir + '_' + self.split
        map_dir = os.path.join(data_dir, 'map')
        prior_dir = os.path.join(data_dir, 'prior')
        # fake_dir = os.path.join(data_dir, 'fake')
        real_dir = os.path.join(data_dir, 'real')

        if not (os.path.isdir(data_dir)):
            os.makedirs(data_dir)
            os.makedirs(map_dir)
            os.makedirs(prior_dir)
            # os.makedirs(fake_dir)
            os.makedirs(real_dir)

            for idx in tqdm(range(len(self.dataset))):
                map_masks, map_img, agent_mask, xy_local, idx = self.dataset[idx]

                agent_past, agent_future, agent_translation = xy_local
                # fake_past, fake_future, fake_translation = virtual_xy_local

                map_image, prior = self.generateDistanceMaskFromColorMap(map_masks[0], scene_size=(64, 64))
                with open('{}/{}.bin'.format(map_dir, idx), 'wb') as f:
                    pickle.dump(map_image, f, pickle.HIGHEST_PROTOCOL)
                with open('{}/{}.bin'.format(prior_dir, idx), 'wb') as f:
                    pickle.dump(prior, f, pickle.HIGHEST_PROTOCOL)

                # # 1) fake agents
                # episode_fake = self.get_episode(fake_past, fake_future, fake_translation)
                # with open('{}/{}.bin'.format(fake_dir, idx), 'wb') as f:
                #     pickle.dump(episode_fake, f, pickle.HIGHEST_PROTOCOL)

                # 2) real agents
                episode_real = self.get_episode(agent_past, agent_future, agent_translation)
                with open('{}/{}.bin'.format(real_dir, idx), 'wb') as f:
                    pickle.dump(episode_real, f, pickle.HIGHEST_PROTOCOL)

        else:
            print('directory exists')

    @staticmethod
    def get_episode(agent_past, agent_future, agent_translation, map_width=64):
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
            if np.max(pose) > side_length or np.min(pose) < -side_length or len(past) == 0:
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
    def __init__(self, version='v1.0-mini', load_dir='../nus_dataset', split='train',
                 shuffle=False, val_ratio=0.3, data_type='real', min_angle=None, max_angle=None):

        self.data_dir = os.path.join(load_dir, version+'_'+split)
        self.data_type = data_type

        n = len(os.listdir(os.path.join(self.data_dir, 'map')))
        
        self.ids = np.arange(n)
        if data_partition == 'train':
            self.ids = self.ids[: int(n * (1 - val_ratio))]
        elif data_partition == 'val':
            self.ids = self.ids[int(n * val_ratio):]

        if shuffle:
            np.random.shuffle(self.ids)

        self.episodes = []
        self.total_agents = 0
        self.num_agents_list = []
        self.curves = []
        self.speeds = []
        self.distances = []
        
        for idx in self.ids:
            with open('{}/{}/{}.bin'.format(self.data_dir, self.data_type, idx), 'rb') as f:
                # past, past_len, future, future_len, agent_mask, vel, pos, sample_idx = episode
                episode = pickle.load(f)
                for i, points_i in enumerate(episode[2]):
                    curve = calculateCurve(points_i[:episode[3][i]])
                    if min_angle is not None and abs(curve) < min_angle:
                        episode[4][i] = False
                    elif max_angle is not None and abs(curve) > max_angle:
                        episode[4][i] = False
                    else:
                        # self.curves.append(np.sign(curve) * np.rad2deg(abs(curve)))
                        self.curves.append(np.rad2deg(curve))
                        self.speeds.append(np.linalg.norm(episode[5][i]))
                        self.distances.append(np.linalg.norm(points_i[episode[3][i]-1] - points_i[0]))

                if np.sum(episode[4]) != 0:
                    episode.append(idx)
                    self.episodes.append(episode)
                    self.total_agents += np.sum(episode[4])
                    self.num_agents_list.append(np.sum(episode[4]))

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

            # if 'test' in self.data_partition:
            #     data = (data[0], data[1], data[4], data[5], data[6], data[7], data[8], data[9])

            return data

    def show_distribution(self):
        print("?????? episodes: {}, ???????????? ??????: {}".format(len(self.episodes), self.total_agents))
        print("episode ??? ?????? ???????????? ??????: {:.2f}".format(np.mean(self.num_agents_list)))
        print("?????? ?????? ??????: {:.2f} (Deg)".format(np.mean(np.abs(self.curves))))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.curves, bins=90, color='royalblue', range=(-90, 90))
        plt.xlabel('Path Curvature (Deg)')
        plt.ylabel('count')
        plt.xlim([-90, 90])
        plt.show()

    def show_speed_distribution(self):
        print("?????? episodes: {}, ???????????? ??????: {}".format(len(self.episodes), self.total_agents))
        print("episode ??? ?????? ???????????? ??????: {:.2f}".format(np.mean(self.num_agents_list)))
        print("?????? ???????????? ??????: {:.2f} (m/s)".format(np.mean(np.abs(self.speeds))))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.speeds, bins=90, color='royalblue', range=(0, 20))
        plt.xlabel('Agent speed (m/s)')
        plt.ylabel('count')
        plt.xlim([0, 20])
        plt.show()

    def show_distance_distribution(self):
        print("?????? episodes: {}, ???????????? ??????: {}".format(len(self.episodes), self.total_agents))
        print("episode ??? ?????? ???????????? ??????: {:.2f}".format(np.mean(self.num_agents_list)))
        print("?????? ???????????? ?????? ????????????: {:.2f} (m)".format(np.mean(np.abs(self.distances))))

        plt.figure(figsize=(10, 5))
        plt.title('Distribution')
        plt.hist(self.distances, bins=90, color='royalblue', range=(0.5, 20))
        plt.xlabel('Future Path Length (m)')
        plt.ylabel('count')
        plt.xlim([0.5, 20])
        plt.show()

        return self.distances


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
        past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id = list(
            zip(*batch))

    else:
        past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, map_image, prior, scene_id = list(
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

    # max_agents = batch_size * 20
    #
    # data = (
    #     map_image[:max_agents], prior[:max_agents],
    #     future_agent_masks[:max_agents],
    #     num_past_agents[:max_agents], past_agents_traj[:max_agents],
    #     past_agents_traj_len[:max_agents], past_agents_traj_len_idx[:max_agents],
    #     num_future_agents[:max_agents], future_agents_traj[:max_agents],
    #     future_agents_traj_len[:max_agents], future_agents_traj_len_idx[:max_agents],
    #     future_agents_two_mask[:max_agents], future_agents_three_mask[:max_agents],
    #     decode_start_vel[:max_agents], decode_start_pos[:max_agents],
    #     scene_id[:max_agents]
    # )

    data = (
        map_image, prior,
        future_agent_masks,
        num_past_agents, past_agents_traj, past_agents_traj_len, past_agents_traj_len_idx,
        num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx,
        future_agents_two_mask, future_agents_three_mask,
        decode_start_vel, decode_start_pos,
        scene_id
    )

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', type=str, default='/datasets/nuscene/v1.0-mini')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--load_dir', type=str, default='../nus_dataset')
    parser.add_argument('--min', type=float, default=None)
    parser.add_argument('--max', type=float, default=None)
    # args = parser.parse_args()
    args = parser.parse_args('--root ../nus_dataset/original_small/v1.0-test_meta \
--version v1.0-test --load_dir ../nus_dataset'.split(' '))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    print('min: {}, max: {}'.format(args.min, args.max))
    pkyLoader = NusToolkit(root=args.root, version=args.version, load_dir=args.load_dir)
    pkyLoader.save_dataset()
    print("finished...")
else:
    print("import:")
    print(__name__)
