import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

from rasterization_q10.generator_dev import NusLoaderQ10

import pickle
from torchvision import transforms
from nuscenes.prediction.input_representation.combinators import Rasterizer

import argparse
from tqdm import tqdm

dataset, map_masks, map_img, agent_mask, xy_local, \
virtual_mask, virtual_xy_local, \
agent_past, agent_future, agent_translation, \
virtual_past, virtual_future, virtual_translation, idx = [None] * 14

p_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226]),
    transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226])
])


def generateDistanceMaskFromColorMap(src, scene_size=(64, 64)):
    img = cv2.resize(src, scene_size)
    raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(raw_image, 100, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    raw_image = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    raw_map_image[raw_map_image < 0] = 0  # Uniform on drivable area
    raw_map_image = raw_map_image.max() - raw_map_image  # Invert values so that non-drivable area has smaller values

    image = img_transform(raw_image)
    prior = p_transform(raw_map_image)

    return image, prior


def get_agent_mask(agent_past, agent_future, agent_translation):
    map_width = 50
    map_height = 50

    num_agents = len(agent_past)
    future_agent_masks = [True] * num_agents

    past_agents_traj = [[[0., 0.]] * 4] * num_agents
    future_agents_traj = [[[0., 0.]] * 6] * num_agents

    past_agents_traj = np.array(past_agents_traj)
    future_agents_traj = np.array(future_agents_traj)

    past_agents_traj_len = [4] * num_agents
    future_agents_traj_len = [6] * num_agents

    decode_start_vel = [[0., 0.]] * num_agents
    decode_start_pos = [[0., 0.]] * num_agents

    for idx, path in enumerate(zip(agent_past, agent_future)):
        past = path[0]
        future = path[1]
        pose = agent_translation[idx]

        # agent filtering
        side_length = map_width // 2
        if len(past) == 0 or len(future) == 0 \
                or np.max(pose) > side_length or np.min(pose) < -side_length:
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

    return past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
           future_agent_masks, decode_start_vel, decode_start_pos


def load_data(idx, thres_min, thres_max):
    global dataset, map_masks, map_img, agent_mask, xy_local, \
        virtual_mask, virtual_xy_local, \
        agent_past, agent_future, agent_translation, \
        virtual_past, virtual_future, virtual_translation

    dataset.thres_min = thres_min
    dataset.thres_max = thres_max

    map_masks, map_img, agent_mask, xy_local, \
    virtual_mask, virtual_xy_local, idx = dataset[idx]

    agent_past = xy_local[0]
    agent_future = xy_local[1]
    agent_translation = xy_local[2]
    virtual_past = virtual_xy_local[0]
    virtual_future = virtual_xy_local[1]
    virtual_translation = virtual_xy_local[2]


def dataProcessing(virtual=False):
    global idx, map_masks, agent_past, agent_future, agent_translation, virtual_past, virtual_future, virtual_translation

    scene_id = idx

    # map mask & prior mask
    map_image, prior = generateDistanceMaskFromColorMap(map_masks[0], scene_size=(64, 64))

    # agent mask
    past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
    future_agent_masks, decode_start_vel, decode_start_pos = get_agent_mask(agent_past, agent_future, agent_translation)

    # virtual agent mask
    past_agents_traj2, past_agents_traj_len2, future_agents_traj2, future_agents_traj_len2, \
    future_agent_masks2, decode_start_vel2, decode_start_pos2 = get_agent_mask(virtual_past, virtual_future,
                                                                               virtual_translation)

    episode = None
    if virtual:
        episode = [past_agents_traj2, past_agents_traj_len2, future_agents_traj2,
                   future_agents_traj_len2, future_agent_masks2,
                   np.array(decode_start_vel2), np.array(decode_start_pos2),
                   map_image, prior, scene_id]
    else:
        episode = [past_agents_traj, past_agents_traj_len, future_agents_traj,
                   future_agents_traj_len, future_agent_masks,
                   np.array(decode_start_vel), np.array(decode_start_pos),
                   map_image, prior, scene_id]

    return episode


def dataGeneration(thres=0.02, curved_ratio=0.3):
    episodes = []

    num_linear = 0
    num_curved = 0

    N = len(dataset)
    print("{} number of samples".format(N))

    # count the number of curved agents
    global agent_past
    for idx in tqdm(range(N), desc='count the number of curved agents'):
        load_data(idx, -1, thres)
        num_linear += len(agent_past)
        load_data(idx, thres, 99999)
        num_curved += len(agent_past)

    # original data
    for idx in tqdm(range(N), desc='load data'):
        load_data(idx, -1, 99999)
        episode = dataProcessing()
        if sum(episode[4]) > 0:
            episodes.append(episode)

    # generate curved data
    curved_target = int(num_linear / (1 - curved_ratio)) * curved_ratio - num_curved
    print("{} curved agents should be generated more...".format(curved_target))
    index = 0
    while curved_target > 0:
        load_data(index, thres, 99999)
        episode = dataProcessing(virtual=True)

        n = sum(episode[4])

        if n > 0:
            episodes.append(episode)
            curved_target -= n

        index += 1
        if index > N - 1:
            index = 0

    print("--- generation finished ---")
    print("number of linear agents: {}".format(num_linear))
    num_curved_total = int(num_linear / (1 - curved_ratio)) * curved_ratio
    print("number of curved agents: {}".format(num_curved_total))
    print("curved agents: {}, generated curved agents: {}".format(num_curved, num_curved_total - num_curved))

    return episodes


parser = argparse.ArgumentParser(description='load details')
parser.add_argument('--root', type=str, help='root of data', default='/datasets/nuscene/v1.0-mini')
parser.add_argument('--version', type=str, help='root of data', default='v1.0-mini')
parser.add_argument('--curvature', type=float, help='curvature threshold', default=0.02)
parser.add_argument('--curve_ratio', type=float, help='curve ratio of dataset', default=0.3)

args = parser.parse_args()

if __name__ == "__main__":
    DATAROOT = args.root
    thres = args.curvature
    curved_ratio = args.curve_ratio
    version = args.version

    sampling_time = 3
    agent_time = 0  # zero for static mask, non-zero for overlap

    layer_names = ['drivable_area', 'lane']
    colors = [(255, 255, 255), (100, 255, 255)]

    dataset = NusLoaderQ10(
        root=DATAROOT,
        sampling_time=sampling_time,
        agent_time=agent_time,
        layer_names=layer_names,
        colors=colors,
        resolution=0.1,
        meters_ahead=25,
        meters_behind=25,
        meters_left=25,
        meters_right=25,
        version=version)

    combinator = Rasterizer()

    # test
    load_data(100, -1, 99999999)
    episode = dataProcessing()
    print("test 100: {}".format(episode))
    print("Generation start...")

    # main
    parsed_data = dataGeneration(thres=thres, curved_ratio=curved_ratio)

    print("Number of Data: {}".format(len(parsed_data)))

    filename = 'nuscene_' + version + '_' + str(thres) + '_' + str(curved_ratio)
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(parsed_data, f, pickle.HIGHEST_PROTOCOL)

    print("--- finished ---")
    print("number of data: {}".format(len(parsed_data)))

