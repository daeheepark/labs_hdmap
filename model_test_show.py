import warnings

warnings.filterwarnings(action='ignore')

import argparse

import pkyutils as pky
from dataset.nuscenes import NuscenesDataset, nuscenes_collate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

import os
import pickle
from tqdm import tqdm

from Proposed.models import Global_Scene_CAM_NFDecoder
from Proposed.utils import ModelTest
from R2P2_MA.model_utils import Interpolated_Ploss

from rasterization_q10.generator_dev import NusLoaderQ10
from nuscenes.prediction.input_representation.combinators import Rasterizer

import cv2
import natsort

combinator = Rasterizer()

DATAROOT = '/datasets/nuscene/v1.0-mini'

sampling_time = 3
agent_time = 0  # zero for static mask, non-zero for overlap

layer_names = ['drivable_area', 'road_segment', 'road_block',
               'lane', 'ped_crossing', 'walkway', 'stop_line',
               'carpark_area', 'road_divider', 'lane_divider']
colors = [(255, 255, 255), (100, 255, 255), (255, 100, 255),
          (255, 255, 100), (100, 100, 255), (100, 255, 100), (255, 100, 100),
          (100, 100, 100), (50, 100, 50), (200, 50, 50), ]

dataset_show = NusLoaderQ10(
    root=DATAROOT,
    sampling_time=sampling_time,
    agent_time=agent_time,
    layer_names=layer_names,
    colors=colors,
    resolution=0.1,
    meters_ahead=25,
    meters_behind=25,
    meters_left=25,
    meters_right=25)


def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)
    return logdet


def show_dataset(idx, ax, dataset=dataset_show):
    def draw_paths(ax, local_paths):
        past = local_paths[0]
        future = local_paths[1]
        translation = local_paths[2]
        for i in range(len(past)):
            if len(past[i]) != 0:
                path = np.append(past[i][-6:], [translation[i]], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='steelblue', linewidth=6, alpha=0.3)
            if len(future[i]) != 0:
                path = np.append([translation[i]], future[i][:6], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='salmon', linewidth=6, alpha=0.3)

    map_masks, map_img, agent_mask, xy_local, _, _, idx = dataset[idx]

    agent_past = xy_local[0]
    agent_future = xy_local[1]
    agent_translation = xy_local[2]

    agents_combined = combinator.combine(
        np.append(map_masks[[0, 5, 8, 9]], agent_mask[np.newaxis, ...], axis=0))

    ax.set_title("Predicted")
    ax.imshow(agents_combined, extent=[-25, 25, -25, 25], alpha=0.3)
    if len(xy_local[0]) != 0:
        draw_paths(ax, xy_local)
        ax.scatter(agent_translation[:, 0], agent_translation[:, 1], c='b', alpha=0.3)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)


def predict_path(dataloader, batch_size=1, ploss_criterion=None):
    results_idx = []
    results_predicted = []
    results_ploss = []
    results_qloss = []
    results_loss = []
    results_pose = []

    with torch.no_grad():
        H = W = 64
        coordinate_2d = np.indices((H, W))
        coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
        coordinate = torch.FloatTensor(coordinate)
        coordinate = coordinate.reshape((1, 1, H, W))

        coordinate_std, coordinate_mean = torch.std_mean(coordinate)
        coordinate = (coordinate - coordinate_mean) / coordinate_std

        distance_2d = coordinate_2d - np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
        distance = np.sqrt((distance_2d ** 2).sum(axis=0))
        distance = torch.FloatTensor(distance)
        distance = distance.reshape((1, 1, H, W))

        distance_std, distance_mean = torch.std_mean(distance)
        distance = (distance - distance_mean) / distance_std

        coordinate = coordinate.to(device)
        distance = distance.to(device)

        c1 = -nfuture * np.log(2 * np.pi)

        for b, batch in tqdm(enumerate(dataloader), total=len(dataloader) // batch_size, desc='predict'):
            scene_images, log_prior, \
            agent_masks, \
            num_src_trajs, src_trajs, src_lens, src_len_idx, \
            num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
            tgt_two_mask, tgt_three_mask, \
            decode_start_vel, decode_start_pos, scene_id = batch
            #             print(batch[-1])

            # Detect dynamic batch size)
            batch_size = scene_images.size(0)
            num_three_agents = torch.sum(tgt_three_mask)

            if num_three_agents == 0:
                results_predicted.append(np.empty([0, 6, 6, 2]))
                results_ploss.append(np.empty(0))
                results_qloss.append(np.empty(0))
                results_loss.append(np.empty(0))

                results_idx.append(-1 * scene_id)
                results_pose.append(np.empty([0, 2]))
                continue

            coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
            distance_batch = distance.repeat(batch_size, 1, 1, 1)
            scene_images = torch.cat((scene_images.to(device), coordinate_batch, distance_batch),
                                     dim=1)

            src_trajs = src_trajs.to(device)
            src_lens = src_lens.to(device)
            tgt_trajs = tgt_trajs.to(device)[tgt_three_mask]
            tgt_lens = tgt_lens.to(device)[tgt_three_mask]

            num_tgt_trajs = num_tgt_trajs.to(device)
            episode_idx = torch.arange(batch_size, device=device).repeat_interleave(num_tgt_trajs)[
                tgt_three_mask]

            agent_masks = agent_masks.to(device)
            agent_tgt_three_mask = torch.zeros_like(agent_masks)
            agent_masks_idx = torch.arange(len(agent_masks), device=device)[agent_masks][tgt_three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            decode_start_vel = decode_start_vel.to(device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(device)[agent_tgt_three_mask]

            log_prior = log_prior.to(device)

            perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=device)

            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = \
                model.infer(tgt_trajs + perterb,
                            src_trajs, src_lens,
                            agent_tgt_three_mask,
                            episode_idx,
                            decode_start_vel,
                            decode_start_pos,
                            num_src_trajs,
                            scene_images)
            z_ = z_.reshape((num_three_agents, -1))  # A X (Td*2)
            log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))
            logdet_sigma = log_determinant(sigma_)
            log_qpi = log_q0 - logdet_sigma.sum(dim=1)
            qloss = -log_qpi
            batch_qloss = qloss.mean()

            gen_trajs, z, mu, sigma = \
                model(motion_encoding_, src_lens, agent_tgt_three_mask,
                      episode_idx, decode_start_vel, decode_start_pos,
                      num_src_trajs, scene_encoding_, agent_encoded=True,
                      scene_encoded=True)

            # ploss = ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)
            ploss = ploss_criterion(gen_trajs, tgt_trajs)

            batch_ploss = ploss.mean()
            batch_loss = batch_qloss + beta * batch_ploss

            results_idx.append(scene_id)
            results_predicted.append(gen_trajs.cpu().numpy())
            results_ploss.append(batch_ploss.cpu().numpy())
            results_qloss.append(batch_qloss.cpu().numpy())
            results_loss.append(batch_loss.cpu().numpy())
            results_pose.append(decode_start_pos.cpu().numpy())

        return results_idx, results_predicted, results_ploss, results_qloss, results_loss, results_pose


def test_idx(idx, ax):
    scene_id = scene_ids[idx][0]

    if scene_id < 0:
        show_dataset(-scene_id, ax, dataset_show)
        return
    else:
        show_dataset(scene_id, ax, dataset_show)

    ax.scatter(start[idx][:, 0], start[idx][:, 1], color='r')

    for i in range(len(predicted[idx])):
        paths = np.insert(predicted[idx][i], 0, start[idx][i], axis=1)
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], color='r')
    ax.text(-24, 22, 'ploss: {:.3f}\nqloss: {:.3f}'.format(ploss[idx], qloss[idx]), fontsize=15, color='r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', type=str, default='/datasets/nuscene/v1.0-mini')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--min', type=float, default=None)
    parser.add_argument('--max', type=float, default=None)
    args = parser.parse_args()

    dataset = pky.DatasetQ10(root='../generated/v1.0-mini/gen_None_None/real', data_partition='train')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                            collate_fn=lambda x: nuscenes_collate(x), num_workers=1)
    print(f'Test Examples: {len(dataset)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = 3
    nfuture = int(3 * 2)
    crossmodal_attention = False
    ploss_type = 'map'
    beta = 0.1

    model = Global_Scene_CAM_NFDecoder(
        device=device,
        agent_embed_dim=128,
        nfuture=nfuture,
        att_dropout=0.1,
        velocity_const=0.5,
        num_candidates=6,
        decoding_steps=nfuture,
        att=crossmodal_attention)
    model = model.to(device)

    ckpt = 'experiment/transfer_None_None/epoch100.pth.tar'
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state'], strict=False)

    ploss_criterion = Interpolated_Ploss()
    ploss_criterion = ploss_criterion.to(device)

    # predict
    scene_ids, predicted, ploss, qloss, loss, start = predict_path(dataloader)

    results_idx = len(os.listdir('results'))
    results_dir = 'results/{}'.format(results_idx)
    os.mkdir(results_dir)

    num_data = len(dataset)

    plt.figure(figsize=(10, 10))
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    for i in tqdm(range(num_data), desc='plot'):
        test_idx(i, plt.gca())
        plt.savefig(results_dir + '/{}.png'.format(i), dpi=300)
        plt.pause(0.001)
        plt.cla()

    video_name = 'results/{}.avi'.format(results_idx)

    images = [img for img in os.listdir(results_dir) if img.endswith(".png")]
    images = natsort.natsorted(images)
    frame = cv2.imread(os.path.join(results_dir, images[5]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 2, (width, height))
    for image in tqdm(images, total=len(images), desc='video processing'):
        video.write(cv2.imread(os.path.join(results_dir, image)))
    cv2.destroyAllWindows()
    video.release()


else:
    print("import:")
    print(__name__)
