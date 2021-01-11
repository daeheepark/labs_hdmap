import os
import argparse

import torch
from torch.utils.data import DataLoader, ChainDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from Proposed.models import Global_Scene_CAM_NFDecoder
from Proposed.utils import ModelTrainer, ModelTest
from R2P2_MA.model_utils import MSE_Ploss, Interpolated_Ploss

from pkyutils import DatasetQ10, nuscenes_collate, NusCustomParser
from nuscenes.prediction.input_representation.combinators import Rasterizer

import cv2
import natsort

combinator = Rasterizer()

np.random.seed(777)
torch.manual_seed(777)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # 1) training options
    nfuture = int(3 * args.sampling_rate)
    ploss_type = args.ploss_type
    if ploss_type == 'mseloss':
        ploss_criterion = MSE_Ploss()
    else:
        ploss_criterion = Interpolated_Ploss()
    ploss_criterion = ploss_criterion.to(device)

    # 2) define model
    if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        crossmodal_attention = True
    else:
        crossmodal_attention = False

    model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                       att_dropout=args.att_dropout,
                                       velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                       decoding_steps=nfuture, att=crossmodal_attention)

    # ckpt = 'experiment/None_None_fake_AttGlobal_Scene_CAM_NFDecoder__03_January__22_25_/epoch40.pth.tar'
    # if ckpt is not None:
    #     checkpoint = torch.load(ckpt)
    #     model.load_state_dict(checkpoint['model_state'], strict=False)

    model = model.to(device)

    # 3) load dataset
    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    train_dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='train',
                               shuffle=True, val_ratio=0.3, data_type=data_type, min_angle=min_angle,
                               max_angle=max_angle)

    val_dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                             shuffle=False, val_ratio=0.3, data_type='real', min_angle=min_angle,
                             max_angle=max_angle)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                              collate_fn=lambda x: nuscenes_collate(x), num_workers=1)

    print(f'Train Examples: {len(train_dataset)} | Valid Examples: {len(val_dataset)}')

    # 4) training
    # Model optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        print('unknown optimizer:', args.optimizer)
        return

    # Trainer
    exp_path = args.exp_path

    # Training Runner
    trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion)

    trainer.train(args.num_epochs)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfuture = int(3 * args.sampling_rate)

    if args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
            crossmodal_attention = True
        else:
            crossmodal_attention = False

        model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                           att_dropout=args.att_dropout,
                                           velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                           decoding_steps=nfuture, att=crossmodal_attention)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    # Send model to Device:
    model = model.to(device)

    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                         shuffle=False, val_ratio=0.3, data_type=data_type, min_angle=min_angle, max_angle=max_angle)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)

    print(f'Test Examples: {len(dataset)}')

    if not os.path.isdir(args.test_dir):
        os.mkdir(args.test_dir)

    if args.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or "NFDecoder" in args.model_type:
        ploss_criterion = ploss_criterion.to(device)
        tester = ModelTest(model, data_loader, args, device, ploss_criterion)
    else:
        tester = ModelTest(model, data_loader, args, device)

    tester.run()


def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)
    return logdet


class Visualizer:
    def __init__(self, model, ploss_criterion, root='/datasets/nuscene/v1.0-mini', version='v1.0-mini'):
        self.root = root
        self.version = version
        self.sampling_time = 3
        self.agent_time = 0  # zero for static mask, non-zero for overlap
        self.layer_names = ['drivable_area', 'road_segment', 'road_block',
                       'lane', 'ped_crossing', 'walkway', 'stop_line',
                       'carpark_area', 'road_divider', 'lane_divider']
        self.colors = [(255, 255, 255), (100, 255, 255), (255, 100, 255),
                  (255, 255, 100), (100, 100, 255), (100, 255, 100), (255, 100, 100),
                  (100, 100, 100), (50, 100, 50), (200, 50, 50), ]

        self.dataset = NusCustomParser(
            root=self.root,
            version=self.version,
            sampling_time=self.sampling_time,
            agent_time=self.agent_time,
            layer_names=self.layer_names,
            colors=self.colors,
            resolution=0.1,
            meters_ahead=32,
            meters_behind=32,
            meters_left=32,
            meters_right=32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_channels = 3
        self.nfuture = int(3 * 2)
        self.crossmodal_attention = True
        self.ploss_type = 'mseloss'
        self.beta = 0.1

        self.ploss_criterion = ploss_criterion
        self.model = model

    def save_to_video(self, dataloader):
        results_idx = len(os.listdir('results'))
        results_dir = 'results/{}'.format(results_idx)
        os.mkdir(results_dir)
        print('save path: {}'.format(results_dir))

        # predict
        scene_ids, predicted, p_loss, q_loss, loss, start = self.predict_path(dataloader)

        plt.figure(figsize=(10, 10))
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        for i, sample_idx in tqdm(enumerate(scene_ids), desc='plot'):
            # load data
            # print('############## sample_idx {}'.format(sample_idx))
            map_masks, map_img, agent_mask, xy_local, _, _, scene_id = self.dataset[sample_idx[0]]
            # agent_past, agent_future, agent_translation = xy_local
            combined_img = combinator.combine(np.append(map_masks[[0, 5, 8, 9]], agent_mask[np.newaxis, ...], axis=0))
            # visualize
            plt.title("Predicted")
            plt.imshow(combined_img, extent=[-32, 32, -32, 32], alpha=0.3)
            if len(xy_local[0]) != 0:
                self.draw_paths(plt.gca(), xy_local)
                plt.scatter(xy_local[2][:, 0], xy_local[2][:, 1], c='b', alpha=0.3)
            plt.xlim(-32, 32)
            plt.ylim(-32, 32)

            plt.scatter(start[i][:, 0], start[i][:, 1], color='r')

            for j in range(len(predicted[i])):
                paths = np.insert(predicted[i][j], 0, start[i][j], axis=1)
                for path in paths:
                    plt.plot(path[:, 0], path[:, 1], color='r')
            plt.text(-31, 28, 'ploss: {:.3f}\nqloss: {:.3f}'.format(p_loss[i], q_loss[i]), fontsize=15, color='r')

            plt.savefig(results_dir + '/{}.png'.format(i), dpi=150)
            plt.pause(0.001)
            plt.cla()
            if i > 120:
                break

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

    @staticmethod
    def draw_paths(ax, local_paths):
        past = local_paths[0]
        future = local_paths[1]
        translation = local_paths[2]
        for i in range(len(past)):
            if len(past[i]) != 0:
                path = np.append([translation[i]], past[i][-6:], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='steelblue', linewidth=6, alpha=0.3)
            if len(future[i]) != 0:
                path = np.append([translation[i]], future[i][:6], axis=0)
                ax.plot(path[:, 0], path[:, 1], color='salmon', linewidth=6, alpha=0.3)

    def predict_path(self, dataloader, batch_size=1):
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

            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)

            c1 = -self.nfuture * np.log(2 * np.pi)

            for b, batch in tqdm(enumerate(dataloader), total=len(dataloader) // batch_size, desc='predict'):
                scene_images, log_prior, \
                agent_masks, \
                num_src_trajs, src_trajs, src_lens, src_len_idx, \
                num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                tgt_two_mask, tgt_three_mask, \
                decode_start_vel, decode_start_pos, scene_id = batch

                # Detect dynamic batch size
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
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch),
                                         dim=1)

                src_trajs = src_trajs.to(self.device)
                src_lens = src_lens.to(self.device)
                tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]

                num_tgt_trajs = num_tgt_trajs.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[
                    tgt_three_mask]

                agent_masks = agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(agent_masks)
                agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][tgt_three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                log_prior = log_prior.to(self.device)

                perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)

                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = \
                    self.model.infer(tgt_trajs + perterb,
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
                    self.model(motion_encoding_, src_lens, agent_tgt_three_mask,
                          episode_idx, decode_start_vel, decode_start_pos,
                          num_src_trajs, scene_encoding_, agent_encoded=True,
                          scene_encoded=True)

                if self.ploss_type == 'map':
                    ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)
                elif self.ploss_type == 'mseloss':
                    ploss = self.ploss_criterion(gen_trajs, tgt_trajs)

                batch_ploss = ploss.mean()
                batch_loss = batch_qloss + self.beta * batch_ploss

                results_idx.append(scene_id)
                results_predicted.append(gen_trajs.cpu().numpy())
                results_ploss.append(batch_ploss.cpu().numpy())
                results_qloss.append(batch_qloss.cpu().numpy())
                results_loss.append(batch_loss.cpu().numpy())
                results_pose.append(decode_start_pos.cpu().numpy())

        return results_idx, results_predicted, results_ploss, results_qloss, results_loss, results_pose


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfuture = int(3 * args.sampling_rate)

    if args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
            crossmodal_attention = True
        else:
            crossmodal_attention = False

        model = Global_Scene_CAM_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                           att_dropout=args.att_dropout,
                                           velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                           decoding_steps=nfuture, att=crossmodal_attention)
        ploss_type = args.ploss_type

        if ploss_type == 'mseloss':
            from R2P2_MA.model_utils import MSE_Ploss
            ploss_criterion = MSE_Ploss()
        else:
            from R2P2_MA.model_utils import Interpolated_Ploss
            ploss_criterion = Interpolated_Ploss()

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    ckpt = args.test_ckpt
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['model_state'], strict=False)

    # Send model to Device:
    model = model.to(device)

    version = args.version
    data_type = args.data_type
    load_dir = args.load_dir
    min_angle = args.min_angle
    max_angle = args.max_angle
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    dataset = DatasetQ10(version=version, load_dir=load_dir, data_partition='val',
                         shuffle=False, val_ratio=0.3, data_type=data_type, min_angle=min_angle, max_angle=max_angle)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             collate_fn=lambda x: nuscenes_collate(x), num_workers=args.num_workers)

    print(f'Test Examples: {len(dataset)}')

    ploss_criterion = ploss_criterion.to(device)

    viz = Visualizer(model, ploss_criterion, root='{}/original_small/{}'.format(load_dir, version), version=version)
    viz.save_to_video(data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Tag
    parser.add_argument('--tag', type=str, default='AttGlobal_Scene_CAM_NFDecoder',
                        help="Add a tag to the saved folder")
    parser.add_argument('--exp_path', type=str, default='./experiment', help='Experient Directory')

    # Model type
    parser.add_argument('--model_type', type=str, default='AttGlobal_Scene_CAM_NFDecoder',
                        help="SimpleEncDec | SocialPooling | MATF | MATF_GAN | CAM | MACAM | R2P2_RNN | "
                             "R2P2_SimpleRNN | Desire")

    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=20, help="")
    parser.add_argument('--gpu_devices', type=str, default='0', help="GPU IDs for model running")

    # Dataset Parameters
    # parser.add_argument('--dataset', type=str, default='nuscenes', help="argoverse | nuscenes | carla")
    # parser.add_argument('--train_cache', default='./data/nuscenes_train_cache.pkl', help="")
    # parser.add_argument('--val_cache', default='./data/nuscenes_val_cache.pkl', help="")

    # Episode sampling parameters
    parser.add_argument('--sample_stride', type=int, default=1,
                        help="Stride between reference frames in a single episode")

    # Trajectory Parameters
    parser.add_argument('--sampling_rate', type=int, default=2,
                        help="Sampling Rate for Encoding/Decoding sequences")  # Hz | 10 frames per sec % sampling_interval=5 => 2 Hz

    # Scene Context Parameters
    # parser.add_argument('--map_version', type=str, default='2.0', help="Map version")
    # ## Only used for MATFs
    # parser.add_argument('--scene_dropout', type=float, default=0.5, help="")
    # parser.add_argument('--scene_encoder', type=str, default='ShallowCNN', help="ShallowCNN | ResNet")
    # parser.add_argument('--freeze_resnet', type=bool, default=True, help="")

    # Agent Encoding
    # (Not used for R2P2 and Desire)
    parser.add_argument('--agent_embed_dim', type=int, default=128, help="Agent Embedding Dimension")
    parser.add_argument('--lstm_layers', type=int, default=1, help="")
    parser.add_argument('--lstm_dropout', type=float, default=0.3, help="")

    # the number of candidate futures in generative models
    parser.add_argument('--num_candidates', type=int, default=6, help="Number of trajectory candidates sampled")

    # CSP Models
    parser.add_argument('--pooling_size', type=int, default=30, help="Map grid H and W dimension")

    # Attention Models
    parser.add_argument('--att_dropout', type=float, default=0.1, help="")

    # Normalizing Flow Models
    parser.add_argument('--multi_agent', type=int, default=1, help="Enables multi-agent setting for dataset")
    parser.add_argument('--beta', type=float, default=0.1, help="Ploss beta parameter")
    parser.add_argument('--velocity_const', type=float, default=0.5,
                        help="Constant multiplied to dx in verlet integration")
    parser.add_argument('--ploss_type', type=str, default='map', help="Ploss Type, \"mseloss | logistic | map\"")

    # GAN Models
    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--noise_dim', type=int, default=16, help="")
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5],
                        help="Adversarial Training Alpha")
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200],
                        help="Decaying Gan Weight by Epoch")
    parser.add_argument('--disc_hidden', type=int, default=512, help="")
    parser.add_argument('--disc_dropout', type=float, default=0.5, help="")

    # Optimization Parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--load_ckpt', default=None, help='Load Model Checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Resume Model Training')

    # Model Testing Parameters
    parser.add_argument('--test_partition', type=str,
                        default='test_obs',
                        help="Data partition to perform test")
    parser.add_argument('--test_cache', type=str, help="")
    parser.add_argument('--test_dir', type=str, help="Test output dir")
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_times', type=int, default=10, help='Number of test trials to calculate std.')
    parser.add_argument('--test_render', type=int, default=1, help='Whether to render the outputs as figure')

    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--data_type', type=str, default='real')
    parser.add_argument('--min_angle', type=float, default=None)
    parser.add_argument('--max_angle', type=float, default=None)
    parser.add_argument('--load_dir', type=str, default='../nus_dataset')
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    if args.viz:
        visualize(args)
    elif args.test_ckpt is not None:
        test(args)
    else:
        train(args)
