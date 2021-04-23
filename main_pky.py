import os
import argparse

import torch
from torch.utils.data import DataLoader, ChainDataset, random_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from Proposed.models import Global_Scene_CAM_NFDecoder, Global_Scene_CAM_DSF_NFDecoder
from Proposed.utils import ModelTrainer, ModelTest
from R2P2_MA.model_utils import MSE_Ploss, Interpolated_Ploss

from pkyutils import DatasetQ10, nuscenes_collate, NusCustomParser
from nuscenes.prediction.input_representation.combinators import Rasterizer

import cv2
import natsort
import hydra
from omegaconf import OmegaConf
import pickle

combinator = Rasterizer()

np.random.seed(777)
torch.manual_seed(777)


def get_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = args.scene_channels
    num_future = int(args.sampling_time * args.sampling_rate)

    discriminator = None  # MATF_GAN
    ioc = None  # DESIRE

    if args.model_type == 'SimpleEncDec':
        from MATF.models import SimpleEncoderDecoder
        from MATF.utils import ModelTrainer
        model = SimpleEncoderDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                     lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
                                     noise_dim=args.noise_dim)

    elif args.model_type == 'SocialPooling':
        from MATF.models import SocialPooling
        from MATF.utils import ModelTrainer
        model = SocialPooling(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                              lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                              pooling_size=args.pooling_size)

    elif args.model_type == 'MATF':
        from MATF.models import MATF
        from MATF.utils import ModelTrainer
        model = MATF(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                     lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                     pooling_size=args.pooling_size,
                     encoder_type=args.scene_encoder, scene_channels=scene_channels,
                     scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)

    elif args.model_type == 'MATF_GAN':
        from MATF_GAN.models import MATF_Gen, MATF_Disc
        from MATF_GAN.utils import ModelTrainer
        model = MATF_Gen(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                         lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout, noise_dim=args.noise_dim,
                         pooling_size=args.pooling_size, encoder_type=args.scene_encoder, scene_channels=scene_channels,
                         scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet)
        discriminator = MATF_Disc(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                  lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
                                  noise_dim=args.noise_dim,
                                  pooling_size=args.pooling_size, encoder_type=args.scene_encoder,
                                  scene_channels=scene_channels,
                                  scene_dropout=args.scene_dropout, freeze_resnet=args.freeze_resnet,
                                  disc_hidden=args.disc_hidden, disc_dropout=args.disc_dropout)

    elif args.model_type == 'R2P2_SimpleRNN':
        from R2P2_MA.models import R2P2_SimpleRNN
        from R2P2_MA.utils import ModelTrainer
        model = R2P2_SimpleRNN(
            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=num_future)

    elif args.model_type == 'R2P2_RNN':
        from R2P2_MA.models import R2P2_RNN
        from R2P2_MA.utils import ModelTrainer
        model = R2P2_RNN(scene_channels=scene_channels, velocity_const=args.velocity_const,
                         num_candidates=args.num_candidates, decoding_steps=num_future)

    elif args.model_type == "Desire":
        from Desire.models import DESIRE_SGM, DESIRE_IOC
        from Desire.utils import ModelTrainer
        model = DESIRE_SGM(decoding_steps=num_future,
                           num_candidates=args.num_candidates)
        ioc = DESIRE_IOC(in_channels=scene_channels, decoding_steps=num_future)

    elif args.model_type == 'CAM':
        from Proposed.models import CAM
        from Proposed.utils import ModelTrainer
        model = CAM(device=device, embedding_dim=args.agent_embed_dim,
                    nfuture=num_future, att_dropout=args.att_dropout)

    elif args.model_type == 'CAM_NFDecoder':
        from Proposed.models import CAM_NFDecoder
        from Proposed.utils import ModelTrainer
        model = CAM_NFDecoder(
            device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future, att_dropout=args.att_dropout,
            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=num_future)

    elif args.model_type == 'Scene_CAM_NFDecoder':
        from Proposed.models import Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer
        model = Scene_CAM_NFDecoder(
            device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future, att_dropout=args.att_dropout,
            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=num_future)

    elif args.model_type == 'Global_Scene_CAM_NFDecoder' or args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        from Proposed.models import Global_Scene_CAM_NFDecoder
        from Proposed.utils import ModelTrainer
        cross_modal_attention = True if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder' else False
        model = Global_Scene_CAM_NFDecoder(
            device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future, att_dropout=args.att_dropout,
            velocity_const=args.velocity_const, num_candidates=args.num_candidates, decoding_steps=num_future,
            att=cross_modal_attention)

    elif args.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
        from Proposed.models import Global_Scene_CAM_DSF_NFDecoder
        from Proposed.utils import ModelTrainer
        model = Global_Scene_CAM_DSF_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                               att_dropout=args.att_dropout,
                                               velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                               decoding_steps=num_future, att=False)

    elif args.model_type == 'AttGlobal_Scene_CAM_DSF_NFDecoder':
        from Proposed.models import Global_Scene_CAM_DSF_NFDecoder
        from Proposed.utils import ModelTrainer
        model = Global_Scene_CAM_DSF_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                               att_dropout=args.att_dropout,
                                               velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                               decoding_steps=num_future, att=True)

    elif args.model_type == 'goalMLP':
        from PathMLP.models import Global_Scene_CAM_Goal_NFDecoder, Only_Global_Scene_CAM_Goal_NFDecoder, Global_Scene_CAM_Path_NFDecoder
        crossmodal_attention = True
        goal_model = Global_Scene_CAM_Goal_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                                     att_dropout=args.att_dropout,
                                                     velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                                     att=crossmodal_attention)
        path_model = Global_Scene_CAM_Path_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=num_future,
                                                     att_dropout=args.att_dropout,
                                                     velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                                     att=crossmodal_attention)
        model = (goal_model, path_model)

    else:
        raise ValueError("Unknown model type {:s}.".format(args.model_type))

    if discriminator is not None:
        return model, discriminator
    elif ioc is not None:
        return model, ioc

    return model


def get_ploss_criterion(args):
    ploss_type = args.ploss_type
    if ploss_type == 'mseloss':
        from R2P2_MA.model_utils import MSE_Ploss
        ploss_criterion = MSE_Ploss()
    else:
        from R2P2_MA.model_utils import Interpolated_Ploss
        ploss_criterion = Interpolated_Ploss()
    return ploss_criterion


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Unknown optimizer {:s}.".format(args.optimizer))
    return optimizer


def get_dataloader(cfg):
    from pkyutils_ import NusCustomDataset, nuscenes_collate

    # filtered angle
    min_angle = cfg.model.min_angle if cfg.model.min_angle > 0 else None
    max_angle = cfg.model.max_angle if cfg.model.max_angle > 0 else None
    print('min angle:', str(min_angle), ', max angle:', str(max_angle))

    # dataset
    train_loader, valid_loader, test_loader = None, None, None
    if cfg.model.mode == 'train':
        train_dataset = NusCustomDataset(
            load_dir=cfg.dataset.load_dir, split='train', shuffle=True, min_angle=min_angle, max_angle=max_angle)
        val_dataset = NusCustomDataset(
            load_dir=cfg.dataset.load_dir, split='train_val', shuffle=False, min_angle=min_angle, max_angle=max_angle)
        train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=lambda x: nuscenes_collate(x), num_workers=cfg.model.num_workers)
        valid_loader = DataLoader(val_dataset, batch_size=cfg.model.batch_size, shuffle=False, pin_memory=True,
                                  collate_fn=lambda x: nuscenes_collate(x), num_workers=1)
        print(
            f'Train Examples: {len(train_dataset)} | Valid Examples: {len(val_dataset)}')
    else:
        test_dataset = NusCustomDataset(
            load_dir=cfg.dataset.load_dir, split='val', shuffle=False, min_angle=min_angle, max_angle=max_angle)
        test_loader = DataLoader(test_dataset, batch_size=cfg.model.batch_size, shuffle=False, pin_memory=True,
                                 collate_fn=lambda x: nuscenes_collate(x), num_workers=1)
        print(f'Test Examples: {len(test_dataset)}')

    return train_loader, valid_loader, test_loader


def train(cfg, model, train_loader, valid_loader, optimizer, device, ploss_criterion, discriminator=None, ioc=None):
    exp_path = os.path.join(hydra.utils.get_original_cwd(),
                            cfg.model.exp_path)  # checkpoint path

    # Trainer
    if cfg.model.model_type in ['SimpleEncDec', 'SocialPooling', 'MATF']:
        from MATF.utils import ModelTrainer
        trainer = ModelTrainer(
            model, train_loader, valid_loader, optimizer, exp_path, cfg.model, device)

    elif cfg.model.model_type == 'MATF_GAN':
        from MATF_GAN.utils import ModelTrainer
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=cfg.model.learning_rate, weight_decay=1e-4)
        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path,
                               cfg.model, device, discriminator, optimizer_d)

    elif cfg.model.model_type in ['R2P2_SimpleRNN', 'R2P2_RNN']:
        from R2P2_MA.utils import ModelTrainer
        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path,
                               cfg.model, device, ploss_criterion)

    elif cfg.model.model_type == "Desire":
        from Desire.utils import ModelTrainer
        optimizer_ioc = torch.optim.Adam(
            ioc.parameters(), lr=cfg.model.learning_rate, weight_decay=1e-4)
        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path,
                               cfg.model, device, ioc.to(device), optimizer_ioc)

    elif cfg.model.model_type in ['CAM', 'Scene_CAM', 'CAM_NFDecoder', 'Scene_CAM_NFDecoder',
                                  'Global_Scene_CAM_NFDecoder', 'AttGlobal_Scene_CAM_NFDecoder',
                                  'Global_Scene_CAM_DSF_NFDecoder', 'AttGlobal_Scene_CAM_DSF_NFDecoder']:
        from Proposed.utils import ModelTrainer
        trainer = ModelTrainer(model, train_loader, valid_loader, optimizer, exp_path,
                               cfg.model, device, ploss_criterion)
    else:
        raise ValueError(
            "Unknown model type {:s}.".format(cfg.model.model_type))

    trainer.train(cfg.model.num_epochs)


def test(cfg, model, test_loader, device, ploss_criterion, ioc=None):
    test_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.model.test_dir)
    if not os.path.isdir(test_path):
        os.mkdir(test_path)

    # Tester
    if cfg.model.model_type in ['SimpleEncDec', 'SocialPooling', 'MATF']:
        from MATF.utils import ModelTest
        tester = ModelTest(model, test_loader, cfg.model, device)

    elif cfg.model.model_type == 'MATF_GAN':
        from MATF_GAN.utils import ModelTest
        tester = ModelTest(model, test_loader, cfg.model, device)

    elif cfg.model.model_type in ['R2P2_SimpleRNN', 'R2P2_RNN']:
        from R2P2_MA.utils import ModelTest
        tester = ModelTest(model, test_loader, cfg.model,
                           device, ploss_criterion)

    elif cfg.model.model_type == "Desire":
        from Desire.utils import ModelTest
        tester = ModelTest(model, ioc, test_loader, cfg.model, device)

    elif cfg.model.model_type in ['CAM', 'Scene_CAM', 'CAM_NFDecoder', 'Scene_CAM_NFDecoder',
                                  'Global_Scene_CAM_NFDecoder', 'AttGlobal_Scene_CAM_NFDecoder',
                                  'Global_Scene_CAM_DSF_NFDecoder', 'AttGlobal_Scene_CAM_DSF_NFDecoder']:
        from Proposed.utils import ModelTest
        tester = ModelTest(model, test_loader, cfg.model,
                           device, ploss_criterion)

    elif cfg.model.model_type == 'goalMLP':
        from PathMLP.utils import ModelTest
        goal_model, path_model = model
        tester = ModelTest(goal_model, path_model, test_loader,
                           cfg.model, device, ploss_criterion)

    elif args.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
        pass

    else:
        raise ValueError(
            "Unknown model type {:s}.".format(cfg.model.model_type))

    tester.run()


def visualize(cfg, model, test_loader, device):
    print('Starting visualization.....')
    model.eval()  # Set model to evaluate mode.

    checkpoint = torch.load(cfg.model.test_ckpt,
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'], strict=False)

    w = cfg.dataset.meters_left + cfg.dataset.meters_right
    h = cfg.dataset.meters_ahead + cfg.dataset.meters_behind
    scene_size = (w, h)

    results_dir = os.path.join(os.getcwd(), 'visualization')
    os.mkdir(results_dir)
    print('save path: {}'.format(results_dir))

    plt.figure(figsize=(10, 10))
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
        exit(0) if event.key == 'escape' else None])
    with torch.no_grad():
        for b, batch in tqdm(enumerate(test_loader), desc='visualize on batch', total=len(test_loader)):
            # predicted paths
            scene_tks, predicts, poses = predict_path(
                cfg, model, batch, device, scene_size)  # (a, can, trj, 2)
            # load scene
            sample_dir = os.path.join(cfg.dataset.load_dir, scene_tks[0])
            with open('{}/viz.bin'.format(sample_dir), 'rb') as f:
                scene_img = pickle.load(f)
            with open('{}/episode.bin'.format(sample_dir), 'rb') as f:
                # episode: past, past_len, future, future_len, agent_mask, vel, pos
                episode = pickle.load(f)
                past, past_len, future, future_len, agent_mask, vel, pos, _, _ = episode
                future = [future[i]
                          for i in np.arange(len(agent_mask))[agent_mask]]
                total_agent_pose = np.array(pos).reshape(-1, 2)
                decoded_agent_pose = (np.array(pos)[agent_mask]).reshape(-1, 2)
            # draw scene
            plt.title("Predicted")
            plt.imshow(scene_img, extent=[-h // 2,
                       h // 2, -w // 2, w // 2], alpha=0.3)
            plt.xlim(-w // 2, w // 2)
            plt.ylim(-h // 2, h // 2)
            plt.scatter(total_agent_pose[:, 0],
                        total_agent_pose[:, 1], color='r')
            plt.scatter(decoded_agent_pose[:, 0],
                        decoded_agent_pose[:, 1], color='b')
            for gt_past in np.array(past):
                plt.plot(gt_past[:, 0], gt_past[:, 1],
                         color='salmon', alpha=0.5, linewidth=6)
            for gt_future, gt_pose in zip(np.array(future), decoded_agent_pose):
                plt.plot(np.append(gt_pose[0], gt_future[:, 0]), np.append(gt_pose[1], gt_future[:, 1]),
                         color='steelblue', alpha=0.3, linewidth=6)
            # draw predicted
            for i in range(len(predicts)):
                paths = np.insert(predicts[i], 0, poses[i], axis=1)
                for path in paths:
                    plt.plot(path[:, 0], path[:, 1], color='r')

            plt.savefig(results_dir + '/{}.png'.format(b), dpi=150)
            print(results_dir + '/{}.png'.format(b))
            plt.pause(0.1)
            plt.cla()

    print('Visualization finished!')


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(coordinate=None, distance=None)
def predict_path(cfg, model, batch, device, scene_size):
    W, H = scene_size
    if predict_path.coordinate is None or predict_path.distance is None:
        coordinate_2d = np.indices((H, W))
        coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
        coordinate = torch.FloatTensor(coordinate)
        coordinate = coordinate.reshape((1, 1, H, W))

        coordinate_std, coordinate_mean = torch.std_mean(coordinate)
        coordinate = (coordinate - coordinate_mean) / coordinate_std

        distance_2d = coordinate_2d - \
            np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
        distance = np.sqrt((distance_2d ** 2).sum(axis=0))
        distance = torch.FloatTensor(distance)
        distance = distance.reshape((1, 1, H, W))

        distance_std, distance_mean = torch.std_mean(distance)
        distance = (distance - distance_mean) / distance_std

        predict_path.coordinate = coordinate.to(device)
        predict_path.distance = distance.to(device)
    else:
        pass

    scene_images, log_prior, \
        agent_masks, \
        num_src_trajs, src_trajs, src_lens, src_len_idx, \
        num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
        tgt_two_mask, tgt_three_mask, \
        decode_start_vel, decode_start_pos, scene_tks, predict_mask, instance_tokens = batch

    # Detect dynamic batch size
    batch_size = scene_images.size(0)
    num_three_agents = torch.sum(tgt_three_mask)

    coordinate_batch = predict_path.coordinate.repeat(batch_size, 1, 1, 1)
    distance_batch = predict_path.distance.repeat(batch_size, 1, 1, 1)
    scene_images = torch.cat(
        (scene_images.to(device), coordinate_batch, distance_batch), dim=1)

    src_trajs = src_trajs.to(device)
    src_lens = src_lens.to(device)

    tgt_trajs = tgt_trajs.to(device)[tgt_three_mask]

    num_tgt_trajs = num_tgt_trajs.to(device)
    episode_idx = torch.arange(batch_size, device=device).repeat_interleave(num_tgt_trajs)[
        tgt_three_mask]

    agent_masks = agent_masks.to(device)
    agent_tgt_three_mask = torch.zeros_like(agent_masks)
    agent_masks_idx = torch.arange(len(agent_masks), device=device)[
        agent_masks][tgt_three_mask]
    agent_tgt_three_mask[agent_masks_idx] = True

    decode_start_vel = decode_start_vel.to(device)[agent_tgt_three_mask]
    decode_start_pos = decode_start_pos.to(device)[agent_tgt_three_mask]

    gen_trajs = None
    if cfg.model.flow_based_decoder:
        perterb = torch.normal(mean=0.0, std=np.sqrt(
            0.001), size=tgt_trajs.shape, device=device)
        motion_encoding_, scene_encoding_ = None, None
        if cfg.model.model_type == 'R2P2_SimpleRNN':
            z_, mu_, sigma_, motion_encoding_ = model.infer(
                tgt_trajs + perterb, src_trajs, decode_start_vel, decode_start_pos)
            gen_trajs, z, mu, sigma = model(
                motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

        elif cfg.model.model_type == 'R2P2_RNN':
            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = model.infer(
                tgt_trajs + perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)
            gen_trajs, z, mu, sigma = model(
                motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_,
                motion_encoded=True, scene_encoded=True)

        elif cfg.model.model_type == 'CAM_NFDecoder':
            z_, mu_, sigma_, motion_encoding_ = model.infer(tgt_trajs + perterb, src_trajs,
                                                            src_lens, agent_tgt_three_mask,
                                                            decode_start_vel, decode_start_pos,
                                                            num_src_trajs)
            gen_trajs, z, mu, sigma = model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                            decode_start_vel, decode_start_pos, num_src_trajs,
                                            agent_encoded=True)

        elif cfg.model.model_type in ['Scene_CAM_NFDecoder', 'Global_Scene_CAM_NFDecoder',
                                      'AttGlobal_Scene_CAM_NFDecoder']:
            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = model.infer(tgt_trajs + perterb,
                                                                             src_trajs, src_lens,
                                                                             agent_tgt_three_mask,
                                                                             episode_idx,
                                                                             decode_start_vel,
                                                                             decode_start_pos,
                                                                             num_src_trajs,
                                                                             scene_images)
            gen_trajs, z, mu, sigma = model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                            episode_idx, decode_start_vel, decode_start_pos,
                                            num_src_trajs, scene_encoding_, agent_encoded=True,
                                            scene_encoded=True)

    elif 'CAM' == cfg.model.model_type:
        gen_trajs = model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel,
                          decode_start_pos, num_src_trajs)
        gen_trajs = gen_trajs.reshape(
            num_three_agents, cfg.model.num_candidates, cfg.model.decoding_steps, 2)

    else:
        raise ValueError(
            "Unknown model type {:s}.".format(cfg.model.model_type))

    return scene_tks, gen_trajs.cpu().numpy(), decode_start_pos.cpu().numpy()


def posthoc(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_channels = 5 if args.map_version == '2.1' else 3
    nfuture = int(3 * args.sampling_rate)

    from Proposed.models import Global_Scene_CAM_DSF_NFDecoder
    from Proposed.posthoc_utils import DSFTrainer, ModelDSFTest

    if args.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
        crossmodal_attention = True
    else:
        crossmodal_attention = False
    model = Global_Scene_CAM_DSF_NFDecoder(device=device, agent_embed_dim=args.agent_embed_dim, nfuture=nfuture,
                                           att_dropout=args.att_dropout,
                                           velocity_const=args.velocity_const, num_candidates=args.num_candidates,
                                           decoding_steps=nfuture, att=crossmodal_attention)

    use_scene = True
    scene_size = (64, 64)
    ploss_type = args.ploss_type

    if ploss_type == 'mseloss':
        from R2P2_MA.model_utils import MSE_Ploss
        ploss_criterion = MSE_Ploss().to(device)
    elif ploss_type == 'map':
        from R2P2_MA.model_utils import Interpolated_Ploss
        ploss_criterion = Interpolated_Ploss().to(device)
    elif ploss_type == 'all':
        from R2P2_MA.model_utils import MSE_Ploss, Interpolated_Ploss
        ploss_criterion = (MSE_Ploss().to(device),
                           Interpolated_Ploss().to(device))

    model = model.to(device)

    from dataset.nuscenes import NuscenesDataset, nuscenes_collate

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

    print(
        f'Train Examples: {len(train_dataset)} | Valid Examples: {len(val_dataset)}')

    # Model optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None

    # Trainer
    exp_path = args.exp_path

    trainer = DSFTrainer(model, train_loader, valid_loader,
                         optimizer, exp_path, args, device, ploss_criterion)
    trainer.traindsf(args.num_epochs)

    # tester = ModelDSFTest(model, train_loader, args, device, ploss_criterion)
    # tester.run_draw()

    # if not os.path.isfile(args.posthoc_tune):
    #     print('training from initial...')
    #
    #     trainer = DSFTrainer(model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion)
    #     trainer.traindsf(args.num_epochs)
    #
    # else:
    #     print('pretrained weight: {}'.format(args.posthoc_tune))
    #
    #     tester = ModelDSFTest(model, train_loader, args, device, ploss_criterion)
    #     tester.run_draw()


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    print("Task: {}".format(cfg.model.mode))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.model.gpu_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # Dataset
    train_loader, valid_loader, test_loader = get_dataloader(cfg)

    # Model
    model, discriminator, ioc = None, None, None
    if cfg.model.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or "NFDecoder" in cfg.model.model_type:
        model = get_model(cfg.model)
        model = model.to(device)
    elif cfg.model.model_type == "MATF_GAN":
        model, discriminator = get_model(cfg.model)
        discriminator = discriminator.to(device)
        model = model.to(device)
    elif cfg.model.model_type == "Desire":
        model, ioc = get_model(cfg.model)
        ioc = ioc.to(device)
        model = model.to(device)
    elif cfg.model.model_type == 'goalMLP':
        model = get_model(cfg.model)
        model = (model[0].to(device), model[1].to(device))
    else:
        model = get_model(cfg.model)
        model = model.to(device)

    # Model optimizer
    if cfg.model.model_type == 'goalMLP':
        if cfg.model.optimizer == 'adam':
            optimizer = torch.optim.Adam(list(model[0].parameters(
            )) + list(model[1].parameters()), lr=args.learning_rate, weight_decay=1e-4)
        elif cfg.model.optimizer == 'sgd':
            optimizer = torch.optim.SGD(list(model[0].parameters(
            )) + list(model[1].parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = get_optimizer(cfg.model, model)

    ploss_criterion = get_ploss_criterion(cfg.model)
    ploss_criterion = ploss_criterion.to(device)

    if cfg.model.mode == 'train':
        train(cfg, model, train_loader, valid_loader, optimizer,
              device, ploss_criterion, discriminator, ioc)
    elif cfg.model.mode == 'test':
        test(cfg, model, test_loader, device, ploss_criterion, ioc=ioc)
    elif cfg.model.mode == 'viz':
        visualize(cfg, model, test_loader, device)
    else:
        raise ValueError("Unknown mode {:s}.".format(cfg.model.mode))


if __name__ == "__main__":
    main()
