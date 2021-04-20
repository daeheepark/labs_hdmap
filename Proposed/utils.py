import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

import matplotlib.pyplot as plt
import cv2
import torch
import pdb
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import logging

from multiprocessing import Pool


def log_determinant(sigma):
    det = sigma[:, :, 0, 0] * sigma[:, :, 1, 1] - sigma[:, :, 0, 1] ** 2
    logdet = torch.log(det + 1e-9)

    return logdet


class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion=None):

        self.exp_path = os.path.join(exp_path, args.tag + '_' + datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_'))
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info(f'Current Exp Path: {self.exp_path}')

        self.writter = SummaryWriter(os.path.join(self.exp_path, 'logs'))

        self.model_type = args.model_type
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

        self.decoding_steps = int(3 * args.sampling_rate)
        self.encoding_steps = int(2 * args.sampling_rate)

        if self.model_type in ['R2P2_SimpleRNN', 'CAM', 'CAM_NFDecoder']:
            self.map_version = None
        else:
            self.map_version = '2.0'

        self.beta = args.beta
        self.ploss_type = args.ploss_type
        self.ploss_criterion = ploss_criterion

        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=(1 / 2), verbose=True, patience=3)

        if args.load_ckpt:
            self.load_checkpoint(args.load_ckpt)

        # Other Parameters
        self.best_valid_ade = 1e9
        self.best_valid_fde = 1e9
        self.start_epoch = args.start_epoch

        if self.model_type in ["R2P2_SimpleRNN", "R2P2_RNN"] or 'NFDecoder' in self.model_type:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

    def train(self, num_epochs):
        self.logger.info('Model Type: ' + str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info(
                "==========================================================================================")

            train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch()
            valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.inference()

            # unwrapping ADEs/FDEs
            train_minade2, train_avgade2, train_minade3, train_avgade3 = train_ades
            train_minfde2, train_avgfde2, train_minfde3, train_avgfde3 = train_fdes

            valid_minade2, valid_avgade2, valid_minade3, valid_avgade3 = valid_ades
            valid_minfde2, valid_avgfde2, valid_minfde3, valid_avgfde3 = valid_fdes

            self.best_valid_ade = min(valid_avgade3, self.best_valid_ade)
            self.best_valid_fde = min(valid_avgfde3, self.best_valid_fde)
            self.scheduler.step(scheduler_metric)

            logging_msg1 = (
                f'| Epoch: {epoch:02} | Train Loss: {train_loss:0.6f} '
                f'| Train minADE[2/3]: {train_minade2:0.4f} / {train_minade3:0.4f} | Train minFDE[2/3]: {train_minfde2:0.4f} / {train_minfde3:0.4f} '
                f'| Train avgADE[2/3]: {train_avgade2:0.4f} / {train_avgade3:0.4f} | Train avgFDE[2/3]: {train_avgfde2:0.4f} / {train_avgfde3:0.4f}'
            )

            logging_msg2 = (
                f'| Epoch: {epoch:02} | Valid Loss: {valid_loss:0.6f} '
                f'| Valid minADE[2/3]: {valid_minade2:0.4f} / {valid_minade3:0.4f} | Valid minFDE[2/3]: {valid_minfde2:0.4f} /{valid_minfde3:0.4f} '
                f'| Valid avgADE[2/3]: {valid_avgade2:0.4f} / {valid_avgade3:0.4f} | Valid avgFDE[2/3]: {valid_avgfde2:0.4f} /{valid_avgfde3:0.4f} '
                f'| Scheduler Metric: {scheduler_metric:0.4f} | Learning Rate: {self.get_lr():g}\n'
            )

            self.logger.info(
                "------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(
                epoch, qloss=valid_qloss, ploss=valid_ploss, ade=valid_minade3, fde=valid_minfde3)

            # Log values to Tensorboard
            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Train_QLoss', train_qloss, epoch)
            self.writter.add_scalar('data/Train_PLoss', train_ploss, epoch)
            self.writter.add_scalar('data/Learning_Rate', self.get_lr(), epoch)

            self.writter.add_scalar('data/Train_minADE2', train_minade2, epoch)
            self.writter.add_scalar('data/Train_minFDE2', train_minfde2, epoch)
            self.writter.add_scalar('data/Train_minADE3', train_minade3, epoch)
            self.writter.add_scalar('data/Train_minFDE3', train_minfde3, epoch)

            self.writter.add_scalar('data/Train_avgADE2', train_avgade2, epoch)
            self.writter.add_scalar('data/Train_avgFDE2', train_avgfde2, epoch)
            self.writter.add_scalar('data/Train_avgADE3', train_avgade3, epoch)
            self.writter.add_scalar('data/Train_avgFDE3', train_avgfde3, epoch)
            self.writter.add_scalar(
                'data/Scheduler_Metric', scheduler_metric, epoch)

            self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Valid_QLoss', valid_qloss, epoch)
            self.writter.add_scalar('data/Valid_PLoss', valid_ploss, epoch)
            self.writter.add_scalar('data/Valid_minADE2', valid_minade2, epoch)
            self.writter.add_scalar('data/Valid_minFDE2', valid_minfde2, epoch)
            self.writter.add_scalar('data/Valid_minADE3', valid_minade3, epoch)
            self.writter.add_scalar('data/Valid_minFDE3', valid_minfde3, epoch)

            self.writter.add_scalar('data/Valid_avgADE2', valid_avgade2, epoch)
            self.writter.add_scalar('data/Valid_avgFDE2', valid_avgfde2, epoch)
            self.writter.add_scalar('data/Valid_avgADE3', valid_avgade3, epoch)
            self.writter.add_scalar('data/Valid_avgFDE3', valid_avgfde3, epoch)

        self.writter.close()
        self.logger.info("Training Complete! ")
        self.logger.info(
            f'| Best Valid ADE: {self.best_valid_ade} | Best Valid FDE: {self.best_valid_fde} |')

    def train_single_epoch(self):
        """Trains the model for a single round."""
        self.model.train()
        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        if self.map_version == '2.0':
            """ Make position & distance embeddings for map v2.0"""
            with torch.no_grad():
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

            coordinate = coordinate.to(self.device)
            distance = distance.to(self.device)

        c1 = -self.decoding_steps * np.log(2 * np.pi)
        for b, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            scene_images, log_prior, \
                future_agent_masks, \
                num_past_agents, past_agents_traj, past_agents_traj_len, past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch

            # Detect dynamic sizes
            batch_size = scene_images.size(0)
            num_three_agents = torch.sum(three_mask)

            if self.map_version == '2.0':
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                scene_images = torch.cat(
                    (scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

            scene_images = scene_images.to(self.device)

            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)
            future_agents_traj = future_agents_traj.to(self.device)[three_mask]
            future_agents_traj_len = future_agents_traj_len.to(self.device)[
                three_mask]

            num_future_agents = num_future_agents.to(self.device)
            episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[
                three_mask]

            future_agent_masks = future_agent_masks.to(self.device)
            agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
            agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][
                three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            future_agent_masks = future_agent_masks.to(self.device)

            decode_start_vel = decode_start_vel.to(
                self.device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(
                self.device)[agent_tgt_three_mask]

            log_prior = log_prior.to(self.device)
            if self.flow_based_decoder:
                # Normalizing Flow (q loss)
                # z_: A X Td X 2
                # mu_: A X Td X 2
                # sigma_: A X Td X 2 X 2
                # Generate perturbation
                perterb = torch.normal(mean=0.0, std=np.sqrt(
                    0.001), size=future_agents_traj.shape, device=self.device)
                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = None, None, None, None, None
                if self.model_type == 'R2P2_SimpleRNN':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj + perterb, past_agents_traj,
                                                                         decode_start_vel, decode_start_pos)

                elif self.model_type == 'R2P2_RNN':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                          past_agents_traj, episode_idx,
                                                                                          decode_start_vel,
                                                                                          decode_start_pos,
                                                                                          scene_images)

                elif self.model_type == 'CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj + perterb, past_agents_traj,
                                                                         past_agents_traj_len, future_agent_masks,
                                                                         decode_start_vel, decode_start_pos,
                                                                         num_past_agents)

                elif self.model_type == 'Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                          past_agents_traj,
                                                                                          past_agents_traj_len,
                                                                                          future_agent_masks,
                                                                                          episode_idx, decode_start_vel,
                                                                                          decode_start_pos,
                                                                                          num_past_agents, scene_images)

                elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                          past_agents_traj,
                                                                                          past_agents_traj_len,
                                                                                          agent_tgt_three_mask,
                                                                                          episode_idx, decode_start_vel,
                                                                                          decode_start_pos,
                                                                                          num_past_agents, scene_images)

                elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                          past_agents_traj,
                                                                                          past_agents_traj_len,
                                                                                          future_agent_masks,
                                                                                          episode_idx, decode_start_vel,
                                                                                          decode_start_pos,
                                                                                          num_past_agents, scene_images)

                '''
                z_ : torch.Size([agent_num, 6, 2])
                mu_ : torch.Size([agent_num, 6, 2])
                sigma_ : torch.Size([agent_num, 6, 2, 2])
                motion_encoding_ : torch.Size([agent_num, 128])
                scene_encoding_ : (torch.Size([3, 6, 100, 100]), torch.Size([3, 1024, 32]))
                '''

                z_ = z_.reshape((num_three_agents, -1))  # A X (Td*2)
                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))
                logdet_sigma = log_determinant(sigma_)
                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()

                # Prior Loss (p loss)
                gen_trajs, z, mu, sigma = None, None, None, None
                if self.model_type == 'R2P2_SimpleRNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos,
                                                         motion_encoded=True)

                elif self.model_type == 'R2P2_RNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel,
                                                         decode_start_pos, scene_encoding_, motion_encoded=True,
                                                         scene_encoded=True)

                elif self.model_type == 'CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                         decode_start_vel, decode_start_pos, num_past_agents,
                                                         agent_encoded=True)

                elif self.model_type == 'Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                         episode_idx, decode_start_vel, decode_start_pos,
                                                         num_past_agents, scene_encoding_, agent_encoded=True,
                                                         scene_encoded=True)

                elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                         episode_idx, decode_start_vel, decode_start_pos,
                                                         num_past_agents, scene_encoding_, agent_encoded=True,
                                                         scene_encoded=True)

                elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                         episode_idx, decode_start_vel, decode_start_pos,
                                                         num_past_agents, scene_encoding_, agent_encoded=True,
                                                         scene_encoded=True)

                if self.beta != 0.0:
                    if self.ploss_type == 'mseloss':
                        # ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
                        ploss = self.ploss_criterion(
                            gen_trajs, future_agents_traj)
                    else:
                        ploss = self.ploss_criterion(
                            episode_idx, gen_trajs, log_prior, -15.0)

                else:
                    ploss = torch.zeros(size=(1,), device=self.device)

                batch_ploss = ploss.mean()
                batch_loss = batch_qloss + self.beta * batch_ploss

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size

            else:

                if 'CAM' == self.model_type:
                    gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask,
                                           decode_start_vel, decode_start_pos, num_past_agents)

                gen_trajs = gen_trajs.reshape(
                    num_three_agents, self.num_candidates, self.decoding_steps, 2)

            rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(
                dim=-1).sqrt_()  # A X candi X T X 2 >> A X candi X T
            rs_error2 = rs_error3[..., :int(self.decoding_steps * 2 / 3)]

            num_agents = gen_trajs.size(0)
            num_agents2 = rs_error2.size(0)
            num_agents3 = rs_error3.size(0)

            ade2 = rs_error2.mean(-1)  # A X candi X T >> A X candi
            fde2 = rs_error2[..., -1]

            minade2, _ = ade2.min(dim=-1)  # A X candi >> A
            avgade2 = ade2.mean(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            avgfde2 = fde2.mean(dim=-1)

            batch_minade2 = minade2.mean()  # A >> 1
            batch_minfde2 = minfde2.mean()
            batch_avgade2 = avgade2.mean()
            batch_avgfde2 = avgfde2.mean()

            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]

            minade3, _ = ade3.min(dim=-1)
            avgade3 = ade3.mean(dim=-1)
            minfde3, _ = fde3.min(dim=-1)
            avgfde3 = fde3.mean(dim=-1)

            batch_minade3 = minade3.mean()
            batch_minfde3 = minfde3.mean()
            batch_avgade3 = avgade3.mean()
            batch_avgfde3 = avgfde3.mean()

            if self.flow_based_decoder is not True:
                batch_loss = batch_minade3
                epoch_loss += batch_loss.item()
                batch_qloss = torch.zeros(1)
                batch_ploss = torch.zeros(1)

            # Loss backward
            batch_loss.backward()
            self.optimizer.step()

            print("Working on train batch {:d}/{:d}... ".format(b + 1, len(self.train_loader)) +
                  "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(),
                                                                            batch_ploss.item()) +
                  "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

            epoch_minade2 += batch_minade2.item() * num_agents2
            epoch_avgade2 += batch_avgade2.item() * num_agents2
            epoch_minfde2 += batch_minfde2.item() * num_agents2
            epoch_avgfde2 += batch_avgfde2.item() * num_agents2
            epoch_minade3 += batch_minade3.item() * num_agents3
            epoch_avgade3 += batch_avgade3.item() * num_agents3
            epoch_minfde3 += batch_minfde3.item() * num_agents3
            epoch_avgfde3 += batch_avgfde3.item() * num_agents3

            epoch_agents += num_agents
            epoch_agents2 += num_agents2
            epoch_agents3 += num_agents3

        if self.flow_based_decoder:
            epoch_ploss /= epoch_agents
            epoch_qloss /= epoch_agents
            epoch_loss = epoch_qloss + self.beta * epoch_ploss
        else:
            epoch_loss /= epoch_agents

        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = [epoch_minade2, epoch_avgade2,
                      epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2,
                      epoch_minfde3, epoch_avgfde3]

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes

    def inference(self):
        self.model.eval()  # Set model to evaluate mode.

        epoch_loss = 0.0
        epoch_qloss = 0.0
        epoch_ploss = 0.0
        epoch_minade2, epoch_avgade2 = 0.0, 0.0
        epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
        epoch_minade3, epoch_avgade3 = 0.0, 0.0
        epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
        epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

        H = W = 64
        with torch.no_grad():
            if self.map_version == '2.0':
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

                coordinate = coordinate.to(self.device)
                distance = distance.to(self.device)

            c1 = -self.decoding_steps * np.log(2 * np.pi)
            for b, batch in enumerate(self.valid_loader):

                scene_images, log_prior, \
                    future_agent_masks, \
                    num_past_agents, past_agents_traj, past_agents_traj_len, past_agents_traj_len_idx, \
                    num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                    two_mask, three_mask, \
                    decode_start_vel, decode_start_pos, \
                    scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                num_three_agents = torch.sum(three_mask)

                if self.map_version == '2.0':
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat(
                        (scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)[
                    three_mask]
                future_agents_traj_len = future_agents_traj_len.to(self.device)[
                    three_mask]

                num_future_agents = num_future_agents.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[
                    three_mask]

                future_agent_masks = future_agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][
                    three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(
                    self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(
                    self.device)[agent_tgt_three_mask]

                log_prior = log_prior.to(self.device)

                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # z: A X Td X 2
                    # mu: A X Td X 2
                    # sigma: A X Td X 2 X 2
                    # Generate perturbation
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape,
                                           device=self.device)

                    if self.model_type == 'R2P2_SimpleRNN':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                             past_agents_traj, decode_start_vel,
                                                                             decode_start_pos)

                    elif self.model_type == 'R2P2_RNN':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(
                            future_agents_traj + perterb, past_agents_traj, episode_idx, decode_start_vel,
                            decode_start_pos, scene_images)

                    elif self.model_type == 'CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                             past_agents_traj, past_agents_traj_len,
                                                                             agent_tgt_three_mask, decode_start_vel,
                                                                             decode_start_pos, num_past_agents)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(
                            future_agents_traj + perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask,
                            episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(
                            future_agents_traj + perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask,
                            episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(
                            future_agents_traj + perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask,
                            episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    z_ = z_.reshape((num_three_agents, -1))  # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    if self.model_type == 'R2P2_SimpleRNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos,
                                                             motion_encoded=True)

                    elif self.model_type == 'R2P2_RNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel,
                                                             decode_start_pos, scene_encoding_, motion_encoded=True,
                                                             scene_encoded=True)

                    elif self.model_type == 'CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len,
                                                             agent_tgt_three_mask, decode_start_vel, decode_start_pos,
                                                             num_past_agents, agent_encoded=True)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len,
                                                             agent_tgt_three_mask, episode_idx, decode_start_vel,
                                                             decode_start_pos, num_past_agents, scene_encoding_,
                                                             agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len,
                                                             agent_tgt_three_mask, episode_idx, decode_start_vel,
                                                             decode_start_pos, num_past_agents, scene_encoding_,
                                                             agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len,
                                                             agent_tgt_three_mask, episode_idx, decode_start_vel,
                                                             decode_start_pos, num_past_agents, scene_encoding_,
                                                             agent_encoded=True, scene_encoded=True)

                    if self.beta != 0.0:
                        if self.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(
                                gen_trajs, future_agents_traj)
                        else:
                            ploss = self.ploss_criterion(
                                episode_idx, gen_trajs, log_prior, -15.0)

                    else:
                        ploss = torch.zeros(size=(1,), device=self.device)

                    batch_ploss = ploss.mean()
                    batch_loss = batch_qloss + self.beta * batch_ploss

                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size

                else:

                    if 'CAM' == self.model_type:
                        gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask,
                                               decode_start_vel, decode_start_pos, num_past_agents)

                    gen_trajs = gen_trajs.reshape(
                        num_three_agents, self.num_candidates, self.decoding_steps, 2)

                rs_error3 = (
                    (gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps * 2 / 3)]

                num_agents = gen_trajs.size(0)
                num_agents2 = rs_error2.size(0)
                num_agents3 = rs_error3.size(0)

                ade2 = rs_error2.mean(-1)
                fde2 = rs_error2[..., -1]

                minade2, _ = ade2.min(dim=-1)
                avgade2 = ade2.mean(dim=-1)
                minfde2, _ = fde2.min(dim=-1)
                avgfde2 = fde2.mean(dim=-1)

                batch_minade2 = minade2.mean()
                batch_minfde2 = minfde2.mean()
                batch_avgade2 = avgade2.mean()
                batch_avgfde2 = avgfde2.mean()

                ade3 = rs_error3.mean(-1)
                fde3 = rs_error3[..., -1]

                minade3, _ = ade3.min(dim=-1)
                avgade3 = ade3.mean(dim=-1)
                minfde3, _ = fde3.min(dim=-1)
                avgfde3 = fde3.mean(dim=-1)

                batch_minade3 = minade3.mean()
                batch_minfde3 = minfde3.mean()
                batch_avgade3 = avgade3.mean()
                batch_avgfde3 = avgfde3.mean()

                if self.flow_based_decoder is not True:
                    batch_loss = batch_minade3
                    epoch_loss += batch_loss.item()
                    batch_qloss = torch.zeros(1)
                    batch_ploss = torch.zeros(1)

                print("Working on valid batch {:d}/{:d}... ".format(b + 1, len(self.valid_loader)) +
                      "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(),
                                                                                batch_ploss.item()) +
                      "minFDE3: {:.2f}, avgFDE3: {:.2f}".format(batch_minfde3.item(), batch_avgfde3.item()), end='\r')

                epoch_ploss += batch_ploss.item() * batch_size
                epoch_qloss += batch_qloss.item() * batch_size
                epoch_minade2 += batch_minade2.item() * num_agents2
                epoch_avgade2 += batch_avgade2.item() * num_agents2
                epoch_minfde2 += batch_minfde2.item() * num_agents2
                epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                epoch_minade3 += batch_minade3.item() * num_agents3
                epoch_avgade3 += batch_avgade3.item() * num_agents3
                epoch_minfde3 += batch_minfde3.item() * num_agents3
                epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                epoch_agents += num_agents
                epoch_agents2 += num_agents2
                epoch_agents3 += num_agents3

        if self.flow_based_decoder:
            epoch_ploss /= epoch_agents
            epoch_qloss /= epoch_agents
            epoch_loss = epoch_qloss + self.beta * epoch_ploss
        else:
            epoch_loss /= epoch_agents

        # 2-Loss
        epoch_minade2 /= epoch_agents2
        epoch_avgade2 /= epoch_agents2
        epoch_minfde2 /= epoch_agents2
        epoch_avgfde2 /= epoch_agents2

        # 3-Loss
        epoch_minade3 /= epoch_agents3
        epoch_avgade3 /= epoch_agents3
        epoch_minfde3 /= epoch_agents3
        epoch_avgfde3 /= epoch_agents3

        epoch_ades = (epoch_minade2, epoch_avgade2,
                      epoch_minade3, epoch_avgade3)
        epoch_fdes = (epoch_minfde2, epoch_avgfde2,
                      epoch_minfde3, epoch_avgfde3)

        scheduler_metric = epoch_avgade3 + epoch_avgfde3

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric

    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde, qloss=0., ploss=0.):
        """Saves experiment checkpoint.
        Saved state consits of epoch, model state, optimizer state, current
        learning rate and experiment path.
        """

        state_dict = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.get_lr(),
            'exp_path': self.exp_path,
            'val_ploss': ploss,
            'val_qloss': qloss,
            'val_ade': ade,
            'val_fde': fde,
        }

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, qloss, ploss, ade,
                                                                              fde)
        torch.save(state_dict, save_path)

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)


class ModelTest:
    def __init__(self, model, data_loader, args, device, ploss_criterion=None):
        self.model = model
        self.data_loader = data_loader

        self.ploss_type = args.ploss_type
        self.ploss_criterion = ploss_criterion

        self.beta = args.beta
        self.num_candidates = args.num_candidates
        self.decoding_steps = int(3 * args.sampling_rate)
        self.model_type = args.model_type

        if self.model_type in ['R2P2_SimpleRNN', 'CAM', 'CAM_NFDecoder']:
            self.map_version = None
        else:
            self.map_version = '2.0'

        if self.model_type in ["R2P2_SimpleRNN", "R2P2_RNN", "CAM_NFDecoder", "Scene_CAM_NFDecoder",
                               "Global_Scene_CAM_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder",
                               "Global_Scene_CAM_DSF_NFDecoder", "AttGlobal_Scene_CAM_DSF_NFDecoder"]:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

        self.device = device
        self.out_dir = args.test_dir
        self.render = args.test_render
        self.test_times = args.test_times

        import hydra
        # self.data_dir = os.path.join(args.load_dir, args.version)
        self.data_dir = hydra.utils.to_absolute_path(args.load_dir)

        # self.test_ckpt = args.test_ckpt.split('/')[-2] + "_" + args.test_ckpt.split('/')[-1]
        self.test_ckpt = args.test_ckpt
        self.test_partition = self.data_dir.split('/')[-1]

        # if args.dataset == "argoverse":
        #     _data_dir = './data/argoverse'
        #     self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.png' for x
        #                                       in scene_id]
        #
        # elif args.dataset == "nuscenes":
        #     _data_dir = './data/nuscenes'
        #     self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x
        #                                       in scene_id]
        #
        # elif args.dataset == "carla":
        #     _data_dir = './data/carla'
        #     self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x
        #                                       in scene_id]

        self.load_checkpoint(args.test_ckpt)

    def load_checkpoint(self, ckpt):
        print('load checkpoint from:', ckpt)
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

    def run(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.

        list_loss = []
        list_qloss = []
        list_ploss = []
        list_minade2, list_avgade2 = [], []
        list_minfde2, list_avgfde2 = [], []
        list_minade3, list_avgade3 = [], []
        list_minfde3, list_avgfde3 = [], []
        list_minmsd, list_avgmsd = [], []

        list_dao = []
        list_dac = []

        minFSD3 = []
        maxFSD3 = []
        stdFD3 = []
        voAngles = []
        minFSD3_n = []
        maxFSD3_n = []

        miss_or_not = []

        for test_time in range(self.test_times):

            epoch_loss = 0.0
            epoch_qloss = 0.0
            epoch_ploss = 0.0
            epoch_minade2, epoch_avgade2 = 0.0, 0.0
            epoch_minfde2, epoch_avgfde2 = 0.0, 0.0
            epoch_minade3, epoch_avgade3 = 0.0, 0.0
            epoch_minfde3, epoch_avgfde3 = 0.0, 0.0
            epoch_minmsd, epoch_avgmsd = 0.0, 0.0
            epoch_agents, epoch_agents2, epoch_agents3 = 0.0, 0.0, 0.0

            epoch_dao = 0.0
            epoch_dac = 0.0
            dao_agents = 0.0
            dac_agents = 0.0

            epoch_min_sd = 0.0
            epoch_vo_angle = 0.0

            H = W = 64
            with torch.no_grad():
                if self.map_version == '2.0':
                    coordinate_2d = np.indices((H, W))
                    coordinate = np.ravel_multi_index(
                        coordinate_2d, dims=(H, W))
                    coordinate = torch.FloatTensor(coordinate)
                    coordinate = coordinate.reshape((1, 1, H, W))

                    coordinate_std, coordinate_mean = torch.std_mean(
                        coordinate)
                    coordinate = (coordinate - coordinate_mean) / \
                        coordinate_std

                    distance_2d = coordinate_2d - \
                        np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
                    distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                    distance = torch.FloatTensor(distance)
                    distance = distance.reshape((1, 1, H, W))

                    distance_std, distance_mean = torch.std_mean(distance)
                    distance = (distance - distance_mean) / distance_std

                    coordinate = coordinate.to(self.device)
                    distance = distance.to(self.device)

                c1 = -self.decoding_steps * np.log(2 * np.pi)

                for b, batch in enumerate(self.data_loader):
                    scene_images, log_prior, \
                        agent_masks, \
                        num_src_trajs, src_trajs, src_lens, src_len_idx, \
                        num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                        tgt_two_mask, tgt_three_mask, \
                        decode_start_vel, decode_start_pos, scene_id = batch

                    # scene_images[scene_images != 0] = 0

                    # Detect dynamic batch size
                    batch_size = scene_images.size(0)
                    num_three_agents = torch.sum(tgt_three_mask)

                    if self.map_version == '2.0':
                        coordinate_batch = coordinate.repeat(
                            batch_size, 1, 1, 1)
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
                    agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[
                        agent_masks][tgt_three_mask]
                    agent_tgt_three_mask[agent_masks_idx] = True

                    decode_start_vel = decode_start_vel.to(
                        self.device)[agent_tgt_three_mask]
                    decode_start_pos = decode_start_pos.to(
                        self.device)[agent_tgt_three_mask]

                    log_prior = log_prior.to(self.device)

                    if self.flow_based_decoder:
                        # Normalizing Flow (q loss)
                        # z: A X Td X 2
                        # mu: A X Td X 2
                        # sigma: A X Td X 2 X 2
                        # Generate perturbation
                        perterb = torch.normal(mean=0.0, std=np.sqrt(
                            0.001), size=tgt_trajs.shape, device=self.device)

                        if self.model_type == 'R2P2_SimpleRNN':
                            z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs + perterb, src_trajs,
                                                                                 decode_start_vel, decode_start_pos)

                        elif self.model_type == 'R2P2_RNN':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs + perterb,
                                                                                                  src_trajs,
                                                                                                  episode_idx,
                                                                                                  decode_start_vel,
                                                                                                  decode_start_pos,
                                                                                                  scene_images)

                        elif self.model_type == 'CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs + perterb, src_trajs,
                                                                                 src_lens, agent_tgt_three_mask,
                                                                                 decode_start_vel, decode_start_pos,
                                                                                 num_src_trajs)

                        elif self.model_type == 'Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs + perterb,
                                                                                                  src_trajs, src_lens,
                                                                                                  agent_tgt_three_mask,
                                                                                                  episode_idx,
                                                                                                  decode_start_vel,
                                                                                                  decode_start_pos,
                                                                                                  num_src_trajs,
                                                                                                  scene_images)

                        elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs + perterb,
                                                                                                  src_trajs, src_lens,
                                                                                                  agent_tgt_three_mask,
                                                                                                  episode_idx,
                                                                                                  decode_start_vel,
                                                                                                  decode_start_pos,
                                                                                                  num_src_trajs,
                                                                                                  scene_images)

                        elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs + perterb,
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

                        # Prior Loss (p loss)
                        if self.model_type == 'R2P2_SimpleRNN':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos,
                                                                 motion_encoded=True)

                        elif self.model_type == 'R2P2_RNN':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel,
                                                                 decode_start_pos, scene_encoding_, motion_encoded=True,
                                                                 scene_encoded=True)

                        elif self.model_type == 'CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                                                 decode_start_vel, decode_start_pos, num_src_trajs,
                                                                 agent_encoded=True)

                        elif self.model_type == 'Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                                                 episode_idx, decode_start_vel, decode_start_pos,
                                                                 num_src_trajs, scene_encoding_, agent_encoded=True,
                                                                 scene_encoded=True)

                        elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                                                 episode_idx, decode_start_vel, decode_start_pos,
                                                                 num_src_trajs, scene_encoding_, agent_encoded=True,
                                                                 scene_encoded=True)

                        elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask,
                                                                 episode_idx, decode_start_vel, decode_start_pos,
                                                                 num_src_trajs, scene_encoding_, agent_encoded=True,
                                                                 scene_encoded=True)

                        if self.beta != 0.0:
                            if self.ploss_type == 'mseloss':
                                ploss = self.ploss_criterion(
                                    gen_trajs, tgt_trajs)
                            else:
                                ploss = self.ploss_criterion(
                                    episode_idx, gen_trajs, log_prior, -15.0)

                        else:
                            ploss = torch.zeros(size=(1,), device=self.device)
                        batch_ploss = ploss.mean()
                        batch_loss = batch_qloss + self.beta * batch_ploss

                        epoch_ploss += batch_ploss.item() * batch_size
                        epoch_qloss += batch_qloss.item() * batch_size

                    else:

                        if 'CAM' == self.model_type:
                            gen_trajs = self.model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel,
                                                   decode_start_pos, num_src_trajs)

                        gen_trajs = gen_trajs.reshape(
                            num_three_agents, self.num_candidates, self.decoding_steps, 2)
                    # gen_trajs - [agents_num, 6, 6, 2]
                    rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1))
                                 ** 2).sum(dim=-1).sqrt_()  # [agents_num, 6, 6]
                    rs_error2 = rs_error3[..., :int(
                        self.decoding_steps * 2 / 3)]

                    diff = gen_trajs - tgt_trajs.unsqueeze(1)
                    msd_error = (diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2)

                    # print('######################################')
                    # print(gen_trajs.size())
                    # print(gen_trajs[0][:, 0, :].squeeze())  # start
                    # print(gen_trajs[0][:, 5, :].squeeze())  # final
                    # print(decode_start_pos[0])

                    # Miss Rate
                    rs_error3_max = rs_error3.max(
                        dim=-1).values  # [agents_num, 6]
                    for error3_max in rs_error3_max:
                        # True or False (miss: true)
                        miss_or_not.append(torch.min(error3_max >= 2.))

                    def cal_vo_angle(path1, path2):
                        vo_angles_ = []
                        for i in range(1, len(path1)):
                            u = path1[i] - path1[i - 1]
                            v = path2[i] - path2[i - 1]
                            vo = torch.acos(
                                (u * v).sum() / (torch.norm(u) * torch.norm(v)))
                            if torch.isnan(vo):
                                continue
                            vo_angles_.append(vo)
                        return torch.FloatTensor(vo_angles_).mean()

                    for agent_idx, paths in enumerate(gen_trajs):
                        vo_angles = torch.FloatTensor([cal_vo_angle(paths[i], paths[j])
                                                       for i in range(len(paths) - 1)
                                                       for j in range(i + 1, len(paths))])

                        f_points = paths[:, 5, :].squeeze(
                        ) - decode_start_pos[agent_idx]
                        fsds = torch.FloatTensor([torch.norm(f_points[i] - f_points[j])
                                                  for i in range(len(f_points) - 1)
                                                  for j in range(i + 1, len(f_points))])
                        min_fsd = torch.min(fsds)
                        max_fsd = torch.max(fsds)
                        # std_fsd = torch.std(fsds)
                        std_fd = torch.std(
                            f_points[:, 0]) + torch.std(f_points[:, 1])
                        vo_angle_mean = torch.mean(vo_angles)

                        fsds_n = torch.FloatTensor([torch.norm(f_points[i] - f_points[j])
                                                    / (torch.norm(f_points[i]) + torch.norm(f_points[j]))
                                                    for i in range(len(f_points) - 1)
                                                    for j in range(i + 1, len(f_points))])
                        min_fsd_n = torch.min(fsds_n)
                        max_fsd_n = torch.max(fsds_n)
                        minFSD3_n.append(min_fsd_n)
                        maxFSD3_n.append(max_fsd_n)

                        minFSD3.append(min_fsd)
                        maxFSD3.append(max_fsd)
                        stdFD3.append(std_fd)
                        voAngles.append(vo_angle_mean)

                    num_agents = gen_trajs.size(0)
                    num_agents2 = rs_error2.size(0)
                    num_agents3 = rs_error3.size(0)

                    ade2 = rs_error2.mean(-1)
                    fde2 = rs_error2[..., -1]

                    minade2, _ = ade2.min(dim=-1)
                    avgade2 = ade2.mean(dim=-1)
                    minfde2, _ = fde2.min(dim=-1)
                    avgfde2 = fde2.mean(dim=-1)

                    batch_minade2 = minade2.mean()
                    batch_minfde2 = minfde2.mean()
                    batch_avgade2 = avgade2.mean()
                    batch_avgfde2 = avgfde2.mean()

                    ade3 = rs_error3.mean(-1)
                    fde3 = rs_error3[..., -1]

                    msd = msd_error.mean(-1)
                    minmsd, _ = msd.min(dim=-1)
                    avgmsd = msd.mean(dim=-1)
                    batch_minmsd = minmsd.mean()
                    batch_avgmsd = avgmsd.mean()

                    minade3, _ = ade3.min(dim=-1)
                    avgade3 = ade3.mean(dim=-1)
                    minfde3, _ = fde3.min(dim=-1)
                    avgfde3 = fde3.mean(dim=-1)

                    batch_minade3 = minade3.mean()
                    batch_minfde3 = minfde3.mean()
                    batch_avgade3 = avgade3.mean()
                    batch_avgfde3 = avgfde3.mean()

                    if self.flow_based_decoder is not True:
                        batch_loss = batch_minade3
                        epoch_loss += batch_loss.item()
                        batch_qloss = torch.zeros(1)
                        batch_ploss = torch.zeros(1)

                    print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_time + 1, self.test_times, b + 1,
                                                                                  len(self.data_loader)), end='\r')  # +

                    epoch_ploss += batch_ploss.item() * batch_size
                    epoch_qloss += batch_qloss.item() * batch_size
                    epoch_minade2 += batch_minade2.item() * num_agents2
                    epoch_avgade2 += batch_avgade2.item() * num_agents2
                    epoch_minfde2 += batch_minfde2.item() * num_agents2
                    epoch_avgfde2 += batch_avgfde2.item() * num_agents2
                    epoch_minade3 += batch_minade3.item() * num_agents3
                    epoch_avgade3 += batch_avgade3.item() * num_agents3
                    epoch_minfde3 += batch_minfde3.item() * num_agents3
                    epoch_avgfde3 += batch_avgfde3.item() * num_agents3

                    epoch_minmsd += batch_minmsd.item() * num_agents3
                    epoch_avgmsd += batch_avgmsd.item() * num_agents3

                    epoch_agents += num_agents
                    epoch_agents2 += num_agents2
                    epoch_agents3 += num_agents3

                    map_files = [self.map_file(sample_idx)
                                 for sample_idx in scene_id]
                    # output_files = [self.out_dir + '/' + x[2] + '_' + x[3] + '.jpg' for x in scene_id]

                    cum_num_tgt_trajs = [
                        0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
                    cum_num_src_trajs = [
                        0] + torch.cumsum(num_src_trajs, dim=0).tolist()

                    src_trajs = src_trajs.cpu().numpy()
                    src_lens = src_lens.cpu().numpy()

                    tgt_trajs = tgt_trajs.cpu().numpy()
                    tgt_lens = tgt_lens.cpu().numpy()

                    zero_ind = np.nonzero(tgt_three_mask.numpy() == 0)[0]
                    zero_ind -= np.arange(len(zero_ind))

                    tgt_three_mask = tgt_three_mask.numpy()
                    agent_tgt_three_mask = agent_tgt_three_mask.cpu().numpy()

                    gen_trajs = gen_trajs.cpu().numpy()

                    src_mask = agent_tgt_three_mask

                    gen_trajs = np.insert(gen_trajs, zero_ind, 0, axis=0)

                    tgt_trajs = np.insert(tgt_trajs, zero_ind, 0, axis=0)
                    tgt_lens = np.insert(tgt_lens, zero_ind, 0, axis=0)

                    for i in range(batch_size):
                        candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                        tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]
                        tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]

                        src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]
                        src_lens_i = src_lens[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]
                        map_file_i = map_files[i]
                        # output_file_i = output_files[i]

                        candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                        tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                        tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]

                        src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]
                        src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]

                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)
                        # dao_i, dao_mask_i = self.dao(candidate_i, map_file_i, img=batch[0][i])
                        # dac_i, dac_mask_i = self.dac(candidate_i, map_file_i, img=batch[0][i])

                        epoch_dao += dao_i.sum()
                        dao_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_agents += dac_mask_i.sum()

                        epoch_min_sd += self.self_distance(candidate_i, tgt_traj_i, tgt_lens_i)
                        epoch_vo_angle += self.vo_angle(candidate_i, tgt_traj_i, tgt_lens_i)

            if self.flow_based_decoder:
                list_ploss.append(epoch_ploss / epoch_agents)
                list_qloss.append(epoch_qloss / epoch_agents)
                list_loss.append(epoch_qloss + self.beta * epoch_ploss)

            else:
                list_loss.append(epoch_loss / epoch_agents)

            # 2-Loss
            list_minade2.append(epoch_minade2 / epoch_agents2)
            list_avgade2.append(epoch_avgade2 / epoch_agents2)
            list_minfde2.append(epoch_minfde2 / epoch_agents2)
            list_avgfde2.append(epoch_avgfde2 / epoch_agents2)

            # 3-Loss
            list_minade3.append(epoch_minade3 / epoch_agents3)
            list_avgade3.append(epoch_avgade3 / epoch_agents3)
            list_minfde3.append(epoch_minfde3 / epoch_agents3)
            list_avgfde3.append(epoch_avgfde3 / epoch_agents3)

            list_minmsd.append(epoch_minmsd / epoch_agents3)
            list_avgmsd.append(epoch_avgmsd / epoch_agents3)

            list_dao.append(epoch_dao / dao_agents)
            list_dac.append(epoch_dac / dac_agents)

            list_sd.append(epoch_min_sd / epoch_agents)
            list_angle.append(epoch_vo_angle / epoch_agents)

        if self.flow_based_decoder:
            test_ploss = [np.mean(list_ploss), np.std(list_ploss)]
            test_qloss = [np.mean(list_qloss), np.std(list_qloss)]
            test_loss = [np.mean(list_loss), np.std(list_loss)]

        else:
            test_ploss = [0.0, 0.0]
            test_qloss = [0.0, 0.0]
            test_loss = [np.mean(list_loss), np.std(list_loss)]

        test_minade2 = [np.mean(list_minade2), np.std(list_minade2)]
        test_avgade2 = [np.mean(list_avgade2), np.std(list_avgade2)]
        test_minfde2 = [np.mean(list_minfde2), np.std(list_minfde2)]
        test_avgfde2 = [np.mean(list_avgfde2), np.std(list_avgfde2)]

        test_minade3 = [np.mean(list_minade3), np.std(list_minade3)]
        test_avgade3 = [np.mean(list_avgade3), np.std(list_avgade3)]
        test_minfde3 = [np.mean(list_minfde3), np.std(list_minfde3)]
        test_avgfde3 = [np.mean(list_avgfde3), np.std(list_avgfde3)]

        test_minmsd = [np.mean(list_minmsd), np.std(list_minmsd)]
        test_avgmsd = [np.mean(list_avgmsd), np.std(list_avgmsd)]

        test_dao = [np.mean(list_dao), np.std(list_dao)]
        test_dac = [np.mean(list_dac), np.std(list_dac)]

        test_ades = (test_minade2, test_avgade2, test_minade3, test_avgade3)
        test_fdes = (test_minfde2, test_avgfde2, test_minfde3, test_avgfde3)

        print("\n--Final Performance Report--")
        # print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(
        #     test_minade3[0], test_minade3[1], test_minfde3[0], test_minfde3[1]))
        # print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(
        #     test_avgade3[0], test_avgade3[1], test_avgfde3[0], test_avgfde3[1]))
        # print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(
        #     test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0], test_dac[1]))

        print("minADE3: {:.5f}".format(test_minade3[0]))
        print("minFDE3: {:.5f}".format(test_minfde3[0]))
        print("avgADE3: {:.5f}".format(test_avgade3[0]))
        print("avgFDE3: {:.5f}".format(test_avgfde3[0]))
        print("DAO: {:.5f}".format(test_dao[0] * 10000.0))
        print("DAC: {:.5f}".format(test_dac[0]))
        print("OffRoad rate: {:.5f}".format(1 - test_dac[0]))
        RF3 = test_avgfde3[0] / test_minfde3[0]
        print("RF3: {:.5f}".format(RF3))

        minFSD3 = torch.FloatTensor(minFSD3)
        maxFSD3 = torch.FloatTensor(maxFSD3)
        stdFD3 = torch.FloatTensor(stdFD3)
        voAngles = torch.FloatTensor(voAngles)
        minFSD3_n = torch.FloatTensor(minFSD3_n)
        maxFSD3_n = torch.FloatTensor(maxFSD3_n)
        miss_or_not = torch.FloatTensor(miss_or_not)
        print("minFSD3_n: {:.5f}".format(minFSD3_n.mean()))
        print("maxFSD3_n: {:.5f}".format(maxFSD3_n.mean()))
        print("minFSD3: {:.5f}".format(minFSD3.mean()))
        print("maxFSD3: {:.5f}".format(maxFSD3.mean()))
        print("stdFD3: {:.5f}".format(stdFD3.mean()))
        print("voAngles: {:.5f}".format(voAngles.mean()))
        print("miss rate: {:.5f}".format(miss_or_not.mean()))

        results_data = {
            'minADE3': list_minade3,
            'minFDE3': list_minfde3,
            'avgADE3': list_avgade3,
            'avgFDE3': list_avgfde3,
            'DAO': list_dao,
            'DAC': list_dac,
            'RF3': RF3,

            'minFSD3_n': minFSD3_n.cpu().numpy(),
            'maxFSD3_n': maxFSD3_n.cpu().numpy(),
            'minFSD3': minFSD3.cpu().numpy(),
            'maxFSD3': maxFSD3.cpu().numpy(),
            'stdFD3': stdFD3.cpu().numpy(),
            'voAngles': voAngles.cpu().numpy(),
            'miss_rate': miss_or_not.cpu().numpy()
        }

        with open('results.pkl', 'wb') as f:
            pkl.dump(results_data, f)

        plt.figure(figsize=(36, 4))
        plt.subplot(1, 4, 1)
        plt.title('minFSD3')
        plt.hist(minFSD3.view(-1).numpy(), bins=25)
        plt.xlabel('minFSD3')
        plt.ylabel('count')

        plt.subplot(1, 4, 2)
        plt.title('maxFSD3')
        plt.hist(maxFSD3.view(-1).numpy(), bins=25)
        plt.xlabel('maxFSD3')
        plt.ylabel('count')

        plt.subplot(1, 4, 3)
        plt.title('stdFD3')
        plt.hist(stdFD3.view(-1).numpy(), bins=25)
        plt.xlabel('stdFD3')
        plt.ylabel('count')

        plt.subplot(1, 4, 4)
        plt.title('voAngles')
        plt.hist(voAngles.view(-1).numpy(), bins=25)
        plt.xlabel('voAngles')
        plt.ylabel('count')

        plt.show()

        results_save_path = '{}_{}_metric.txt'.format(
            self.test_ckpt.split('/')[-2], self.test_ckpt.split('/')[-1].split('.')[0])
        with open(results_save_path, 'w') as f:
            f.write('ADEs: {} \n FDEs: {} \n Qloss: {} \n Ploss: {} \n DAO: {} \n DAC: {}'.format(
                test_ades, test_fdes, test_qloss, test_ploss, test_dao, test_dac))

    @staticmethod
    def q10testsingle(model, batch, device):
        print('Starting model test.....')
        model.eval()  # Set model to evaluate mode.

        H = W = 64
        with torch.no_grad():
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

            coordinate = coordinate.to(device)
            distance = distance.to(device)

            scene_images, log_prior, \
                agent_masks, \
                num_src_trajs, src_trajs, src_lens, src_len_idx, \
                num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                tgt_two_mask, tgt_three_mask, \
                decode_start_vel, decode_start_pos, scene_id = batch

            # Detect dynamic batch size
            batch_size = scene_images.size(0)

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
            agent_masks_idx = torch.arange(len(agent_masks), device=device)[
                agent_masks][tgt_three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            decode_start_vel = decode_start_vel.to(
                device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(
                device)[agent_tgt_three_mask]

            perterb = torch.normal(mean=0.0, std=np.sqrt(
                0.001), size=tgt_trajs.shape, device=device)

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

            rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1))
                         ** 2).sum(dim=-1).sqrt_()
            rs_error2 = rs_error3[..., :int(6 * 2 / 3)]
            diff = gen_trajs - tgt_trajs.unsqueeze(1)
            msd_error = (diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2)
            ade2 = rs_error2.mean(-1)
            fde2 = rs_error2[..., -1]
            minade2, _ = ade2.min(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            ade3 = rs_error3.mean(-1)
            fde3 = rs_error3[..., -1]
            msd = msd_error.mean(-1)
            minmsd, _ = msd.min(dim=-1)
            minade3, _ = ade3.min(dim=-1)
            minfde3, _ = fde3.min(dim=-1)

        return gen_trajs.cpu().numpy(), \
            [z.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()], \
            [minade3.cpu().numpy(), minfde3.cpu().numpy()]

    @staticmethod
    def dac(gen_trajs, map_file, img=None):
        map_array = None
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        elif '.bin' in map_file:
            if img is not None:
                import copy
                map_array = copy.deepcopy(img)
                map_array = np.asarray(map_array)[0]
                map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]
            else:
                with open(map_file, 'rb') as f:
                    map_img = pkl.load(f)
                    drivable_area, road_divider, lane_divider = map_img
                _, drivable_area = cv2.threshold(
                    drivable_area, 0, 255, cv2.THRESH_BINARY)
                _, road_divider = cv2.threshold(
                    road_divider, 0, 255, cv2.THRESH_BINARY)
                drivable_area = drivable_area - road_divider
                map_array = cv2.resize(
                    drivable_area, (128, 128))[..., np.newaxis]
            # else:
            #     with open(map_file, 'rb') as pnt:
            #         map_array = pkl.load(pnt)
            #         map_array = np.asarray(map_array)[0]
            #         map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]

        da_mask = np.any(map_array > 0, axis=-1)
        # da_mask = np.any(map_array > np.min(map_array), axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dac = []

        # gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)
        gen_trajs = ((gen_trajs + 32) * 2).astype(np.int64)

        stay_in_da_count = [0 for i in range(num_agents)]
        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            stay_in_da = [True for i in range(num_agents)]

            oom_mask = np.any(np.logical_or(
                gen_trajs_k >= 128, gen_trajs_k < 0), axis=-1)

            diregard_mask = oom_mask.sum(axis=-1) > 2
            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x * 128 + y)
                lin_xy[oom_mask_t] = -1
                for i in range(num_agents):
                    xi, yi = x[i], y[i]
                    _lin_xy = lin_xy.tolist()
                    lin_xyi = _lin_xy.pop(i)

                    if diregard_mask[i]:
                        continue

                    if oom_mask_t[i]:
                        continue

                    if not da_mask[yi, xi] or (lin_xyi in _lin_xy):
                        stay_in_da[i] = False

            for i in range(num_agents):
                if stay_in_da[i]:
                    stay_in_da_count[i] += 1

        for i in range(num_agents):
            if diregard_mask[i]:
                dac.append(0.0)
            else:
                dac.append(stay_in_da_count[i] / num_candidates)

        dac_mask = np.logical_not(diregard_mask)

        return np.array(dac), dac_mask

    @staticmethod
    def dao(gen_trajs, map_file, img=None):
        map_array = None
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        elif '.bin' in map_file:
            if img is not None:
                import copy
                map_array = copy.deepcopy(img)
                map_array = np.asarray(map_array)[0]
                map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]

            else:
                with open(map_file, 'rb') as f:
                    map_img = pkl.load(f)
                    drivable_area, road_divider, lane_divider = map_img
                _, drivable_area = cv2.threshold(
                    drivable_area, 0, 255, cv2.THRESH_BINARY)
                _, road_divider = cv2.threshold(
                    road_divider, 0, 255, cv2.THRESH_BINARY)
                drivable_area = drivable_area - road_divider
                map_array = cv2.resize(
                    drivable_area, (128, 128))[..., np.newaxis]

            # else:
            #     with open(map_file, 'rb') as pnt:
            #         map_array = pkl.load(pnt)
            #         map_array = np.asarray(map_array)[0]
            #         map_array = cv2.resize(map_array, (128, 128))[..., np.newaxis]

        da_mask = np.any(map_array > 0, axis=-1)
        # da_mask = np.any(map_array > np.min(map_array), axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dao = [0 for i in range(num_agents)]

        occupied = [[] for i in range(num_agents)]

        # gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)
        gen_trajs = ((gen_trajs + 32) * 2).astype(np.int64)

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            oom_mask = np.any(np.logical_or(
                gen_trajs_k >= 128, gen_trajs_k < 0), axis=-1)
            diregard_mask = oom_mask.sum(axis=-1) > 2

            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x * 128 + y)
                lin_xy[oom_mask_t] = -1
                for i in range(num_agents):
                    xi, yi = x[i], y[i]
                    _lin_xy = lin_xy.tolist()
                    lin_xyi = _lin_xy.pop(i)

                    if diregard_mask[i]:
                        continue

                    if oom_mask_t[i]:
                        continue

                    if lin_xyi in occupied[i]:
                        continue

                    if da_mask[yi, xi] and (lin_xyi not in _lin_xy):
                        occupied[i].append(lin_xyi)
                        dao[i] += 1

        for i in range(num_agents):
            if diregard_mask[i]:
                dao[i] = 0.0
            else:
                dao[i] /= da_mask.sum()

        dao_mask = np.logical_not(diregard_mask)

        return np.array(dao), dao_mask

    @staticmethod
    def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
        """abcd"""
        if '.png' in map_file:
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2RGB)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        H, W = map_array.shape[:2]
        fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                         facecolor='k', dpi=80)

        ax = plt.axes()
        ax.imshow(map_array, extent=[-56, 56, 56, -56])
        ax.set_aspect('equal')
        ax.set_xlim([-56, 56])
        ax.set_ylim([-56, 56])

        plt.gca().invert_yaxis()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)

        num_tgt_agents, num_candidates = gen_trajs.shape[:2]
        num_src_agents = len(src_trajs)

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            x_pts_k = []
            y_pts_k = []
            for i in range(num_tgt_agents):
                gen_traj_ki = gen_trajs_k[i]
                tgt_len_i = tgt_lens[i]
                x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
                y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

            ax.scatter(x_pts_k, y_pts_k, s=0.5, marker='o', c='b')

        x_pts = []
        y_pts = []
        for i in range(num_src_agents):
            src_traj_i = src_trajs[i]
            src_len_i = src_lens[i]
            x_pts.extend(src_traj_i[:src_len_i, 0])
            y_pts.extend(src_traj_i[:src_len_i, 1])

        ax.scatter(x_pts, y_pts, s=2.0, marker='x', c='g')

        x_pts = []
        y_pts = []
        for i in range(num_tgt_agents):
            tgt_traj_i = tgt_trajs[i]
            tgt_len_i = tgt_lens[i]
            x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
            y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

        ax.scatter(x_pts, y_pts, s=2.0, marker='o', c='r')

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((H, W, 3))

        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, buffer)
        ax.clear()
        plt.close(fig)

    def map_file(self, scene_id):
        return '{}/{}/map.bin'.format(self.data_dir, scene_id)
        # return '{}/map/{}.bin'.format(self.data_dir, scene_id)
