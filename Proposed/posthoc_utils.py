import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cv2
import torch
import pdb
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from Proposed.utils import *

import logging

from multiprocessing import Pool


class DSFTrainer(ModelTrainer):
    def __init__(self, model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion=None):
        super().__init__(model, train_loader, valid_loader, optimizer, exp_path, args, device, ploss_criterion)
        self.lamb = args.lamb
        self.gamma = args.gamma
        self.load_checkpoint(args.posthoc_tune)

    def init_setting(self):
        for param in self.model.named_parameters():
            if not param[0].startswith('dsfnet'):
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

    def traindsf(self, num_epochs):
        self.init_setting()
        self.logger.info('Model Type: ' + str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info(
                "==========================================================================================")

            train_loss, train_qloss, train_ploss, train_ades, train_fdes, train_z = self.train_single_epoch()
            valid_loss, valid_qloss, valid_ploss, valid_ades, valid_fdes, scheduler_metric = self.inference()

            ## unwrapping ADEs/FDEs
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

            self.save_checkpoint(epoch, qloss=valid_qloss, ploss=valid_ploss, ade=valid_minade3, fde=valid_minfde3)

            # Log values to Tensorboard
            self.writter.add_histogram('data/Z_generated', train_z, epoch)

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
            self.writter.add_scalar('data/Scheduler_Metric', scheduler_metric, epoch)

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
        self.logger.info(f'| Best Valid ADE: {self.best_valid_ade} | Best Valid FDE: {self.best_valid_fde} |')

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

                distance_2d = coordinate_2d - np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
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
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

            scene_images = scene_images.to(self.device)

            past_agents_traj = past_agents_traj.to(self.device)
            past_agents_traj_len = past_agents_traj_len.to(self.device)

            future_agents_traj = future_agents_traj.to(self.device)[three_mask]
            future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

            num_future_agents = num_future_agents.to(self.device)
            episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

            future_agent_masks = future_agent_masks.to(self.device)
            agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
            agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][three_mask]
            agent_tgt_three_mask[agent_masks_idx] = True

            future_agent_masks = future_agent_masks.to(self.device)

            decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
            decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

            log_prior = log_prior.to(self.device)
            # Normalizing Flow (q loss)
            # z_: A X Td X 2
            # mu_: A X Td X 2
            # sigma_: A X Td X 2 X 2
            # Generate perturbation
            perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)

            if self.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
                z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                      past_agents_traj,
                                                                                      past_agents_traj_len,
                                                                                      agent_tgt_three_mask, episode_idx,
                                                                                      decode_start_vel,
                                                                                      decode_start_pos, num_past_agents,
                                                                                      scene_images)

            z_ = z_.reshape((num_three_agents, -1))  # A X (Td*2)
            log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

            logdet_sigma = log_determinant(sigma_)

            log_qpi = log_q0 - logdet_sigma.sum(dim=1)
            qloss = -log_qpi
            batch_qloss = qloss.mean()

            # Prior Loss (p loss)

            if self.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
                gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                     episode_idx, decode_start_vel, decode_start_pos, num_past_agents,
                                                     scene_encoding_, agent_encoded=True, scene_encoded=True)

            if self.beta != 0.0:
                if self.ploss_type == 'mseloss':
                    ploss = self.ploss_criterion(gen_trajs, future_agents_traj)  # past_agents_traj)
                elif self.ploss_type == 'map':
                    ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)
                elif self.ploss_type == 'all':
                    ploss = self.ploss_criterion[0](gen_trajs, future_agents_traj) + self.gamma * self.ploss_criterion[1](
                        episode_idx, gen_trajs, log_prior, -15.0)

            else:
                ploss = torch.zeros(size=(1,), device=self.device)

            batch_ploss = ploss.mean()
            # batch_loss = batch_qloss + self.beta * batch_ploss

            epoch_ploss += batch_ploss.item() * batch_size
            epoch_qloss += batch_qloss.item() * batch_size

            gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)

            # TODO : trajs 가 끝 점만을 포함할 것인지, 모든 timestep 에 대해 거리를 계산할 것인지
            # trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, -1)
            trajs = gen_trajs[:, :, -1, :]

            # TODO: Choice : custom_NN_loss or batch_NN_loss
            # divloss = custom_NN_loss(trajs).to(self.device)
            divloss = batch_NN_loss(trajs, trajs)
            batch_divloss = divloss.mean().to(self.device)
            # batch_divloss = torch.clamp(batch_divloss, 0, 80)

            batch_loss = self.beta * batch_ploss - self.lamb * batch_divloss  # + 0.5 * ((z ** 2).sum(dim=1))

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
                  "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:.2f}, divloss: {:.3f}, ".format(batch_loss.item(),
                                                                                               batch_qloss.item(),
                                                                                               batch_ploss.item(),
                                                                                               batch_divloss.item()) +
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

        epoch_ades = [epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3]
        epoch_fdes = [epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3]

        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, z

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

                distance_2d = coordinate_2d - np.array([(H - 1) / 2, (H - 1) / 2]).reshape((2, 1, 1))
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
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)

                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)[three_mask]
                future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

                num_future_agents = num_future_agents.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[
                    three_mask]

                future_agent_masks = future_agent_masks.to(self.device)
                agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks][
                    three_mask]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                log_prior = log_prior.to(self.device)

                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # z: A X Td X 2
                    # mu: A X Td X 2
                    # sigma: A X Td X 2 X 2
                    # Generate perturbation
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape,
                                           device=self.device)

                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj + perterb,
                                                                                          past_agents_traj,
                                                                                          past_agents_traj_len,
                                                                                          agent_tgt_three_mask,
                                                                                          episode_idx, decode_start_vel,
                                                                                          decode_start_pos,
                                                                                          num_past_agents, scene_images)

                    z_ = z_.reshape((num_three_agents, -1))  # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask,
                                                         episode_idx, decode_start_vel, decode_start_pos,
                                                         num_past_agents, scene_encoding_, agent_encoded=True,
                                                         scene_encoded=True)

                    if self.beta != 0.0:
                        if self.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(gen_trajs, future_agents_traj)  # past_agents_traj)
                        elif self.ploss_type == 'map':
                            ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)
                        elif self.ploss_type == 'all':
                            ploss = self.ploss_criterion[0](gen_trajs, future_agents_traj) + self.gamma * self.ploss_criterion[
                                1](episode_idx, gen_trajs, log_prior, -15.0)

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

                    gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)

                rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
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

        epoch_ades = (epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3)
        epoch_fdes = (epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3)

        scheduler_metric = epoch_avgade3 + epoch_avgfde3

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric


class ModelDSFTest(ModelTest):
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
            self.map_version = args.map_version

        if self.model_type in ["R2P2_SimpleRNN", "R2P2_RNN", "CAM_NFDecoder", "Scene_CAM_NFDecoder",
                               "Global_Scene_CAM_DSF_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder"]:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1

        self.device = device
        self.expname = args.posthoc_tune.split('/')[-2]
        self.out_dir = args.test_dir + '/' + self.expname
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.render = args.test_render
        self.test_times = args.test_times
        self.dataset = args.dataset

        _data_dir = './data/nuscenes_shpark'
        self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x
                                          in scene_id]

        self.load_checkpoint(args.posthoc_tune)

    @staticmethod
    def write_img_output(gen_trajs, src_trajs, src_lens, tgt_trajs, tgt_lens, map_file, output_file):
        """abcd"""
        # print(map_file)
        # print(output_file)
        if '.png' in map_file or '.jpg' in map_file or map_file.lower().endswith('.jpg'):
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(map_array, 60, 255, cv2.THRESH_BINARY)
            map_array = cv2.bitwise_not(thresh)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_GRAY2BGR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        scale = 3
        H, W = map_array.shape[:2]
        H, W = H * scale, W * scale
        fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                         facecolor='k', dpi=80)

        ax = plt.axes()
        ax.imshow(map_array, extent=[-H / 4, H / 4, H / 4, -H / 4])
        ax.set_aspect('equal')
        ax.set_xlim([-H / 4, H / 4])
        ax.set_ylim([-H / 4, H / 4])

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
                ax.plot(scale * gen_traj_ki[:tgt_len_i, 0], scale * gen_traj_ki[:tgt_len_i, 1], marker='o', c='b',
                        linewidth=2.5, markersize=4, linestyle="-")
                # ax.plot(gen_traj_ki[:tgt_len_i, 0], gen_traj_ki[:tgt_len_i, 1], marker='o', c='b', linewidth=1, markersize=2, linestyle="-")
                # x_pts_k.extend(gen_traj_ki[:tgt_len_i, 0])
                # y_pts_k.extend(gen_traj_ki[:tgt_len_i, 1])

            # ax.scatter([i * scale for i in x_pts_k], [i * scale for i in y_pts_k], s=0.5 * scale, marker='o', c='b')

        x_pts = []
        y_pts = []
        for i in range(num_src_agents):
            # src_traj_i = np.concatenate((src_trajs[i], np.array([tgt_trajs[i,0]])), axis=0)
            # src_len_i = src_lens[i] + 1
            src_traj_i = src_trajs[i]
            src_len_i = src_lens[i]
            ax.plot(scale * src_traj_i[:src_len_i, 0], scale * src_traj_i[:src_len_i, 1], marker='x', c='g',
                    linewidth=5, markersize=8, linestyle="-")
            # ax.plot(src_traj_i[:src_len_i, 0], src_traj_i[:src_len_i, 1], marker='x', c='g', linewidth=1, markersize=2, linestyle="-")
            # x_pts.extend(src_traj_i[:src_len_i, 0])
            # y_pts.extend(src_traj_i[:src_len_i, 1])

        # ax.plot([i for i in x_pts], [i for i in y_pts], marker='x', c='g')
        # ax.scatter([i * scale for i in x_pts], [i * scale for i in y_pts], s=0.5 * scale , marker='x', c='g')

        x_pts = []
        y_pts = []
        for i in range(num_tgt_agents):
            tgt_traj_i = tgt_trajs[i]
            tgt_len_i = tgt_lens[i]
            ax.plot(scale * tgt_traj_i[:tgt_len_i, 0], scale * tgt_traj_i[:tgt_len_i, 1], marker='o', c='r',
                    linewidth=5, markersize=8, linestyle="-")
            # ax.plot(tgt_traj_i[:tgt_len_i, 0], tgt_traj_i[:tgt_len_i, 1], marker='o', c='r', linewidth=1, markersize=2, linestyle="-")
            # x_pts.extend(tgt_traj_i[:tgt_len_i, 0])
            # y_pts.extend(tgt_traj_i[:tgt_len_i, 1])

        # ax.plot([i for i in x_pts], [i * scale for i in y_pts], marker='o', c='r')
        # ax.scatter([i * scale for i in x_pts], [i * scale for i in y_pts], s=0.5 * scale, marker='o', c='r')

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((H, W, 3))

        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        buffer = cv2.resize(buffer, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_file, buffer)
        ax.clear()
        plt.close(fig)

    def run_draw(self):
        print('Starting model test.....')
        self.model.eval()  # Set model to evaluate mode.
        self.flow_based_decoder = True

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

            H = W = 64
            with torch.no_grad():
                if self.map_version == '2.0':
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

                c1 = -self.decoding_steps * np.log(2 * np.pi)

                for b, batch in enumerate(self.data_loader):

                    scene_images, log_prior, \
                    agent_masks, \
                    num_src_trajs, src_trajs, src_lens, src_len_idx, \
                    num_tgt_trajs, tgt_trajs, tgt_lens, tgt_len_idx, \
                    tgt_two_mask, tgt_three_mask, \
                    decode_start_vel, decode_start_pos, scene_id = batch

                    # Detect dynamic batch size
                    batch_size = scene_images.size(0)

                    if self.map_version == '2.0':
                        coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                        distance_batch = distance.repeat(batch_size, 1, 1, 1)
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch),
                                                 dim=1)

                    src_trajs = src_trajs.to(self.device)
                    src_lens = src_lens.to(self.device)

                    if self.dataset == 'nuscenes_carla':
                        # print('tgt', len(tgt_three_mask))
                        # print('am ', len(agent_masks))
                        # num_three_agents = torch.sum(tgt_three_mask)
                        # tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                        # tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]
                        # num_tgt_trajs = num_tgt_trajs.to(self.device)
                        # episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[tgt_three_mask]

                        # agent_masks = agent_masks.to(self.device)
                        # agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        # agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][tgt_three_mask]
                        # agent_tgt_three_mask[agent_masks_idx] = True

                        # decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                        # decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                        num_three_agents = torch.sum(agent_masks)
                        tgt_trajs = tgt_trajs.to(self.device)[agent_masks]
                        tgt_lens = tgt_lens.to(self.device)[agent_masks]

                        num_tgt_trajs = num_tgt_trajs.to(self.device)
                        episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[
                            agent_masks]

                        agent_masks = agent_masks.to(self.device)
                        agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks]
                        agent_tgt_three_mask[agent_masks_idx] = True

                        decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                        decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]
                    else:
                        # print('tgt', len(tgt_three_mask))
                        # print('am ', len(agent_masks))
                        num_three_agents = torch.sum(tgt_three_mask)
                        tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                        tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]
                        num_tgt_trajs = num_tgt_trajs.to(self.device)
                        episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[
                            tgt_three_mask]

                        agent_masks = agent_masks.to(self.device)
                        agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][
                            tgt_three_mask]
                        agent_tgt_three_mask[agent_masks_idx] = True

                        decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                        decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                    log_prior = log_prior.to(self.device)

                    if self.flow_based_decoder:
                        # Normalizing Flow (q loss)
                        # z: A X Td X 2
                        # mu: A X Td X 2
                        # sigma: A X Td X 2 X 2
                        # Generate perturbation
                        perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=tgt_trajs.shape, device=self.device)

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

                        elif self.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
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

                        elif self.model_type == 'Global_Scene_CAM_DSF_NFDecoder':
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
                                ploss = self.ploss_criterion(gen_trajs, tgt_trajs)
                            else:
                                ploss = self.ploss_criterion(episode_idx, gen_trajs, log_prior, -15.0)

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

                        gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)

                    rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                    rs_error2 = rs_error3[..., :int(self.decoding_steps * 2 / 3)]

                    diff = gen_trajs - tgt_trajs.unsqueeze(1)
                    msd_error = (diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2)

                    num_agents = gen_trajs.size(0)
                    # print(num_agents)
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

                    map_files = self.map_file(scene_id)
                    if self.dataset == 'nuscenes_carla':
                        output_files = [self.out_dir + x[1:] for x in scene_id]
                    else:
                        output_files = [self.out_dir + '/' + x[2] + '_' + x[3] + '.jpg' for x in scene_id]

                    # print(num_tgt_trajs, num_src_trajs)
                    cum_num_tgt_trajs = [0] + torch.cumsum(num_tgt_trajs, dim=0).tolist()
                    cum_num_src_trajs = [0] + torch.cumsum(num_src_trajs, dim=0).tolist()

                    src_trajs = src_trajs.cpu().numpy()
                    src_lens = src_lens.cpu().numpy()

                    tgt_trajs = tgt_trajs.cpu().numpy()
                    tgt_lens = tgt_lens.cpu().numpy()

                    agent_masks = agent_masks.cpu().numpy()

                    if self.dataset == 'nuscenes_carla':
                        zero_ind = np.nonzero(agent_masks == 0)[0]
                        zero_ind -= np.arange(len(zero_ind))
                    else:
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
                        output_file_i = output_files[i]
                        # print(output_file_i)
                        if self.dataset == 'nuscenes_carla':
                            # print(len(candidate_i))
                            # print(len(tgt_traj_i))
                            # print(len(agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]))
                            # if len(candidate_i)<len(agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]):
                            #     continue
                            # candidate_i = candidate_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]][:len(candidate_i)]]
                            # tgt_traj_i = tgt_traj_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]][:len(tgt_traj_i)]]
                            # tgt_lens_i = tgt_lens_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]][:len(tgt_lens_i)]]
                            candidate_i = candidate_i[
                                agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                            tgt_traj_i = tgt_traj_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                            tgt_lens_i = tgt_lens_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                        else:
                            candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                            tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]
                            tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i + 1]]]

                        src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]
                        src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i + 1]]]

                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)
                        self.write_img_output(candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i,
                                              output_file_i)

                        epoch_dao += dao_i.sum()
                        dao_agents += dao_mask_i.sum()

                        epoch_dac += dac_i.sum()
                        dac_agents += dac_mask_i.sum()

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

        print("--Final Performane Report--")
        print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(test_minade3[0], test_minade3[1], test_minfde3[0],
                                                                      test_minfde3[1]))
        print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(test_avgade3[0], test_avgade3[1], test_avgfde3[0],
                                                                      test_avgfde3[1]))
        print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0],
                                                              test_dac[1]))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": test_ades,
                      "FDEs": test_fdes,
                      "Qloss": test_qloss,
                      "Ploss": test_ploss,
                      "DAO": test_dao,
                      "DAC": test_dac}, f)


def batch_pairwise_dist(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    P[P == 0] = 1000
    return P


def batch_NN_loss(x, y, dim=1):
    assert dim != 0
    # pdb.set_trace()
    dist = batch_pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    v, i = values.min(dim=1)
    # v = torch.clamp(v, 0, 40)
    v = torch.clamp(v, 0, 32)
    return v


def custom_NN_loss(trajs):
    trajs_ = trajs.transpose(0, 1)
    distance = torch.zeros(trajs_.size(0), trajs_.size(0), trajs_.size(1))
    for i, tr1 in enumerate(trajs_):
        for j, tr2 in enumerate(trajs_):
            agents = torch.zeros(tr1.size(0))
            if i != j:
                for k in range(tr1.size(0)):
                    dis = torch.norm(tr1[k, :] - tr2[k, :], 2)
                    agents[k] = dis
            distance[i, j, :] = agents
    distance[distance == 0] = 1000

    v, i = distance.min(dim=0)
    v_, i_ = v.min(dim=0)
    return v_
