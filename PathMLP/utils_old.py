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
from itertools import combinations

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
        self.dataset = args.dataset
        self.exp_path = os.path.join(exp_path, args.tag + '_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).strftime('_%d_%B__%H_%M_'))
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S'))
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
            self.map_version = args.map_version

        self.beta = args.beta
        self.ploss_type = args.ploss_type        
        self.ploss_criterion = ploss_criterion

        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=(1/2), verbose=True, patience=3)

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
        self.logger.info('Model Type: '+str(self.model_type))
        self.logger.info('TRAINING .....')

        for epoch in tqdm(range(self.start_epoch, self.start_epoch + num_epochs)):
            self.logger.info("==========================================================================================")

            train_loss, train_qloss, train_ploss, train_ades, train_fdes = self.train_single_epoch()
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

            self.logger.info("------------------------------------------------------------------------------------------")
            self.logger.info(logging_msg1)
            self.logger.info(logging_msg2)

            self.save_checkpoint(epoch, qloss=valid_qloss, ploss=valid_ploss, ade=valid_minade3, fde=valid_minfde3)

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
                
                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
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
            num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
            num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
            two_mask, three_mask, \
            decode_start_vel, decode_start_pos, \
            scene_id = batch

            # Detect dynamic sizes
            batch_size = scene_images.size(0)

            if self.map_version == '2.0':
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
            
            scene_images = scene_images.to(self.device)

            if self.dataset == "nuscenes_carla":
                num_three_agents = torch.sum(future_agent_masks)
                future_agent_masks = future_agent_masks.to(self.device)

                past_agents_traj = past_agents_traj.to(self.device)
                past_agents_traj_len = past_agents_traj_len.to(self.device)

                future_agents_traj = future_agents_traj.to(self.device)[future_agent_masks]
                future_agents_traj_len = future_agents_traj_len.to(self.device)[future_agent_masks]

                num_future_agents = num_future_agents.to(self.device)
                episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[future_agent_masks]

                agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks]
                agent_tgt_three_mask[agent_masks_idx] = True

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                # for i in [future_agents_traj, future_agents_traj_len, past_agents_traj, past_agents_traj_len, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_future_agents, num_past_agents, scene_images]:
                #     print(len(i))
                # print('###' * 10)

                log_prior = log_prior.to(self.device)


                # mask_comb = three_mask * future_agent_masks
                # mask_comb = mask_comb.to(self.device)

                # future_agent_masks = future_agent_masks.to(self.device)
                # agent_tgt_three_mask = torch.zeros_like(future_agent_masks)

                # agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[mask_comb]
                # agent_tgt_three_mask[agent_masks_idx] = True

                # past_agents_traj = past_agents_traj.to(self.device)
                # past_agents_traj_len = past_agents_traj_len.to(self.device)

                # future_agents_traj = future_agents_traj.to(self.device)[mask_comb]
                # future_agents_traj_len = future_agents_traj_len.to(self.device)[mask_comb]

                # num_future_agents = num_future_agents.to(self.device)
                # episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[mask_comb]
                # # episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]
                # # print(three_mask)

                # future_agent_masks = future_agent_masks.to(self.device)

                # decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                # decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]
                # print('###' * 10)

                # log_prior = log_prior.to(self.device)

            else:
                num_three_agents = torch.sum(three_mask)
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

                decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                # for i in [future_agents_traj, future_agents_traj_len, past_agents_traj, past_agents_traj_len, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_future_agents, num_past_agents, scene_images]:
                #     print(len(i))
                # print('###' * 10)
                log_prior = log_prior.to(self.device)

            if self.flow_based_decoder:
                # Normalizing Flow (q loss)
                # z_: A X Td X 2
                # mu_: A X Td X 2
                # sigma_: A X Td X 2 X 2
                # Generate perturbation
                perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
                
                if self.model_type == 'R2P2_SimpleRNN':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)

                elif self.model_type == 'R2P2_RNN':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                elif self.model_type == 'CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, future_agent_masks, decode_start_vel, decode_start_pos, num_past_agents)

                elif self.model_type == 'Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj_+perterb, past_agents_traj, past_agents_traj_len, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)


                elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                    z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, future_agent_masks, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)
                
                # print(z_.size())
                z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                logdet_sigma = log_determinant(sigma_)

                log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                qloss = -log_qpi
                batch_qloss = qloss.mean()
                
                # Prior Loss (p loss)
                if self.model_type == 'R2P2_SimpleRNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                elif self.model_type == 'R2P2_RNN':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                elif self.model_type == 'CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)

                elif self.model_type == 'Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                    gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                if self.beta != 0.0:
                    if self.ploss_type == 'mseloss':
                        ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
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
                    gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)

            rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_() # A X candi X T X 2 >> A X candi X T
            rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
            
            num_agents = gen_trajs.size(0)
            num_agents2 = rs_error2.size(0)
            num_agents3 = rs_error3.size(0)

            ade2 = rs_error2.mean(-1) #  A X candi X T >> A X candi
            fde2 = rs_error2[..., -1]

            minade2, _ = ade2.min(dim=-1) # A X candi >> A
            avgade2 = ade2.mean(dim=-1)
            minfde2, _ = fde2.min(dim=-1)
            avgfde2 = fde2.mean(dim=-1)

            batch_minade2 = minade2.mean() # A >> 1
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

            print("Working on train batch {:d}/{:d}... ".format(b+1, len(self.train_loader)) +
            "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
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

                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
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
                num_past_agents,   past_agents_traj,   past_agents_traj_len,   past_agents_traj_len_idx, \
                num_future_agents, future_agents_traj, future_agents_traj_len, future_agents_traj_len_idx, \
                two_mask, three_mask, \
                decode_start_vel, decode_start_pos, \
                scene_id = batch

                # Detect dynamic batch size
                batch_size = scene_images.size(0)
                if self.map_version == '2.0':
                    coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                    distance_batch = distance.repeat(batch_size, 1, 1, 1)
                    scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                if self.dataset == "nuscenes_carla":
                    num_three_agents = torch.sum(future_agent_masks)
                    future_agent_masks = future_agent_masks.to(self.device)

                    past_agents_traj = past_agents_traj.to(self.device)
                    past_agents_traj_len = past_agents_traj_len.to(self.device)

                    future_agents_traj = future_agents_traj.to(self.device)[future_agent_masks]
                    future_agents_traj_len = future_agents_traj_len.to(self.device)[future_agent_masks]

                    num_future_agents = num_future_agents.to(self.device)
                    episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[future_agent_masks]

                    agent_tgt_three_mask = torch.zeros_like(future_agent_masks)
                    agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[future_agent_masks]
                    agent_tgt_three_mask[agent_masks_idx] = True

                    decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                    decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                    log_prior = log_prior.to(self.device)
                    
                    # mask_comb = future_agent_masks * three_mask
                    # mask_comb = mask_comb.to(self.device)

                    # future_agent_masks = future_agent_masks.to(self.device)
                    # agent_tgt_three_mask = torch.zeros_like(future_agent_masks)

                    # agent_masks_idx = torch.arange(len(future_agent_masks), device=self.device)[mask_comb]
                    # agent_tgt_three_mask[agent_masks_idx] = True

                    # past_agents_traj = past_agents_traj.to(self.device)
                    # past_agents_traj_len = past_agents_traj_len.to(self.device)

                    # future_agents_traj = future_agents_traj.to(self.device)[three_mask]
                    # future_agents_traj_len = future_agents_traj_len.to(self.device)[three_mask]

                    # num_future_agents = num_future_agents.to(self.device)
                    # episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_future_agents)[three_mask]

                    # future_agent_masks = future_agent_masks.to(self.device)

                    # decode_start_vel = decode_start_vel.to(self.device)[mask_comb]
                    # decode_start_pos = decode_start_pos.to(self.device)[mask_comb]
                    # log_prior = log_prior.to(self.device)
                else:
                    num_three_agents = torch.sum(three_mask)
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

                    decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                    decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]

                    log_prior = log_prior.to(self.device)

                if self.flow_based_decoder:
                    # Normalizing Flow (q loss)
                    # z: A X Td X 2
                    # mu: A X Td X 2
                    # sigma: A X Td X 2 X 2
                    # Generate perturbation
                    perterb = torch.normal(mean=0.0, std=np.sqrt(0.001), size=future_agents_traj.shape, device=self.device)
                    
                    if self.model_type == 'R2P2_SimpleRNN':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, decode_start_vel, decode_start_pos)

                    elif self.model_type == 'R2P2_RNN':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                    elif self.model_type == 'CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(future_agents_traj+perterb, past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_images)

                    z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                    log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                    logdet_sigma = log_determinant(sigma_)

                    log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                    qloss = -log_qpi
                    batch_qloss = qloss.mean()

                    # Prior Loss (p loss)
                    if self.model_type == 'R2P2_SimpleRNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                    elif self.model_type == 'R2P2_RNN':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                    elif self.model_type == 'CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents, agent_encoded=True)

                    elif self.model_type == 'Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                        gen_trajs, z, mu, sigma = self.model(motion_encoding_, past_agents_traj_len, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_past_agents, scene_encoding_, agent_encoded=True, scene_encoded=True)

                    if self.beta != 0.0:
                        if self.ploss_type == 'mseloss':
                            ploss = self.ploss_criterion(gen_trajs, past_agents_traj)
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
                        gen_trajs = self.model(past_agents_traj, past_agents_traj_len, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_past_agents)

                    gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                rs_error3 = ((gen_trajs - future_agents_traj.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                
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

                print("Working on valid batch {:d}/{:d}... ".format(b+1, len(self.valid_loader)) +
                "batch_loss: {:.2f}, qloss: {:.2f}, ploss: {:g}, ".format(batch_loss.item(), batch_qloss.item(), batch_ploss.item()) +
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

        epoch_ades = ( epoch_minade2, epoch_avgade2, epoch_minade3, epoch_avgade3 )
        epoch_fdes = ( epoch_minfde2, epoch_avgfde2, epoch_minfde3, epoch_avgfde3 )

        scheduler_metric = epoch_avgade3 + epoch_avgfde3 

        return epoch_loss, epoch_qloss, epoch_ploss, epoch_ades, epoch_fdes, scheduler_metric


    def get_lr(self):
        """Returns Learning Rate of the Optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, epoch, ade, fde, qloss=0, ploss=0):
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

        save_path = "{}/ck_{}_{:0.4f}_{:0.4f}_{:0.4f}_{:0.4f}.pth.tar".format(self.exp_path, epoch, qloss, ploss, ade, fde)
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

        self.decoding_steps = int(3 *  args.sampling_rate)
        
        self.model_type = args.model_type

        if self.model_type in ['R2P2_SimpleRNN', 'CAM', 'CAM_NFDecoder']:
            self.map_version = None
        else:
            self.map_version = args.map_version

        if self.model_type in ["R2P2_SimpleRNN", "R2P2_RNN", "CAM_NFDecoder", "Scene_CAM_NFDecoder", "Global_Scene_CAM_NFDecoder", "AttGlobal_Scene_CAM_NFDecoder"]:
            self.flow_based_decoder = True
            self.num_candidates = args.num_candidates

        else:
            self.flow_based_decoder = False
            self.num_candidates = 1


        self.device = device
        self.out_dir = args.test_dir
        self.render = args.test_render
        self.test_times = args.test_times
        self.dataset = args.dataset
                
        if args.dataset == "argoverse":
            _data_dir = './data/argoverse'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.png' for x in scene_id]

        elif args.dataset == "nuscenes":
            _data_dir = './data/nuscenes_shpark'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3]) + '.pkl' for x in scene_id]

        elif args.dataset == "carla":
            _data_dir = './data/carla'
            self.map_file = lambda scene_id: [os.path.join(_data_dir, x[0], x[1], x[2], 'map/v1.3', x[3] ) + '.pkl' for x in scene_id]

        elif args.dataset == "nuscenes_carla":
            # _data_dir = './data/carla'
            self.map_file = lambda scene_id: [os.path.join('./cache', x) for x in scene_id]

        self.load_checkpoint(args.test_ckpt)


    def load_checkpoint(self, ckpt):
        # print(torch.cuda.device_count())
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)

    @staticmethod
    def dac(gen_trajs, map_file):
        if '.png' in map_file or '.jpg' in map_file or map_file.lower().endswith('.jpg'):
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(map_array, 60, 255, cv2.THRESH_BINARY)
            map_array = cv2.bitwise_not(thresh)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_GRAY2BGR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        da_mask = np.any(map_array > 0, axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dac = []

        gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

        stay_in_da_count = [0 for i in range(num_agents)]
        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            stay_in_da = [True for i in range(num_agents)]

            oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
            diregard_mask = oom_mask.sum(axis=-1) > 2
            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x*224+y)
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
    def dao(gen_trajs, map_file):
        # print(map_file)
        if '.png' in map_file or '.jpg' in map_file or map_file.lower().endswith('.jpg'):
            map_array = cv2.imread(map_file, cv2.IMREAD_COLOR)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(map_array, 60, 255, cv2.THRESH_BINARY)
            map_array = cv2.bitwise_not(thresh)
            map_array = cv2.cvtColor(map_array, cv2.COLOR_GRAY2BGR)

        elif '.pkl' in map_file:
            with open(map_file, 'rb') as pnt:
                map_array = pkl.load(pnt)

        da_mask = np.any(map_array > 0, axis=-1)

        num_agents, num_candidates, decoding_timesteps = gen_trajs.shape[:3]
        dao = [0 for i in range(num_agents)]

        occupied = [[] for i in range(num_agents)]

        gen_trajs = ((gen_trajs + 56) * 2).astype(np.int64)

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            oom_mask = np.any( np.logical_or(gen_trajs_k >= 224, gen_trajs_k < 0), axis=-1 )
            diregard_mask = oom_mask.sum(axis=-1) > 2

            for t in range(decoding_timesteps):
                gen_trajs_kt = gen_trajs_k[:, t]
                oom_mask_t = oom_mask[:, t]
                x, y = gen_trajs_kt.T

                lin_xy = (x*224+y)
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
    def vo_angle(gen_trajs, tgt_trajs, tgt_lens):
        def angle_between(v1, v2):
            v1_u = v1 / (np.linalg.norm(v1) + 1e-5)
            v2_u = v2 / (np.linalg.norm(v2) + 1e-5)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        num_tgt_agents, num_candidates = gen_trajs.shape[:2]
        angle_sum = 0

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]
            for i in range(num_tgt_agents):
                gen_traj_ki = gen_trajs_k[i]
                tgt_traj_i = tgt_trajs[i]
                tgt_len_i = tgt_lens[i]

                gen_init_x = gen_traj_ki[:tgt_len_i-1, 0]
                gen_init_y = gen_traj_ki[:tgt_len_i-1, 1]
                gen_fin_x = gen_traj_ki[1:tgt_len_i, 0]
                gen_fin_y = gen_traj_ki[1:tgt_len_i, 1]

                tgt_init_x = tgt_traj_i[:tgt_len_i-1, 0]
                tgt_init_y = tgt_traj_i[:tgt_len_i-1, 1]
                tgt_fin_x = tgt_traj_i[1:tgt_len_i, 0]
                tgt_fin_y = tgt_traj_i[1:tgt_len_i, 1]

                gen_to_tgt_x = tgt_init_x - gen_init_x
                gen_to_tgt_y = tgt_init_y - gen_init_y

                gen_fin_x = gen_fin_x + gen_to_tgt_x
                gen_fin_y = gen_fin_y + gen_to_tgt_y

                tgt_vector_x = tgt_fin_x - tgt_init_x
                tgt_vector_y = tgt_fin_y - tgt_init_y
                gen_vector_x = gen_fin_x - tgt_init_x
                gen_vector_y = gen_fin_y - tgt_init_y

                tgt_vectors = (np.concatenate((tgt_vector_x, tgt_vector_y), axis= 0)).T
                gen_vectors = (np.concatenate((gen_vector_x, gen_vector_y), axis= 0)).T
                
                for tgt_vec, gen_vec in zip(tgt_vectors, gen_vectors):
                    angle_sum += angle_between(tgt_vec, gen_vec)
        angle_sum = angle_sum/num_candidates/num_tgt_agents
        return angle_sum
    
    def distance_between(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @staticmethod
    def self_distance(gen_trajs, tgt_trajs, tgt_lens):
        num_tgt_agents, num_candidates = gen_trajs.shape[:2]
        multi_fin = np.zeros((num_tgt_agents, num_candidates, 2))

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]
            for i in range(num_tgt_agents):
                gen_traj_ki = gen_trajs_k[i]
                tgt_len_i = tgt_lens[i]
                multi_fin[i,k] = gen_traj_ki[tgt_len_i-1]
        
        sd_total = 0
        min_sd = 10000000
        for i in range(num_tgt_agents):
            curr_agent = multi_fin[i]
            for k in range(num_candidates-1):
                curr_candidate = np.tile(curr_agent[k], (num_candidates-k-1, 1))
                sd = np.min(np.sqrt(np.sum(np.power(curr_candidate - curr_agent[k+1:], 2), axis=1)))
                if min_sd>sd:
                    min_sd = sd
            sd_total+=min_sd
        sd_total /=num_tgt_agents
        return sd_total

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
        H, W = H*scale, W*scale
        fig = plt.figure(figsize=(float(H) / float(80), float(W) / float(80)),
                        facecolor='k', dpi=80)

        ax = plt.axes()
        ax.imshow(map_array, extent=[-H/4, H/4, H/4, -H/4])
        ax.set_aspect('equal')
        ax.set_xlim([-H/4, H/4])
        ax.set_ylim([-H/4, H/4])

        plt.gca().invert_yaxis()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
        plt.margins(0,0)
        
        num_tgt_agents, num_candidates = gen_trajs.shape[:2]
        num_src_agents = len(src_trajs)

        for k in range(num_candidates):
            gen_trajs_k = gen_trajs[:, k]

            x_pts_k = []
            y_pts_k = []
            for i in range(num_tgt_agents):
                gen_traj_ki = gen_trajs_k[i]
                tgt_len_i = tgt_lens[i]
                ax.plot(scale * gen_traj_ki[:tgt_len_i, 0], scale * gen_traj_ki[:tgt_len_i, 1], marker='o', c='b', linewidth=2.5, markersize=4, linestyle="-")
        
        x_pts = []
        y_pts = []
        for i in range(num_src_agents):
                src_traj_i = src_trajs[i]
                src_len_i = src_lens[i]
                ax.plot(scale * src_traj_i[:src_len_i, 0], scale * src_traj_i[:src_len_i, 1], marker='x', c='g', linewidth=5, markersize=8, linestyle="-")


        x_pts = []
        y_pts = []
        for i in range(num_tgt_agents):
                tgt_traj_i = tgt_trajs[i]
                tgt_len_i = tgt_lens[i]
                ax.plot(scale * tgt_traj_i[:tgt_len_i, 0], scale * tgt_traj_i[:tgt_len_i, 1], marker='o', c='r', linewidth=5, markersize=8, linestyle="-")

        fig.canvas.draw()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((H, W, 3))

        buffer = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        buffer = cv2.resize(buffer, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_file, buffer)
        ax.clear()
        plt.close(fig)
        
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

        list_sd =[]
        list_angle = []

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
                    coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                    coordinate = torch.FloatTensor(coordinate)
                    coordinate = coordinate.reshape((1, 1, H, W))

                    coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                    coordinate = (coordinate - coordinate_mean) / coordinate_std

                    distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
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
                        scene_images = torch.cat((scene_images.to(self.device), coordinate_batch, distance_batch), dim=1)
                    
                    src_trajs = src_trajs.to(self.device)
                    src_lens = src_lens.to(self.device)

                    if self.dataset == 'nuscenes_carla':
                        num_three_agents = torch.sum(agent_masks)
                        tgt_trajs = tgt_trajs.to(self.device)[agent_masks]
                        tgt_lens = tgt_lens.to(self.device)[agent_masks]

                        num_tgt_trajs = num_tgt_trajs.to(self.device)
                        episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[agent_masks]

                        agent_masks = agent_masks.to(self.device)
                        agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks]
                        agent_tgt_three_mask[agent_masks_idx] = True

                        decode_start_vel = decode_start_vel.to(self.device)[agent_tgt_three_mask]
                        decode_start_pos = decode_start_pos.to(self.device)[agent_tgt_three_mask]
                    else:
                        num_three_agents = torch.sum(tgt_three_mask)
                        tgt_trajs = tgt_trajs.to(self.device)[tgt_three_mask]
                        tgt_lens = tgt_lens.to(self.device)[tgt_three_mask]
                        num_tgt_trajs = num_tgt_trajs.to(self.device)
                        episode_idx = torch.arange(batch_size, device=self.device).repeat_interleave(num_tgt_trajs)[tgt_three_mask]

                        agent_masks = agent_masks.to(self.device)
                        agent_tgt_three_mask = torch.zeros_like(agent_masks)
                        agent_masks_idx = torch.arange(len(agent_masks), device=self.device)[agent_masks][tgt_three_mask]
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
                            z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, decode_start_vel, decode_start_pos)

                        elif self.model_type == 'R2P2_RNN':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, episode_idx, decode_start_vel, decode_start_pos, scene_images)

                        elif self.model_type == 'CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)

                        elif self.model_type == 'Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                        elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                        elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                            z_, mu_, sigma_, motion_encoding_, scene_encoding_ = self.model.infer(tgt_trajs+perterb, src_trajs, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_images)

                        z_ = z_.reshape((num_three_agents, -1)) # A X (Td*2)
                        log_q0 = c1 - 0.5 * ((z_ ** 2).sum(dim=1))

                        logdet_sigma = log_determinant(sigma_)

                        log_qpi = log_q0 - logdet_sigma.sum(dim=1)
                        qloss = -log_qpi
                        batch_qloss = qloss.mean()

                        # Prior Loss (p loss)
                        if self.model_type == 'R2P2_SimpleRNN':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, decode_start_vel, decode_start_pos, motion_encoded=True)

                        elif self.model_type == 'R2P2_RNN':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, episode_idx, decode_start_vel, decode_start_pos, scene_encoding_, motion_encoded=True, scene_encoded=True)

                        elif self.model_type == 'CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs, agent_encoded=True)

                        elif self.model_type == 'Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                        elif self.model_type == 'Global_Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

                        elif self.model_type == 'AttGlobal_Scene_CAM_NFDecoder':
                            gen_trajs, z, mu, sigma = self.model(motion_encoding_, src_lens, agent_tgt_three_mask, episode_idx, decode_start_vel, decode_start_pos, num_src_trajs, scene_encoding_, agent_encoded=True, scene_encoded=True)

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
                            gen_trajs = self.model(src_trajs, src_lens, agent_tgt_three_mask, decode_start_vel, decode_start_pos, num_src_trajs)                                                    

                        gen_trajs = gen_trajs.reshape(num_three_agents, self.num_candidates, self.decoding_steps, 2)


                    rs_error3 = ((gen_trajs - tgt_trajs.unsqueeze(1)) ** 2).sum(dim=-1).sqrt_()
                    rs_error2 = rs_error3[..., :int(self.decoding_steps*2/3)]
                    
                    
                    diff = gen_trajs - tgt_trajs.unsqueeze(1)
                    msd_error = (diff[:,:,:,0] ** 2 + diff[:,:,:,1] ** 2)
                    
                    num_agents = gen_trajs.size(0)
                    num_agents2 = rs_error2.size(0)
                    num_agents3 = rs_error3.size(0)
                    # print(num_agents, num_agents2, num_agents3)

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

                    print("Working on test {:d}/{:d}, batch {:d}/{:d}... ".format(test_time+1, self.test_times, b+1, len(self.data_loader)), end='\r')# +

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
                        candidate_i = gen_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        tgt_traj_i = tgt_trajs[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]
                        tgt_lens_i = tgt_lens[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]

                        src_traj_i = src_trajs[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                        src_lens_i = src_lens[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]
                        map_file_i = map_files[i]
                        output_file_i = output_files[i]
                        # print(output_file_i)
                        if self.dataset == 'nuscenes_carla':
                            candidate_i = candidate_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_traj_i = tgt_traj_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_lens_i = tgt_lens_i[agent_tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                        else:
                            candidate_i = candidate_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_traj_i = tgt_traj_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]
                            tgt_lens_i = tgt_lens_i[tgt_three_mask[cum_num_tgt_trajs[i]:cum_num_tgt_trajs[i+1]]]

                        src_traj_i = src_traj_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]
                        src_lens_i = src_lens_i[agent_tgt_three_mask[cum_num_src_trajs[i]:cum_num_src_trajs[i+1]]]

                        dao_i, dao_mask_i = self.dao(candidate_i, map_file_i)
                        dac_i, dac_mask_i = self.dac(candidate_i, map_file_i)
                        if self.render==0:
                            self.write_img_output(candidate_i, src_traj_i, src_lens_i, tgt_traj_i, tgt_lens_i, map_file_i, output_file_i)

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

        test_ades = ( test_minade2, test_avgade2, test_minade3, test_avgade3 )
        test_fdes = ( test_minfde2, test_avgfde2, test_minfde3, test_avgfde3 )

        test_sd = [np.mean(list_sd), np.std(list_sd)]
        test_angle = [np.mean(list_angle), np.std(list_angle)]

        print("--Final Performane Report--")
        print("minADE3: {:.5f}±{:.5f}, minFDE3: {:.5f}±{:.5f}".format(test_minade3[0], test_minade3[1], test_minfde3[0], test_minfde3[1]))
        print("avgADE3: {:.5f}±{:.5f}, avgFDE3: {:.5f}±{:.5f}".format(test_avgade3[0], test_avgade3[1], test_avgfde3[0], test_avgfde3[1]))
        print("DAO: {:.5f}±{:.5f}, DAC: {:.5f}±{:.5f}".format(test_dao[0] * 10000.0, test_dao[1] * 10000.0, test_dac[0], test_dac[1]))
        print("SD: {:.5f}±{:.5f}".format(test_sd[0], test_sd[1]))
        print("Angle: {:.5f}±{:.5f}".format(test_angle[0]*180/np.pi, test_angle[1]*180/np.pi))
        with open(self.out_dir + '/metric.pkl', 'wb') as f:
            pkl.dump({"ADEs": test_ades,
                      "FDEs": test_fdes,
                      "Qloss": test_qloss,
                      "Ploss": test_ploss, 
                      "DAO": test_dao,
                      "DAC": test_dac,
                      "SD": test_sd,
                      "Angle": test_angle}, f)