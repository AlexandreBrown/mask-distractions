import torch
import torch.nn.functional as F
import madi.utils as utils
from madi.algorithms.vit.svea_vit import SVEA_ViT
import madi.algorithms.modules as m
from madi.augmentations import strong_augment
from segdac.agents.stochastic_agent import StochasticAgent
from segdac.agents.sampling_strategy import SamplingStrategy
from tensordict import TensorDict


class MaDi_ViT(SVEA_ViT, StochasticAgent):
    """MaDi: Masking Distractions for Generalization in Reinforcement Learning
    Vision Transformer backbone in this version"""

    def __init__(
        self,
        obs_shape,
        action_dim,
        gamma,
        critic_tau,
        encoder_tau,
        actor_update_freq,
        critic_target_update_freq,
        num_shared_layers,
        num_filters,
        num_head_layers,
        projection_dim,
        hidden_dim,
        actor_log_std_min,
        actor_log_std_max,
        init_temperature,
        actor_lr,
        actor_beta,
        critic_lr,
        critic_beta,
        critic_weight_decay,
        alpha_lr,
        alpha_beta,
        svea_alpha,
        svea_beta,
        augment,
        overlay_alpha,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        qvk_bias,
        env_action_scaler,
        frame_stack,
        masker_num_filters,
        masker_num_layers,
        mask_type,
        mask_threshold,
        mask_threshold_type,
        masker_lr,
        masker_beta,
        places365_data_dir_path,
    ):
        StochasticAgent.__init__(self=self, env_action_scaler=env_action_scaler)
        SVEA_ViT.__init__(
            self,
            obs_shape,
            action_dim,
            gamma,
            critic_tau,
            encoder_tau,
            actor_update_freq,
            critic_target_update_freq,
            num_shared_layers,
            num_filters,
            num_head_layers,
            projection_dim,
            hidden_dim,
            actor_log_std_min,
            actor_log_std_max,
            init_temperature,
            actor_lr,
            actor_beta,
            critic_lr,
            critic_beta,
            critic_weight_decay,
            alpha_lr,
            alpha_beta,
            svea_alpha,
            svea_beta,
            augment,
            overlay_alpha,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qvk_bias,
        )
        self.places365_data_dir_path = places365_data_dir_path
        self.masker = m.MaskerNet(
            obs_shape,
            frame_stack,
            masker_num_filters,
            masker_num_layers,
            mask_type,
            mask_threshold,
            mask_threshold_type,
        ).cuda()
        self.masker_optimizer = torch.optim.Adam(
            self.masker.parameters(), lr=masker_lr, betas=(masker_beta, 0.999)
        )
        self.num_masks = frame_stack

    def apply_mask(self, obs, test_env=False):
        # obs: tensor shaped as (B, 9, H, W)
        frames = obs.chunk(
            self.num_masks, dim=1
        )  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        frames_cat = torch.cat(
            frames, dim=0
        )  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
        masks_cat = self.masker(
            frames_cat, test_env=test_env
        )  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
        masks = masks_cat.chunk(
            self.num_masks, dim=0
        )  # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
        masked_frames = [
            m * f for m, f in zip(masks, frames)
        ]  # element-wise multiplication. uses broadcasting. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        return torch.cat(
            masked_frames, dim=1
        )  # concat in channel dim. returns: tensor shaped (B, 9, H, W)

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        if self.sampling_strategy == SamplingStrategy.RANDOM:
            action = self.sample_action(observation)
        else:
            action = self.select_action(observation, test_env=True)

        return action

    def select_action(self, obs, test_env=False):
        _obs = obs
        _obs = self.apply_mask(_obs, test_env)
        mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu

    def sample_action(self, obs):
        _obs = obs
        _obs = self.apply_mask(_obs)
        mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi

    def update(self, train_data: TensorDict, env_step: int) -> dict:
        self.train()
        obs = train_data["pixels_transformed"]
        action = train_data["action"]
        reward = train_data["next"]["reward"]
        next_obs = train_data["next"]["pixels_transformed"]
        done = train_data["next"]["done"]
        not_done = 1 - done.float()

        logs = dict()

        logs.update(self.update_critic(obs, action, reward, next_obs, not_done))

        if env_step % self.actor_update_freq == 0:
            logs.update(self.update_actor_and_alpha(obs))

        if env_step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        return logs

    def update_critic(self, obs, action, reward, next_obs, not_done) -> dict:
        self.actor.eval()
        self.critic_target.eval()
        self.critic.train()

        with torch.no_grad():
            next_obs = self.apply_mask(next_obs)
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs_aug = strong_augment(
            obs, self.places365_data_dir_path, self.augment, self.overlay_alpha
        )

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, obs_aug)
            obs = self.apply_mask(obs)
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * (
                F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            )
        else:
            # chance to speedup: apply_mask just once here as well
            # just concat(obs, obs_aug), apply, then split back again
            # actually: could probably do the same with the 2 critic forward passes in regular SVEA
            obs = self.apply_mask(obs)
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * (
                F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            )
            obs_aug = self.apply_mask(obs_aug)
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * (
                F.mse_loss(current_Q1_aug, target_Q)
                + F.mse_loss(current_Q2_aug, target_Q)
            )

        logs = dict()

        logs["critic_target_q"] = target_Q.mean().item()
        logs["critic_q1"] = current_Q1.mean().item()
        logs["critic_q2"] = current_Q2.mean().item()
        logs["critic_loss"] = critic_loss.item()

        self.masker_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.masker_optimizer.step()

        return logs

    def update_actor_and_alpha(self, obs, update_alpha=True) -> dict:
        self.critic.eval()
        self.actor.train()

        obs = self.apply_mask(obs)
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        logs = dict()
        logs["actor_loss"] = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            logs["entropy_loss"] = alpha_loss.item()

        logs["log_alpha"] = self.log_alpha.item()
        logs["alpha"] = self.alpha.item()

        return logs
