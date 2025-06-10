import madi.algorithms.modules as m
import torch
import torch.nn.functional as F
import madi.utils as utils
from madi.algorithms.svea import SVEA
from madi.augmentations import strong_augment
from segdac.agents.agent import Agent
from segdac.action_scaling.env_action_scaler import TanhEnvActionScaler
from segdac.agents.action_sampling_strategy import ActionSamplingStrategy
from segdac.data.mdp import MdpData
from tensordict import TensorDict


class MaDiActionSamplingStrategy(ActionSamplingStrategy):
    def __init__(self, actor, masker, num_masks):
        super().__init__(actor)
        self.masker = masker
        self.num_masks = num_masks

    @torch.no_grad()
    def forward(self, mdp_data: MdpData) -> TensorDict:
        b, s, c, h, w = mdp_data.data["pixels"].shape
        obs = mdp_data.data["pixels"].reshape(b, s * c, h, w)

        test_env = not self.is_exploration_enabled and not self.is_stochasticity_enabled

        obs = self.apply_mask(obs, test_env=test_env)

        if test_env:
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)

            action = mu
        else:
            mu, pi, _, _ = self.actor(obs, compute_pi=True, compute_log_pi=False)

            action = pi

        return TensorDict(
            {"unscaled_action": action}, batch_size=torch.Size([action.shape[0]])
        )

    def apply_mask(self, obs, test_env=False):
        # obs: tensor shaped as (B, 9, H, W)
        frames = obs.chunk(self.num_masks, dim=1)  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        frames_cat = torch.cat(frames, dim=0)  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
        masks_cat = self.masker(frames_cat, test_env=test_env)  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
        # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
        masks = masks_cat.chunk(self.num_masks, dim=0)
        # element-wise multiplication, uses broadcasting over the 3 RGB channels within 1 frame. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        masked_frames = [m * f for m, f in zip(masks, frames)]
        return torch.cat(masked_frames, dim=1)  # concat in channel dim. returns: tensor shaped (B, 9, H, W)


class MaDi(SVEA, Agent):
    """MaDi: Masking Distractions for Generalization in Reinforcement Learning"""

    def __init__(
        self,
        obs_shape,
        action_shape,
        discount,
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
        frame_stack,
        masker_num_filters,
        masker_num_layers,
        mask_type,
        mask_threshold,
        mask_threshold_type,
        masker_lr,
        masker_beta,
        dataset_dir,
        env_action_scaler: TanhEnvActionScaler,
    ):
        Agent.__init__(
            self,
            env_action_scaler,
            None
        )
        SVEA.__init__(
            self,
            obs_shape,
            action_shape,
            discount,
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
            dataset_dir=dataset_dir
        )
        self.masker = m.MaskerNet(
            obs_shape,
            frame_stack,
            masker_num_filters,
            masker_num_layers,
            mask_type,
            mask_threshold,
            mask_threshold_type,
        ).cuda()
        self.action_sampling_strategy = MaDiActionSamplingStrategy(
            actor=self.actor,
            masker=self.masker,
            num_masks=frame_stack
        )

        self.masker_optimizer = torch.optim.Adam(
            self.masker.parameters(), lr=masker_lr, betas=(masker_beta, 0.999)
        )
        self.num_masks = frame_stack

    @property
    def actor(self):
        return self.action_sampling_strategy.actor

    def apply_mask(self, obs, test_env=False):
        # obs: tensor shaped as (B, 9, H, W)
        frames = obs.chunk(self.num_masks, dim=1)  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        frames_cat = torch.cat(frames, dim=0)  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
        masks_cat = self.masker(frames_cat, test_env=test_env)  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
        # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
        masks = masks_cat.chunk(self.num_masks, dim=0)
        # element-wise multiplication, uses broadcasting over the 3 RGB channels within 1 frame. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
        masked_frames = [m * f for m, f in zip(masks, frames)]
        return torch.cat(masked_frames, dim=1)  # concat in channel dim. returns: tensor shaped (B, 9, H, W)

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            next_obs = self.apply_mask(next_obs)
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs_aug = strong_augment(obs, self.augment, dataset_dir=self.dataset_dir, overlay_alpha=self.overlay_alpha)

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, obs_aug)
            obs = self.apply_mask(obs)
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        else:
            obs = self.apply_mask(obs)
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
            obs_aug = self.apply_mask(obs_aug)
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * \
                (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        if L is not None:
            L["critic_target_q"] = target_Q.mean()
            L["critic_q1"] = current_Q1.detach().mean()
            L["critic_q2"] = current_Q2.detach().mean()
            L["critic_loss"] = critic_loss.detach()

        self.masker_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.masker_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        obs = self.apply_mask(obs)
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L["actor_loss"] = actor_loss.detach()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L['entropy_loss'] = alpha_loss.detach()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()
