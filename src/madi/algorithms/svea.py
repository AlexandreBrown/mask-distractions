import torch
import torch.nn.functional as F
import madi.utils as utils
from madi.algorithms.sac import SAC
from madi.augmentations import strong_augment
from segdac.data.mdp import MdpData
from tensordict import TensorDict


class SVEA(SAC):
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
        dataset_dir
    ):
        super().__init__(
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
            alpha_beta
        )
        self.svea_alpha = svea_alpha
        self.svea_beta = svea_beta
        self.augment = augment
        self.overlay_alpha = overlay_alpha
        self.dataset_dir = dataset_dir

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs_aug = strong_augment(obs, self.augment, self.dataset_dir, self.overlay_alpha)

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, obs_aug)
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * \
                (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * \
                (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

        if L is not None:
            L["critic_target_q"] = target_Q.mean()
            L["critic_q1"] = current_Q1.detach().mean()
            L["critic_q2"] = current_Q2.detach().mean()
            L["critic_loss"] = critic_loss.detach()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(
        self, train_mdp_data: MdpData, env_step: int, is_time_to_evaluate: bool
    ) -> TensorDict:
        logs_data = {}

        b, s, c, h, w = train_mdp_data.data["pixels"].shape
        obs = train_mdp_data.data["pixels"].reshape(b, s * c, h, w)
        action = train_mdp_data.data["action"]
        reward = train_mdp_data.next.data["reward"].reshape(-1, 1)
        next_obs = train_mdp_data.next.data["pixels"].reshape(b, s * c, h, w)
        not_done = ~train_mdp_data.next.data["done"].reshape(-1, 1)

        if is_time_to_evaluate:
            L = logs_data
        else:
            L = None

        self.update_critic(obs, action, reward, next_obs, not_done, L, env_step)

        if env_step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, env_step)

        if env_step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        return TensorDict(logs_data, batch_size=torch.Size([]))
