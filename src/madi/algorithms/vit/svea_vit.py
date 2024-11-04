import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import madi.utils as utils
import madi.algorithms.modules as m
from madi.algorithms.sac import SAC
from madi.augmentations import strong_augment


class SVEA_ViT(SAC):
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
    ):
        super().__init__(
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
        )
        self.svea_alpha = svea_alpha
        self.svea_beta = svea_beta
        self.augment = augment
        self.overlay_alpha = overlay_alpha

        self.discount = gamma
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        shared = m.SharedTransformer(
            obs_shape,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qvk_bias,
        ).cuda()
        head = m.HeadCNN(shared.out_shape, num_head_layers, num_filters).cuda()
        actor_encoder = m.Encoder(
            shared, head, m.RLProjection(head.out_shape, projection_dim)
        )
        critic_encoder = m.Encoder(
            shared, head, m.RLProjection(head.out_shape, projection_dim)
        )

        self.actor = m.Actor(
            actor_encoder,
            action_dim,
            hidden_dim,
            actor_log_std_min,
            actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(critic_encoder, action_dim, hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=(critic_beta, 0.999),
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )
        self.train()

        print("Shared ViT:", utils.count_parameters(shared))
        print("Head:", utils.count_parameters(head))
        print("Projection:", utils.count_parameters(critic_encoder.projection))
        print("Critic: 2x", utils.count_parameters(self.critic.Q1))

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_vit()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def update_critic(self, obs, action, reward, next_obs, not_done, step=None):
        metrics = dict()

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)

        if self.svea_alpha == self.svea_beta:
            obs = utils.cat(obs, obs_aug)
            action = utils.cat(action, action)
            target_Q = utils.cat(target_Q, target_Q)

            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = (self.svea_alpha + self.svea_beta) * (
                F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            )
        else:
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = self.svea_alpha * (
                F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            )
            current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
            critic_loss += self.svea_beta * (
                F.mse_loss(current_Q1_aug, target_Q)
                + F.mse_loss(current_Q2_aug, target_Q)
            )

        metrics["critic_loss_mean"] = critic_loss.item() / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics
