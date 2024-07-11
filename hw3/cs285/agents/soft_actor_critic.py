from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            # obs -> tensor -> add dim [1, len(self.action_dim)]
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)  # (1, self.action_dim) -> (action_dim, )

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        # iterate for critics
        # output : (num_critics, batch_size)
        # (self.num_critic_networks, self.num_actor_samples, batch_size)
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"

        num_critic_networks, batch_size = next_qs.shape

        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            # Double Q : action 선택과 value 평가를 서로 다른 네트워크 에서 수행
            # 여러 critic networks 중 하나의 q value 추출
            next_qs = next_qs[torch.randperm(self.num_critic_networks)]

        elif self.target_critic_backup_type == "min":
            # next_qs : (num of critic networks, batch_size)
            # num of critic networks 차원에서 가장 작은 값들을 각 batch 별로 선정
            next_qs = next_qs.min(dim=0, keepdim=True)  # (1, batch_size)
            # expand q values to critics networks
            # 값의 참조값 복사, memory efficient
            next_qs = next_qs.expand(self.num_critic_networks, batch_size)  # (num_critic_networks, batch_size)

        elif self.target_critic_backup_type == "mean":
            next_qs = next_qs.mean(dim=0, keepdim=True)
            next_qs = next_qs.expand(self.num_critic_networks, batch_size)
        else:
            # Default, we don't need to do anything.
            pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            # (1, batch_size) -> (num_critic_networks, batch_size)
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape

        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,       # (batch_size, obs_dim)
        action: torch.Tensor,    # (batch_size, action_dim)
        reward: torch.Tensor,    # (batch_size, )
        next_obs: torch.Tensor,  # (batch_size, obs_dim)
        done: torch.Tensor,      # (batch_size, )
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            # Sample from the actor
            # (batch_size, obs_dim)
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()  # output (batch_size, act_dim)

            # Compute the next Q-values for the sampled actions
            # next_obs : batch_size, obs_dim
            # next_action : batch_size, actiom_dim
            next_qs = self.target_critic(next_obs, next_action)  # Q(s,a)  -> (num critic networks, batch size)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs = self.q_backup_strategy(next_qs)

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                next_action_entropy = self.entropy(next_action_distribution)
                next_qs = next_qs + self.backup_entropy * next_action_entropy

            # Compute the target Q-value
            target_values: torch.Tensor = reward + self.discount * next_qs * (~done)
            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Update the critic
        # Predict Q-values
        q_values = self.critic(obs, action)

        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        loss: torch.Tensor = self.critic_loss(
            q_values.mean(dim=0), target_values.mean(dim=0))

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        # reparameterize sampling -> enable gradient backpropagation
        # continuous : (batch_size, action_dim)
        # discrete : (batch_size, )
        action = action_distribution.rsample()
        # c : (batch_size, 1)
        # d : (batch_size, )
        # log probability of p(x)
        log_probs = action_distribution.log_prob(action)
        if log_probs.dim() == 2:
            log_probs = log_probs.squeeze(dim=-1)  # (batch_size, 1) -> (batch_size, )
        # -log(p)
        entropy = -log_probs.mean()
        return entropy

    def actor_loss_reinforce(self, obs: torch.Tensor):
        """
        - 여러번 샘플링을 추출하는 방법 (num_actor_samples)
        - 여러 크리틱 네트워크의 값을 활용하고, 각 상태에 대해 여러 행동을 평가함.
        - 불안정할 수 있음.

        """
        batch_size = obs.shape[0]   # (batch_size, observation_dim)

        # TODO(student): Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            # TODO(student): draw num_actor_samples samples from the action distribution for each batch element
            action = action_distribution.sample((self.num_actor_samples, ))   # 각 배치에 대해 샘플링할 샘플의 수
            # batch_size 별로 가능한 act_dim 만큼

            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # TODO(student): Compute Q-values for the current state-action pair
            # obs (batch_size, observation_dim) -> obs[:, None] -> (batch_size, 1, obesrvation_dim)
            # expand(-1, self.num_actor_samples, -1) -> -1 해당 차원 변경 없음. -> (batch_size, num_actor_samples, obs_dim)
            # 각 관찰에 대해 num_actor_samples 만큼 복사
            q_values = self.critic(obs[:, None].expand(-1, self.num_actor_samples, -1), action)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        # TODO(student)
        log_probs = action_distribution.log_prob(action)
        loss = -log_probs * advantage

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        """
        - reinforce 방식과 다르게 한번만 샘플링

        """
        batch_size = obs.shape[0]

        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # TODO(student): Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action = action_distribution.rsample()

        # TODO(student): Compute Q-values for the sampled state-action pair
        q_values = self.critic(obs, action)

        # TODO(student): Compute the actor loss
        loss = -torch.mean(q_values)

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        for _ in range(self.num_critic_updates):
            critic_infos.append(self.update_critic(
                observations, actions, rewards, next_observations, dones
            ))

        # TODO(student): Update the actor
        actor_info = {}

        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.target_update_period is None:   # Soft update
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif step % self.target_update_period == 0:    # Hard Update
            self.update_target_critic()

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
