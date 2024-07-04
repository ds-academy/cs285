from typing import Sequence, Callable, Tuple, Optional
import numpy as np
import torch
from torch import nn

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        # current network : 학습 중 지속적으로 업데이트 됨
        self.critic = make_critic(observation_shape, num_actions)
        # target network : 현재 네트워크의 파라메터가 주기적으로 복사됨
        self.target_critic = make_critic(observation_shape, num_actions)
        # critic's optimizer and scheduler
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        # target network update periods
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        # init 할때 update target_network's param <- critic network's param
        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if np.random.random() < epsilon:
            # Exploration
            return np.random.randint(self.num_actions)
        else:
            # Exploitation
            observation = ptu.from_numpy(np.asarray(observation))[None]
            with torch.no_grad():
                q_values = self.critic(observation)
                # q_values dim (batch, num_actions)
                # 각 배체 항목에 대해 최대 Q 값 인덱스 추출
                # item () -> 정수로 변환
            return q_values.argmax(dim=1).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        # get batch_size of reward tensor

        # Data Flow
        # use_double_q : next_obs -> self.critic -> choose action -> self.target_critic -> Q evaluation -> target_q
        # dqn : next_obs -> self.target_critic -> choose action and evaluation Q -> target_q

        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            # it use target_network
            # next_qa_values dim : (batch, num_actions)
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                # next_action dim : (batch_size, 1), keepdim : keep dimension of input dim.
                next_action = self.critic(next_obs).argmax(dim=1, keepdim=True)
            else:
                next_action = next_qa_values.argmax(dim=1, keepdim=True)

            # gather(input, dim, index)
            # next_qa_values dim : batch_size, num_actions
            # next_action dim : batch_size, 1
            # gather(1, next_action) -> 각 배치 항목에 대해 next_action에 해당하는 Q값 선택
            # (batch_size, 1) -> (bath_size, ) 각 배치에 대한 Q 값만 선택
            next_q_values = next_qa_values.gather(1, next_action).squeeze(1)

            assert next_q_values.shape == (batch_size, )
            target_values = reward + self.discount * next_q_values * (~done)

        # TODO(student): train the critic with the target values

        qa_values = self.critic(obs)
        # qa_values (batch_size, num_actions)
        # action (batch_size, ) -> (batch_size, 1)
        # 실제 action 에 대한 Q 값 선택 -> (batch_size, )
        q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)  # Compute from the data actions; see torch.gather

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()
        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        return critic_stats
