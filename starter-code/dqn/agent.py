from gym import spaces
import numpy as np

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

from torch.optim import Adam

device = "cuda"


class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        # TODO: Initialise agent's networks, optimiser and replay buffer -> Done 
        self.lr = lr
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        #agents networks
        self.target_network = DQN(observation_space,action_space).to(device)
        self.policy_network = DQN(observation_space,action_space).to(device)
        #agents optimiser 
        self.optimiser = Adam(self.policy_network.parameters(),lr=self.lr)
        
        # raise NotImplementedError

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        raise NotImplementedError

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters -> Done 
        self.target_network.load_state_dict(self.policy_network.state_dict())
        raise NotImplementedError

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        def policy_fn(sess, observation):
            # A = np.ones(nA, dtype=float) * epsilon / nA
            # q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn
        # raise NotImplementedError
