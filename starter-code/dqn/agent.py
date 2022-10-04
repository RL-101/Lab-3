from gym import spaces
import numpy as np

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

from torch.optim import Adam
from torch import nn

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
        self.use_double_dqn = use_double_dqn
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
        # TODO -> done
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        # get batch from replay buffer and convert info into tensors
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        tensor_states = torch.from_numpy(np.array(states)/255.0).float().to(device)
        tensor_next_states = torch.from_numpy(np.array(next_states)/255.0).float().to(device)
        tensor_actions = torch.from_numpy(actions).long().to(device)
        tensor_rewards = torch.from_numpy(rewards).float().to(device)
        tensor_dones = torch.from_numpy(dones).float().to(device)

        # don't track gradients
        with torch.no_grad():
            # calculate target
            estimate = None
            if self.use_double_dqn:
                # get next action from the other network
                _, next_action = self.policy_network(tensor_next_states).max(1)
                # get next q from target network
                estimate = self.target_network(tensor_next_states).gather(1, next_action.unsqueeze(1)).squeeze()
            else:
                # get next q from target network
                estimate = self.target_network(tensor_next_states).max(1)

            target = tensor_rewards
            if tensor_dones != 1:
                # if not done
                target += self.gamma * estimate


        # backpropagation
        new_estimate = self.policy_network(tensor_next_states).gather(1, tensor_actions.unsqueeze(1)).squeeze()
        loss = nn.MSELoss(new_estimate, target)
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()

        # remove from GPU
        del tensor_states
        del tensor_next_states

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters -> Done 
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # make rgb values have a range of [0,1]
        state = state/255.0
        # convert state to tensor object and put on GPU
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # making sure gradients aren't saved for the following calculations
        with torch.no_grad():
            # get action-state values using the foward pass of the network
            qs = self.policy_network(state)
            # get max action
            _, action = qs.max(1)
            # return action from tensor object
            return action.item()
