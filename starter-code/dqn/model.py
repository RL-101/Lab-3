from gym import spaces
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"
        
        
        #  input image is 84 x 84 X 4 
        #  first hidden layer convolves 16 8 x 8 filters with stride 4
        #  it means we do a 8 x 8 convolution without padding and stride=4
        #  stride shrinks the size to ceil((n+f-1)/s) where ’n’ is input dimensions ‘f’ is filter size and ‘s’ is stride length
        #  the input image will be strinked to ((84+8-1)/4)
        #  output layer is fully connected with single output for each valid action
        
        # num out features in current layer is equal to num in features in lext layer
  
        self.convolutions = nn.Sequential(
            
            #  first hidden layer convolves 16 8 x 8 filters with stride 4
            nn.Conv2d(observation_space.shape[0], 32, 8, stride=4),
            nn.ReLU(),
            #  second layer convolves 32 4x4 filters with stride 2 -> num in_features is 32
            #  because num out_features in current layer has to be equal to num in_features in lext layer
            #  i then said num out_fearures for the first layer is 32, idk
            #  So for the num out_features i said because Deep Q-Learning uses 2 neural networks 
            #  so at every step weights from the main network are copied to the target network
            #  so at each step 2 copies of the input are produced 
            #  so the output will be twice the input
            #  idk if this understanding is correct
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            #  final hidden layer is fully connected with 256 rectifier units
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        self.fully_connected_layer = nn.Sequential(
            # 49 games were considered -> 7x7
            # 
            nn.Linear(in_features=64*7*7 , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )

    def forward(self, x):
        output = self.convolutions(x)
        output = self.fully_connected_layer(output.view(x.size()[0],-1))
        return output
