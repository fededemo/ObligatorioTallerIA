import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super().__init__()  # super(FeedForwardModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=env_inputs[0], out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        
        # pooling respeta los canales de entrada
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)   # Regiones de 2x2 con paso 2.
           
        self.fc1 = nn.Linear(in_features=21*21*8, out_features=128) 
        
        self.output = nn.Linear(in_features=128, out_features=n_actions)
  


    def forward(self, env_input):
        #((n+2p-f)/s) + 1
        # input = 84*84*3
        result = self.conv1(env_input)
        # luego de la convl 84*84*16
        result = F.relu(self.pooling_layer(result))
        # luego del pooling tenemos 42*42*16
        result = self.conv2(result)
        # luego de la conv2 42*42*8
        result = F.relu(self.pooling_layer(result))     
        # luego de pooling tenemos 21*21*8

        # "Achatamos" los feature maps (flatten)
        result = result.reshape((-1, self.fc1.in_features))
        result = F.relu(self.fc1(result))
        
        return self.output(result)
