from typing import Tuple

import torch
import torch.nn as nn


class DQN_CNN_Model(nn.Module):
    def __init__(self, env_inputs: Tuple[int], n_actions: int):
        super().__init__()
        # Architecture adapted from https://github.com/jasonbian97/Deep-Q-Learning-Atari-Pytorch/blob/master/DQNs.py
        self.cnn = nn.Sequential(
            nn.Conv2d(env_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, n_actions)
        )

    def forward(self, env_input):
        x = self.cnn(env_input)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
