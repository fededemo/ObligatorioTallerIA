import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from abstract_agent import Agent
from replay_memory import ReplayMemory, Transition


class DQNAgent(Agent):
    policy_net: nn.Module
    loss_function: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block):
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i,
                         epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block)

        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.policy_net = model
        self.policy_net.to(self.device)

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss()
        self.loss_function.to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def _predict_action(self, state):
        with torch.no_grad():
            state_t = self.state_processing_function(state).to(self.device)
            state_t = state_t.unsqueeze(0)
            action_t = torch.argmax(self.policy_net(state_t), dim=1)
            action = action_t.item()
        return action

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            # Resetear gradientes

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados. 

            # Enviar los tensores al dispositivo correspondiente.
            states = ?
            actions = ?
            rewards = ?
            dones = ?  # Dones deberia ser 0 y 1; no True y False. Pueden usar .float() en un tensor para convertirlo
            next_states = ?

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = ?

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            max_q_next_state = ?

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.    
            target = ?

            # Compute el costo y actualice los pesos.
            # En Pytorch la función de costo se llaman con (predicciones, objetivos) en ese orden.
