import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from abstract_agent import Agent
from replay_memory import ReplayMemory, Transition


class DoubleDQNAgent(Agent):
    q_a: nn.Module
    q_b: nn.Module
    loss_function: nn.Module
    optimizer_A: torch.optim.Optimizer
    optimizer_B: torch.optim.Optimizer

    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate,
                 gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, sync_target=100):
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                         epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block)

        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.q_a = model_a
        self.q_a.to(self.device)
        self.q_b = model_b
        self.q_b.to(self.device)

        # Asignar una función de costo (MSE) (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss()
        self.loss_function.to(self.device)

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = Adam(self.q_a.parameters(), lr=self.learning_rate)
        self.optimizer_B = Adam(self.q_b.parameters(), lr=self.learning_rate)

    def _predict_action(self, state):
        with torch.no_grad():
            state_t = self.state_processing_function(state).to(self.device)
            state_t = state_t.unsqueeze(0)
            action_t = torch.argmax(self.q_a(state_t) + self.q_b(state_t), dim=1)
            action = action_t.item()
        return action

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados.

            # Enviar los tensores al dispositivo correspondiente.
            states = ?
            actions = ?
            rewards = ?
            dones = ?  # Dones deberia ser 0 y 1; no True y False. Pueden usar .float() en un tensor para convertirlo
            next_states = ?

            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.

            # Para el Q elegido:
            # Obetener el valor estado-accion (Q) de acuerdo al Q seleccionado.
            q_actual = ?

            # Obtener max a' Q para los siguientes estados (del minibatch) (Usando el Q no seleccionado).
            # Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            max_q_next_state = ?

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            target = ?

            # Resetear gradientes

            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
