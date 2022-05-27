import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from abstract_agent import Agent
from utils import to_tensor


class DQNAgent(Agent):
    policy_net: nn.Module
    loss_function: nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block):
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i,
                         epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block)

        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.policy_net = model.to(self.device)

        # Asignar una función de costo (MSE) (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def _predict_action(self, state):
        # with torch.no_grad():
        state_t = self.state_processing_function(state).to(self.device)
        state_t = state_t.unsqueeze(0)
        action_t = torch.argmax(self.policy_net(state_t), dim=1)
        action = action_t.item()
        return action

    def _predict_rewards(self, states: np.array) -> np.array:
        """
        Dado una serie de estados devuelve las rewards para cada action.
        :param states: states dados.
        :returns: la lista de rewards para cada action de cada estado.
        """
        # with torch.no_grad():
        state_t = self.state_processing_function(states).to(self.device)
        rewards = self.policy_net(state_t)
        return rewards

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminacion y siguentes estados.
            # Transition: ('state', 'action', 'reward', 'done', 'next_state')
            mini_batch = self.memory.sample(self.batch_size)

            # Enviar los tensores al dispositivo correspondiente.
            states, actions, rewards, dones, next_states = zip(*mini_batch)

            states = to_tensor(states).to(self.device)
            actions = to_tensor(actions).long().to(self.device)
            rewards = to_tensor(rewards).to(self.device)
            dones = to_tensor(dones).to(self.device)
            next_states = to_tensor(next_states).to(self.device)

            # Obtener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = self._predict_rewards(states)
            predicted = q_actual[torch.arange(self.batch_size), actions]

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            max_q_next_state = torch.max(self._predict_rewards(next_states), dim=1).values.detach()

            # Compute el target de DQN de acuerdo a la Ecuación (3) del paper.
            # y_true = R + (1 - done) * self.gamma * max(self._predict_rewards(S_prime))
            target = rewards + (1 - dones) * self.gamma * max_q_next_state

            # Compute el costo y actualice los pesos.
            loss = self.loss_function(predicted, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            self.optimizer.step()

    def _save_net(self) -> None:
        """
        Guarda los pesos de la red a disco.
        """
        torch.save(self.policy_net.state_dict(), "./weights/DQNAgent.pt")
