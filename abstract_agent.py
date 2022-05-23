from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from replay_memory import ReplayMemory, Transition
from utils import show_video


class Agent:
    device: torch.device
    state_processing_function: Callable[[np.array], torch.Tensor]
    memory = ReplayMemory
    env: object
    batch_size: int
    learning_rate: float
    gamma: float
    epsilon_i: float
    epsilon_f: float
    epsilon_anneal: float
    epsilon_decay: float
    episode_block: int
    total_steps: int

    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Function phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente 
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0

    def train(self, number_episodes=50000, max_steps_episode=10000, max_steps=1000000,
              writer_name="default_writer_name"):
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment="-" + writer_name)

        for ep in tqdm(range(number_episodes), unit=' episodes'):
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            observation = self.env.reset()

            current_episode_reward = 0.0

            for s in range(max_steps):

                # Seleccionar action usando una política epsilon-greedy.
                A = self.select_action(observation, total_steps, False)

                # Ejecutar la action, observar resultado y procesarlo como indica el algoritmo.

                current_episode_reward += reward
                total_steps += 1

                # Guardar la transicion en la memoria

                # Actualizar el estado

                # Actualizar el modelo

                if done:
                    break

            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)

            # Report on the training rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                print(
                    f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

        print(
            f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}")

        torch.save(self.policy_net.state_dict(), "GenericDQNAgent.dat")
        writer.close()

        return rewards

    def compute_epsilon(self, steps_so_far: int) -> float:
        return self.epsilon_i + (self.epsilon_f - self.epsilon_i) * min(1, steps_so_far / self.epsilon_anneal)

    def record_test_episode(self, env):
        done = False

        # Observar estado inicial como indica el algoritmo 

        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.
            observation = self.state_processing_function(self.env.reset())
            # Seleccione una accion de forma completamente greedy.
            A = self.select_action(observation, train=False)
            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.

            if done:
                break

                # Actualizar el estado

        env.close()
        show_video()

    @abstractmethod
    def _predict_action(self, state):
        pass

    def select_action(self, state, current_steps, train=True):
        """
        Se selecciona la action epsilongreedy-mente si se esta entrenando y completamente greedy en otro caso.
        :param state: es la observacion.
        :param current_steps: cantidad de pasos llevados actualmente. En el caso de Train=False no se tiene en
         consideracion.
        :param train: si se está entrenando, True para indicar que si, False para indicar que no.
        """
        if not train:
            action = self._predict_action(state)
        else:
            random_number = np.random.uniform()
            if random_number >= self.compute_epsilon(steps_so_far=current_steps):
                action = self._predict_action(state)
            else:
                action = np.random.choice(self.env.action_space.n)
        return action

