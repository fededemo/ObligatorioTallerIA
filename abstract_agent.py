from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from replay_memory import ReplayMemory
from utils import show_video


class Agent(ABC):
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
    episode_block: int
    total_steps: int
    use_pretrained: bool
    model_weights_dir_path: str

    def __init__(self, gym_env: object, obs_processing_func: Callable, memory_buffer_size: int, batch_size: int,
                 learning_rate: float, gamma: float, epsilon_i: float, epsilon_f: float,
                 epsilon_anneal_time: int, episode_block: int,
                 use_pretrained: Optional[bool] = False, model_weights_dir_path: Optional[str] = './weights',
                 save_between_steps: Optional[int] = None):
        self.use_pretrained = use_pretrained
        self.model_weights_dir_path = model_weights_dir_path

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Function phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente 
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        self.save_between_steps = save_between_steps
        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.episode_block = episode_block

        self.total_steps = 0

    def train(self, number_episodes=50000, max_steps_episode=10000, max_steps=1000000,
              writer_name="default_writer_name"):
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment=f">>> {writer_name}")

        for ep in tqdm(range(number_episodes), unit=' episodes'):
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            S = self.env.reset()

            current_episode_reward = 0.0
            steps_in_episode = 0
            for s in range(max_steps_episode):

                # Seleccionar action usando una política epsilon-greedy.
                A = self.select_action(S, current_steps=total_steps)

                # Ejecutar la action, observar resultado y procesarlo como indica el algoritmo.
                S_prime, R, done, _ = self.env.step(A)
                current_episode_reward += R
                total_steps += 1
                steps_in_episode += 1

                # Guardar la transition en la memoria
                # Transition: ('state', 'action', 'reward', 'done', 'next_state')
                self.memory.add(S, A, R, done, S_prime)

                # Actualizar el estado
                S = S_prime

                # Actualizar el modelo
                self.update_weights()

                if self.save_between_steps is not None and total_steps % self.save_between_steps == 0:
                    self._save_net(suffix=total_steps)

                if done:
                    break

            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.compute_epsilon(total_steps), total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)

            # Report on the training rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                print(f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])}, "
                      f"epsilon {self.compute_epsilon(total_steps):.5f}, total steps {total_steps}")

        print(f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])}, "
              f"epsilon {self.compute_epsilon(total_steps):.5f}, total steps {total_steps}")

        # persist this with a function
        self._save_net()
        writer.close()

        return rewards

    def compute_epsilon(self, steps_so_far: int) -> float:
        # 1 + (0.02 - 1) * min (1, 882/1000)
        return self.epsilon_i + (self.epsilon_f - self.epsilon_i) * min(1, steps_so_far / self.epsilon_anneal)

    def record_test_episode(self, env):
        done = False

        # Observar estado inicial como indica el algoritmo 
        S = self.state_processing_function(env.reset())
        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            A = self.select_action(S, train=False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            S_prime, R, done, _ = env.step(A)

            # Actualizar el estado
            S = S_prime

            if done:
                break

        env.close()
        show_video()

    @abstractmethod
    def _save_net(self, suffix: Optional[str] = None) -> None:
        """
        Guarda los pesos de la red a disco.
        :param suffix: sufijo a agregar al archivo.
        """
        pass

    @abstractmethod
    def _load_net(self) -> None:
        """
        Carga los pesos de la red desde disco.
        """
        pass

    @abstractmethod
    def _predict_rewards(self, state: np.array) -> np.array:
        """
        Dado un estado devuelve las rewards de cada action.
        :param state: state dado.
        :returns: la lista de rewards para cada action.
        """
        pass

    @abstractmethod
    def _predict_action(self, state: np.array) -> int:
        """
        Dado un estado me predice el action de mayor reward (greedy).
        :param state: state dado.
        :returns: el action con mayor reward.
        """
        pass

    def select_action(self, state: np.array, current_steps: Optional[int] = None, train: bool = True) -> int:
        """
        Se selecciona la action epsilongreedy-mente si se esta entrenando y completamente greedy en otro caso.
        :param state: es la observacion.
        :param current_steps: cantidad de pasos llevados actualmente. En el caso de Train=False no se tiene en
         consideracion.
        :param train: si se está entrenando, True para indicar que si, False para indicar que no.
        :returns: an action.
        """
        if not train:
            with torch.no_grad():
                action = self._predict_action(state)
        else:
            random_number = np.random.uniform()
            if random_number >= self.compute_epsilon(steps_so_far=current_steps):
                action = self._predict_action(state)
            else:
                action = np.random.choice(self.env.action_space.n)
        return action

    @abstractmethod
    def update_weights(self):
        pass

