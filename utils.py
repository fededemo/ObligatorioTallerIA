from base64 import b64encode
import collections
from glob import glob
from os.path import getmtime
from os import remove, rename
import io
import uuid
from typing import List
import subprocess

import cv2
import gym.spaces
import numpy as np
import torch
from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from IPython.display import HTML


def show_video():
    """
    Utility function to enable video recording of gym environment and displaying it
    To enable video, just do "env = wrap_env(env)""
    """
    mp4list = sorted(glob('./video/*/*.mp4'), key=getmtime, reverse=True)
    if len(mp4list) > 0:
        mp4 = mp4list[0]

        try:
            # Workaround para gym monitor 0.19.0 que usa mpeg4 como codec y no h264.
            command = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{mp4}"'
            codec = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True).stdout.strip()

            if codec == 'mpeg4':
                h264_file = mp4.replace('.mp4', '_h264.mp4')
                command = f'ffmpeg -i "{mp4}" -y "{h264_file}"'
                output = subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                remove(mp4)
                rename(h264_file, mp4)

            video = io.open(mp4, 'r+b').read()
            encoded = b64encode(video)
            ipythondisplay.display(HTML(
                data='''
                    <video alt="test" autoplay loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                     </video>
                '''.format(encoded.decode('ascii'))))
        except Exception as e:
            print("No se pudo crear el video")
    else:
        print("No se encontró el video.")


def wrap_env(env):
    """
    Wrapper del ambiente donde definimos un Monitor que guarda la visualización como un archivo de video.
    """

    env = Monitor(env, './video/' + str(uuid.uuid4()), force=True)
    return env


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# OpenAI Gym Wrappers
# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1],
                                                                          old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


def to_tensor(elements: np.array) -> torch.Tensor:
    """
    Transforma los elements en un tensor de floats
    :param elements: numpy array de elementos.
    :returns: un tensor cargado con los elementos dados.
    """
    return torch.tensor(elements, dtype=torch.float32)


def process_state(obs: List[float]) -> torch.Tensor:
    """
    Transforma la observación en un tensor de floats
    :param obs: array con la secuencia de imágenes de la observación.
    :returns: un tensor cargado con la observación dada.
    """
    return to_tensor(obs)
