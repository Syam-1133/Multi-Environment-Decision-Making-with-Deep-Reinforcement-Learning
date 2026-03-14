import os
import random
from numpy import mod
import yaml
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo


class HighwayEnv:
    def __init__(self, env, config_dir, offscreen_rendering=True, video_path=None, render_mode=None):
        self.envs = []
        for env_name in env.names:
            sub_env = gym.make(env_name, render_mode=render_mode)
            sub_env.unwrapped.configure({'offscreen_rendering': offscreen_rendering})
            if env.custom_config:
                sub_env.unwrapped.configure(self._load_config(os.path.join(config_dir, env_name + '.yaml')))
            if video_path is not None:
                os.makedirs(video_path, exist_ok=True)
                sub_env = self._record_videos(sub_env, os.path.join(video_path, env_name), env.record_frequency)
            self.envs.append(sub_env)

        self.current_env = None
        self.current_env_idx = -1
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _record_videos(self, env, path, record_frequency):
        monitor = RecordVideo(
            env, path, episode_trigger=lambda episode: episode % record_frequency == 0
        )
        return monitor
    
    def _validate(self):
        if self.current_env is None:
            raise AssertionError('No environment selected.')
    
    @property
    def current_env_name(self):
        self._validate()
        return self.current_env.__str__().split('-v0')[0].split('<')[-1]
    
    @property
    def num_envs(self):
        return len(self.envs)
    
    @property
    def observation_space(self):
        # FIXME: This currently assumes that all environments have the same observation space shape
        return self.envs[0].observation_space
    
    @property
    def action_space(self):
        # FIXME: This currently assumes that all environments have the same action space shape
        return self.envs[0].action_space

    def close(self):
        self.current_env = None
        self.current_env_idx = -1
        for env in self.envs:
            env.close()
    
    def _set_seed(self):
        # seed is passed directly to reset() in gymnasium API
        pass

    def reset(self):
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.current_env = self.envs[self.current_env_idx]
        self._set_seed()
        obs, _ = self.current_env.reset(seed=random.randint(1, 100))
        return obs

    def step(self, action):
        self._validate()
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self):
        self._validate()
        self.current_env.render()


def create_env(config, config_dir, output_dir, mode='train', offscreen_rendering=True):
    # Enable rgb_array rendering for eval/test modes to support video recording
    render_mode = 'rgb_array' if config.save_video and mode in ['eval', 'test'] else None
    return HighwayEnv(
        config, config_dir,
        offscreen_rendering=offscreen_rendering,
        video_path=output_dir if mode == 'test' and config.save_video else None,
        render_mode=render_mode
    )
