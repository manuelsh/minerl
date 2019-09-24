import numpy as np
from typing import Deque, Dict, List, Tuple, Union
import os
import numpy
import torch
import random
import matplotlib.pyplot as plt
import joblib

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int,
        act_dim: int,
        batch_size: int
#         n_step: int = 1, 
#         gamma: float = 0.99
    ):
        
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
#         # for N-step Learning
#         self.n_step_buffer = deque(maxlen=n_step)
#         self.n_step = n_step
#         self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> None:
        
        transition = (obs, act, rew, next_obs, done)
#         self.n_step_buffer.append(transition)

#         # single step transition is not ready
#         if len(self.n_step_buffer) < self.n_step:
#             return ()
        
#         # make a n-step transition
#         rew, next_obs, done = self._get_n_step_info(
#             self.n_step_buffer, self.gamma
#         )
#         obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
#         return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            dones=self.done_buf[idxs],
#             # for N-step Learning
#             indices=indices,
        )
    
    def __len__(self) -> int:
        return self.size
    


def seed_everything(seed, env):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

    
class Monitor:
    def __init__(
        self, 
        filename: str,
        path: str
    ):
        self.filename = filename
        self.path = path
        self.data = {}
        
    def add(
        self, 
        name: Union[str, List[str]], 
        data: any
    ):
        if isinstance(name, str):
            self._add(name, data)
        if isinstance(name, list):
            for n, d in zip(name, data):
                self._add(n, d)
            
    def _add(
        self, 
        name: str,
        data: any
    ):
        if name in self.data:
            if isinstance(data, List):
                self.data[name].append(*data)
            else:
                self.data[name].append(data)
        else:
            if isinstance(data, List):
                self.data[name] = data 
            else:
                self.data[name] = [data] 
            
    def save(self):
        joblib.dump(self.data, self.path+'/'+self.filename)
    
    def load(self):
        self.data = joblib.load(self.path+'/'+self.filename)
    
    def plot(self, names: Union[str, List[str]]):
        plt.figure(figsize=(20, 5))
        if isinstance(names, List):
            for n in names:
                plt.plot(self.data[n], label=n)
        elif isinstance(names, str):
            plt.plot(self.data[names], label=names)
            
        plt.grid()
        plt.title(str(names))
        plt.legend()
        
    def plot_all(self):
        n = 1
        plt.figure(figsize=(20, 5))
        for name in self.data.keys():
            plt.subplot(1,3,n)
            plt.plot(self.data[name])
            plt.title(name)
            plt.grid()
            if n==3:
                plt.figure(figsize=(20, 5))
            n = n+1 if n<3 else 1
        plt.show()
        
        
class MonitorV2:
    def __init__(
        self, 
        filename: str,
        path: str
    ):
        self.filename = filename
        self.path = path
        self.data = {}
        
    def add(
        self,
        family: Union[str, List[str]],
        name: Union[str, List[str]], 
        data: any
    ):
        if isinstance(name, str):
            self._add(family, name, data)
        if isinstance(family, name, list):
            for f, n, d in zip(family, name, data):
                self._add(n, d)
            
    def _add(
        self, 
        family: str,
        name: str,
        data: any
    ):
        if family not in self.data:
            self.data[family] = {}
        if name not in self.data[family]:
            self.data[family][name] = []
        
        if isinstance(data, List):
            self.data[family][name].append(*data)
        else:
            self.data[family][name].append(data)
            
    def save(self):
        pickle.dump(self.data, open(self.path+'/'+self.filename, 'wb'))
    
    def load(self):
        self.data = pickle.load(open(self.path+'/'+self.filename, 'rb'))
    
    def plot(self, names: Union[str, List[str]]):
        plt.figure(figsize=(20, 5))
        if isinstance(names, List):
            for n in names:
                plt.plot(self.data[n], label=n)
        elif isinstance(names, str):
            plt.plot(self.data[names], label=names)
            
        plt.grid()
        plt.title(str(names))
        plt.legend()
        
    def plot_all(self):
        n = 1
        plt.figure(figsize=(20, 5))
        for name in self.data.keys():
            plt.subplot(1,3,n)
            plt.plot(self.data[name])
            plt.title(name)
            plt.grid()
            if n==3:
                plt.figure(figsize=(20, 5))
            n = n+1 if n<3 else 1
        plt.show()