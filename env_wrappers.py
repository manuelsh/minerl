import numpy as np

class StackFrames():
    def __init__(self, env, steps, only_last=False, obs_rest=False):
        self.env = env
        self.steps = steps
        self.only_last = only_last # returns only last frame
        self.obs_rest = False
        assert not obs_rest, "Not implemented yet"
    
    def step(self, action):
        end_rew = 0
        end_obs = {'pov':[]}
            
        for i in range(self.steps):
            obs, rew, done, _ = self.env.step(action)
            end_rew += rew
            end_obs['pov'].append(obs['pov'])
            if done:
                break
        
        if self.only_last:
            end_obs['pov'] = end_obs['pov'][-1]
        else:
            end_obs['pov'] = np.stack(end_obs['pov'])
        
        return end_obs, end_rew, done
    
    def reset(self):
        obs = self.env.reset()
        return obs        
        
    def close(self):
        self.env.close()
        
    def seed(self, seed):
        self.env.seed(seed)