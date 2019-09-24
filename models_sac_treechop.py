
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class CriticNetwork(nn.Module):   
    def __init__(self, 
                 acts_dim=10,
                 num_filters=64, 
                 use_bn=False, 
                 pov_scaling=255,
                 lin_1_dim=128,
                 lin_2_dim=64):
        super(CriticNetwork, self).__init__()

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filters, padding=0,
                               kernel_size=9, stride=1, bias=not use_bn)   # output dim: 56
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)         # output dim: 28
        if use_bn: self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, padding=1,stride=2,
                               kernel_size=4, bias=not use_bn)    # output dim: 14
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output dim: 7
        if use_bn: self.bn2 = nn.BatchNorm2d(num_features=num_filters)

        # Fully connected layer
        self.linear1 = nn.Linear(num_filters*7*7+acts_dim, lin_1_dim) #todo: automatically calculate this
        self.linear2 = nn.Linear(lin_1_dim, lin_2_dim)
        self.linear3 = nn.Linear(lin_2_dim, 1)
        
        self.non_lin_1 = self.non_lin_2 = self.non_lin_3 = None 
        self.use_bn = use_bn
        self.pov_scaling = pov_scaling
        
    def forward(self, obs, acts):
        
        x = self.conv1(obs.permute(0,3,1,2) / self.pov_scaling)
        x = self.max_pool1(x)
        if self.use_bn: x = self.bn1(x)
        self.non_lin_1 = F.relu(x)
        
        x = self.conv2(self.non_lin_1)
        x = self.max_pool2(x)
        if self.use_bn: x = self.bn2(x)
        self.non_lin_2 = F.relu(x)
        
        x = x.view(self.non_lin_2.size(0), -1)
        x = self.linear1(torch.cat([x, acts], dim=1))
        self.non_lin_3 = F.relu(x)
        x = self.linear2(self.non_lin_3)
        self.non_lin_4 = F.relu(x)
        out = self.linear3(self.non_lin_4)
        return out
    

class ActorNetwork(nn.Module):
    def __init__(self, 
                 acts_dim=10, 
                 num_filters=64, 
                 use_bn=False, 
                 pov_scaling=255, 
                 compass_scaling=180,
                 lin_1_dim=128,
                 lin_2_dim=64):
        
        super(ActorNetwork, self).__init__()

        # Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filters, 
                               padding=0, kernel_size=9, stride=1, bias=not use_bn)   # output dim: 56
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)         # output dim: 28
        if use_bn: self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, 
                               padding=1,stride=2, kernel_size=4, bias=not use_bn)    # output dim: 14
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output dim: 7
        if use_bn: self.bn2 = nn.BatchNorm2d(num_features=num_filters)

        # Fully connected layer
        self.linear1 = nn.Linear(num_filters*7*7, lin_1_dim) #todo: automatically calculate this
        self.linear2 = nn.Linear(lin_1_dim, lin_2_dim)
        self.mean_linear= nn.Linear(lin_2_dim, acts_dim)
        self.log_std_linear= nn.Linear(lin_2_dim, acts_dim)

        self.normal = Normal(0, 1)
        
        self.use_bn = use_bn
        self.pov_scaling = pov_scaling

    def forward(self, obs):
        
        x = self.conv1( obs.permute(0,3,1,2) / self.pov_scaling)
        x = self.max_pool1(x)
        if self.use_bn: x = self.bn1(x)
        self.non_lin_1 = F.relu(x)
        
        x = self.conv2(self.non_lin_1)
        x = self.max_pool2(x)
        if self.use_bn: x = self.bn2(x)
        self.non_lin_2 = F.relu(x)
        
        x = x.view(self.non_lin_2.size(0), -1)
        x = self.linear1(x)
        self.non_lin_3 = F.relu(x)
        x = self.linear2(self.non_lin_3)
        self.non_lin_4 = F.relu(x)
        
        self.mean = self.mean_linear(self.non_lin_4)
        log_std = self.log_std_linear(self.non_lin_4)
        self.log_std = torch.clamp(log_std, -20, 2)
        return self.mean, self.log_std
    
    def get_log_probs(self, obs, epsilon=1e-6):
        mean, log_std = self.forward(obs)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        action_logit =   Normal(mean, std).sample()
        action = torch.tanh(action_logit)
        log_prob = Normal(mean, std).log_prob(action_logit) - torch.log(1. - action.pow(2) + epsilon)
        #assert float(log_prob.mean())==float(log_prob.mean()), "Log_prob is nan"
        return log_prob.sum(dim=1, keepdim=True), action
    
    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp() # no clip in evaluation, clip affects gradients flow
            action_logit =  mean + std * self.normal.sample()
            action = torch.tanh(action_logit) # TanhNormal distribution as actions; reparameterization trick     
        return action