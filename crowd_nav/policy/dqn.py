import torch
import torch.nn as nn
import numpy as np
import itertools
import random
from collections import deque
import logging
import torch.nn.functional as F
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.state import JointState

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value

# =============== Replay ===============
class TensorReplay:
    def __init__(self, cap, feat_dim, device):
        self.cap = cap
        self.device = device
        self.ptr = 0
        self.size = 0

        # preallocate tensors on device
        self.S  = torch.empty((cap, feat_dim), device=device, dtype=torch.float32)
        self.S2 = torch.empty((cap, feat_dim), device=device, dtype=torch.float32)
        self.A  = torch.empty((cap,), device=device, dtype=torch.long)
        self.R  = torch.empty((cap,), device=device, dtype=torch.float32)
        self.D  = torch.empty((cap,), device=device, dtype=torch.float32)

    @torch.no_grad()
    def push(self, policy, s_js, a_idx, r, s2_js, d):
        """Pre-transform to feature tensors once, then store directly on device."""
        s_t  = policy.transform(s_js)   # (F,) on policy.device
        s2_t = policy.transform(s2_js)  # (F,)

        i = self.ptr
        self.S[i].copy_(s_t)
        self.S2[i].copy_(s2_t)
        self.A[i] = int(a_idx)
        self.R[i] = float(r)
        self.D[i] = float(d)

        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    @torch.no_grad()
    def sample(self, n):
        n = min(n, self.size)
        idx = torch.randint(self.size, (n,), device=self.device)
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]

    def __len__(self):
        return self.size


class DQN(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'DQN'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('dqn', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.target = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.rb = TensorReplay(cap=10_000, feat_dim=self.joint_state_dim, device=self.device)
        self.multiagent_training = config.getboolean('dqn', 'multiagent_training')
        logging.info('Policy: DQN without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.tau = config.getfloat('rl', 'tau')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)
        self.target.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    @torch.no_grad()
    def predict(self, state, return_index=False):
        """
        Returns the selected ActionXY/ActionRot.
        If return_index=True, returns (action, index).
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # --- reached goal
        if self.reach_destination(state):
            act = ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
            return (act, 0) if return_index else act

        # --- ensure action space exists
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        # --- epsilon-greedy
        if self.phase == 'train' and np.random.random() < self.epsilon:
            idx = np.random.choice(len(self.action_space))
            act = self.action_space[idx]
            return (act, idx) if return_index else act

        # --- greedy lookahead (same as before)
        self_state = state.self_state
        human_states = state.human_states

        rows = [self_state + h for h in human_states]  # assumes these are lists/tuples
        batch_next_states = torch.tensor(rows, dtype=torch.float32, device=self.device)

        outputs = self.model(self.rotate(batch_next_states))  # (H, n_actions)
        min_per_action, _ = torch.min(outputs, dim=0)         # (n_actions,)
        idx = int(min_per_action.argmax().item())
        act = self.action_space[idx]
        return (act, idx) if return_index else act

    def update(self, batch):
        s, a, r, s2, d = batch
        s  = s.to(self.device, non_blocking=True)
        a  = a.to(self.device, non_blocking=True)
        r  = r.to(self.device, non_blocking=True)
        s2 = s2.to(self.device, non_blocking=True)
        d  = d.to(self.device, non_blocking=True)

        self.model.train()
        q_sa = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.model(s2).argmax(1, keepdim=True)
            q_next = self.target(s2).gather(1, a_star).squeeze(1)
            target = r + (1.0 - d) * self.gamma * q_next

        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()

        with torch.no_grad():
            for p_t, p in zip(self.target.parameters(), self.model.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return float(loss.item())



    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        state = torch.Tensor(state.self_state + state.human_states[0]).to(self.device)
        state = self.rotate(state.unsqueeze(0)).squeeze(dim=0)
        return state


    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
