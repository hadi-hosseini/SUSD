# noqa: D212, D415
"""
Modified from simple_spread.py
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym import spaces
import torch
from matplotlib import colormaps

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        img_encoder=None,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_heterogenous_v3"

        self.img_encoder = img_encoder
        if img_encoder is None:
            state_dim = N * 7
            self.state_space = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(state_dim,),
                dtype=np.float32,
            )
            self.use_img = False
        else:
            state_dim = N * 5
            self.state_space = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(state_dim,),
                dtype=np.float32,
            )
            self.use_img = True
            from slot_attention.data import MPTransforms
            self.img_transforms = MPTransforms([64,64])

    def observe(self, agent):
        return None

    def state(self):
        speed_idx = [10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39, 42, 43, 46, 47]
        if self.use_img:
            img = self.render()
            with torch.no_grad():
                # Format and process the img using the given encoder
                img = self.img_transforms(img)
                img = img.to(device=self.img_encoder.load_device).unsqueeze(0)
                out_dict = self.img_encoder(img)
                rel_dist, pos, _, _ = out_dict["regression"]
                state = torch.cat([rel_dist.flatten(), pos.flatten()]).cpu().numpy().astype(np.float32)

            use_gt = False
            if use_gt:
                candidate_idx = range(70)
                idx = []
                for id in candidate_idx:
                    if id not in speed_idx:
                        idx.append(id)
                state = self.scenario.get_state(self.world).astype(np.float32)[idx]

        else:
            state = self.scenario.get_state(self.world).astype(np.float32)

            mask_out_velocity = False
            if mask_out_velocity:
                state[speed_idx] = 0
        return state


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class clipWorld(World):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        super().step()
        for agent in self.agents:
            agent.state.p_pos = np.clip(
                agent.state.p_pos, -1, 1
            )  # clip position

class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = clipWorld()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True
        self.agent_size = 0.00916  # 0.03
        self.landmark_size = self.agent_size / 3 * 5

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = self.agent_size

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = np.array([0.75, 0.75, 0.75])
            landmark.size = self.landmark_size
        return world

    def reset_world(self, world, np_random):
        colors = np.array([
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [0.5, 0.5, 0],  # Olive
            [0, 0.5, 0.5],  # Teal
            [0.5, 0, 0.5],  # Purple
            [0.5, 0.5, 0.5]  # Gray
        ])
        # colors = np.array(colormaps['tab20'].colors)

        for i, agent in enumerate(world.agents):
            # black, gray, white, red, blue, green, yellow, orange, brown, purple, and pink
            agent.color = colors[i] # np.array([1, 0, 0]) * i / len(world.landmarks)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            # landmark.color = colors[i+2]  # np.array([0, 0, 1]) * i / len(world.landmarks)
            landmark.color = colors[i]  # np.array([0, 0, 1]) * i / len(world.landmarks)


        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0
        return rew

    def global_reward(self, world):
        rew = 0
        return rew

    def get_state(self, world):
        agent_stats = []

        for agt in world.agents:
            agent_stats.append(agt.state.p_vel)
            agent_stats.append(agt.state.p_pos)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)
            
        diayn_states = []
        upper_threshold = np.inf
        for idx, lm in enumerate(world.landmarks):
            other_idx = (idx + 1) % len(world.landmarks)
            ag_idx_list = [idx]

            dists = [
                np.sqrt(np.sum(np.square(world.agents[a_i].state.p_pos - lm.state.p_pos)))
                for a_i in ag_idx_list
            ]

            # Cap the distance
            dists += [upper_threshold]
            m_dist = min(dists)

            diayn_states.append(m_dist)

        return np.concatenate([diayn_states] + agent_stats + entity_pos)

    def observation(self, agent, world):
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + other_pos + comm
        )
