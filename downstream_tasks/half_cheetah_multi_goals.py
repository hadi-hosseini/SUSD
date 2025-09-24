import numpy as np


from envs.mujoco.half_cheetah_env import HalfCheetahEnv


class HalfCheetahGoal(HalfCheetahEnv):
    def __init__(self, **kwargs):
            self.goal_radius = 3.0
            self.current_goal = None
            super().__init__(**kwargs)
            self.sample_new_goal()


    def reset_model(self):
        obs = super().reset_model()
        self.sample_new_goal()
        return obs

    def sample_new_goal(self):
        self.current_goal = np.random.uniform(-10, 10)


    def compute_reward(self, **kwargs):
        xpos = self.sim.data.qpos[0]
        if self.current_goal is None:
            return 0.0
    
        else:
            distance = np.linalg.norm(np.array([xpos]) - self.current_goal)
            if distance <= self.goal_radius:
                self.sample_new_goal()
                return 10.0
            return 0

    def step(self, action, render=False):
        obs, reward, done, info = super().step(action, render=render)

        done = False
        
        if self.current_goal is not None:

            agent_pos = self.sim.data.qpos[0]
            distance = np.linalg.norm(agent_pos - self.current_goal)

            if distance <= self.goal_radius:
                self.sample_new_goal()

            info["current_goal"] = self.current_goal

        return obs, reward, done, info
