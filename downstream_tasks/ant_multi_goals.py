import numpy as np

from envs.mujoco.ant_env import AntEnv

class AntMultiGoalsEnv(AntEnv):
    def __init__(self, **kwargs):
        self.goal_radius = 3.0
        self.max_steps_per_goal = 50 # 50
        self.num_goals_per_episode = 4
        self.goal_count = 0
        self.steps_since_goal = 0
        self.current_goal = None

        super().__init__(task="goal", **kwargs)

        self.sample_new_goal()


    def reset_model(self):
        obs = super().reset_model()
        self.goal_count = 0
        self.steps_since_goal = 0
        self.sample_new_goal()
        return obs

    def sample_new_goal(self):
        sx = self.sim.data.qpos.flat[0]
        sy = self.sim.data.qpos.flat[1]
        self.current_goal = np.random.uniform(
            low=[sx - 7.5, sy - 7.5],
            high=[sx + 7.5, sy + 7.5]
        )
        self.steps_since_goal = 0
        self.goal_count += 1

    def compute_reward(self, **kwargs):
        xpos = kwargs.get('xposafter')
        ypos = kwargs.get('yposafter')

        if self.current_goal is None:
            return 0.0
    
        else:
            distance = np.linalg.norm(np.array([xpos, ypos]) - self.current_goal)
            if distance <= self.goal_radius:
                self.sample_new_goal()
                return 2.5
            return 0

    def step(self, action, render=False):
        obs, reward, done, info = super().step(action, render=render)

        done = False
        
        if self.current_goal is not None:
            self.steps_since_goal += 1

            # Check if goal reached
            agent_pos = self.sim.data.qpos.flat[:2]
            distance = np.linalg.norm(agent_pos - self.current_goal)

            if distance <= self.goal_radius or self.steps_since_goal >= self.max_steps_per_goal:
                if self.goal_count < self.num_goals_per_episode:
                    self.sample_new_goal()
                else:
                    done = True 
                
            if self.goal_count > self.num_goals_per_episode:
                done = True

            info["current_goal"] = self.current_goal

        return obs, reward, done, info
