import os
import time
import random
import numpy as np
from copy import deepcopy
from gym import spaces

from envs.elden_kitchen_new.kitchen import Kitchen
from envs.elden_kitchen_new.kitchen_skills import DropSkill, MoveSkill, GraspSkill, ToggleSkill


class KitchenWithMid(Kitchen):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 0.8, 0.05),
        table_offset=(-0.2, 0, 0.90),
        butter_x_range=(0.2, 0.3),
        butter_y_range=(-0.3, 0.0),
        meatball_x_range=(0.2, 0.3),
        meatball_y_range=(-0.3, 0.0),
        pot_x_range=(0.07, 0.07),
        pot_y_range=(-0.05, -0.05),
        button_x_range=(0.07, 0.07),
        button_y_range=(-0.05, -0.05),
        stove_x_range=(0.07, 0.07),
        stove_y_range=(-0.05, -0.05),
        normalization_range=((-0.5, -0.5, 0.7), (0.5, 0.5, 1.1))
    ):

        self.skill_timestep = 0

        self.pre_interact_xy_dist_thre = 0.15
        self.pre_grasp_dist_thre = 0.025
        self.pre_toggle_dist_thre = 0.05

        self.move_to_drop_pot_z_offset = 0.05
        self.move_to_drop_in_pot_z_offset = 0.10
        self.move_to_toggle_z_offset = 0.1

        self.slice_dict = None

        self.has_renderer = has_renderer

        self.prev_skill_action = np.zeros(4)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            placement_initializer=placement_initializer,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=True,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            table_full_size=table_full_size,
            table_offset=table_offset,
            butter_x_range=butter_x_range,
            butter_y_range=butter_y_range,
            meatball_x_range=meatball_x_range,
            meatball_y_range=meatball_y_range,
            pot_x_range=pot_x_range,
            pot_y_range=pot_y_range,
            button_x_range=button_x_range,
            button_y_range=button_y_range,
            stove_x_range=stove_x_range,
            stove_y_range=stove_y_range,
            normalization_range=normalization_range
        )


        global_act_low, global_act_high = self.global_low + 0.1, self.global_high - 0.1
        global_act_low[2] = self.table_offset[2] + 0.1
        self.global_act_mean = (global_act_low + global_act_high) / 2
        self.global_act_scale = (global_act_high - global_act_low) / 2


    def normalize_obs(self, obs, out_of_range_warning=True):
        self.prev_unnormalized_obs = deepcopy(obs)
        return super().normalize_obs(obs, out_of_range_warning)

    def reset(self):
        self.skill_timestep = 0
        obs = super().reset()
        return obs
    
    @property
    def action_spec(self):
        low=np.array([-1, -1, -1, -1, -1], dtype=np.float32)
        high=np.array([1, 1, 1, 1, 1], dtype=np.float32)
        return low, high

    def get_grasp_status(self, state):
        if state["butter_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "butter_pos"
            in_hand_obj_touch_key = "butter_grasped"
        elif state["meatball_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "meatball_pos"
            in_hand_obj_touch_key = "meatball_grasped"
        elif state["pot_grasped"]:
            thing_in_hand = True
            in_hand_obj_pos_key = "pot_pos"
            in_hand_obj_touch_key = "pot_touched"
        else:
            thing_in_hand = False
            in_hand_obj_pos_key = "robot0_eef_pos"
            in_hand_obj_touch_key = None
        return thing_in_hand, in_hand_obj_pos_key, in_hand_obj_touch_key

    def get_nearby_interact_obj(self, state, goal_pos, obj_pos_keys):
        for obj_pos_key in obj_pos_keys:
            obj_pos = state[obj_pos_key]
            xy_dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
            if xy_dist <= self.pre_interact_xy_dist_thre:
                return obj_pos_key
        return None

    def get_nearby_drop_obj(self, state, goal_pos, in_hand_obj_pos_key):
        if in_hand_obj_pos_key == "pot_pos":
            drop_obj_pos_keys = ["stove_pos"]
        else:
            drop_obj_pos_keys = ["pot_pos"]
        return self.get_nearby_interact_obj(state, goal_pos, drop_obj_pos_keys)

    def get_nearby_grasp_obj(self, state, goal_pos):
        grasp_obj_pos_keys = ["pot_handle_pos"]
        if not self.butter_in_pot:
            grasp_obj_pos_keys.append("butter_pos")
        if not self.meatball_in_pot:
            grasp_obj_pos_keys.append("meatball_pos")
        return self.get_nearby_interact_obj(state, goal_pos, grasp_obj_pos_keys)

    def get_nearby_toggle_obj(self, state, goal_pos):
        return self.get_nearby_interact_obj(state, goal_pos, ["button_handle_pos"])

    def step(self, action):
        state = self.prev_unnormalized_obs

        # action [-1, 1] -> global xyz + binary grasp or not + binary toggle or not
        target_pos = action[:3] * self.global_act_scale + self.global_act_mean
        grasp = action[3] > 0
        toggle = action[4] > 0

        # update cooking status
        if self.button_on and self.pot_on_stove:
            if self.butter_in_pot:
                prev_butter_melt_status = self.butter_melt_status
                self.butter_melt_status = min(self.butter_melt_status + 0.2, 1)
                if prev_butter_melt_status < 1 and self.butter_melt_status == 1:
                    butter_obj = self.objects_dict["butter"]
                    body_id = self.obj_body_id["butter"]
                    self.sim.data.set_joint_qpos(butter_obj.joints[0], np.concatenate([np.array((0, 0, 0)), self.sim.model.body_quat[body_id]]))
            if self.meatball_in_pot:
                if self.butter_melt_status != 1:
                    self.meatball_overcooked = True
                elif not self.meatball_overcooked:
                    self.meatball_cook_status = min(self.meatball_cook_status + 0.2, 1)

        do_nothing = False
        thing_in_hand, in_hand_obj_pos_key, in_hand_obj_touch_key = self.get_grasp_status(state)

        eef_pos = state["robot0_eef_pos"]

        """
        move to xyz
        if thing_in_hand:
            if grasp:
                keep holding
            else:
                drop
            toggle: do nothing
        else:
            if grasp:
                grasp thing nearby
            if toggle:
                toggle
        """
        z_offset = 0
        if thing_in_hand:
            drop_key = self.get_nearby_drop_obj(state, target_pos, in_hand_obj_pos_key)
            if not grasp and drop_key is not None:
                if in_hand_obj_pos_key == "pot_pos":
                    z_offset = self.move_to_drop_pot_z_offset
                else:
                    z_offset = self.move_to_drop_in_pot_z_offset
                skills_to_execute = [MoveSkill(in_hand_obj_pos_key, drop_key, z_offset, thing_in_hand),
                                     DropSkill(in_hand_obj_touch_key)]
            else:
                skills_to_execute = [MoveSkill(in_hand_obj_pos_key, target_pos, z_offset, thing_in_hand)]
        else:
            grasp_key = self.get_nearby_grasp_obj(state, target_pos)
            toggle_key = self.get_nearby_toggle_obj(state, target_pos)
            if grasp and grasp_key is not None:
                if grasp_key == "butter_pos":
                    obj_grasped_key = "butter_grasped"
                elif grasp_key == "meatball_pos":
                    obj_grasped_key = "meatball_grasped"
                elif grasp_key == "pot_handle_pos":
                    obj_grasped_key = "pot_grasped"
                skills_to_execute = [MoveSkill(in_hand_obj_pos_key, grasp_key, z_offset, thing_in_hand),
                                     GraspSkill(grasp_key, obj_grasped_key)]
            elif toggle and toggle_key is not None:
                skills_to_execute = [MoveSkill(in_hand_obj_pos_key, toggle_key, self.move_to_toggle_z_offset, thing_in_hand),
                                     ToggleSkill(not self.button_on)]
            else:
                skills_to_execute = [MoveSkill(in_hand_obj_pos_key, target_pos, z_offset, thing_in_hand)]

        for skill in skills_to_execute:
            skill_done = False
            while not skill_done:
                skill_action, skill_done = skill.step(self.prev_unnormalized_obs)
                next_obs, reward, _, info = super().step(skill_action)
                self.prev_skill_action = skill_action
                if self.has_renderer:
                    self.render()
                    time.sleep(0.01)

        self.skill_timestep += 1
        done = self.skill_timestep >= self.horizon

        return next_obs, reward, done, info