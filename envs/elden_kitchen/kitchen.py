import os
import random
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject, BallObject, CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.wrappers import DomainRandomizationWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

from envs.elden_kitchen.kitchen_objects import TargetObject, ButtonObject, StoveObject, PotObject

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Kitchen(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
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
        target_x_range=(0.07, 0.07),
        target_y_range=(-0.05, -0.05),
        normalization_range=((-0.5, -0.5, 0.7), (0.5, 0.5, 1.1)),
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = table_full_size
        self.table_offset = table_offset

        # reward configuration
        self.reward_scale = reward_scale

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.objects_dict = {}
        self.fixtures_dict = {}

        self.objects = []
        self.fixtures = []

        self.butter_x_range = butter_x_range
        self.butter_y_range = butter_y_range
        self.meatball_x_range = meatball_x_range
        self.meatball_y_range = meatball_y_range
        self.pot_x_range = pot_x_range
        self.pot_y_range = pot_y_range
        self.button_x_range = button_x_range
        self.button_y_range = button_y_range
        self.stove_x_range = stove_x_range
        self.stove_y_range = stove_y_range
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range

        # global position range for normalization
        global_low, global_high = normalization_range
        self.global_low = np.array(global_low)
        self.global_high = np.array(global_high)
        self.global_mean = (self.global_high + self.global_low) / 2
        self.global_scale = (self.global_high - self.global_low) / 2

        # eef velocity range for normalization
        self.eef_vel_scale = np.array([2, 2, 2])

        # gripper angle range for normalization
        self.gripper_qpos_scale = np.array([0.03, 0.03])

        # gripper angular velocity range for normalization
        self.gripper_qvel_scale = np.array([0.5, 0.5])

        self.button_joint_qpos_scale = 0.4

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the task succeeds.

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """

        meatball_in_pot = self.check_contact(self.objects_dict["meatball"], "pot_body_bottom")
        pot_on_stove = self.check_contact("stove_collision_burner", "pot_body_bottom")

        pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]
        target_pos = self.sim.data.body_xpos[self.obj_body_id["target"]]
        target_pot_xy_dist = np.linalg.norm(pot_pos[:2] - target_pos[:2])

        pot_touched = int(self.check_contact(self.robots[0].gripper, self.objects_dict["pot"]))

        pot_on_target = target_pot_xy_dist < 0.07 and not pot_touched

        if self.meatball_cook_status < 1:
            self.stage = int(meatball_in_pot) + int(self.butter_melt_status == 1) + int(pot_on_stove) + int(self.button_on)
        else:
            self.stage = 5 + int(pot_on_target) + int(not self.button_on)

        return int(self._check_success())

    def _check_success(self):
        # pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]
        # target_pos = self.sim.data.body_xpos[self.obj_body_id["target"]]
        # target_pot_xy_dist = np.linalg.norm(pot_pos[:2] - target_pos[:2])
        # pot_touched = int(self.check_contact(self.robots[0].gripper, self.objects_dict["pot"]))
        # pot_on_target = target_pot_xy_dist < 0.07 and not pot_touched
        return self.meatball_in_pot and self.meatball_cook_status == 1 and not self.button_on # and pot_on_target

    def _load_fixtures_in_arena(self, mujoco_arena):
        self.table_body = mujoco_arena.table_body
        wall_z = 0.075
        wall_type = "collision"     # all, collision
        self.wall_left = BoxObject(
            name="wall_left",
            size=[self.global_scale[0], 0.01, wall_z],
            density=100.,
            friction=0.0,
            obj_type=wall_type,
            joints=None,
        )
        self.wall_left.get_obj().set("pos", array_to_string((0, self.global_low[1] + 0.1, wall_z)))
        mujoco_arena.table_body.append(self.wall_left.get_obj())

        self.wall_right = BoxObject(
            name="wall_right",
            size=[self.global_scale[0], 0.01, wall_z],
            density=100.,
            friction=0.0,
            obj_type=wall_type,
            joints=None,
        )
        self.wall_right.get_obj().set("pos", array_to_string((0, self.global_high[1] - 0.1, wall_z)))
        mujoco_arena.table_body.append(self.wall_right.get_obj())

        self.wall_forward = BoxObject(
            name="wall_forward",
            size=[0.01, self.global_scale[1], wall_z],
            density=100.,
            friction=0.0,
            obj_type=wall_type,
            joints=None,
        )
        self.wall_forward.get_obj().set("pos", array_to_string((self.global_high[0] - 0.1, 0, wall_z)))
        mujoco_arena.table_body.append(self.wall_forward.get_obj())

        self.wall_back = BoxObject(
            name="wall_back",
            size=[0.01, self.global_scale[1], wall_z],
            density=100.,
            friction=0.0,
            obj_type=wall_type,
            joints=None,
        )
        self.wall_back.get_obj().set("pos", array_to_string((self.global_low[0] + 0.1, 0, wall_z)))
        mujoco_arena.table_body.append(self.wall_back.get_obj())

        self.fixtures_dict["button"] = ButtonObject(name="button")
        button_object = self.fixtures_dict["button"].get_obj()
        button_object.set("pos", array_to_string((0, 0, 0.003)))
        button_object.set("quat", array_to_string((0., 0., 0., 1.)))
        mujoco_arena.table_body.append(button_object)

        self.fixtures_dict["stove"] = StoveObject(name="stove")
        stove_object = self.fixtures_dict["stove"].get_obj()
        stove_object.set("pos", array_to_string((0, 0, 0.003)))
        mujoco_arena.table_body.append(stove_object)

        self.fixtures_dict["target"] = TargetObject(name="target")
        target_object = self.fixtures_dict["target"].get_obj()
        target_object.set("pos", array_to_string((0, 0, 0.003)))
        mujoco_arena.table_body.append(target_object)

    def _load_objects_in_arena(self, mujoco_arena):
        self.objects_dict["pot"] = PotObject(name="pot")

        butter_size = [0.015, 0.015, 0.015]
        self.objects_dict["butter"] = BoxObject(
            name="butter",
            size_min=butter_size,
            size_max=butter_size,
            rgba=[1, 0, 0, 1],
            material=self.custom_material_dict["lemon"],
            density=100.,
        )
        meatball_size = [0.02, 0.02, 0.02]
        self.objects_dict["meatball"] = BallObject(
            name="meatball",
            size_min=meatball_size,
            size_max=meatball_size,
            rgba=[1, 0, 0, 1],
            material=self.custom_material_dict["bread"],
            density=100.,
        )
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001),
        )
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        self._load_custom_material()

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena)

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())

        self.fixtures = list(self.fixtures_dict.values())

        for fixture in self.fixtures:
            if issubclass(type(fixture), CompositeObject):
                continue
            for material_name, material in self.custom_material_dict.items():
                tex_element, mat_element, _, used = add_material(root=fixture.worldbody,
                                                                 naming_prefix=fixture.naming_prefix,
                                                                 custom_material=deepcopy(material))
                fixture.asset.append(tex_element)
                fixture.asset.append(mat_element)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.objects,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict()

        for object_name, object_body in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(object_body.root_body)
        for fixture_name, fixture_body in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(fixture_body.root_body)

        self.button_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.fixtures_dict["button"].joints[0])
        self.pot_right_handle_id = self.sim.model.geom_name2id('pot_handle_right_0')
        self.button_switch_pad_id = self.sim.model.geom_name2id('button_switch_pad')

    def _setup_placement_initializer(self, mujoco_arena):
        """Function to define the placement"""
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-butter",
            mujoco_objects=self.objects_dict["butter"],
            x_range=self.butter_x_range,
            y_range=self.butter_y_range,
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-meatball",
            mujoco_objects=self.objects_dict["meatball"],
            x_range=self.meatball_x_range,
            y_range=self.meatball_y_range,
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler=UniformRandomSampler(
            name="ObjectSampler-pot",
            mujoco_objects=self.objects_dict["pot"],
            x_range=self.pot_x_range,
            y_range=self.pot_y_range,
            rotation=(-0.1, 0.1),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))

    def _load_custom_material(self):
        """
        Define all the textures
        """
        self.custom_material_dict = dict()

        tex_attrib = {
            "type": "cube"
        }

        self.custom_material_dict["bread"] = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        self.custom_material_dict["darkwood"] = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        self.custom_material_dict["lightwood"] = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )
        
        self.custom_material_dict["metal"] = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib=tex_attrib,
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }

        self.custom_material_dict["greenwood"] = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.custom_material_dict["redwood"] = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.custom_material_dict["bluewood"] = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="MatBlueWood",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["lemon"] = CustomMaterial(
            texture="Lemon",
            tex_name="lemon",
            mat_name="MatLemon",
            tex_attrib=tex_attrib,
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        self.custom_material_dict["steel"] = CustomMaterial(
            texture="SteelScratched",
            tex_name="steel_scratched_tex",
            mat_name="MatSteelScratched",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix

            modality = "object"
            @sensor(modality=modality)
            def robot0_eef_vel(obs_cache):
                return self.robots[0]._hand_vel

            @sensor(modality=modality)
            def butter_melt_status(obs_cache):
                return self.butter_melt_status

            @sensor(modality=modality)
            def butter_in_pot(obs_cache):
                return self.butter_in_pot

            @sensor(modality=modality)
            def meatball_overcooked(obs_cache):
                return self.meatball_overcooked

            @sensor(modality=modality)
            def meatball_cook_status(obs_cache):
                return self.meatball_cook_status

            @sensor(modality=modality)
            def meatball_in_pot(obs_cache):
                return self.meatball_in_pot

            @sensor(modality=modality)
            def button_joint_qpos(obs_cache):
                return self.sim.data.qpos[self.button_qpos_addrs]

            @sensor(modality="object")
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

            sensors = [robot0_eef_vel, world_pose_in_gripper, button_joint_qpos,
                       butter_melt_status, butter_in_pot,
                       meatball_cook_status, meatball_overcooked, meatball_in_pot]
            names = ["robot0_eef_vel", "world_pose_in_gripper", "button_joint_qpos",
                     "butter_melt_status", "butter_in_pot",
                     "meatball_cook_status", "meatball_overcooked", "meatball_in_pot"]

            for obj in self.objects + self.fixtures:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")
                sensors += obj_sensors
                names += obj_sensor_names

            for obj_name in ["pot_handle", "button_handle"]:
                obj_sensors, obj_sensor_names = self._create_geom_sensors(obj_name=obj_name, modality="object")

                sensors += obj_sensors
                names += obj_sensor_names

            for name, s in zip(names, sensors):
                if name == "world_pose_in_gripper":
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                        enabled=True,
                        active=False,
                    )
                else:
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq
                    )
    
        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        if obj_name in self.objects_dict:
            obj = self.objects_dict[obj_name]
        elif obj_name in self.fixtures_dict:
            obj = self.fixtures_dict[obj_name]
        else:
            raise NotImplementedError

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            pos = self.sim.data.body_xpos[self.obj_body_id[obj_name]]
            return pos

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_grasped(obs_cache):
            grasped = int(self._check_grasp(gripper=self.robots[0].gripper,
                                            object_geoms=[g for g in obj.contact_geoms]))
            return grasped

        @sensor(modality=modality)
        def object_touched(obs_cache):
            touched = int(self.check_contact(self.robots[0].gripper, obj))
            return touched

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        if not obj_name in ["stove", "target"]:
            sensors += [obj_grasped, object_touched]
            names += [f"{obj_name}_grasped", f"{obj_name}_touched"]

        return sensors, names

    def _create_geom_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        if obj_name == "pot_handle":
            geom_id = self.pot_right_handle_id
        elif obj_name == "button_handle":
            geom_id = self.button_switch_pad_id
        else:
            raise NotImplementedError

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return self.sim.data.geom_xpos[geom_id]

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], np.array([1.0, 0.0, 0.0, 0.0])))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, _ = T.mat2pose(rel_pose)
            return rel_pos

        sensors = [obj_pos, obj_to_eef_pos]
        names = [f"{obj_name}_pos", f"{obj_name}_to_{pf}eef_pos"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # fixtures reset
        for obj_name, obj_x_range, obj_y_range in [["button", self.button_x_range, self.button_y_range],
                                                   ["stove", self.stove_x_range, self.stove_y_range],
                                                   ["target", self.target_x_range, self.target_y_range]]:
            obj = self.fixtures_dict[obj_name]
            body_id = self.sim.model.body_name2id(obj.root_body)
            obj_x = np.random.uniform(obj_x_range[0], obj_x_range[1])
            obj_y = np.random.uniform(obj_y_range[0], obj_y_range[1])
            obj_z = 0.005 if obj_name == "target" else 0.02
            self.sim.model.body_pos[body_id] = (obj_x, obj_y, obj_z)
            if obj_name == "button":
                self.sim.model.body_quat[body_id] = (0., 0., 0., 1.)

        # button and meatball state reset
        self.sim.data.set_joint_qpos(self.fixtures_dict["button"].joints[0], np.array([-0.3]))
        self.butter_melt_status = np.random.uniform(-1, -0.5)
        self.meatball_cook_status = np.random.uniform(-1, -0.5)

        self.meatball_overcooked = False

        self.button_on = False
        self.butter_in_pot = False
        self.meatball_in_pot = False
        self.pot_on_stove = False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def normalize_obs(self, obs, out_of_range_warning=True):
        for k, v in obs.items():
            normalize = True
            mean, scale = 0, 1
            if k.endswith("pos") and not k.endswith("qpos"):
                scale = self.global_scale
                if k.endswith("to_robot0_eef_pos"):
                    scale = 2 * self.global_scale
                else:
                    mean = self.global_mean
            elif k == "robot0_eef_vel":
                scale = self.eef_vel_scale
            elif k == "robot0_gripper_qpos":
                scale = self.gripper_qpos_scale
            elif k == "robot0_gripper_qvel":
                scale = self.gripper_qvel_scale
            elif k == "button_joint_qpos":
                scale = self.button_joint_qpos_scale
            else:
                normalize = False

            if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
                obs[k] = v = np.array([v])

            if normalize:
                butter_melted = self.butter_melt_status == 1
                butter_ok = "butter" in k and butter_melted
                # if out_of_range_warning and not k.endswith("to_robot0_eef_pos") and ((v < mean - scale) | (v > mean + scale)).any():
                #     if not butter_ok:
                #         print(k, "out of range")
                #         print("value", v)
                #         print("range", mean - scale, mean + scale)

                obs[k] = (v - mean) / scale

        return obs

    def reset(self):
        obs = super().reset()
        obs = self.normalize_obs(obs)
        # obs["step_count"] = np.array([0.])
        return obs

    def observation_spec(self):
        obs_spec = super().observation_spec()
        obs_spec = self.normalize_obs(obs_spec, out_of_range_warning=False)
        # obs_spec["step_count"] = np.array([0])
        return obs_spec

    def step(self, action):
        # pre-process action
        assert action.shape == (4,)
        global_act_low, global_act_high = self.global_low + 0.1, self.global_high - 0.1
        global_act_low[2] = self.table_offset[2] + 0.01                                     # table height
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        controller_scale = 0.05
        action[:3] = np.clip(action[:3],
                             (global_act_low - eef_pos) / controller_scale,
                             (global_act_high - eef_pos) / controller_scale)
        action = np.clip(action, -1, 1)

        next_obs, reward, done, info = super().step(action)
        next_obs = self.normalize_obs(next_obs)
        # next_obs["step_count"] = np.array([float(self.timestep) / self.horizon])

        info["success"] = self._check_success()
        info["stage_completion"] = self.stage
        return next_obs, reward, done, info

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self._post_process()

        return reward, done, info

    def _post_process(self):
        self.butter_in_pot = self.check_contact(self.objects_dict["butter"], "pot_body_bottom")
        self.meatball_in_pot = self.check_contact(self.objects_dict["meatball"], "pot_body_bottom")
        self.pot_on_stove = self.check_contact("stove_collision_burner", "pot_body_bottom")

        if self.sim.data.qpos[self.button_qpos_addrs] < 0.0:
            self.button_on = False
        else:
            self.button_on = True

        self.fixtures_dict["stove"].set_sites_visibility(sim=self.sim, visible=self.button_on)

    def obs_delta_range(self):
        max_delta_eef_pos = 0.1 * np.ones(3) / (2 * self.global_scale)
        max_delta_vel = 2 * np.ones(3) / (2 * self.eef_vel_scale)
        max_delta_gripper_qpos = 0.02 * np.ones(2) / (2 * self.gripper_qpos_scale)
        max_delta_gripper_qvel = 0.5 * np.ones(2) / (2 * self.gripper_qvel_scale)
        max_delta_obj_pos = 0.1 * np.ones(3) / (2 * self.global_scale)
        max_delta_obj_quat = 2 * np.ones(4)

        obs_delta_range = {"robot0_eef_pos": [-max_delta_eef_pos, max_delta_eef_pos],
                           "robot0_eef_vel": [-max_delta_vel, max_delta_vel],
                           "robot0_gripper_qpos": [-max_delta_gripper_qpos, max_delta_gripper_qpos],
                           "robot0_gripper_qvel": [-max_delta_gripper_qvel, max_delta_gripper_qvel],
                           "butter_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "butter_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "butter_melt_status": [np.array([-1]), np.array([1])],
                           "meatball_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "meatball_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "meatball_cook_status": [np.array([-1]), np.array([1])],
                           "meatball_overcooked": [np.array([0]), np.array([1])],
                           "stove_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "stove_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "pot_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "pot_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "target_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "target_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "button_pos": [-max_delta_obj_pos, max_delta_obj_pos],
                           "button_quat": [-max_delta_obj_quat, max_delta_obj_quat],
                           "button_joint_qpos": [np.array([-self.button_joint_qpos_scale]), np.array([self.button_joint_qpos_scale])],}
        return obs_delta_range

    def render(self, mode="rgb_array"):
        frame = self.sim.render(width=640, height=480, camera_name="frontview")
        frame = np.flip(frame, axis=0)
        return frame
