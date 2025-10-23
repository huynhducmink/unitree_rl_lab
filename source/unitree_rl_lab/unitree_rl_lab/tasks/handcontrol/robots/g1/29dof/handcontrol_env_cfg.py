import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg 
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, CameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG
# from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.handcontrol import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=3.0,
    # num_rows=9,
    # num_cols=21,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # terrain = sim_utils.GroundPlaneCfg(  # flat plane instead of generator
    #     prim_path="/World/ground",
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/Basic/Wood.mdl",
    #         project_uvw=True,
    #         texture_scale=(1.0, 1.0),
    #     ),
    # )
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contacts for sparse success detection
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)

    # A simple table (cube) and a red ball (sphere)
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.73),  # L, W, H (m)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,     # fixed table
                disable_gravity=False,
            ),
            # Give it a collider + mass so PhysX is happy
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                # either mass or density; pick one. Kinematic still needs a valid value
                mass=5.0,
                # density=None
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(
                # diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.375),  # z = H/2
        ),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        collision_group=-1,
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,    # dynamic
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                # choose mass or density; here density for a small ball
                density=500.0,  # kg/m^3 (plastic-ish)
                # mass=None
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.2
            ),
            visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.82),
        ),
    )

    # Head camera pitched down 45 degrees, 64x64 RGB
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
        # parent_prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        # position=(0.15, 0.0, 0.20),   # forward & up from torso
        # orientation=(math.radians(-45.0), 0.0, 0.0),  # pitch down
        spawn=sim_utils.PinholeCameraCfg(
            # focal_length=None,                   # let default derive from FOV
            # horizontal_aperture=None,
            clipping_range=(0.01, 5.0),
            # f_stop=None,
            # focus_distance=None,
            visible=True,
        ),
        offset = CameraCfg.OffsetCfg(
            pos = (0.15, 0.0, 0.20),   # forward & up from torso
            rot =(0.9238795, 0.3826834, 0.0, 0.0),  # pitch down
            convention = "ros",
        ),
        width=64, height=64,
        # fov=60.0,  # there is no FOV parameter?
        data_types=["rgb"], #https://isaac-sim.github.io/IsaacLab/release/2.2.0/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.Camera
        history_length=0,
        update_period=0.1,
        debug_vis = True,
    )

    # Neutral skylight
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )

# ---------- Events ----------

@configclass
class EventCfg:
    # keep material & mass randomization to generalize
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.0), "dynamic_friction_range": (0.6, 1.0),
                "restitution_range": (0.0, 0.1), "num_buckets": 32}
    )

    # Reset robot at origin facing +X; small random yaw to avoid overfit
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.2, 0.2)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}}
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.5, 0.5)}
    )

    # Randomize ball XY within a corridor in front; keep Z near table top
    respawn_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table", body_names="Table"),
            "pose_range": {"x": (0.25, 0.25), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}}
    )
    respawn_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names="Ball"),
            "pose_range": {"x": (0.3, 0.5), "y": (-0.3, 0.3), "yaw": (0.0, 0.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}}
    )

# ---------- Commands (no velocity tracking; keep a Null command) ----------

@configclass
class CommandsCfg:
    stand = mdp.UniformLevelVelocityCommandCfg(  # reuse cfg class but keep near-zero
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )

# ---------- Actions (unchanged; joint position control) ----------

@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

# ---------- Observations ----------

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprio
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.0, n_max=1.0))
        last_action = ObsTerm(func=mdp.last_action)
        # Pixels (RGB 64x64x3)
        rgb_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera")},
            clip=(0.0, 1.0),
        )

        def __post_init__(self):
            self.history_length = 4   # small frame stack helps motion cues
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Keep critic vector-only (sample-efficient)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # Pixels (RGB 64x64x3)
        rgb_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera")},
            clip=(0.0, 1.0),
        )
        def __post_init__(self):
            self.history_length = 4

    critic: CriticCfg = CriticCfg()

# ---------- Rewards: reach & touch ----------

@configclass
class RewardsCfg:
    # Dense shaping: end-effector (right wrist/hand) to ball distance (exp kernel)
    reach_ball = RewTerm(
        func=mdp.ee_to_target_distance_exp,
        weight=2.0,
        params={"ee_body_name": "right_wrist_roll_link", "target_asset": "ball", "std": 0.10},
    )

    # Success: contact between any right hand link and the ball
    touch_ball = RewTerm(
        func=mdp.touch_target_sparse,
        weight=5.0,
        params={"ee_regex": "right_.*(wrist|hand).*", "target_asset": "ball", "sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-3.0, params={"target_height": 0.78})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)

# ---------- Terminations ----------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.9})

# ---------- Curriculum: ball distance / table height ----------

@configclass
class CurriculumCfg:
    reach_levels = CurrTerm(func=mdp.reach_cmd_levels)
    table_levels = CurrTerm(func=mdp.table_height_levels)

# ---------- Env ----------

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=2048, env_spacing=2.5)  # fewer envs to offset pixels
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.head_camera.update_period = self.decimation * self.sim.dt

@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
