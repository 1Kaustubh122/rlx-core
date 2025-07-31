# from isaacsim import SimulationApp

# simulation_app = SimulationApp({"headless": False})

# import sys
# from isaacsim.storage.native import get_assets_root_path
# from isaacsim.core.api import World
# from isaacsim.core.utils.viewport import set_camera_view

# asset_root_path = get_assets_root_path()

# world = World(stage_units_in_meters=1.0)
# world.scene.add_default_ground_plane()

# set_camera_view(
#     eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
# )

# world.reset()


# # while True:
# #     pass




from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
# simulation_app = SimulationApp({"headless": True})


import sys

import carb # pyright: ignore[reportMissingImports]
import numpy as np
from isaacsim.core.api import World 
import isaacsim.replicator as rep
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view   
from isaacsim.storage.native import get_assets_root_path  
from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper
from pxr import Usd, UsdPhysics
import omni.usd
import numpy as np
import time

assets_root_path = get_assets_root_path()
stage = omni.usd.get_context().get_stage()

if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()  
set_camera_view(
    eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
)  

franka_usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
franka_prim_path = "/World/franka"

table_usd_path = assets_root_path + "/Isaac/Props/Mounts/ThorlabsTable/table_instanceable.usd"
table_prim_path = "/World/table_instanceable/visuals"

cylinder_usd_path = assets_root_path + "/Isaac/Props/Shapes/cylinder.usd"
cylinder_prim_path = "/World/cylinder"

base_usd_path = assets_root_path + "/Isaac/IsaacLab/Robots/UniversalRobots/UR10/Props/ur10_wrist_3.usd"
base_prim_path = "/World/ur10_wrist_3" 

add_reference_to_stage(usd_path=franka_usd_path, prim_path=franka_prim_path) 
franka_arm = Articulation(prim_paths_expr=franka_prim_path, name="franka")  

add_reference_to_stage(usd_path=table_usd_path, prim_path=table_prim_path)  
table = RigidPrim(prim_paths_expr=table_prim_path, name="table")

add_reference_to_stage(usd_path=base_usd_path, prim_path=base_prim_path)
base = RigidPrim(prim_paths_expr=base_prim_path, name="base")

add_reference_to_stage(usd_path=cylinder_usd_path, prim_path=cylinder_prim_path)
cylinder = RigidPrim(prim_paths_expr=cylinder_prim_path, name="cylinder")

base.enable_rigid_body_physics()
base_prim = stage.GetPrimAtPath(base_prim_path)    
UsdPhysics.CollisionAPI.Apply(base_prim)

# cylinder.enable_rigid_body_physics()
cylinder_prim = stage.GetPrimAtPath(cylinder_prim_path)   

mat_api = UsdPhysics.MaterialAPI.Apply(cylinder_prim)
mat_api.CreateStaticFrictionAttr(2.5)
mat_api.CreateDynamicFrictionAttr(2.0)
mat_api.CreateRestitutionAttr(0.0) 
UsdPhysics.CollisionAPI.Apply(cylinder_prim)


gripper = SurfaceGripper(
    end_effector_prim_path="/World/franka/panda_hand",  
    translate=0.031, 
    direction="x", 
    grip_threshold=0.003,
    force_limit=1e6,   
    torque_limit=1e4,
    bend_angle=np.pi/24,
    kp=1e2,
    kd=1e2,
    disable_gravity=True,
)



left_finger_joint_path = "/World/franka/panda_hand/panda_finger_joint1"
right_finger_joint_path = "/World/franka/panda_hand/panda_finger_joint2"


left_finger_joint_prim = stage.GetPrimAtPath(left_finger_joint_path)
right_finger_joint_prim = stage.GetPrimAtPath(right_finger_joint_path)

left_joint_api = UsdPhysics.PrismaticJoint(left_finger_joint_prim)
right_joint_api = UsdPhysics.PrismaticJoint(right_finger_joint_prim)

def set_gripper_position(pos):
    pos = max(0.0, min(0.04, pos))
    left_joint_api.CreateDriveTargetPositionAttr(pos)
    right_joint_api.CreateDriveTargetPositionAttr(pos)

# set_gripper_position(0.0)


franka_arm.set_world_poses(positions=np.array([[0.0, 0.0, 0.0]]) / get_stage_units())
# print(franka_arm.dof_names)

# simulation_app.close()

base.set_local_scales(scales=np.array([[1.0, 3.0, 1.0]]))
base.set_world_poses(positions=np.array([[0.65, 0.25, 0.45373]]) / get_stage_units(), orientations=np.array([[0.70711, 0.70711, 0.0, 0.0]]))

table.set_local_scales(scales=np.array([[0.6, 1.0, 0.5]]))
table.set_world_poses(positions=np.array([[0.69757, 0.0, 0.37025]]) / get_stage_units())

cylinder.set_local_scales(scales=np.array([[0.03, 0.03, 0.05]]))
cylinder.set_world_poses(positions=np.array([[0.65633, 0.0, 0.3875]]) / get_stage_units())

my_world.reset()

def print_robot_info(articulation):
    print("Robot name:", articulation.name)
    print("DOF count:", articulation.num_dof)
    print("Joint Names:", articulation.dof_names)
    print("Joint Limits:", articulation.get_dof_limits())
    print("Joint Types:", articulation.get_dof_types())
    print("Body Names:", articulation.body_names)
    print("Prim Paths:", articulation.prim_paths)
    

def read_joints(filename, num_joints=9):
    try:
        with open(filename, "r") as f:
            vals = [float(x) for x in f.read().strip().split()]
            if len(vals) == num_joints:
                return np.array([vals])
    except Exception as e:
        print("File read error:", e)
    return None
gripper.initialize(articulation_num_dofs=franka_arm.num_dof)

print_robot_info(franka_arm)
i = 0
while True:
    i += 1 
    if i % 10 == 0:  
        joints = read_joints("envs/joints.txt", 9)

        if joints is not None:
            franka_arm.set_joint_positions(joints)
            if (joints[-1][-1]) < 0.04:
                gripper.close()
            else:
                gripper.open()
            
            gripper.update()

                
    my_world.step(render=True)
    time.sleep(0.01)
    
    # joint_positions = franka_arm.get_joint_positions()
       
       
       
       
       
       