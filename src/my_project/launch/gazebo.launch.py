import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share = get_package_share_directory("my_project")

    world = os.path.join(
        pkg_share,
        "worlds",
        "dynamic_corridor.world"
    )

    # Make models, worlds and meshes discoverable
    resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=pkg_share,
    )

    # NOTE: we intentionally do NOT set a custom GZ_PARTITION here.
    # spawn_robot.launch.py runs later in a separate terminal/process tree
    # and needs to discover this same gz sim server over Gazebo Transport
    # on the *default* partition. A per-launch random GZ_PARTITION isolates
    # this server from any other terminal, breaking spawn_robot.launch.py
    # (it would loop forever on "Requesting list of world names" because
    # it can't find a server on the default partition).
    #
    # The original problem this was meant to fix -- a stale leftover
    # "gz sim server" process colliding with a new one and causing a
    # duplicate-entity segfault -- should instead be handled by killing
    # any stale process before relaunching:
    #
    #   ps aux | grep "gz sim" | grep -v grep
    #   pkill -9 -f "gz sim server"
    #
    # and/or restarting the ROS 2 daemon if discovery seems stuck:
    #
    #   ros2 daemon stop && ros2 daemon start

    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"-r {world}",
        }.items(),
    )

    # FIX: ros_gz_bridge does NOT expose any Gazebo world services to ROS
    # by default -- not even SetEntityPose. Without this bridge,
    # train_agent.py's reset_client.service_is_ready() always returns
    # False, so reset_simulation() silently skips teleporting the robot
    # back to spawn and every episode after the first starts from
    # wherever the robot happened to stop, instead of a fixed start
    # state. That makes the PPO reward signal inconsistent across
    # episodes and will stall or destabilize training.
    #
    # IMPORTANT: the world name in the path below ("default") must match
    # the <world name="..."> attribute inside dynamic_corridor.world, NOT
    # the filename. Verify with:
    #   gz service -l | grep set_pose
    # and update the argument below if your world's <world name="..."> is
    # something other than "default".
    set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="set_pose_bridge",
        arguments=[
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        output="screen",
    )

    return LaunchDescription([
        resource_path,
        gz_launch,
        set_pose_bridge,
    ])