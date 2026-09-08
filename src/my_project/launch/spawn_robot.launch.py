import os
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

import xacro


def launch_setup(context, *args, **kwargs):
    pkg_path = get_package_share_directory('my_project')
    robot_file = os.path.join(pkg_path, 'urdf', 'robot.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- Expand xacro to a plain URDF string (done once, at launch time) ---
    robot_description_raw = xacro.process_file(robot_file).toxml()

    # Write expanded URDF to a temp file so `create -file` can read it directly.
    # This sidesteps the robot_state_publisher latched-topic QoS race that was
    # causing `ros_gz_sim create -topic robot_description` to hang forever.
    tmp_urdf = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', delete=False
    )
    tmp_urdf.write(robot_description_raw)
    tmp_urdf.close()

    robot_description = ParameterValue(robot_description_raw, value_type=str)

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    jsp_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Spawn from the expanded file instead of waiting on the robot_description
    # topic. This is the key fix — `-topic` requires create's subscriber QoS
    # to line up with RSP's transient-local publisher, which doesn't always
    # happen in Jazzy/Harmonic and causes the indefinite "Waiting messages on
    # topic [robot_description]" hang you saw.
    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-file', tmp_urdf.name,
            '-x', '-5',
            '-y', '-3',
            '-z', '0.1'
        ],
        output='screen'
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return [rsp_node, jsp_node, spawn_node, bridge_node]


def generate_launch_description():
    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    return LaunchDescription([
        declare_sim_time,
        OpaqueFunction(function=launch_setup),
    ])
