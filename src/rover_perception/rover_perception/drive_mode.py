"""Control ownership shared by launch files and the live camera bridge."""

GAZEBO_DRIVE_TOPIC = '/rover/live/gazebo_cmd_vel'
GAZEBO_COMMAND_TIMEOUT = 1.0


def control_mode(value):
    if value not in ('browser', 'gazebo'):
        raise ValueError('control_mode must be browser or gazebo')
    return value


def gazebo_bridge_argument():
    # '[' selects Gazebo -> ROS only; commands never echo back into Gazebo.
    return GAZEBO_DRIVE_TOPIC + '@geometry_msgs/msg/Twist[gz.msgs.Twist'
