from nav_msgs.msg import Odometry

from four_wheel_sim.odom_distance import valid_position


def test_valid_position_returns_position_for_valid_odom():
    msg = Odometry()
    msg.header.frame_id = 'odom'
    msg.pose.pose.orientation.w = 1.0
    msg.pose.pose.position.x = 1.25
    msg.pose.pose.position.y = -0.5
    msg.pose.pose.position.z = 0.2

    assert valid_position(msg) == (1.25, -0.5, 0.2)


def test_valid_position_rejects_null_pose_and_missing_frame():
    msg = Odometry()
    msg.header.frame_id = 'odom'

    assert valid_position(msg) is None

    msg.pose.pose.orientation.w = 1.0
    msg.header.frame_id = ''
    assert valid_position(msg) is None
