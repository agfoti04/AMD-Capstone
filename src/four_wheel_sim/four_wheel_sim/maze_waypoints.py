import math
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator


def pose(nav, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = 'odom'
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z = math.sin(yaw / 2)
    p.pose.orientation.w = math.cos(yaw / 2)
    return p


def main():
    rclpy.init()
    nav = BasicNavigator()
    nav.waitUntilNav2Active(localizer='bt_navigator')
    h = math.pi / 2
    route = [
        pose(nav, 1.0, 0.0, 0.0),
        pose(nav, 1.7, 0.0, 0.0),
        pose(nav, 2.0, 0.3, h),
        pose(nav, 2.0, 1.0, h),
        pose(nav, 2.0, 1.7, h),
        pose(nav, 2.3, 2.0, 0.0),
        pose(nav, 3.0, 2.0, 0.0),
        pose(nav, 4.0, 2.0, 0.0),
    ]
    nav.followWaypoints(route)
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            print(f'Waypoint {fb.current_waypoint + 1}/{len(route)}')
    print('Result:', nav.getResult())
    nav.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()