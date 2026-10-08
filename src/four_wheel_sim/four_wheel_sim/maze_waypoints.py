import math
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator
from four_wheel_sim.worlds import GOAL


def pose(nav, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = 'map' #change back to odom if needed. Propbably will not be needed due to the use of the map frame in the navigation stack.
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z = math.sin(yaw / 2)
    p.pose.orientation.w = math.cos(yaw / 2)
    return p


def main():
    rclpy.init()
    nav = BasicNavigator()
    nav.waitUntilNav2Active(localizer='bt_navigator')
    nav.goToPose(pose(nav, GOAL[0], GOAL[1], 0.0))
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            print(f'Distance remaining: {fb.distance_remaining:.2f} m')
    print('Result:', nav.getResult())
    nav.lifecycleShutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()