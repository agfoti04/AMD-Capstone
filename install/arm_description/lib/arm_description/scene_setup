#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

FRAME = 'base_link'   # coincides with world for your arm


def make_object(oid, prim_type, dims, x, y, z):
    obj = CollisionObject()
    obj.header.frame_id = FRAME
    obj.id = oid
    prim = SolidPrimitive(type=prim_type, dimensions=dims)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.w = 1.0
    obj.primitives = [prim]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


def main():
    rclpy.init()
    node = Node('scene_setup')
    for name, default in (('rod_x', 0.25), ('rod_y', 0.0), ('rod_height', 0.12),
                          ('rod_radius', 0.01), ('margin', 0.01), ('remove_rod', False)):
        node.declare_parameter(name, default)
    p = lambda k: node.get_parameter(k).value

    client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error('/apply_planning_scene not available. Is move_group running?')
        return

    scene = PlanningScene()
    scene.is_diff = True

    if p('remove_rod'):
        rod = CollisionObject()
        rod.header.frame_id = FRAME
        rod.id = 'rod'
        rod.operation = CollisionObject.REMOVE
        scene.world.collision_objects = [rod]
    else:
        # Floor top sits 2 mm below z=0 so the base mesh doesn't start "in collision"
        floor = make_object('floor', SolidPrimitive.BOX, [2.0, 2.0, 0.05], 0.0, 0.0, -0.027)
        # Rod is inflated by `margin` so the jaws and wrist clear it
        h = p('rod_height') + p('margin')
        r = p('rod_radius') + p('margin')
        rod = make_object('rod', SolidPrimitive.CYLINDER, [h, r],
                          p('rod_x'), p('rod_y'), h / 2.0)
        scene.world.collision_objects = [floor, rod]

    req = ApplyPlanningScene.Request(scene=scene)
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    ok = fut.result() is not None and fut.result().success
    node.get_logger().info('Planning scene updated' if ok else 'Planning scene update FAILED')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()