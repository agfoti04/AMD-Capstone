import depthai as dai
import numpy as np
import open3d as o3d
from depth_config import configureDepthPostProcessing

class BasicPointCloudMapper:
    def __init__(self):
        self.global_map = o3d.geometry.PointCloud()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            "Basic Point Cloud Map",
            width=960,
            height=540
        )

        self.map_added = False

    def add_point_cloud(self, pcl_data):
        # Convert DepthAI points to Open3D point cloud
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(pcl_data)

        # Correctly remove invalid points
        new_cloud, _ = new_cloud.remove_non_finite_points()

        if len(new_cloud.points) == 0:
            return

        # Filter points by distance from camera
        points = np.asarray(new_cloud.points)
        distances = np.linalg.norm(points, axis=1)

        valid = (distances > 0.7) & (distances < 4.0)
        new_cloud = new_cloud.select_by_index(
            np.where(valid)[0]
        )

        if len(new_cloud.points) == 0:
            return

        # Reduce point density
        new_cloud = new_cloud.voxel_down_sample(
            voxel_size=0.03
        )

        # Add this frame to the global map
        self.global_map += new_cloud

        # Downsample the accumulated map
        self.global_map = self.global_map.voxel_down_sample(
            voxel_size=0.03
        )

        # Update visualization
        if not self.map_added:
            self.vis.add_geometry(self.global_map)
            self.map_added = True
        else:
            self.vis.update_geometry(self.global_map)

        self.vis.poll_events()
        self.vis.update_renderer()

    def save_map(self, filename="basic_map.ply"):
        o3d.io.write_point_cloud(
            filename,
            self.global_map
        )

        print(f"Saved map to {filename}")

    def close(self):
        self.vis.destroy_window()


# --------------------------------------------------
# DepthAI pipeline
# --------------------------------------------------

COLOR = True
# Depth resolution
resolution = (640, 400)

pipeline = dai.Pipeline()


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------

left = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_B
)

right = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_C
)


# --------------------------------------------------------------------------
# Stereo depth
# --------------------------------------------------------------------------

stereo = pipeline.create(dai.node.StereoDepth)

configureDepthPostProcessing(stereo)


# Request 640x400 mono frames
leftOutput = left.requestOutput(
    resolution,
    type=dai.ImgFrame.Type.GRAY8
)

rightOutput = right.requestOutput(
    resolution,
    type=dai.ImgFrame.Type.GRAY8
)

leftOutput.link(stereo.left)
rightOutput.link(stereo.right)


# --------------------------------------------------------------------------
# Point cloud
# --------------------------------------------------------------------------

pointCloud = pipeline.create(dai.node.PointCloud)

# Output coordinates in meters
pointCloud.initialConfig.setLengthUnit(
    dai.LengthUnit.METER
)

# Use the stereo depth as the point cloud source
stereo.depth.link(pointCloud.inputDepth)


# Create point-cloud output queue
pclQueue = pointCloud.outputPointCloud.createOutputQueue(
    maxSize=4,
    blocking=False
)


# --------------------------------------------------------------------------
# RGB camera
# --------------------------------------------------------------------------

if COLOR:

    color = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )

    colorOutput = color.requestOutput(
        resolution,
        type=dai.ImgFrame.Type.BGR888p
    )

    rgbQueue = colorOutput.createOutputQueue(
        maxSize=1,
        blocking=False
    )


# --------------------------------------------------
# Run mapper
# --------------------------------------------------

mapper = BasicPointCloudMapper()

with pipeline:

    pipeline.start()

    print("Mapping started.")
    print("Keep the camera still or move it slowly.")
    print("Press Ctrl+C to stop and save the map.")

    try:
        while True:
            pcl_packet = pclQueue.get()

            pcl_data = np.asarray(pcl_packet.getPoints())

            mapper.add_point_cloud(pcl_data)

    except KeyboardInterrupt:
        print("\nStopping mapper...")

    finally:
        mapper.save_map("basic_map.ply")
        mapper.close()