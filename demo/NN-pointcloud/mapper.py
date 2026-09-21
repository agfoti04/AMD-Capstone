import numpy as np
import open3d as o3d
import depthai as dai

from depth_config import configure_depth_post_processing


COLOR = True
RESOLUTION = (640, 400)
MAP_VOXEL_SIZE = 0.04
MIN_DISTANCE = 0.0
MAX_DISTANCE = 2.0


class BasicPointCloudMapper:
    def __init__(self) -> None:
        self.global_map = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Basic Point Cloud Map", width=960, height=540)
        self.map_added = False

    def add_point_cloud(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return

        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(points)
        new_cloud, _ = new_cloud.remove_non_finite_points()

        valid_points = np.asarray(new_cloud.points)
        if len(valid_points) == 0:
            return

        distances = np.linalg.norm(valid_points, axis=1)
        valid = (distances >= MIN_DISTANCE) & (distances <= MAX_DISTANCE)
        new_cloud = new_cloud.select_by_index(np.flatnonzero(valid))
        if len(new_cloud.points) == 0:
            return

        new_cloud = new_cloud.voxel_down_sample(MAP_VOXEL_SIZE)
        self.global_map += new_cloud
        downsampled_map = self.global_map.voxel_down_sample(MAP_VOXEL_SIZE)
        self.global_map.points = downsampled_map.points

        if not self.map_added:
            self.vis.add_geometry(self.global_map)
            self.map_added = True
        else:
            self.vis.update_geometry(self.global_map)

        self.vis.poll_events()
        self.vis.update_renderer()

    def save_map(self, filename: str = "basic_map.ply") -> None:
        if not o3d.io.write_point_cloud(filename, self.global_map):
            raise RuntimeError(f"Failed to save point cloud map to {filename}")
        print(f"Saved map to {filename}")

    def close(self) -> None:
        self.vis.destroy_window()


def create_pipeline() -> tuple[dai.Pipeline, dai.DataOutputQueue]:
    pipeline = dai.Pipeline()

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    configure_depth_post_processing(stereo)

    left_output = left.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    right_output = right.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    left_output.link(stereo.left)
    right_output.link(stereo.right)

    point_cloud = pipeline.create(dai.node.PointCloud)
    point_cloud.initialConfig.setLengthUnit(dai.LengthUnit.METER)
    stereo.depth.link(point_cloud.inputDepth)
    pcl_queue = point_cloud.outputPointCloud.createOutputQueue(
        maxSize=4,
        blocking=False,
    )

    if COLOR:
        color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_output = color.requestOutput(
            RESOLUTION,
            type=dai.ImgFrame.Type.BGR888p,
        )
        color_output.createOutputQueue(maxSize=1, blocking=False)

    return pipeline, pcl_queue


def main() -> None:
    pipeline, pcl_queue = create_pipeline()
    mapper = BasicPointCloudMapper()

    with pipeline:
        pipeline.start()
        print("Mapping started. Move the camera slowly.")
        print("Press Ctrl+C to stop and save the map.")

        try:
            while pipeline.isRunning():
                packet = pcl_queue.get()
                mapper.add_point_cloud(packet.getPoints())
        except KeyboardInterrupt:
            print("\nStopping mapper...")
        finally:
            mapper.save_map()
            mapper.close()


if __name__ == "__main__":
    main()
