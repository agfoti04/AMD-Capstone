import numpy as np
import open3d as o3d
from functools import partial

class PointCloudVisualizer():
    def __init__(self):
        self.pcl = None
        # transform from camera to world orientation (Note the absolute position won't be correct)
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("[DepthAI] Open3D integration demo", 960, 540)
        self.isstarted = False

    def visualize_pcl(self, pcl_data, downsample=True):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_data)

        # 1. Remove NaN and infinite points
        pcd.remove_non_finite_points()

        # 2. Remove points outside useful depth range
        points = np.asarray(pcd.points)

        if len(points) == 0:
            return

        # Keep points between 0.7m and 4.0m
        distances = np.linalg.norm(points, axis=1)

        mask = (distances >= 0.0) & (distances <= 2.0)

        pcd = pcd.select_by_index(
            np.where(mask)[0]
        )

        # 3. Downsample
        if downsample:
            pcd = pcd.voxel_down_sample(
                voxel_size=0.04
            )

        # 4. Remove statistical outliers
        if len(pcd.points) > 30:

            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=1.0
            )

        # 5. Update stored point cloud
        if self.pcl is None:
            self.pcl = pcd
        else:
            self.pcl.points = pcd.points

        # 6. Rotate camera coordinates for visualization
        self.pcl.rotate(
            self.R_camera_to_world,
            center=np.array(
                [0, 0, 0],
                dtype=np.float64
            )
        )

        # 7. Display
        if not self.isstarted:

            self.vis.add_geometry(self.pcl)

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1,
                origin=[0, 0, 0]
            )

            self.vis.add_geometry(origin)

            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.5)

            self.isstarted = True

        else:

            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()