#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from depth_config import configure_depth_post_processing

try:
    from projector_device import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occurred when importing PCL projector: {e}"
    )

############################################################################
# USER CONFIGURABLE PARAMETERS

COLOR = True

# Depth resolution
resolution = (640, 400)

# Downsample point cloud before visualization
downsample_pcl = True

# StereoDepth configuration
############################################################################
# CREATE PIPELINE

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

configure_depth_post_processing(stereo)


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


############################################################################
# MAIN

if __name__ == "__main__":

    print("Opening device...")

    # DepthAI v3 pipeline
    with pipeline:

        pipeline.start()

        print("Pipeline started!")

        pcl_converter = PointCloudVisualizer()

        while pipeline.isRunning():

            # --------------------------------------------------------------
            # Color image
            # --------------------------------------------------------------

            if COLOR and rgbQueue.has():

                frame = rgbQueue.get().getCvFrame()

                cv2.imshow(
                    "color",
                    frame
                )


            # --------------------------------------------------------------
            # Point cloud
            # --------------------------------------------------------------

            if pclQueue.has():

                pclData = pclQueue.get()

                # DepthAI v3 returns an Nx3 float32 NumPy array
                points = pclData.getPoints()

                # Make sure we have points
                if points is not None and len(points) > 0:

                    pcl_data = np.asarray(
                        points,
                        dtype=np.float64
                    )

                    # Your visualizer expects meters,
                    # and PointCloud is configured for meters.
                    pcl_converter.visualize_pcl(
                        pcl_data,
                        downsample=downsample_pcl
                    )


            # --------------------------------------------------------------
            # Exit
            # --------------------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                pcl_converter.close_window()

                break

        cv2.destroyAllWindows()