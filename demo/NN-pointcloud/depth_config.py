import cv2
import depthai as dai
import numpy as np


lrcheck = True
extended = False
subpixel = True

def configureDepthPostProcessing(stereoDepthNode):
    """
    Configure StereoDepth post-processing.
    """

    stereoDepthNode.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.FAST_DENSITY
    )

    config = stereoDepthNode.initialConfig

    config.postProcessing.decimationFilter.decimationFactor = 1

    # Speckle filtering
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 28

    # Temporal filtering
    config.postProcessing.temporalFilter.enable = True

    # Spatial filtering
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1

    # Depth range
    config.postProcessing.thresholdFilter.minRange = 100
    config.postProcessing.thresholdFilter.maxRange = 3000

    # Census transform
    config.censusTransform.enableMeanMode = True

    # Cost matching
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2


    # Stereo configuration
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)

    # Black fill for invalid rectified pixels
    stereoDepthNode.setRectifyEdgeFillColor(0)