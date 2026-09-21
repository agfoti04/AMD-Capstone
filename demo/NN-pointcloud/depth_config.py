import depthai as dai


LR_CHECK = True
EXTENDED_DISPARITY = False
SUBPIXEL = True


def configure_depth_post_processing(stereo_depth_node: dai.node.StereoDepth) -> None:
    stereo_depth_node.setDefaultProfilePreset(
        dai.node.StereoDepth.PresetMode.FAST_DENSITY
    )

    config = stereo_depth_node.initialConfig
    config.postProcessing.decimationFilter.decimationFactor = 1
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 28
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 0
    config.postProcessing.thresholdFilter.maxRange = 2000
    config.censusTransform.enableMeanMode = True
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2

    stereo_depth_node.setLeftRightCheck(LR_CHECK)
    stereo_depth_node.setExtendedDisparity(EXTENDED_DISPARITY)
    stereo_depth_node.setSubpixel(SUBPIXEL)
    stereo_depth_node.setRectifyEdgeFillColor(0)
