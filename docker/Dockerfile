# Ubuntu 22.04 + ROS 2 Humble + Gazebo Classic 11.
# Do not base this on Ubuntu 24.04 / Jazzy / Gazebo Harmonic.
# The rover description uses Classic plugins (libgazebo_ros_camera.so,
# libgazebo_ros2_control.so) that those newer versions do not provide.

FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-gazebo-ros2-control \
        ros-humble-controller-manager \
        ros-humble-joint-state-broadcaster \
        ros-humble-diff-drive-controller \
        ros-humble-topic-tools \
        ros-humble-cv-bridge \
        ros-humble-vision-msgs \
        ros-humble-image-transport \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-opencv \
        xvfb \
        mesa-utils \
        libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
