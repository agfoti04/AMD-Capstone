cmake_minimum_required(VERSION 3.8)
project(arm_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(robot_state_publisher REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY urdf launch config meshes DESTINATION share/${PROJECT_NAME})

install(
  PROGRAMS src/arm_teleop.py 
  DESTINATION lib/${PROJECT_NAME}
  RENAME arm_teleop
)

install(PROGRAMS src/arm_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME arm_control)

install(DIRECTORY urdf launch config meshes models worlds
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS src/pick_place.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME pick_place)

install(PROGRAMS src/goto_xyz.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME goto_xyz)

install(PROGRAMS src/gripper_fan.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gripper_fan)

  install(PROGRAMS src/scene_setup.py DESTINATION lib/${PROJECT_NAME} RENAME scene_setup)

install(PROGRAMS src/goto_xyz_dual.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME goto_xyz_dual)

  ament_package()