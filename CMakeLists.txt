cmake_minimum_required(VERSION 3.8)
project(rsf_3d_localization)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}_core SHARED
  src/eskf.cpp
  src/likelihood_grid.cpp
  src/ndt_map.cpp
  src/pcd_io.cpp
)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}_core Eigen3::Eigen)
target_compile_features(${PROJECT_NAME}_core PUBLIC cxx_std_17)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/rsf_3d_localization_node.cpp
)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_core)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "rsf_3d_localization::Rsf3dLocalization"
  EXECUTABLE rsf_3d_localization_node
)

install(TARGETS ${PROJECT_NAME}_core EXPORT export_${PROJECT_NAME}_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(export_${PROJECT_NAME}_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Formatting is checked with the package .clang-format, which conflicts with uncrustify and cpplint
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_uncrustify ament_cmake_cpplint)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_${PROJECT_NAME}
    test/test_pose_math.cpp
    test/test_pcd_io.cpp
    test/test_ndt_map.cpp
    test/test_eskf.cpp
    test/test_likelihood_grid.cpp
  )
  target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME}_core)
endif()

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE config launch schema
)
