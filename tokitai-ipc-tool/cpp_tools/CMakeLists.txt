cmake_minimum_required(VERSION 3.10)
project(lidar_cpp_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 jsoncpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# 可选：查找 PCL
# find_package(PCL REQUIRED)

# 可选：查找 CUDA
# find_package(CUDA)

include_directories(${JSONCPP_INCLUDE_DIRS})
# include_directories(${PCL_INCLUDE_DIRS})

link_directories(${JSONCPP_LIBRARY_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})

add_executable(pointcloud_tools_cpp
    cpp_tools/pointcloud_tools.cpp
)

target_link_libraries(pointcloud_tools_cpp
    ${JSONCPP_LIBRARIES}
    # ${PCL_LIBRARIES}
)

# 安装目标
install(TARGETS pointcloud_tools_cpp DESTINATION bin)
