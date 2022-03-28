cmake_minimum_required(VERSION 3.1)
# ①改工程名字
project(test)

set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

# ②添加库文件
find_package(Ceres REQUIRED)
message("ceres: " ${Ceres_INCLUDE_DIRS})

# ☂include
include_directories(${Ceres_INCLUDE_DIRS})

# ④生成可执行文件
add_executable( LearnCeres cereslearning1.cpp)

# ⑤添加软连接
target_link_libraries(LearnCeres ${CERES_LIBRARIES})
