cmake_minimum_required(VERSION 3.1)
# ①改工程名字
project(test)

set( CMAKE_CXX_FLAGS "-std=c++14 -O3" )

# ②添加库文件
find_package(Ceres REQUIRED)
find_package(Python2 COMPONENTS Development NumPy)
message("ceres: " ${Ceres_INCLUDE_DIRS})

# ☂include
include_directories(ThirdParty
${Ceres_INCLUDE_DIRS}
${Python2_INCLUDE_DIRS}
${Python2_NumPy_INCLUDE_DIRS}
)

# ④生成可执行文件
add_executable( LearnCeres4 src/No4_CurveFitting.cpp)

# ⑤添加软连接
target_link_libraries(LearnCeres4 ${CERES_LIBRARIES} Python2::Python Python2::NumPy)
