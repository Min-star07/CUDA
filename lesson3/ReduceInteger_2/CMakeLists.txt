# 最低版本要求
cmake_minimum_required(VERSION 3.22)
# 项目信息
project(ReductionComputation LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 17)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

find_package(CUDA 12.2 REQUIRED)
if (CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0 -arch=sm_86")   # CUDA调试
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")                    # C++调试
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
set(INC "${PROJECT_SOURCE_DIR}/inc")

# 添加可执行文件
add_executable(reduction ${SRCS})
target_include_directories(reduction PRIVATE ${INC})
target_include_directories(reduction PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(reduction PRIVATE ${CUDA_LIBRARIES})


