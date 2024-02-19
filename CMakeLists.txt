# CMakeList.txt : CMake project for NeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
# set(CMAKE_VERBOSE_MAKEFILE ON)

project ("NeuralNetwork")

# Set LibTorch and OpenCV paths
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/include/libtorch;${CMAKE_SOURCE_DIR}/include/opencv/build" ${CMAKE_PREFIX_PATH})

# find torch package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")

# Add source to this project's executable.
add_executable(NeuralNetwork "../NeuralNetwork.cpp" "../NeuralNetwork.h")

# Include OpenCV directories
include_directories("${CMAKE_SOURCE_DIR}/include/opencv/build/include")

# Link OpenCV libraries
target_link_libraries(NeuralNetwork PRIVATE ${OpenCV_LIBS})
# Link Torch libraries
target_link_libraries(NeuralNetwork PRIVATE ${TORCH_LIBRARIES})

# Set C++ standard
set_property(TARGET NeuralNetwork PROPERTY CXX_STANDARD 17)

# Handle Windows-specific DLL copying
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET NeuralNetwork
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:NeuralNetwork>)
endif (MSVC)