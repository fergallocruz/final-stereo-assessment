cmake_minimum_required(VERSION 3.0.0)
project(final-assessment VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

#add_library(stereo_calibration src/stereo_calibration.cpp)
add_executable(final-assessment src/main.cpp)

#target_link_libraries(stereo_calibration PRIVATE ${OpenCV_LIBS})
target_link_libraries(final-assessment PRIVATE ${OpenCV_LIBS} jsoncpp_lib jsoncpp_object)