cmake_minimum_required(VERSION 3.25)
project(cyka VERSION 0.0 LANGUAGES C CXX)

find_package(Eigen3 3.4.0 REQUIRED)

file(GLOB_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/*.h;${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" headers)

add_library(cyka INTERFACE)
target_include_directories(cyka INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(cyka INTERFACE Eigen3::Eigen)
target_compile_features(cyka INTERFACE cxx_std_23)
target_sources(cyka INTERFACE $<BUILD_INTERFACE:${headers}>)

add_subdirectory(test)