cmake_minimum_required(VERSION 3.25)
project(cyka VERSION 0.0 LANGUAGES C CXX)

if (${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
    option(CYKA_INSTALL "Install cyka" ON)
else ()
    option(CYKA_INSTALL "Install cyka" OFF)
endif ()

option(CYKA_INCLUDE_TESTS "Build test files" ON)

message(STATUS "CYKA_INSTALL = ${CYKA_INSTALL}")

find_package(Eigen3 3.4.0 REQUIRED)

file(GLOB_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/cyka/*.h;${CMAKE_CURRENT_SOURCE_DIR}/cyka/*.hpp" headers)

add_library(cyka INTERFACE)
target_include_directories(cyka INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(cyka INTERFACE Eigen3::Eigen)
target_compile_features(cyka INTERFACE cxx_std_23)
target_precompile_headers(cyka INTERFACE $<BUILD_INTERFACE:${headers}>)

add_library(cyka::cyka ALIAS cyka)

if (${CYKA_INCLUDE_TESTS})
    enable_testing()
    add_subdirectory(test)
endif ()

if (${CYKA_INSTALL})
    include(cmake/install.cmake)
endif ()