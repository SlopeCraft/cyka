cmake_minimum_required(VERSION 3.25)
project(cyka VERSION 0.0 LANGUAGES C CXX)

find_package(Eigen3 3.4.0 REQUIRED)

add_subdirectory(cyka/genetic)