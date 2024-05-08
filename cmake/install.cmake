include(CMakePackageConfigHelpers)


install(DIRECTORY ${CMAKE_SOURCE_DIR}/cyka
    DESTINATION include)
install(TARGETS cyka
    EXPORT cykaTargets
    LIBRARY DESTINATION lib)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cykaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cykaConfigVersion.cmake" DESTINATION lib/cmake/cyka)

install(EXPORT cykaTargets DESTINATION lib/cmake/cyka)

include(CPack)