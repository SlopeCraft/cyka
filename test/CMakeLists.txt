function(cyka_add_test src_name)
    cmake_path(GET src_name STEM stem)
    add_executable(test_${stem} ${src_name})
    target_link_libraries(test_${stem} PRIVATE cyka::cyka)

    add_test(NAME ${stem} COMMAND "test_${stem}.exe"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

cyka_add_test(compiler_features.cpp)
cyka_add_test(initiate.cpp)
cyka_add_test(crossovers.cpp)