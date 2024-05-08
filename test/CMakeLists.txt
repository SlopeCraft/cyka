function(cyka_add_test src_name)
    cmake_path(GET src_name STEM stem)
    add_executable(test_${stem} ${src_name})
    target_link_libraries(test_${stem} PRIVATE cyka::cyka)
endfunction()

cyka_add_test(compiler_features.cpp)
cyka_add_test(initiate.cpp)
cyka_add_test(crossovers.cpp)