add_library(genetic STATIC
        GA.h
        GA.cpp
        population.cpp
        population.h)
target_compile_features(genetic PUBLIC cxx_std_23)
target_link_libraries(genetic PUBLIC Eigen3::Eigen)