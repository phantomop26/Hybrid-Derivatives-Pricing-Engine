# Enable testing
enable_testing()

# Find GTest package
find_package(GTest REQUIRED)

# Add test executable
add_executable(pricer_tests
    test_european_option.cpp
    test_asian_option.cpp
    test_barrier_option.cpp
    test_lookback_option.cpp
    test_two_asset_option.cpp
    test_monte_carlo.cpp
    test_pde_solver.cpp
    test_sobol.cpp
    test_gpu.cpp
)

# Define binary directory for runtime
target_compile_definitions(pricer_tests PRIVATE BUILD_DIR="${CMAKE_BINARY_DIR}")

# Link libraries
target_link_libraries(pricer_tests 
    PRIVATE 
    hybridpricer 
    GTest::GTest 
    GTest::Main
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pricer_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    target_link_libraries(pricer_tests PRIVATE ${OPENCL_LIBRARY})
else()
    target_link_libraries(pricer_tests PRIVATE ${OpenCL_LIBRARIES})
endif()

# Copy OpenCL kernel file to build directory
file(COPY ${CMAKE_SOURCE_DIR}/src/monte_carlo_kernels.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/src/monte_carlo_kernels.cl DESTINATION ${CMAKE_BINARY_DIR})

# Add tests
include(GoogleTest)
gtest_discover_tests(pricer_tests)
