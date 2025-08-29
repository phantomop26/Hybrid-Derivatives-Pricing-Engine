# Source directory CMakeLists.txt
add_library(pricer_core OBJECT
    monte_carlo_pricer.cpp
    pde_solver.cpp
    gpu_accelerator.cpp
    sobol_generator.cpp
)

# Monte Carlo components
add_library(monte_carlo OBJECT
    monte_carlo_pricer.cpp
    gpu_accelerator.cpp
    sobol_generator.cpp
)

# PDE solver components
add_library(pde_solver OBJECT
    pde_solver.cpp
)

# Set include directories for all targets
target_include_directories(pricer_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(monte_carlo PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pde_solver PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add OpenMP flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(pricer_core PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(monte_carlo PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(pde_solver PUBLIC OpenMP::OpenMP_CXX)
endif()

# Build main executable
add_executable(pricer main.cpp)
target_link_libraries(pricer PRIVATE hybridpricer)

# Add OpenCL library to the executable
if(APPLE)
    target_link_libraries(pricer PRIVATE ${OPENCL_LIBRARY})
else()
    target_link_libraries(pricer PRIVATE ${OpenCL_LIBRARIES})
endif()
