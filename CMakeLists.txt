cmake_minimum_required(VERSION 3.15)
project(HybridPricer VERSION 1.0)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for nlohmann/json
include_directories(${CMAKE_BINARY_DIR}/include)

# macOS specific settings
if(APPLE)
    # OpenMP configuration
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE BREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_C_INCLUDE_DIR "${BREW_LIBOMP_PREFIX}/include")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIR "${BREW_LIBOMP_PREFIX}/include")
    endif()
    
    # OpenCL configuration for macOS
    find_library(OPENCL_LIBRARY OpenCL)
    if(NOT OPENCL_LIBRARY)
        message(FATAL_ERROR "OpenCL library not found")
    endif()
    include_directories(SYSTEM "/System/Library/Frameworks/OpenCL.framework/Headers")
else()
    # Non-macOS OpenCL configuration
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
endif()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)

# Set build type to Release by default for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add compiler flags
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories("${OpenMP_CXX_INCLUDE_DIRS}")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Create static library
add_library(hybridpricer STATIC
    $<TARGET_OBJECTS:pricer_core>
    $<TARGET_OBJECTS:monte_carlo>
    $<TARGET_OBJECTS:pde_solver>
)

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(hybridpricer PUBLIC OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    target_link_libraries(hybridpricer PUBLIC ${OPENCL_LIBRARY})
else()
    target_link_libraries(hybridpricer PUBLIC ${OpenCL_LIBRARIES})
endif()
