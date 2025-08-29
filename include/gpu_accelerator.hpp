#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include "option.hpp"

namespace hybrid_pricer {

class GPUAccelerator {
public:
    GPUAccelerator(const std::string& kernelPath);
    ~GPUAccelerator();
    
    // Price options using GPU
    double priceOption(std::shared_ptr<Option> option,
                      size_t numPaths,
                      size_t numSteps,
                      bool useSobol = true);
    
private:
    // OpenCL objects
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel generatePathsKernel_;
    cl_kernel calculatePayoffsKernel_;
    
    // Initialization helpers
    void initOpenCL();
    void buildProgram(const std::string& kernelPath);
    void createKernels();
    
    // Clean up OpenCL resources
    void cleanup();
    
    // Generate Sobol sequences on CPU for use in GPU
    std::vector<double> generateSobolSequence(size_t numPaths, size_t numDims);
    
    // Check OpenCL errors
    static void checkError(cl_int err, const char* operation);
};

} // namespace hybrid_pricer
