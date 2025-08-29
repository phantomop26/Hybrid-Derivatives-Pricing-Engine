#include "gpu_accelerator.hpp"
#include "sobol_generator.hpp"
#include "vanilla_options.hpp"
#include "two_asset_options.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace hybrid_pricer {

GPUAccelerator::GPUAccelerator(const std::string& kernelFile) {
    cl_int err;

    // Get platform
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platform count");
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platforms");
    }

    // Find Apple platform
    cl_platform_id selectedPlatform = nullptr;
    for (cl_platform_id platform : platforms) {
        char platformName[128];
        err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        if (err == CL_SUCCESS && strstr(platformName, "Apple") != nullptr) {
            selectedPlatform = platform;
            break;
        }
    }

    if (!selectedPlatform) {
        selectedPlatform = platforms[0]; // Fall back to first platform
    }

    // Get device
    cl_uint numDevices;
    err = clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get GPU device count");
    }

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get GPU devices");
    }

    // Create context
    context_ = clCreateContext(nullptr, 1, &devices[0], nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }

    // Create command queue
#ifdef __APPLE__
    queue_ = clCreateCommandQueue(context_, devices[0], 0, &err);  // Use legacy function on macOS
#else
    queue_ = clCreateCommandQueueWithProperties(context_, devices[0], nullptr, &err);
#endif
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create command queue");
    }

    device_ = devices[0];

    // Load and build kernel
    std::ifstream kernelFile_(kernelFile);
    if (!kernelFile_.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + kernelFile);
    }

    std::stringstream buffer;
    buffer << kernelFile_.rdbuf();
    std::string kernelSource = buffer.str();
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();

    program_ = clCreateProgramWithSource(context_, 1, &source, &sourceSize, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
    }

    // Build options for macOS OpenCL compatibility
    const char* buildOptions = "-cl-std=CL1.2 -D CL_TARGET_OPENCL_VERSION=120 -cl-mad-enable -cl-fast-relaxed-math -cl-denorms-are-zero -cl-fp32-correctly-rounded-divide-sqrt";
    err = clBuildProgram(program_, 1, &device_, buildOptions, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build status
        cl_build_status buildStatus;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, nullptr);

        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        std::string errMsg = "Failed to build program (status " + std::to_string(buildStatus) + "): ";
        switch (err) {
            case CL_BUILD_PROGRAM_FAILURE:
                errMsg += "Build failed:\n";
                break;
            case CL_INVALID_PROGRAM:
                errMsg += "Invalid program object\n";
                break;
            case CL_INVALID_VALUE:
                errMsg += "Invalid build options\n";
                break;
            default:
                errMsg += "Unknown error " + std::to_string(err) + "\n";
        }
        errMsg += buildLog.data();
        throw std::runtime_error("Failed to build program: " + std::string(buildLog.data()));
    }

    // Create kernels with detailed error handling
    cl_int kernel_err;
    generatePathsKernel_ = clCreateKernel(program_, "generatePaths", &kernel_err);
    if (kernel_err != CL_SUCCESS) {
        std::string errMsg = "Failed to create generatePaths kernel: ";
        switch (kernel_err) {
            case CL_INVALID_PROGRAM:
                errMsg += "Invalid program object";
                break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                errMsg += "Program executable not loaded for device";
                break;
            case CL_INVALID_KERNEL_NAME:
                errMsg += "Kernel name not found in program";
                break;
            case CL_INVALID_KERNEL_DEFINITION:
                errMsg += "Invalid kernel definition";
                break;
            default:
                errMsg += "Error code " + std::to_string(kernel_err);
        }
        throw std::runtime_error(errMsg);
    }

    calculatePayoffsKernel_ = clCreateKernel(program_, "calculatePayoffs", &kernel_err);
    if (kernel_err != CL_SUCCESS) {
        std::string errMsg = "Failed to create calculatePayoffs kernel: ";
        switch (kernel_err) {
            case CL_INVALID_PROGRAM:
                errMsg += "Invalid program object";
                break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                errMsg += "Program executable not loaded for device";
                break;
            case CL_INVALID_KERNEL_NAME:
                errMsg += "Kernel name not found in program";
                break;
            case CL_INVALID_KERNEL_DEFINITION:
                errMsg += "Invalid kernel definition";
                break;
            default:
                errMsg += "Error code " + std::to_string(kernel_err);
        }
        throw std::runtime_error(errMsg);
    }
}

GPUAccelerator::~GPUAccelerator() {
    cleanup();
}

void GPUAccelerator::checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::stringstream ss;
        ss << "Error during operation '" << operation << "': ";
        switch (err) {
            case CL_DEVICE_NOT_FOUND:
                ss << "Device not found";
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                ss << "Device not available";
                break;
            case CL_COMPILER_NOT_AVAILABLE:
                ss << "Compiler not available";
                break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                ss << "Memory object allocation failure";
                break;
            case CL_OUT_OF_RESOURCES:
                ss << "Out of resources";
                break;
            case CL_OUT_OF_HOST_MEMORY:
                ss << "Out of host memory";
                break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                ss << "Profiling info not available";
                break;
            case CL_MEM_COPY_OVERLAP:
                ss << "Memory copy overlap";
                break;
            case CL_IMAGE_FORMAT_MISMATCH:
                ss << "Image format mismatch";
                break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                ss << "Image format not supported";
                break;
            case CL_BUILD_PROGRAM_FAILURE:
                ss << "Build program failure";
                break;
            case CL_MAP_FAILURE:
                ss << "Map failure";
                break;
            case CL_INVALID_VALUE:
                ss << "Invalid value";
                break;
            case CL_INVALID_DEVICE_TYPE:
                ss << "Invalid device type";
                break;
            case CL_INVALID_PLATFORM:
                ss << "Invalid platform";
                break;
            case CL_INVALID_DEVICE:
                ss << "Invalid device";
                break;
            case CL_INVALID_CONTEXT:
                ss << "Invalid context";
                break;
            case CL_INVALID_QUEUE_PROPERTIES:
                ss << "Invalid queue properties";
                break;
            case CL_INVALID_COMMAND_QUEUE:
                ss << "Invalid command queue";
                break;
            case CL_INVALID_HOST_PTR:
                ss << "Invalid host pointer";
                break;
            case CL_INVALID_MEM_OBJECT:
                ss << "Invalid memory object";
                break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                ss << "Invalid image format descriptor";
                break;
            case CL_INVALID_IMAGE_SIZE:
                ss << "Invalid image size";
                break;
            case CL_INVALID_SAMPLER:
                ss << "Invalid sampler";
                break;
            case CL_INVALID_BINARY:
                ss << "Invalid binary";
                break;
            case CL_INVALID_BUILD_OPTIONS:
                ss << "Invalid build options";
                break;
            case CL_INVALID_PROGRAM:
                ss << "Invalid program";
                break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                ss << "Invalid program executable";
                break;
            case CL_INVALID_KERNEL_NAME:
                ss << "Invalid kernel name";
                break;
            case CL_INVALID_KERNEL_DEFINITION:
                ss << "Invalid kernel definition";
                break;
            case CL_INVALID_KERNEL:
                ss << "Invalid kernel";
                break;
            case CL_INVALID_ARG_INDEX:
                ss << "Invalid argument index";
                break;
            case CL_INVALID_ARG_VALUE:
                ss << "Invalid argument value";
                break;
            case CL_INVALID_ARG_SIZE:
                ss << "Invalid argument size";
                break;
            case CL_INVALID_KERNEL_ARGS:
                ss << "Invalid kernel arguments";
                break;
            case CL_INVALID_WORK_DIMENSION:
                ss << "Invalid work dimension";
                break;
            case CL_INVALID_WORK_GROUP_SIZE:
                ss << "Invalid work group size";
                break;
            case CL_INVALID_WORK_ITEM_SIZE:
                ss << "Invalid work item size";
                break;
            case CL_INVALID_GLOBAL_OFFSET:
                ss << "Invalid global offset";
                break;
            case CL_INVALID_EVENT_WAIT_LIST:
                ss << "Invalid event wait list";
                break;
            case CL_INVALID_EVENT:
                ss << "Invalid event";
                break;
            case CL_INVALID_OPERATION:
                ss << "Invalid operation";
                break;
            case CL_INVALID_GL_OBJECT:
                ss << "Invalid OpenGL object";
                break;
            case CL_INVALID_BUFFER_SIZE:
                ss << "Invalid buffer size";
                break;
            case CL_INVALID_MIP_LEVEL:
                ss << "Invalid mip-map level";
                break;
            default:
                ss << "Unknown error " << err;
                break;
        }
        throw std::runtime_error(ss.str());
    }
}

void GPUAccelerator::cleanup() {
    if (generatePathsKernel_) clReleaseKernel(generatePathsKernel_);
    if (calculatePayoffsKernel_) clReleaseKernel(calculatePayoffsKernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

std::vector<double> GPUAccelerator::generateSobolSequence(size_t numPaths, size_t numDims) {
    if (numPaths == 0 || numDims == 0) {
        throw std::invalid_argument("numPaths and numDims must be positive");
    }

    const size_t totalSize = numPaths * numDims;
    if (totalSize / numPaths != numDims) {
        throw std::runtime_error("Sequence size would exceed memory limits");
    }

    std::vector<double> sequence;
    sequence.reserve(totalSize);

    hybrid_pricer::SobolGenerator sobol(numDims);
    for (size_t i = 0; i < numPaths; ++i) {
        auto point = sobol.next();
        for (auto& x : point) {
            x = std::max(std::min(x, 1.0 - 1e-10), 1e-10);
        }
        sequence.insert(sequence.end(), point.begin(), point.end());
    }

    return sequence;
}

double GPUAccelerator::priceOption(std::shared_ptr<Option> option,
                                 size_t numPaths,
                                 size_t numSteps,
                                 bool useSobol) {
    cl_int err;

    // Validate inputs
    if (numPaths == 0 || numSteps == 0) {
        throw std::invalid_argument("numPaths and numSteps must be positive");
    }

    if (option->timeToMaturity <= 0.0 || option->volatility <= 0.0) {
        throw std::invalid_argument("Time to maturity and volatility must be positive");
    }

    // Path generation parameters with improved numerical stability
    const double dt = option->timeToMaturity / static_cast<double>(numSteps);
    const double sigma2 = option->volatility * option->volatility;
    const double drift = (option->riskFreeRate - 0.5 * sigma2) * dt;
    const size_t pathStride = numSteps + 1;

    // Check for buffer size overflow
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
    const size_t pathElements = numPaths * pathStride;
    if (pathElements / numPaths != pathStride || pathElements > maxElements) {
        throw std::runtime_error("Buffer size would exceed memory limits");
    }

    // Create buffers with overflow checks
    size_t pathsSize = numPaths * pathStride * sizeof(double);
    size_t payoffsSize = numPaths * sizeof(double);

    cl_mem pathsBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, pathsSize, nullptr, &err);
    checkError(err, "creating paths buffer");

    cl_mem payoffsBuffer = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, payoffsSize, nullptr, &err);
    checkError(err, "creating payoffs buffer");

    // Generate and upload Sobol numbers if requested
    if (useSobol) {
        auto sobolSeq = generateSobolSequence(numPaths, 2 * numSteps);
        size_t sobolSize = sobolSeq.size() * sizeof(double);

        cl_mem sobolBuffer = clCreateBuffer(context_, CL_MEM_READ_ONLY, sobolSize, nullptr, &err);
        checkError(err, "creating sobol buffer");

        err = clEnqueueWriteBuffer(queue_, sobolBuffer, CL_TRUE, 0, sobolSize, sobolSeq.data(), 0, nullptr, nullptr);
        checkError(err, "writing sobol numbers");

        // Set kernel arguments for path generation
        err = clSetKernelArg(generatePathsKernel_, 0, sizeof(cl_mem), &pathsBuffer);
        err |= clSetKernelArg(generatePathsKernel_, 1, sizeof(cl_mem), &sobolBuffer);
        err |= clSetKernelArg(generatePathsKernel_, 2, sizeof(double), &option->spot);
        err |= clSetKernelArg(generatePathsKernel_, 3, sizeof(double), &drift);
        err |= clSetKernelArg(generatePathsKernel_, 4, sizeof(double), &option->volatility);
        err |= clSetKernelArg(generatePathsKernel_, 5, sizeof(double), &dt);
        err |= clSetKernelArg(generatePathsKernel_, 6, sizeof(size_t), &numSteps);
        err |= clSetKernelArg(generatePathsKernel_, 7, sizeof(size_t), &pathStride);
        checkError(err, "setting generate_paths kernel arguments");

        // Launch path generation kernel
        size_t globalWorkSize = numPaths;
        err = clEnqueueNDRangeKernel(queue_, generatePathsKernel_, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        checkError(err, "enqueueing generate_paths kernel");

        clReleaseMemObject(sobolBuffer);
    }

    // Determine option type
    int optionType = 0;  // Default to European
    if (std::dynamic_pointer_cast<AsianOption>(option)) {
        optionType = 1;
    } else if (std::dynamic_pointer_cast<BarrierOption>(option)) {
        optionType = 2;
    }

    // Set kernel arguments for payoff calculation
    // Get barrier level if this is a barrier option
    double barrier = 0.0;
    if (auto barrierOption = std::dynamic_pointer_cast<BarrierOption>(option)) {
        barrier = barrierOption->getBarrier();
    }

    err = clSetKernelArg(calculatePayoffsKernel_, 0, sizeof(cl_mem), &pathsBuffer);
    err |= clSetKernelArg(calculatePayoffsKernel_, 1, sizeof(cl_mem), &payoffsBuffer);
    err |= clSetKernelArg(calculatePayoffsKernel_, 2, sizeof(int), &optionType);
    err |= clSetKernelArg(calculatePayoffsKernel_, 3, sizeof(double), &option->strike);
    err |= clSetKernelArg(calculatePayoffsKernel_, 4, sizeof(double), &barrier);
    err |= clSetKernelArg(calculatePayoffsKernel_, 5, sizeof(size_t), &numSteps);
    err |= clSetKernelArg(calculatePayoffsKernel_, 6, sizeof(size_t), &pathStride);
    checkError(err, "setting calculate_payoffs kernel arguments");

    // Launch payoff calculation kernel
    size_t globalWorkSize = numPaths;
    err = clEnqueueNDRangeKernel(queue_, calculatePayoffsKernel_, 1, nullptr,
                                &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "enqueueing calculate_payoffs kernel");

    // Read back results
    std::vector<double> payoffs(numPaths);
    err = clEnqueueReadBuffer(queue_, payoffsBuffer, CL_TRUE, 0,
                            payoffsSize, payoffs.data(), 0, nullptr, nullptr);
    checkError(err, "reading payoffs");

    // Calculate mean
    double sum = 0.0;
    for (double payoff : payoffs) {
        sum += payoff;
    }
    double mean = sum / static_cast<double>(numPaths);

    // Apply discounting
    double price = mean * exp(-option->riskFreeRate * option->timeToMaturity);

    // Cleanup
    clReleaseMemObject(pathsBuffer);
    clReleaseMemObject(payoffsBuffer);

    return price;
}

} // namespace hybrid_pricer
