#include <gtest/gtest.h>
#include "gpu_accelerator.hpp"
#include "monte_carlo_pricer.hpp"
#include "vanilla_options.hpp"
#include <cmath>
#include <chrono>

using namespace hybrid_pricer;
using namespace std::chrono;

TEST(GPUTest, CompareWithCPU) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto option = std::make_shared<EuropeanOption>(
        Option::OptionType::Call,
        K, T, S, sigma, r);
    
    const size_t numPaths = 1000000;
    const size_t numSteps = 252;
    
    // Price with CPU Monte Carlo
    MonteCarloPricer cpu_pricer(option, numPaths, numSteps);
    auto start = high_resolution_clock::now();
    double cpu_price = cpu_pricer.price(false, false);
    auto cpu_duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    
    // Price with GPU
    const char* kernelPath = "monte_carlo_kernels.cl";
    std::string buildPath = std::string(BUILD_DIR) + "/tests/" + kernelPath;
    GPUAccelerator gpu_pricer(buildPath);
    start = high_resolution_clock::now();
    double gpu_price = gpu_pricer.priceOption(option, numPaths, numSteps, true);
    auto gpu_duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    
    // Calculate Black-Scholes price
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S * std::erf(d1/sqrt(2.0)) - 
                     K * std::exp(-r*T) * std::erf(d2/sqrt(2.0));
    
    // Compare results
    EXPECT_NEAR(cpu_price, bs_price, 0.1);
    EXPECT_NEAR(gpu_price, bs_price, 0.1);
    EXPECT_NEAR(cpu_price, gpu_price, 0.1);
    
    // GPU should be significantly faster
    std::cout << "CPU time: " << cpu_duration.count() << "ms\n";
    std::cout << "GPU time: " << gpu_duration.count() << "ms\n";
    EXPECT_LT(gpu_duration.count(), cpu_duration.count());
}

TEST(GPUTest, AsianOption) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto option = std::make_shared<AsianOption>(
        AsianOption::Type::Arithmetic,
        Option::OptionType::Call,
        K, T, S, sigma, r);
    
    const size_t numPaths = 1000000;
    const size_t numSteps = 252;
    
    // Price with CPU Monte Carlo
    MonteCarloPricer cpu_pricer(option, numPaths, numSteps);
    double cpu_price = cpu_pricer.price(false, false);
    
    // Price with GPU
    GPUAccelerator gpu_pricer("monte_carlo_kernels.cl");
    double gpu_price = gpu_pricer.priceOption(option, numPaths, numSteps, true);
    
    // Results should be close
    EXPECT_NEAR(cpu_price, gpu_price, 0.1);
}

TEST(GPUTest, BarrierOption) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double barrier = 120.0;
    
    auto option = std::make_shared<BarrierOption>(
        BarrierOption::Type::UpAndOut,
        Option::OptionType::Call,
        K, T, S, sigma, r, barrier);
    
    const size_t numPaths = 1000000;
    const size_t numSteps = 252;
    
    // Price with CPU Monte Carlo
    MonteCarloPricer cpu_pricer(option, numPaths, numSteps);
    double cpu_price = cpu_pricer.price(false, false);
    
    // Price with GPU
    GPUAccelerator gpu_pricer("monte_carlo_kernels.cl");
    double gpu_price = gpu_pricer.priceOption(option, numPaths, numSteps, true);
    
    // Results should be close
    EXPECT_NEAR(cpu_price, gpu_price, 0.1);
}
