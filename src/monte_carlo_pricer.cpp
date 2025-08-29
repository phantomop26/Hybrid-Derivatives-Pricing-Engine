#include "sobol_generator.hpp"
#include "monte_carlo_pricer.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <limits>
#include <iostream>

// Remove using declarations to avoid namespace issues

namespace hybrid_pricer {

namespace {
// Thread-local random number generation helper
std::vector<double> generateNormals(std::size_t count, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = normal(rng);
    }
    return result;
}

// Convert uniform numbers from Sobol to standard normal using inverse CDF
inline double uniformToNormal(double u) {
    static const double ROOT_2 = std::sqrt(2.0);
    // Moro's algorithm for inverse normal CDF approximation
    if (u < 0.5) {
        u = 2.0 * u;
        double y = std::sqrt(-2.0 * std::log(u));
        return -y + ((0.010328 * y + 0.802853) * y + 2.515517) /
               (((0.001308 * y + 0.189269) * y + 1.432788) * y + 1.0);
    } else {
        u = 2.0 * (1.0 - u);
        double y = std::sqrt(-2.0 * std::log(u));
        return y - ((0.010328 * y + 0.802853) * y + 2.515517) /
               (((0.001308 * y + 0.189269) * y + 1.432788) * y + 1.0);
    }
}
} // anonymous namespace

// Thread-local Sobol generator to ensure thread safety
thread_local SobolGenerator sobol1D(1);
thread_local SobolGenerator sobol2D(2);

MonteCarloPricer::MonteCarloPricer(std::shared_ptr<Option> option, size_t numPaths, size_t numSteps)
    : option_(option), numPaths_(numPaths), numSteps_(numSteps) {
    pathPrices_.reserve(numPaths_);
}

std::vector<double> MonteCarloPricer::generatePath(bool useAntithetic, bool useQuasiRandom) {
    alignas(64) std::vector<double> path(numSteps_ + 1);
    alignas(64) std::vector<double> randomNums(numSteps_);
    alignas(64) std::vector<double> logReturns(numSteps_);
    
    const double dt = option_->timeToMaturity / numSteps_;
    const double drift = (option_->riskFreeRate - 0.5 * option_->volatility * option_->volatility) * dt;
    const double vol = option_->volatility * std::sqrt(dt);
    
    // Initialize first value
    path[0] = option_->spot;
    
    // Pre-generate random numbers (non-vectorizable part)
    if (useQuasiRandom) {
        for (size_t j = 0; j < numSteps_; ++j) {
            double uniform = sobol1D.next()[0];
            randomNums[j] = uniformToNormal(uniform);
        }
    } else {
        for (size_t j = 0; j < numSteps_; ++j) {
            randomNums[j] = normalDist_(gen_);
        }
    }
    
    // Compute log returns (vectorizable)
    #pragma omp simd
    for (size_t j = 0; j < numSteps_; ++j) {
        logReturns[j] = drift + vol * randomNums[j];
    }
    
    // Apply antithetic variates if requested (vectorizable)
    if (useAntithetic) {
        #pragma omp simd
        for (size_t j = 1; j < numSteps_; j += 2) {
            logReturns[j] = drift - vol * randomNums[j]; // Negate only the random component
        }
    }
    
    // Pre-compute all exponentials (vectorizable)
    #pragma omp simd
    for (size_t j = 0; j < numSteps_; ++j) {
        logReturns[j] = std::exp(logReturns[j]);
    }
    
    // Apply the returns (non-vectorizable due to dependency)
    double current = path[0];
    for (size_t j = 0; j < numSteps_; ++j) {
        current *= logReturns[j];
        path[j + 1] = current;
    }
    
    return path;
}

std::vector<double> MonteCarloPricer::generateTwoAssetPath(bool useAntithetic, bool useQuasiRandom,
                                                          double spot2, double sigma2, double correlation) {
    alignas(64) std::vector<double> path(2 * (numSteps_ + 1));
    alignas(64) std::vector<double> z1(numSteps_);
    alignas(64) std::vector<double> z2(numSteps_);
    alignas(64) std::vector<double> logReturns1(numSteps_);
    alignas(64) std::vector<double> logReturns2(numSteps_);
    
    const double dt = option_->timeToMaturity / numSteps_;
    const double drift1 = (option_->riskFreeRate - 0.5 * option_->volatility * option_->volatility) * dt;
    const double drift2 = (option_->riskFreeRate - 0.5 * sigma2 * sigma2) * dt;
    const double vol1 = option_->volatility * std::sqrt(dt);
    const double vol2 = sigma2 * std::sqrt(dt);
    const double sqrtOneMinusRhoSq = std::sqrt(1.0 - correlation * correlation);
    
    // Initialize first values
    path[0] = option_->spot;
    path[1] = spot2;
    
    // Pre-generate random numbers (non-vectorizable part)
    if (useQuasiRandom) {
        for (size_t j = 0; j < numSteps_; ++j) {
            auto point = sobol2D.next();
            z1[j] = uniformToNormal(point[0]);
            z2[j] = uniformToNormal(point[1]);
        }
    } else {
        for (size_t j = 0; j < numSteps_; ++j) {
            z1[j] = normalDist_(gen_);
            z2[j] = normalDist_(gen_);
        }
    }
    
    // Compute correlated random numbers (vectorizable)
    alignas(64) std::vector<double> z2Corr(numSteps_);
    #pragma omp simd
    for (size_t j = 0; j < numSteps_; ++j) {
        z2Corr[j] = correlation * z1[j] + sqrtOneMinusRhoSq * z2[j];
    }
    
    // Compute log returns (vectorizable)
    #pragma omp simd
    for (size_t j = 0; j < numSteps_; ++j) {
        logReturns1[j] = drift1 + vol1 * z1[j];
        logReturns2[j] = drift2 + vol2 * z2Corr[j];
    }
    
    // Apply antithetic variates if requested (vectorizable)
    if (useAntithetic) {
        #pragma omp simd
        for (size_t j = 1; j < numSteps_; j += 2) {
            logReturns1[j] = drift1 - vol1 * z1[j];
            logReturns2[j] = drift2 - vol2 * z2Corr[j];
        }
    }
    
    // Pre-compute all exponentials (vectorizable)
    #pragma omp simd
    for (size_t j = 0; j < numSteps_; ++j) {
        logReturns1[j] = std::exp(logReturns1[j]);
        logReturns2[j] = std::exp(logReturns2[j]);
    }
    
    // Apply the returns (non-vectorizable due to dependency)
    double current1 = path[0];
    double current2 = path[1];
    for (size_t j = 0; j < numSteps_; ++j) {
        current1 *= logReturns1[j];
        current2 *= logReturns2[j];
        size_t idx = 2 * (j + 1);
        path[idx] = current1;
        path[idx + 1] = current2;
    }
    
    return path;
}

double MonteCarloPricer::price(bool useAntithetic, bool useQuasiRandom) {
    double sum = 0.0;
    double sumSquared = 0.0;
    pathPrices_.clear();
    pathPrices_.reserve(numPaths_);
    
    // Check if we're pricing a two-asset option
    if (auto twoAssetOpt = std::dynamic_pointer_cast<TwoAssetOption>(option_)) {
        #pragma omp parallel for simd reduction(+:sum,sumSquared) schedule(static)
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generateTwoAssetPath(useAntithetic, useQuasiRandom,
                                          twoAssetOpt->getSpot2(),
                                          twoAssetOpt->getSigma2(),
                                          twoAssetOpt->getCorrelation());
            double payoff = option_->payoff(path);
            
            pathPrices_.push_back(payoff);
            sum += payoff;
            sumSquared += payoff * payoff;
        }
    } else {
        // Original single-asset pricing code
        #pragma omp parallel for simd reduction(+:sum,sumSquared) schedule(static)
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generatePath(useAntithetic, useQuasiRandom);
            double payoff = option_->payoff(path);
            
            pathPrices_.push_back(payoff);
            sum += payoff;
            sumSquared += payoff * payoff;
        }
    }
    
    // Calculate price and error
    double mean = sum / numPaths_;
    double variance = (sumSquared / numPaths_ - mean * mean) / (numPaths_ - 1);
    standardError_ = std::sqrt(variance / numPaths_);
    
    return std::exp(-option_->riskFreeRate * option_->timeToMaturity) * mean;
}

std::vector<double> MonteCarloPricer::generateAntitheticPath(const std::vector<double>& path) {
    std::vector<double> antitheticPath = path;
    const double dt = option_->timeToMaturity / static_cast<double>(numSteps_);
    const double drift = (option_->riskFreeRate - 0.5 * option_->volatility * option_->volatility) * dt;
    const double vol = option_->volatility * std::sqrt(dt);
    
    // Generate antithetic path by negating the random components
    antitheticPath[0] = option_->spot;
    for (size_t i = 1; i < numSteps_; ++i) {
        const double Z = -((std::log(path[i]/path[i-1]) - drift) / vol);  // Negate the random component
        antitheticPath[i] = antitheticPath[i-1] * std::exp(drift + vol * Z);
    }
    return antitheticPath;
}

std::vector<double> MonteCarloPricer::generateSobolPath(SobolGenerator& sobol) {
    const double dt = option_->timeToMaturity / static_cast<double>(numSteps_);
    const double sqrtDt = std::sqrt(dt);
    const double drift = (option_->riskFreeRate - 0.5 * option_->volatility * option_->volatility) * dt;
    const double diffusion = option_->volatility * sqrtDt;
    
    std::vector<double> path(numSteps_ + 1);
    path[0] = option_->spot;
    
    // Get next point from Sobol sequence and convert to normal variates
    auto uniforms = sobol.next();
    for (size_t i = 1; i <= numSteps_; ++i) {
        path[i] = path[i-1] * std::exp(drift + diffusion * uniformToNormal(uniforms[i-1]));
    }
    return path;
}

double MonteCarloPricer::calculateGeometricMean(const std::vector<double>& path) const {
    double logSum = 0.0;
    for (const auto& price : path) {
        logSum += std::log(price);
    }
    return std::exp(logSum / static_cast<double>(path.size()));
}

std::vector<double> MonteCarloPricer::getConfidenceInterval() const {
    if (pathPrices_.empty()) {
        return {0.0, 0.0};
    }
    
    const double mean = std::accumulate(pathPrices_.begin(), pathPrices_.end(), 0.0) / numPaths_;
    double sumSq = 0.0;
    
    #pragma omp parallel reduction(+:sumSq)
    {
        #pragma omp for nowait
        for (size_t i = 0; i < pathPrices_.size(); ++i) {
            const double diff = pathPrices_[i] - mean;
            sumSq += diff * diff;
        }
    }
    
    const double variance = sumSq / (numPaths_ - 1);
    const double stdError = std::sqrt(variance / numPaths_);
    const double discountFactor = std::exp(-option_->riskFreeRate * option_->timeToMaturity);
    
    // 95% confidence interval
    const double z = 1.96;  // Normal distribution 97.5th percentile
    return {
        (mean - z * stdError) * discountFactor,
        (mean + z * stdError) * discountFactor
    };
}

double MonteCarloPricer::priceWithSobol() {
    pathPrices_.clear();
    pathPrices_.reserve(numPaths_);
    
    // Initialize Sobol sequence generator with dimension = numSteps_
    SobolGenerator sobol(numSteps_);
    
    #pragma omp parallel if(numPaths_ > 10000)
    {
        std::vector<double> localPrices;
        const size_t localSize = numPaths_ / omp_get_max_threads() + 1;
        localPrices.reserve(localSize);
        
        #pragma omp for nowait
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generateSobolPath(sobol);
            double pathPayoff = option_->payoff(path);
            localPrices.push_back(pathPayoff);
        }
        
        #pragma omp critical
        {
            pathPrices_.insert(pathPrices_.end(), localPrices.begin(), localPrices.end());
        }
    }
    
    const double mean = std::accumulate(pathPrices_.begin(), pathPrices_.end(), 0.0) / numPaths_;
    return mean * std::exp(-option_->riskFreeRate * option_->timeToMaturity);
}

double MonteCarloPricer::getOptimalDrift() const {
    // For European options, the optimal drift shifts the mean to the strike price
    if (auto european = std::dynamic_pointer_cast<EuropeanOption>(option_)) {
        const double T = option_->timeToMaturity;
        const double sigma = option_->volatility;
        const double S = option_->spot;
        const double K = option_->strike;
        const double r = option_->riskFreeRate;
        
        // Optimal drift to reduce variance
        return (std::log(K/S) - (r - 0.5*sigma*sigma)*T) / (sigma*T);
    }
    return 0.0;  // Default for other option types
}

std::vector<double> MonteCarloPricer::generatePathWithImportanceSampling() {
    const double dt = option_->timeToMaturity / static_cast<double>(numSteps_);
    const double sqrtDt = std::sqrt(dt);
    const double sigma = option_->volatility;
    const double r = option_->riskFreeRate;
    
    // Calculate drift adjustment for importance sampling
    const double driftAdjust = sigma * getOptimalDrift();
    const double adjustedDrift = (r - 0.5 * sigma * sigma + driftAdjust) * dt;
    const double diffusion = sigma * sqrtDt;
    
    std::vector<double> path(numSteps_ + 1);
    path[0] = option_->spot;
    
    auto normals = generateNormals(numSteps_, gen_);
    for (size_t i = 1; i <= numSteps_; ++i) {
        path[i] = path[i-1] * std::exp(adjustedDrift + diffusion * normals[i-1]);
    }
    return path;
}

double MonteCarloPricer::getImportanceSamplingWeight(const std::vector<double>& path) const {
    const double dt = option_->timeToMaturity / static_cast<double>(numSteps_);
    const double sigma = option_->volatility;
    const double driftAdjust = sigma * getOptimalDrift();
    
    // Calculate likelihood ratio for importance sampling
    double logLikelihoodRatio = 0.0;
    for (size_t i = 1; i <= numSteps_; ++i) {
        const double logReturn = std::log(path[i] / path[i-1]);
        logLikelihoodRatio -= driftAdjust * (logReturn / (sigma * std::sqrt(dt)));
        logLikelihoodRatio -= 0.5 * driftAdjust * driftAdjust * dt;
    }
    return std::exp(logLikelihoodRatio);
}

double MonteCarloPricer::priceWithImportanceSampling() {
    pathPrices_.clear();
    pathPrices_.reserve(numPaths_);
    
    optimalDrift_ = getOptimalDrift();
    
    #pragma omp parallel if(numPaths_ > 10000)
    {
        std::vector<double> localPrices;
        const size_t localSize = numPaths_ / omp_get_max_threads() + 1;
        localPrices.reserve(localSize);
        
        #pragma omp for nowait
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generatePathWithImportanceSampling();
            double pathPayoff = option_->payoff(path);
            double weight = getImportanceSamplingWeight(path);
            localPrices.push_back(pathPayoff * weight);
        }
        
        #pragma omp critical
        {
            pathPrices_.insert(pathPrices_.end(), localPrices.begin(), localPrices.end());
        }
    }
    
    const double mean = std::accumulate(pathPrices_.begin(), pathPrices_.end(), 0.0) / numPaths_;
    return mean * std::exp(-option_->riskFreeRate * option_->timeToMaturity);
}

double MonteCarloPricer::priceWithControlVariate() {
    // For European options, we can use geometric average as control variate
    pathPrices_.clear();
    pathPrices_.reserve(numPaths_);
    
    std::vector<double> controlVariates;
    controlVariates.reserve(numPaths_);
    
    double sumXY = 0.0;  // Sum of product of payoff and control
    double sumX = 0.0;   // Sum of payoffs
    double sumY = 0.0;   // Sum of controls
    double sumYY = 0.0;  // Sum of squared controls
    
    #pragma omp parallel reduction(+:sumXY,sumX,sumY,sumYY)
    {
        std::vector<double> localPrices;
        std::vector<double> localControls;
        const size_t localSize = numPaths_ / omp_get_max_threads() + 1;
        localPrices.reserve(localSize);
        localControls.reserve(localSize);
        
        #pragma omp for nowait
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generatePath(false, true);  // Use quasi-random numbers
            double payoff = option_->payoff(path);
            double control = calculateGeometricMean(path);  // Geometric mean as control
            
            sumXY += payoff * control;
            sumX += payoff;
            sumY += control;
            sumYY += control * control;
            
            localPrices.push_back(payoff);
            localControls.push_back(control);
        }
        
        #pragma omp critical
        {
            pathPrices_.insert(pathPrices_.end(), localPrices.begin(), localPrices.end());
            controlVariates.insert(controlVariates.end(), localControls.begin(), localControls.end());
        }
    }
    
    // Calculate optimal beta for variance reduction
    const double n = static_cast<double>(numPaths_);
    const double covXY = (sumXY - sumX * sumY / n) / (n - 1);
    const double varY = (sumYY - sumY * sumY / n) / (n - 1);
    const double beta = -covXY / varY;
    
    // Calculate geometric average option price (closed form)
    const double T = option_->timeToMaturity;
    const double r = option_->riskFreeRate;
    const double sigma = option_->volatility;
    const double S0 = option_->spot;
    const double K = option_->strike;
    
    const double sigmaSq = sigma * sigma;
    const double muG = (r - 0.5 * sigmaSq) * T;
    const double sigmaG = sigma * std::sqrt(T / 3.0);  // Reduced variance for geometric average
    
    const double d1 = (std::log(S0/K) + (muG + 0.5 * sigmaG * sigmaG)) / (sigmaG);
    const double d2 = d1 - sigmaG;
    
    const double geometricPrice = std::exp(-r * T) * 
        (S0 * std::exp(muG + 0.5 * sigmaG * sigmaG) * normalCDF(d1) - 
         K * normalCDF(d2));
    
    // Apply control variate correction
    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum)
    for (size_t i = 0; i < numPaths_; ++i) {
        sum += pathPrices_[i] + beta * (controlVariates[i] - geometricPrice);
    }
    
    double mean = sum / numPaths_;
    return std::exp(-r * T) * mean;
}

} // namespace hybrid_pricer
