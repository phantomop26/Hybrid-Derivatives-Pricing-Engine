#pragma once

#include "option.hpp"
#include "vanilla_options.hpp"
#include "two_asset_options.hpp"
#include "sobol_generator.hpp"
#include <vector>
#include <memory>
#include <random>
#include <limits>
#include <cmath>
#include <ostream>
#include <istream>
#include <algorithm>

namespace hybrid_pricer {

class MonteCarloPricer {
public:
    MonteCarloPricer(std::shared_ptr<Option> option, size_t numPaths, size_t numSteps);
    
    double price(bool useAntithetic = false, bool useQuasiRandom = false);
    double priceWithSobol();  // Quasi-Monte Carlo with Sobol sequences
    
    // Add importance sampling to pricing
    double priceWithImportanceSampling();
    
    // Control variate methods
    double priceWithControlVariates(bool useAntithetic = false, bool useQuasiRandom = false) {
        if (auto asian = std::dynamic_pointer_cast<AsianOption>(option_)) {
            if (asian->getType().find("Arithmetic") != std::string::npos) {
                return priceAsianWithGeometricControl(useAntithetic, useQuasiRandom);
            }
        }
        return price(useAntithetic, useQuasiRandom);
    }
    
    double priceWithControlVariate();
    
    std::vector<double> getConfidenceInterval() const;
    double getStandardError() const { return standardError_; }
    
private:
    std::vector<double> generatePath(bool useAntithetic, bool useQuasiRandom);
    std::vector<double> generateAntitheticPath(const std::vector<double>& path);
    std::vector<double> generateSobolPath(SobolGenerator& sobol);
    
    // Importance sampling helpers
    std::vector<double> generatePathWithImportanceSampling();
    double getImportanceSamplingWeight(const std::vector<double>& path) const;
    double getOptimalDrift() const;

    double calculateGeometricMean(const std::vector<double>& path) const;

    double priceAsianWithGeometricControl(bool useAntithetic, bool useQuasiRandom) {
        double sum = 0.0;
        double sumControlVariate = 0.0;
        double sumSquared = 0.0;
        std::vector<double> controlVariateValues;
        controlVariateValues.reserve(numPaths_);
        
        // Create geometric Asian option for control variate
        auto asian = std::dynamic_pointer_cast<AsianOption>(option_);
        auto geometricAsian = std::make_shared<AsianOption>(
            AsianOption::Type::Geometric,
            asian->getOptionType(),
            asian->strike,
            asian->timeToMaturity,
            asian->spot,
            asian->volatility,
            asian->riskFreeRate
        );

        // Calculate paths and control variate
        for (size_t i = 0; i < numPaths_; ++i) {
            auto path = generatePath(useAntithetic, useQuasiRandom);
            double payoff = option_->payoff(path);
            double controlPayoff = geometricAsian->payoff(path);
            
            controlVariateValues.push_back(controlPayoff);
            sum += payoff;
            sumControlVariate += controlPayoff;
            sumSquared += payoff * payoff;
        }

        // Calculate means
        double meanY = sum / numPaths_;
        double meanC = sumControlVariate / numPaths_;
        
        // Calculate analytical value for geometric Asian
        double analyticalGeometric = calculateGeometricAsianPrice(geometricAsian);
        
        // Calculate optimal beta
        double beta = calculateOptimalBeta(controlVariateValues, analyticalGeometric);
        
        // Apply control variate correction
        double price = meanY - beta * (meanC - analyticalGeometric);
        
        // Calculate error
        double variance = (sumSquared / numPaths_ - meanY * meanY) / (numPaths_ - 1);
        standardError_ = std::sqrt(variance / numPaths_);
        
        return std::exp(-option_->riskFreeRate * option_->timeToMaturity) * price;
    }

    double calculateOptimalBeta(const std::vector<double>& controlVariates, double expectedValue) {
        double meanC = std::accumulate(controlVariates.begin(), controlVariates.end(), 0.0) / controlVariates.size();
        double covariance = 0.0;
        double varianceC = 0.0;
        
        for (size_t i = 0; i < numPaths_; ++i) {
            double devC = controlVariates[i] - meanC;
            covariance += devC * (pathPrices_[i] - expectedValue);
            varianceC += devC * devC;
        }
        
        covariance /= (numPaths_ - 1);
        varianceC /= (numPaths_ - 1);
        
        return -covariance / varianceC;
    }

    double calculateGeometricAsianPrice(std::shared_ptr<AsianOption> geometricAsian) {
        // Analytical formula for geometric Asian option
        double dt = geometricAsian->timeToMaturity / numSteps_;
        double adjSigma = geometricAsian->volatility * std::sqrt((numSteps_ + 1) * (2 * numSteps_ + 1) / (6 * numSteps_ * numSteps_));
        double adjDrift = (geometricAsian->riskFreeRate - 0.5 * geometricAsian->volatility * geometricAsian->volatility) * 
                         (numSteps_ + 1) / (2 * numSteps_);
        
        double d1 = (std::log(geometricAsian->spot / geometricAsian->strike) + 
                    (adjDrift + 0.5 * adjSigma * adjSigma) * geometricAsian->timeToMaturity) / 
                   (adjSigma * std::sqrt(geometricAsian->timeToMaturity));
        double d2 = d1 - adjSigma * std::sqrt(geometricAsian->timeToMaturity);
        
        if (geometricAsian->getOptionType() == Option::OptionType::Call) {
            return std::exp(-geometricAsian->riskFreeRate * geometricAsian->timeToMaturity) * 
                   (geometricAsian->spot * std::exp(adjDrift * geometricAsian->timeToMaturity) * normalCDF(d1) - 
                    geometricAsian->strike * normalCDF(d2));
        } else {
            return std::exp(-geometricAsian->riskFreeRate * geometricAsian->timeToMaturity) * 
                   (geometricAsian->strike * normalCDF(-d2) - 
                    geometricAsian->spot * std::exp(adjDrift * geometricAsian->timeToMaturity) * normalCDF(-d1));
        }
    }

    double normalCDF(double x) const {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    std::vector<double> generateTwoAssetPath(bool useAntithetic, bool useQuasiRandom,
                                           double spot2, double sigma2, double correlation);

    std::shared_ptr<Option> option_;
    std::size_t numPaths_;
    std::size_t numSteps_;
    std::vector<double> pathPrices_;
    
    std::random_device rd_;
    std::mt19937_64 gen_{rd_()};
    std::normal_distribution<double> normalDist_{0.0, 1.0};
    
    double standardError_;
    
    // Store optimal drift for importance sampling
    mutable double optimalDrift_;
};

} // namespace hybrid_pricer
