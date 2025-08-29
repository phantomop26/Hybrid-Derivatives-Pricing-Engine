#include <gtest/gtest.h>
#include "sobol_generator.hpp"
#include "monte_carlo_pricer.hpp"
#include "vanilla_options.hpp"
#include <cmath>
#include <vector>
#include <numeric>

using namespace hybrid_pricer;

TEST(SobolTest, FirstDimension) {
    SobolGenerator sobol(1);
    std::vector<double> expected_first_points = {
        0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125
    };
    
    for (const auto& expected : expected_first_points) {
        auto point = sobol.next();
        ASSERT_EQ(point.size(), 1);
        EXPECT_NEAR(point[0], expected, 1e-6);
    }
}

TEST(SobolTest, Dimensionality) {
    const size_t dim = 5;
    SobolGenerator sobol(dim);
    auto point = sobol.next();
    EXPECT_EQ(point.size(), dim);
    
    // All points should be in [0,1]
    for (const auto& x : point) {
        EXPECT_GE(x, 0.0);
        EXPECT_LE(x, 1.0);
    }
}

TEST(SobolTest, SequenceUniformity) {
    SobolGenerator sobol(1);
    const size_t N = 1000;
    double sum = 0.0, sum_sq = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        auto point = sobol.next();
        sum += point[0];
        sum_sq += point[0] * point[0];
    }
    
    // Check mean is close to 0.5
    double mean = sum / N;
    EXPECT_NEAR(mean, 0.5, 0.05);
    
    // Check variance is close to 1/12 (uniform distribution on [0,1])
    double variance = sum_sq/N - mean*mean;
    EXPECT_NEAR(variance, 1.0/12.0, 0.05);
}

TEST(SobolPricingTest, EuropeanCallOption) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<EuropeanOption>(
        Option::OptionType::Call,
        K, T, S, sigma, r);
    
    // Price with Sobol sequences
    MonteCarloPricer mc_sobol(call, 10000, 252);
    double sobol_price = mc_sobol.priceWithSobol();
    
    // Price with standard Monte Carlo
    MonteCarloPricer mc_standard(call, 10000, 252);
    double std_price = mc_standard.price(false, false);
    
    // Calculate Black-Scholes price for reference
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S * std::erf(d1/sqrt(2.0)) - 
                     K * std::exp(-r*T) * std::erf(d2/sqrt(2.0));
    
    // Sobol should be more accurate than standard MC
    double sobol_error = std::abs(sobol_price - bs_price);
    double std_error = std::abs(std_price - bs_price);
    EXPECT_LT(sobol_error, std_error);
}

TEST(SobolPricingTest, AsianOption) {
    // TODO: Add test for Asian option pricing once implemented
    // This should compare convergence rates between standard MC and Sobol
}
