#include <gtest/gtest.h>
#include "monte_carlo_pricer.hpp"
#include "vanilla_options.hpp"

TEST(MonteCarloTest, VarianceReduction) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::MonteCarloPricer mc(call, 100000, 252);
    
    // Price without variance reduction
    double price1 = mc.price(false, false);
    auto ci1 = mc.getConfidenceInterval();
    double width1 = ci1[1] - ci1[0];
    
    // Price with antithetic variates
    double price2 = mc.price(true, false);
    auto ci2 = mc.getConfidenceInterval();
    double width2 = ci2[1] - ci2[0];
    
    // Confidence interval should be narrower with variance reduction
    EXPECT_LT(width2, width1);
    
    // Both prices should be within each other's confidence intervals
    EXPECT_TRUE(price1 >= ci2[0] && price1 <= ci2[1]);
    EXPECT_TRUE(price2 >= ci1[0] && price2 <= ci1[1]);
}

TEST(MonteCarloTest, ImportanceSampling) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    const size_t numPaths = 100000;
    const size_t numSteps = 252;
    
    // Price with standard Monte Carlo
    hybrid_pricer::MonteCarloPricer mc1(call, numPaths, numSteps);
    double price1 = mc1.price(false, false);
    auto ci1 = mc1.getConfidenceInterval();
    double width1 = ci1[1] - ci1[0];
    
    // Price with importance sampling
    hybrid_pricer::MonteCarloPricer mc2(call, numPaths, numSteps);
    double price2 = mc2.priceWithImportanceSampling();
    auto ci2 = mc2.getConfidenceInterval();
    double width2 = ci2[1] - ci2[0];
    
    // Calculate Black-Scholes price for reference
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S * std::erf(d1/sqrt(2.0)) - 
                     K * std::exp(-r*T) * std::erf(d2/sqrt(2.0));
    
    // Both prices should be within each other's confidence intervals
    EXPECT_TRUE(price1 >= ci2[0] && price1 <= ci2[1]);
    EXPECT_TRUE(price2 >= ci1[0] && price2 <= ci1[1]);
    
    // Importance sampling should give narrower confidence interval
    EXPECT_LT(width2, width1);
    
    // Both methods should be close to Black-Scholes price
    EXPECT_NEAR(price1, bs_price, 0.5);
    EXPECT_NEAR(price2, bs_price, 0.5);
}

TEST(MonteCarloTest, ControlVariate) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    const size_t numPaths = 100000;
    const size_t numSteps = 252;
    
    hybrid_pricer::MonteCarloPricer mc(call, numPaths, numSteps);
    
    // Price with standard Monte Carlo
    double price1 = mc.price(false, false);
    auto ci1 = mc.getConfidenceInterval();
    double width1 = ci1[1] - ci1[0];
    
    // Price with control variate
    double price2 = mc.priceWithControlVariate();
    auto ci2 = mc.getConfidenceInterval();
    double width2 = ci2[1] - ci2[0];
    
    // Control variate should reduce variance
    EXPECT_LT(width2, width1);
    
    // Both prices should be close to each other
    EXPECT_NEAR(price1, price2, 0.5);
    
    // Compare with Black-Scholes
    double d1 = (std::log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*std::sqrt(T));
    double d2 = d1 - sigma*std::sqrt(T);
    double bs_price = S*std::erf(d1/std::sqrt(2.0)) - K*std::exp(-r*T)*std::erf(d2/std::sqrt(2.0));
    
    // Both prices should be close to Black-Scholes
    EXPECT_NEAR(price1, bs_price, 0.5);
    EXPECT_NEAR(price2, bs_price, 0.5);
}
