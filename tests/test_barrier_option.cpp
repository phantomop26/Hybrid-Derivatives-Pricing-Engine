#include <gtest/gtest.h>
#include "vanilla_options.hpp"
#include "monte_carlo_pricer.hpp"

TEST(BarrierOptionTest, KnockOutProperties) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double barrier = 120.0;
    
    auto upAndOut = std::make_shared<hybrid_pricer::BarrierOption>(
        hybrid_pricer::BarrierOption::Type::UpAndOut,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r, barrier);
        
    // Test case 1: Path that doesn't hit barrier
    std::vector<double> path1 = {100.0, 105.0, 110.0, 115.0, 110.0};
    double payoff1 = upAndOut->payoff(path1);
    EXPECT_GT(payoff1, 0.0);  // Should pay off as barrier not hit
    
    // Test case 2: Path that hits barrier
    std::vector<double> path2 = {100.0, 110.0, 121.0, 115.0, 110.0};
    double payoff2 = upAndOut->payoff(path2);
    EXPECT_EQ(payoff2, 0.0);  // Should be knocked out
}

TEST(BarrierOptionTest, InOutParity) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double barrier = 120.0;
    size_t numPaths = 100000;
    size_t numSteps = 252;
    
    // Create up-and-in and up-and-out call options
    auto upAndIn = std::make_shared<hybrid_pricer::BarrierOption>(
        hybrid_pricer::BarrierOption::Type::UpAndIn,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r, barrier);
        
    auto upAndOut = std::make_shared<hybrid_pricer::BarrierOption>(
        hybrid_pricer::BarrierOption::Type::UpAndOut,
        hybrid_pricer::BarrierOption::OptionType::Call,
        K, T, S, sigma, r, barrier);
        
    // Create vanilla call for comparison
    auto vanilla = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    // Price all options
    hybrid_pricer::MonteCarloPricer mcIn(upAndIn, numPaths, numSteps);
    hybrid_pricer::MonteCarloPricer mcOut(upAndOut, numPaths, numSteps);
    hybrid_pricer::MonteCarloPricer mcVanilla(vanilla, numPaths, numSteps);
    
    double inPrice = mcIn.price(true, true);
    double outPrice = mcOut.price(true, true);
    double vanillaPrice = mcVanilla.price(true, true);
    
    // Verify in-out parity: price of vanilla = price of up-and-in + price of up-and-out
    EXPECT_NEAR(inPrice + outPrice, vanillaPrice, 0.1);
}
