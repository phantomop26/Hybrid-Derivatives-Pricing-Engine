#include <gtest/gtest.h>
#include "vanilla_options.hpp"
#include "monte_carlo_pricer.hpp"

TEST(LookbackOptionTest, FixedStrikeProperties) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    size_t numPaths = 100000;
    size_t numSteps = 252;
    
    auto fixedCall = std::make_shared<hybrid_pricer::LookbackOption>(
        hybrid_pricer::LookbackOption::Type::Fixed,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
        
    auto fixedPut = std::make_shared<hybrid_pricer::LookbackOption>(
        hybrid_pricer::LookbackOption::Type::Fixed,
        hybrid_pricer::Option::OptionType::Put,
        K, T, S, sigma, r);
    
    // Test basic properties
    std::vector<double> path1 = {100.0, 110.0, 120.0, 115.0, 105.0};
    EXPECT_EQ(fixedCall->payoff(path1), 20.0);  // max = 120, K = 100
    EXPECT_EQ(fixedPut->payoff(path1), 0.0);    // min = 100, K = 100
    
    std::vector<double> path2 = {100.0, 90.0, 80.0, 85.0, 95.0};
    EXPECT_EQ(fixedCall->payoff(path2), 0.0);    // max = 100, K = 100
    EXPECT_EQ(fixedPut->payoff(path2), 20.0);   // min = 80, K = 100
    
    // Price and verify reasonable values
    hybrid_pricer::MonteCarloPricer mcCall(fixedCall, numPaths, numSteps);
    hybrid_pricer::MonteCarloPricer mcPut(fixedPut, numPaths, numSteps);
    
    double callPrice = mcCall.price(true, true);
    double putPrice = mcPut.price(true, true);
    
    EXPECT_GT(callPrice, 0.0);
    EXPECT_GT(putPrice, 0.0);
    EXPECT_LT(callPrice, S * 2);  // Sanity check
    EXPECT_LT(putPrice, K * 2);   // Sanity check
}

TEST(LookbackOptionTest, FloatingStrikeProperties) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto floatingCall = std::make_shared<hybrid_pricer::LookbackOption>(
        hybrid_pricer::LookbackOption::Type::Floating,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
        
    auto floatingPut = std::make_shared<hybrid_pricer::LookbackOption>(
        hybrid_pricer::LookbackOption::Type::Floating,
        hybrid_pricer::Option::OptionType::Put,
        K, T, S, sigma, r);
    
    // Test that floating strike lookback always pays off
    std::vector<double> path1 = {100.0, 110.0, 120.0, 115.0, 105.0};
    EXPECT_GT(floatingCall->payoff(path1), 0.0);
    EXPECT_GT(floatingPut->payoff(path1), 0.0);
    
    std::vector<double> path2 = {100.0, 90.0, 80.0, 85.0, 95.0};
    EXPECT_GT(floatingCall->payoff(path2), 0.0);
    EXPECT_GT(floatingPut->payoff(path2), 0.0);
}

TEST(AsianOptionTest, ControlVariateEfficiency) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    size_t numPaths = 10000;
    size_t numSteps = 52;
    
    auto arithmeticCall = std::make_shared<hybrid_pricer::AsianOption>(
        hybrid_pricer::AsianOption::Type::Arithmetic,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::MonteCarloPricer mc(arithmeticCall, numPaths, numSteps);
    
    // Price with and without control variates
    double priceStandard = mc.price(true, true);
    double priceCV = mc.priceWithControlVariates(true, true);
    
    // Standard error with CV should be lower
    double standardError = mc.getStandardError();
    mc.priceWithControlVariates(true, true);
    double standardErrorCV = mc.getStandardError();
    
    EXPECT_LT(standardErrorCV, standardError);  // Control variate should reduce variance
    EXPECT_NEAR(priceCV, priceStandard, 2.0);  // Prices should be similar
}
