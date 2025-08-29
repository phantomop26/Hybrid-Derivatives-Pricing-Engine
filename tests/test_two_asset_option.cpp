#include <gtest/gtest.h>
#include "two_asset_options.hpp"
#include "monte_carlo_pricer.hpp"
#include <memory>

TEST(TwoAssetOptionTest, SpreadOptionProperties) {
    double K = 0.0;  // ATM spread option
    double T = 1.0;
    double S1 = 100.0;
    double S2 = 100.0;
    double sigma1 = 0.2;
    double sigma2 = 0.3;
    double rho = 0.5;
    double r = 0.05;
    
    auto spreadCall = std::make_shared<hybrid_pricer::TwoAssetOption>(
        hybrid_pricer::TwoAssetOption::Type::Spread,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S1, S2, sigma1, sigma2, rho, r);
    
    // Test basic properties
    std::vector<double> path = {100.0, 100.0, 110.0, 105.0, 120.0, 110.0};
    EXPECT_EQ(spreadCall->payoff(path), 10.0);  // 120 - 110 = 10
    
    // Price and verify reasonable values
    hybrid_pricer::MonteCarloPricer mc(spreadCall, 100000, 252);
    double price = mc.price(true, true);
    
    // Spread option value should increase with:
    // - Higher vol of first asset
    // - Lower vol of second asset
    // - Lower correlation
    
    auto spreadCall2 = std::make_shared<hybrid_pricer::TwoAssetOption>(
        hybrid_pricer::TwoAssetOption::Type::Spread,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S1, S2, sigma1 * 1.2, sigma2, rho, r);
    
    hybrid_pricer::MonteCarloPricer mc2(spreadCall2, 100000, 252);
    double price2 = mc2.price(true, true);
    
    EXPECT_GT(price2, price);  // Higher vol1 should increase price
}

TEST(TwoAssetOptionTest, BasketOptionProperties) {
    double K = 100.0;
    double T = 1.0;
    double S1 = 100.0;
    double S2 = 100.0;
    double sigma1 = 0.2;
    double sigma2 = 0.2;
    double rho = 0.5;
    double r = 0.05;
    
    auto basketCall = std::make_shared<hybrid_pricer::TwoAssetOption>(
        hybrid_pricer::TwoAssetOption::Type::BasketCall,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S1, S2, sigma1, sigma2, rho, r);
    
    // Test payoff calculation
    std::vector<double> path = {100.0, 100.0, 110.0, 90.0, 120.0, 100.0};
    EXPECT_EQ(basketCall->payoff(path), 10.0);  // (120 + 100)/2 - 100 = 10
    
    // Price and verify reasonable values
    hybrid_pricer::MonteCarloPricer mc(basketCall, 100000, 252);
    double price = mc.price(true, true);
    
    EXPECT_GT(price, 0.0);
    EXPECT_LT(price, S1);  // Price should be less than individual asset price
}

TEST(TwoAssetOptionTest, RainbowOptionProperties) {
    double K = 100.0;
    double T = 1.0;
    double S1 = 100.0;
    double S2 = 100.0;
    double sigma1 = 0.2;
    double sigma2 = 0.2;
    double rho = 0.5;
    double r = 0.05;
    
    auto rainbow = std::make_shared<hybrid_pricer::TwoAssetOption>(
        hybrid_pricer::TwoAssetOption::Type::Rainbow,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S1, S2, sigma1, sigma2, rho, r);
    
    // Test that rainbow option takes max of both assets
    std::vector<double> path = {100.0, 100.0, 110.0, 90.0, 120.0, 130.0};
    EXPECT_EQ(rainbow->payoff(path), 30.0);  // max(120, 130) - 100 = 30
    
    // Price and verify reasonable values
    hybrid_pricer::MonteCarloPricer mc(rainbow, 100000, 252);
    double price = mc.price(true, true);
    
    // Rainbow option should be worth more than individual calls
    auto european = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S1, sigma1, r);
    
    hybrid_pricer::MonteCarloPricer mcEuro(european, 100000, 252);
    double euroPrice = mcEuro.price(true, true);
    
    EXPECT_GT(price, euroPrice);
}
