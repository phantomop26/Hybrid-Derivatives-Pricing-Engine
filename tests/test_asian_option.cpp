#include <gtest/gtest.h>
#include "vanilla_options.hpp"
#include "monte_carlo_pricer.hpp"

TEST(AsianOptionTest, GeometricVsArithmetic) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto arithmeticCall = std::make_shared<hybrid_pricer::AsianOption>(
        hybrid_pricer::AsianOption::Type::Arithmetic,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
        
    auto geometricCall = std::make_shared<hybrid_pricer::AsianOption>(
        hybrid_pricer::AsianOption::Type::Geometric,
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::MonteCarloPricer mcArith(arithmeticCall, 100000, 252);
    hybrid_pricer::MonteCarloPricer mcGeom(geometricCall, 100000, 252);
    
    double arithPrice = mcArith.price(true, true);  // Use variance reduction
    double geomPrice = mcGeom.price(true, true);
    
    // Geometric Asian option price should be lower than arithmetic
    // due to Jensen's inequality
    EXPECT_LT(geomPrice, arithPrice);
    
    // The difference should not be too large
    EXPECT_LT(arithPrice - geomPrice, 2.0);
}
