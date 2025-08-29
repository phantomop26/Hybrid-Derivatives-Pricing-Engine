#include <gtest/gtest.h>
#include "vanilla_options.hpp"
#include <cmath>

// Black-Scholes formula for European options
double blackScholesPrice(bool isCall, double S, double K, double T, double r, double sigma) {
    double d1 = (std::log(S/K) + (r + sigma*sigma/2)*T) / (sigma*std::sqrt(T));
    double d2 = d1 - sigma*std::sqrt(T);
    
    double Nd1 = 0.5 * (1 + std::erf(d1/std::sqrt(2)));
    double Nd2 = 0.5 * (1 + std::erf(d2/std::sqrt(2)));
    
    if (isCall) {
        return S*Nd1 - K*std::exp(-r*T)*Nd2;
    } else {
        return K*std::exp(-r*T)*(1-Nd2) - S*(1-Nd1);
    }
}

TEST(EuropeanOptionTest, BlackScholesComparison) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
        
    auto put = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Put,
        K, T, S, sigma, r);
    
    std::vector<double> finalPath = {S * std::exp((r - 0.5*sigma*sigma)*T + sigma*std::sqrt(T))};
    
    double callPayoff = call->payoff(finalPath);
    double putPayoff = put->payoff(finalPath);
    
    double bsCall = blackScholesPrice(true, S, K, T, r, sigma);
    double bsPut = blackScholesPrice(false, S, K, T, r, sigma);
    
    EXPECT_NEAR(callPayoff, std::max(finalPath[0] - K, 0.0), 1e-10);
    EXPECT_NEAR(putPayoff, std::max(K - finalPath[0], 0.0), 1e-10);
    
    // Verify put-call parity
    EXPECT_NEAR(bsCall - bsPut, S - K*std::exp(-r*T), 1e-10);
}
