#include <gtest/gtest.h>
#include "pde_solver.hpp"
#include "vanilla_options.hpp"

TEST(PDESolverTest, EuropeanConvergence) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    // Solve with coarse grid
    hybrid_pricer::PDESolver solver1(call, 50, 100);
    double price1 = solver1.solveCrankNicolson();
    
    // Solve with finer grid
    hybrid_pricer::PDESolver solver2(call, 100, 200);
    double price2 = solver2.solveCrankNicolson();
    
    // Solve with even finer grid
    hybrid_pricer::PDESolver solver3(call, 200, 400);
    double price3 = solver3.solveCrankNicolson();
    
    // Check convergence order
    double ratio = std::abs(price2 - price1) / std::abs(price3 - price2);
    
    // For second-order convergence, ratio should be approximately 4
    EXPECT_NEAR(ratio, 4.0, 0.5);
}

TEST(PDESolverTest, ExplicitEuler) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    // Use fine grid to ensure stability
    hybrid_pricer::PDESolver solver(call, 1000, 200);
    double price = solver.solveExplicitEuler();
    
    // Compare with Black-Scholes price
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S*0.5*(1 + erf(d1/sqrt(2))) - K*exp(-r*T)*0.5*(1 + erf(d2/sqrt(2)));
    
    EXPECT_NEAR(price, bs_price, 0.1);
}

TEST(PDESolverTest, AmericanPutEarlyExercise) {
    double K = 100.0;
    double T = 1.0;
    double S = 90.0;  // Deep ITM put
    double r = 0.05;
    double sigma = 0.2;
    
    auto put = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Put,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(put, 100, 200);
    
    // Price with and without early exercise
    double american_price = solver.solveWithFreeBoundary();
    double european_price = solver.solveCrankNicolson();
    
    // American put should be worth more than European put
    EXPECT_GT(american_price, european_price);
    
    // Early exercise premium should be significant for this case
    EXPECT_GT(american_price - european_price, 0.5);
}

TEST(PDESolverTest, ADIAsianOption) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto asian = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(asian, 100, 100);
    double adi_price = solver.solveADI();
    
    // Asian option should be worth less than vanilla European
    double vanilla_price = solver.solveCrankNicolson();
    EXPECT_LT(adi_price, vanilla_price);
    
    // Price should be positive and reasonable
    EXPECT_GT(adi_price, 0.0);
    EXPECT_LT(adi_price, S);  // Call option price cannot exceed spot
}

TEST(PDESolverTest, MethodComparison) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(call, 200, 400);
    
    double cn_price = solver.solveCrankNicolson();
    double explicit_price = solver.solveExplicitEuler();
    
    // Both methods should give similar results
    EXPECT_NEAR(cn_price, explicit_price, 0.1);
}

TEST(PDESolverTest, StabilityCheck) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    // This should work (stable)
    EXPECT_NO_THROW({
        hybrid_pricer::PDESolver solver1(call, 1000, 200);
        solver1.solveExplicitEuler();
    });
    
    // This should throw (unstable)
    EXPECT_THROW({
        hybrid_pricer::PDESolver solver2(call, 10, 200);
        solver2.solveExplicitEuler();
    }, std::runtime_error);
}

TEST(PDESolverTest, BoundaryConditions) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(call, 100, 200);
    auto grid = solver.getSolutionGrid();
    
    // Check S = 0 boundary
    for (size_t i = 0; i < grid.size(); ++i) {
        EXPECT_NEAR(grid[i][0], 0.0, 1e-10);
    }
    
    // Check S = Smax boundary for call option
    for (size_t i = 0; i < grid.size(); ++i) {
        double tau = T * (grid.size() - 1 - i) / (grid.size() - 1);
        double sMax = 4.0 * S;
        double expected = sMax - K * std::exp(-r * tau);
        EXPECT_NEAR(grid[i][grid[0].size()-1], expected, 1.0);
    }
}

TEST(PDESolverTest, CompareAllMethods) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(call, 1000, 200);
    
    double explicit_price = solver.solveExplicitEuler();
    double cn_price = solver.solveCrankNicolson();
    double adi_price = solver.solveADI();
    
    // Calculate Black-Scholes price for reference
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S*0.5*(1 + erf(d1/sqrt(2))) - K*exp(-r*T)*0.5*(1 + erf(d2/sqrt(2)));
    
    // All methods should agree within reasonable tolerance
    EXPECT_NEAR(explicit_price, bs_price, 0.1);
    EXPECT_NEAR(cn_price, bs_price, 0.1);
    EXPECT_NEAR(adi_price, bs_price, 0.1);
    
    // Crank-Nicolson should be most accurate
    EXPECT_LT(std::abs(cn_price - bs_price), std::abs(explicit_price - bs_price));
}

TEST(PDESolverTest, HighVolatilityCase) {
    double K = 100.0;
    double T = 1.0;
    double S = 100.0;
    double r = 0.05;
    double sigma = 0.8;  // High volatility
    
    auto call = std::make_shared<hybrid_pricer::EuropeanOption>(
        hybrid_pricer::Option::OptionType::Call,
        K, T, S, sigma, r);
    
    hybrid_pricer::PDESolver solver(call, 200, 400);
    
    // Even with high volatility, Crank-Nicolson should be stable
    double cn_price = solver.solveCrankNicolson();
    
    // Calculate Black-Scholes price for reference
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    double bs_price = S*0.5*(1 + erf(d1/sqrt(2))) - K*exp(-r*T)*0.5*(1 + erf(d2/sqrt(2)));
    
    // Price should still be accurate despite high volatility
    EXPECT_NEAR(cn_price, bs_price, 0.5);
}
