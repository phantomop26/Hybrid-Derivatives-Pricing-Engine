#include "pde_solver.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace hybrid_pricer {

using std::vector;
using std::max;
using std::exp;

PDESolver::PDESolver(std::shared_ptr<Option> option, size_t numTimeSteps, size_t numSpaceSteps)
    : option_(option), numTimeSteps_(numTimeSteps), numSpaceSteps_(numSpaceSteps) {
    grid_.resize(numTimeSteps_);
    for (auto& row : grid_) {
        row.resize(numSpaceSteps_);
    }
    setupGrid();
}

void PDESolver::setupGrid() {
    setInitialCondition();
    setBoundaryConditions();
}

void PDESolver::setInitialCondition() {
    const double sMax = 4.0 * option_->spot;
    const double K = option_->strike;
    const double dx = sMax / static_cast<double>(numSpaceSteps_ - 1);
    
    // Set terminal conditions (payoff at maturity)
    for (size_t j = 0; j < numSpaceSteps_; ++j) {
        const double S = j * dx;
        if (option_->getOptionType() == Option::OptionType::Call) {
            grid_[numTimeSteps_ - 1][j] = std::max(S - K, 0.0);
        } else {  // Put option
            grid_[numTimeSteps_ - 1][j] = std::max(K - S, 0.0);
        }
    }
}

void PDESolver::setBoundaryConditions() {
    const double sigma = option_->volatility;
    const double r = option_->riskFreeRate;
    const double T = option_->timeToMaturity;
    const double sMax = 4.0 * option_->spot;
    const double K = option_->strike;
    const double dt = T / static_cast<double>(numTimeSteps_ - 1);
    
    // Set boundary conditions for each time level
    for (size_t i = 0; i < numTimeSteps_; ++i) {
        const double tau = T * (numTimeSteps_ - 1 - i) / (numTimeSteps_ - 1);
        
        if (option_->getOptionType() == Option::OptionType::Call) {
            // At S = 0, call option is worthless
            grid_[i][0] = 0.0;
            // At S = Smax, call option approaches S - K*e^(-rT)
            grid_[i][numSpaceSteps_ - 1] = sMax - K * std::exp(-r * tau);
        } else {  // Put option
            // At S = 0, put option is worth K*e^(-rT)
            grid_[i][0] = K * std::exp(-r * tau);
            // At S = Smax, put option is worthless
            grid_[i][numSpaceSteps_ - 1] = 0.0;
        }
    }
}

void thomasAlgorithm(std::vector<double>& a, std::vector<double>& b, 
                     std::vector<double>& c, std::vector<double>& x,
                     std::vector<double>& d) {
    const size_t n = d.size();
    
    // Forward elimination
    for (size_t i = 1; i < n; i++) {
        const double m = a[i] / b[i-1];
        b[i] -= m * c[i-1];
        d[i] -= m * d[i-1];
    }
    
    // Back substitution
    x[n-1] = d[n-1] / b[n-1];
    for (size_t i = n-1; i-- > 0;) {
        x[i] = (d[i] - c[i] * x[i+1]) / b[i];
    }
}

double PDESolver::solveCrankNicolson() {
    const double sigma = option_->volatility;
    const double r = option_->riskFreeRate;
    const double T = option_->timeToMaturity;
    const double sMax = 4.0 * option_->spot;
    const double K = option_->strike;
    const double dx = sMax / static_cast<double>(numSpaceSteps_ - 1);
    const double dt = T / static_cast<double>(numTimeSteps_ - 1);
    
    const size_t M = numSpaceSteps_ - 2;
    std::vector<double> a(M), b(M), c(M), d(M), x(M);
    
    // Work backwards in time
    for (size_t i = numTimeSteps_ - 1; i > 0; --i) {
        // Set up tridiagonal system
        for (size_t j = 0; j < M; ++j) {
            const double S = (j + 1) * dx;
            const double alpha = 0.25 * sigma * sigma * S * S / (dx * dx);
            const double beta = 0.25 * r * S / dx;
            
            a[j] = -alpha + beta;
            b[j] = 1.0/dt + 2.0*alpha + 0.5*r;
            c[j] = -alpha - beta;
            
            const size_t jp = j + 1;
            d[j] = grid_[i][jp] * (1.0/dt - 2.0*alpha - 0.5*r) + 
                   alpha * (grid_[i][jp+1] + grid_[i][jp-1]) +
                   beta * (grid_[i][jp+1] - grid_[i][jp-1]);
        }
        
        thomasAlgorithm(a, b, c, x, d);
        
        // Copy solution to grid
        for (size_t j = 0; j < M; ++j) {
            grid_[i-1][j+1] = x[j];
            
            // Apply early exercise constraint for American options
            if (option_->getOptionType() == Option::OptionType::Put) {
                const double S = (j + 1) * dx;
                grid_[i-1][j+1] = std::max(grid_[i-1][j+1], K - S);
            }
        }
        setBoundaryConditions();
    }
    
    // Interpolate to get price at spot price
    const size_t idx = std::min(static_cast<size_t>(option_->spot / dx), numSpaceSteps_ - 2);
    const double lambda = std::max(0.0, std::min(1.0, (option_->spot - idx * dx) / dx));
    
    return (1.0 - lambda) * grid_[0][idx] + lambda * grid_[0][idx + 1];
}

double PDESolver::solveExplicitEuler() {
    const double sigma = option_->volatility;
    const double r = option_->riskFreeRate;
    const double T = option_->timeToMaturity;
    const double sMax = 4.0 * option_->spot;
    const double K = option_->strike;
    const double dx = sMax / static_cast<double>(numSpaceSteps_ - 1);
    const double dt = T / static_cast<double>(numTimeSteps_ - 1);
    
    // Check stability condition
    const double maxS = sMax;
    const double stability = (sigma * sigma * maxS * maxS * dt) / (dx * dx) + (std::abs(r) * maxS * dt) / dx;
    if (stability > 0.25) {  // Conservative condition
        throw std::runtime_error("Explicit Euler scheme is unstable. Reduce dt or increase dx.");
    }
    
    // Work backwards in time
    for (size_t i = numTimeSteps_ - 1; i > 0; --i) {
        for (size_t j = 1; j < numSpaceSteps_ - 1; ++j) {
            const double S = j * dx;
            const double alpha = 0.5 * sigma * sigma * S * S / (dx * dx);
            const double beta = r * S / (2.0 * dx);
            
            grid_[i-1][j] = grid_[i][j] + dt * (
                alpha * (grid_[i][j+1] - 2.0 * grid_[i][j] + grid_[i][j-1]) +
                beta * (grid_[i][j+1] - grid_[i][j-1]) -
                r * grid_[i][j]
            );
            
            // Apply early exercise constraint for American options
            if (option_->getOptionType() == Option::OptionType::Put) {
                grid_[i-1][j] = std::max(grid_[i-1][j], K - S);
            }
        }
        setBoundaryConditions();
    }
    
    // Interpolate to get price at spot price
    const size_t idx = std::min(static_cast<size_t>(option_->spot / dx), numSpaceSteps_ - 2);
    const double lambda = std::max(0.0, std::min(1.0, (option_->spot - idx * dx) / dx));
    
    return (1.0 - lambda) * grid_[0][idx] + lambda * grid_[0][idx + 1];
}

double PDESolver::solveADI() {
    const double sigma = option_->volatility;
    const double r = option_->riskFreeRate;
    const double T = option_->timeToMaturity;
    const double sMax = 4.0 * option_->spot;
    const double K = option_->strike;
    const double dx = sMax / static_cast<double>(numSpaceSteps_ - 1);
    const double dt = T / static_cast<double>(numTimeSteps_ - 1);
    
    std::vector<std::vector<double>> temp_grid = grid_;
    const size_t M = numSpaceSteps_ - 2;
    std::vector<double> a(M), b(M), c(M), d(M), x(M);
    
    // Work backwards in time
    for (size_t i = numTimeSteps_ - 1; i > 0; --i) {
        // First half-step: implicit in x-direction
        for (size_t j = 0; j < M; ++j) {
            const double S = (j + 1) * dx;
            const double alpha = 0.5 * sigma * sigma * S * S / (dx * dx);
            const double beta = 0.5 * r * S / dx;
            
            a[j] = -alpha + beta;
            b[j] = 1.0/dt + 2.0*alpha + r;
            c[j] = -alpha - beta;
            
            const size_t jp = j + 1;
            d[j] = grid_[i][jp]/dt;
        }
        
        thomasAlgorithm(a, b, c, x, d);
        
        for (size_t j = 0; j < M; ++j) {
            temp_grid[i-1][j+1] = x[j];
        }
        
        // Second half-step: explicit update
        for (size_t j = 1; j < numSpaceSteps_ - 1; ++j) {
            const double S = j * dx;
            const double alpha = 0.5 * sigma * sigma * S * S / (dx * dx);
            const double beta = 0.5 * r * S / dx;
            
            grid_[i-1][j] = temp_grid[i-1][j] + 0.5 * dt * (
                (alpha + beta) * (temp_grid[i-1][j+1] - temp_grid[i-1][j]) +
                (alpha - beta) * (temp_grid[i-1][j-1] - temp_grid[i-1][j])
            );
            
            // Apply early exercise constraint for American options
            if (option_->getOptionType() == Option::OptionType::Put) {
                grid_[i-1][j] = std::max(grid_[i-1][j], K - S);
            }
        }
        setBoundaryConditions();
    }
    
    // Interpolate to get price at spot price
    const size_t idx = std::min(static_cast<size_t>(option_->spot / dx), numSpaceSteps_ - 2);
    const double lambda = std::max(0.0, std::min(1.0, (option_->spot - idx * dx) / dx));
    
    return (1.0 - lambda) * grid_[0][idx] + lambda * grid_[0][idx + 1];
}

double PDESolver::solveWithFreeBoundary() {
    return solveCrankNicolson();  // For now, use CN with early exercise constraint
}

std::vector<std::vector<double>> PDESolver::getSolutionGrid() const {
    return grid_;
}

} // namespace hybrid_pricer
