#pragma once

#include "option.hpp"
#include <vector>
#include <memory>

namespace hybrid_pricer {

class PDESolver {
public:
    PDESolver(std::shared_ptr<Option> option, size_t numTimeSteps, size_t numSpaceSteps);
    
    // Different numerical schemes
    double solveExplicitEuler();
    double solveCrankNicolson();
    double solveADI();  // For 2D problems
    
    // For American options
    double solveWithFreeBoundary();
    
    // Get the full solution grid for visualization/analysis
    std::vector<std::vector<double>> getSolutionGrid() const;
    
private:
    std::shared_ptr<Option> option_;
    size_t numTimeSteps_;
    size_t numSpaceSteps_;
    std::vector<std::vector<double>> grid_;
    
    void setupGrid();
    void setInitialCondition();
    void setBoundaryConditions();
};

} // namespace hybrid_pricer
