# Hybrid Derivatives Pricing Engine

A high-performance C++ library for pricing financial derivatives using both Monte Carlo simulation and PDE methods.

## Features

- Monte Carlo Simulation Engine
  - European options (calls/puts)
  - Exotic options (Asian, barrier, lookback)
  - Variance reduction techniques (antithetic variates, control variates)
  - Quasi-Monte Carlo with Sobol sequences
  - OpenMP parallelization

- PDE Solver Module
  - Explicit Euler method
  - Crank-Nicolson scheme
  - ADI for 2D PDEs
  - American option pricing via free-boundary conditions

## Building the Project

Prerequisites:
- CMake 3.15 or higher
- C++17 compatible compiler
- OpenMP
- Google Test (for running tests)
- nlohmann/json (for config parsing)

```bash
mkdir build && cd build
cmake ..
make -j
```

## Running the Pricer

Create a JSON configuration file:

```json
{
    "option": "european_call",
    "method": "both",
    "strike": 100.0,
    "spot": 100.0,
    "maturity": 1.0,
    "volatility": 0.2,
    "rate": 0.05,
    "paths": 1000000,
    "steps": 252,
    "timesteps": 1000,
    "spacesteps": 200
}
```

Run the pricer:
```bash
./pricer config.json
```

## Running Tests

```bash
cd build
ctest
```

## Performance Benchmarks

The engine includes performance benchmarks comparing:
- Monte Carlo vs PDE methods
- CPU vs GPU implementations
- Different variance reduction techniques
