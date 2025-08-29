#include "monte_carlo_pricer.hpp"
#include "pde_solver.hpp"
#include "vanilla_options.hpp"
#include "gpu_accelerator.hpp"
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using nlohmann::json;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <config.json>" << endl;
        return 1;
    }
    
    try {
        // Read configuration
        ifstream config_file(argv[1]);
        json config;
        config_file >> config;
        
        // Parse option parameters
        double strike = config["strike"];
        double spot = config["spot"];
        double maturity = config["maturity"];
        double vol = config["volatility"];
        double rate = config["rate"];
        size_t numPaths = config["paths"];
        size_t numSteps = config["steps"];
        
        // Create appropriate option object
        std::shared_ptr<hybrid_pricer::Option> option;
        std::string option_type = config["option"];
        
        if (option_type == "european") {
            option = std::make_shared<hybrid_pricer::EuropeanOption>(
                hybrid_pricer::Option::OptionType::Call,
                strike, maturity, spot, vol, rate);
        } else if (option_type == "asian_arithmetic") {
            option = std::make_shared<hybrid_pricer::AsianOption>(
                hybrid_pricer::AsianOption::Type::Arithmetic,
                hybrid_pricer::Option::OptionType::Call,
                strike, maturity, spot, vol, rate);
        } else if (option_type == "barrier") {
            double barrier = config["barrier"];
            hybrid_pricer::BarrierOption::Type bType = 
                config["barrier_type"] == "up_out" ?
                hybrid_pricer::BarrierOption::Type::UpAndOut :
                hybrid_pricer::BarrierOption::Type::DownAndOut;
            
            option = std::make_shared<hybrid_pricer::BarrierOption>(
                bType,
                hybrid_pricer::Option::OptionType::Call,
                strike, maturity, spot, vol, rate, barrier);
        }
        
        // Price using specified method
        std::string method = config["method"];
        if (method == "montecarlo" || method == "both") {
            cout << "\nMonte Carlo Pricing:\n";
            cout << "------------------\n";
            
            // Standard Monte Carlo
            hybrid_pricer::MonteCarloPricer mc(option, numPaths, numSteps);
            
            auto start = high_resolution_clock::now();
            double price = mc.price(true, true);  // Use both variance reduction techniques
            auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
            
            cout << "CPU price: " << price << endl;
            auto ci = mc.getConfidenceInterval();
            cout << "95% CI: [" << ci[0] << ", " << ci[1] << "]" << endl;
            cout << "CPU time: " << duration.count() << "ms" << endl;
            
            // Importance sampling
            start = high_resolution_clock::now();
            price = mc.priceWithImportanceSampling();
            duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
            
            cout << "\nImportance sampling price: " << price << endl;
            ci = mc.getConfidenceInterval();
            cout << "95% CI: [" << ci[0] << ", " << ci[1] << "]" << endl;
            cout << "CPU time with importance sampling: " << duration.count() << "ms" << endl;
            
            // GPU acceleration
            hybrid_pricer::GPUAccelerator gpu("monte_carlo_kernels.cl");
            start = high_resolution_clock::now();
            price = gpu.priceOption(option, numPaths, numSteps, true);
            duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
            
            cout << "\nGPU price (with Sobol): " << price << endl;
            cout << "GPU time: " << duration.count() << "ms" << endl;
        }
        
        if (method == "pde" || method == "both") {
            cout << "\nPDE Solver:\n";
            cout << "-----------\n";
            
            // Only supported for European options currently
            if (auto euro_option = std::dynamic_pointer_cast<hybrid_pricer::EuropeanOption>(option)) {
                hybrid_pricer::PDESolver solver(euro_option, 100, 200);
                
                auto start = high_resolution_clock::now();
                double cn_price = solver.solveCrankNicolson();
                auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
                
                cout << "Crank-Nicolson price: " << cn_price << endl;
                cout << "PDE time: " << duration.count() << "ms" << endl;
            } else {
                cout << "PDE method currently only supports European options" << endl;
            }
        }
        
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
