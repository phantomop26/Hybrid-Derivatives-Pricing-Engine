#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace hybrid_pricer {

using std::vector;
using std::string;
using std::shared_ptr;
using std::max;

class Option {
public:
    enum class OptionType { Call, Put };
    
    virtual ~Option() = default;
    
    virtual double payoff(const std::vector<double>& path) const = 0;
    virtual bool isPathDependent() const = 0;
    virtual std::string getType() const = 0;
    virtual OptionType getOptionType() const = 0;
    virtual double getBarrier() const { return 0.0; }  // Default implementation for non-barrier options
    
    // Option parameters
    double spot;              // Current spot price
    double strike;            // Strike price
    double volatility;        // Volatility
    double riskFreeRate;      // Risk-free interest rate
    double timeToMaturity;    // Time to maturity in years

protected:
    Option(double s, double k, double v, double r, double t)
        : spot(s), strike(k), volatility(v), riskFreeRate(r), timeToMaturity(t) {}
};

} // namespace hybrid_pricer
