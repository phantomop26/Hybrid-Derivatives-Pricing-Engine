#pragma once

#include "option.hpp"

namespace hybrid_pricer {

class BarrierOption : public Option {
public:
    enum class BarrierType { 
        UpAndOut,
        UpAndIn,
        DownAndOut,
        DownAndIn
    };

    BarrierOption(double spot, double strike, double barrier,
                  double volatility, double riskFreeRate, double timeToMaturity,
                  OptionType optType, BarrierType barrierType)
        : Option(spot, strike, volatility, riskFreeRate, timeToMaturity),
          barrier_(barrier),
          barrierType_(barrierType) {}

    double payoff(const std::vector<double>& path) const override {
        bool barrierHit = false;
        for (double price : path) {
            switch (barrierType_) {
                case BarrierType::UpAndOut:
                case BarrierType::UpAndIn:
                    if (price >= barrier_) barrierHit = true;
                    break;
                case BarrierType::DownAndOut:
                case BarrierType::DownAndIn:
                    if (price <= barrier_) barrierHit = true;
                    break;
            }
        }

        // Final price is the last element in the path
        double finalPrice = path.back();
        double intrinsicValue = (getOptionType() == OptionType::Call) 
            ? max(finalPrice - strike, 0.0)
            : max(strike - finalPrice, 0.0);

        // Determine if the option pays off based on barrier condition
        bool shouldPay = (barrierType_ == BarrierType::UpAndIn || barrierType_ == BarrierType::DownAndIn)
            ? barrierHit
            : !barrierHit;

        return shouldPay ? intrinsicValue : 0.0;
    }

    bool isPathDependent() const override { return true; }

    std::string getType() const override {
        std::string type = (getOptionType() == OptionType::Call) ? "Call" : "Put";
        std::string barrier;
        switch (barrierType_) {
            case BarrierType::UpAndOut: barrier = "Up-and-Out"; break;
            case BarrierType::UpAndIn: barrier = "Up-and-In"; break;
            case BarrierType::DownAndOut: barrier = "Down-and-Out"; break;
            case BarrierType::DownAndIn: barrier = "Down-and-In"; break;
        }
        return barrier + " " + type;
    }

    double getBarrier() const override { return barrier_; }

private:
    double barrier_;
    BarrierType barrierType_;
};

} // namespace hybrid_pricer
