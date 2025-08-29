#pragma once

#include "option.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace hybrid_pricer {

class TwoAssetOption : public Option {
public:
    enum class Type { Spread, Rainbow, BasketCall, BasketPut, BestOf, WorstOf };
    
    TwoAssetOption(Type type, OptionType optType, double K, double T,
                  double S1, double S2, double sigma1, double sigma2,
                  double rho, double r)
        : Option(S1, K, sigma1, r, T), 
          type_(type), optType_(optType),
          spot2_(S2), sigma2_(sigma2), correlation_(rho) {}
    
    double payoff(const std::vector<double>& path) const override {
        // Path contains alternating prices for asset 1 and 2
        std::vector<double> path1, path2;
        for (size_t i = 0; i < path.size(); i += 2) {
            path1.push_back(path[i]);
            path2.push_back(path[i + 1]);
        }
        
        double S1T = path1.back();
        double S2T = path2.back();
        
        switch (type_) {
            case Type::Spread:
                if (optType_ == OptionType::Call)
                    return std::max(S1T - S2T - strike, 0.0);
                else
                    return std::max(strike - (S1T - S2T), 0.0);
                
            case Type::BasketCall:
            case Type::BasketPut: {
                double basketPrice = (S1T + S2T) / 2.0;
                if (type_ == Type::BasketCall)
                    return std::max(basketPrice - strike, 0.0);
                else
                    return std::max(strike - basketPrice, 0.0);
            }
                
            case Type::BestOf:
                if (optType_ == OptionType::Call)
                    return std::max({0.0, S1T - strike, S2T - strike});
                else
                    return std::max({0.0, strike - S1T, strike - S2T});
                
            case Type::WorstOf:
                if (optType_ == OptionType::Call)
                    return std::max(std::min(S1T - strike, S2T - strike), 0.0);
                else
                    return std::max(strike - std::min(S1T, S2T), 0.0);
                
            case Type::Rainbow:
                return std::max(std::max(S1T, S2T) - strike, 0.0);
        }
        return 0.0;
    }
    
    bool isPathDependent() const override { 
        return false; // Most two-asset options are European-style
    }
    
    std::string getType() const override {
        std::string typeStr;
        switch(type_) {
            case Type::Spread: typeStr = "Spread"; break;
            case Type::Rainbow: typeStr = "Rainbow"; break;
            case Type::BasketCall: typeStr = "BasketCall"; break;
            case Type::BasketPut: typeStr = "BasketPut"; break;
            case Type::BestOf: typeStr = "BestOf"; break;
            case Type::WorstOf: typeStr = "WorstOf"; break;
        }
        if (type_ != Type::BasketCall && type_ != Type::BasketPut)
            typeStr += optType_ == OptionType::Call ? "Call" : "Put";
        return typeStr;
    }
    
    OptionType getOptionType() const override { return optType_; }
    
    // Getters for the second asset parameters
    double getSpot2() const { return spot2_; }
    double getSigma2() const { return sigma2_; }
    double getCorrelation() const { return correlation_; }
    
private:
    Type type_;
    OptionType optType_;
    double spot2_;
    double sigma2_;
    double correlation_;
};

} // namespace hybrid_pricer
