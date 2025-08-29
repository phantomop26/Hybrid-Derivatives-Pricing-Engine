#pragma once

#include "option.hpp"
#include <string>
#include <algorithm>
#include <numeric>  // for std::accumulate

namespace hybrid_pricer {

class VanillaOption : public Option {
public:
    VanillaOption(double K, double T, double S0, double sigma, double r, OptionType type)
        : Option(S0, K, sigma, r, T), optType_(type) {}

    OptionType getOptionType() const override { return optType_; }

private:
    OptionType optType_;
};

class EuropeanOption : public Option {
public:
    EuropeanOption(OptionType type, double K, double T, double S0, double sigma, double r)
        : Option(S0, K, sigma, r, T), optType_(type) {}
    
    double payoff(const std::vector<double>& path) const override {
        double ST = path.back();
        if (optType_ == OptionType::Call) {
            return std::max(ST - strike, 0.0);
        } else {
            return std::max(strike - ST, 0.0);
        }
    }
    
    bool isPathDependent() const override { return false; }
    std::string getType() const override { 
        return optType_ == OptionType::Call ? "EuropeanCall" : "EuropeanPut";
    }
    OptionType getOptionType() const override { return optType_; }
    
private:
    OptionType optType_;
};

class BarrierOption : public Option {
public:
    enum class Type { UpAndOut, UpAndIn, DownAndOut, DownAndIn };
    
    BarrierOption(Type type, OptionType optType, double K, double T, double S0, 
                 double sigma, double r, double barrier)
        : Option(S0, K, sigma, r, T), type_(type), optType_(optType), barrier_(barrier) {}
    
    double payoff(const std::vector<double>& path) const override {
        bool barrierHit = false;
        if (type_ == Type::UpAndOut || type_ == Type::UpAndIn) {
            barrierHit = std::any_of(path.begin(), path.end(), 
                                   [this](double price) { return price >= barrier_; });
        } else {
            barrierHit = std::any_of(path.begin(), path.end(), 
                                   [this](double price) { return price <= barrier_; });
        }
        
        bool shouldPay = (type_ == Type::UpAndIn || type_ == Type::DownAndIn) ? barrierHit 
                                                                              : !barrierHit;
        
        if (!shouldPay) return 0.0;
        
        double ST = path.back();
        if (optType_ == OptionType::Call) {
            return std::max(ST - strike, 0.0);
        } else {
            return std::max(strike - ST, 0.0);
        }
    }
    
    bool isPathDependent() const override { return true; }
    std::string getType() const override {
        std::string typeStr;
        switch(type_) {
            case Type::UpAndOut: typeStr = "UpAndOut"; break;
            case Type::UpAndIn: typeStr = "UpAndIn"; break;
            case Type::DownAndOut: typeStr = "DownAndOut"; break;
            case Type::DownAndIn: typeStr = "DownAndIn"; break;
        }
        typeStr += optType_ == OptionType::Call ? "Call" : "Put";
        return typeStr;
    }
    OptionType getOptionType() const override { return optType_; }
    double getBarrier() const override { return barrier_; }
    
private:
    Type type_;
    OptionType optType_;
    double barrier_;
};

class AsianOption : public Option {
public:
    enum class Type { Arithmetic, Geometric };
    
    AsianOption(Type type, OptionType optType, double K, double T, double S0, double sigma, double r)
        : Option(S0, K, sigma, r, T), type_(type), optType_(optType) {}
    
    double payoff(const std::vector<double>& path) const override {
        double avg;
        if (type_ == Type::Arithmetic) {
            avg = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
        } else {  // Geometric
            // Use log-sum-exp trick to prevent numerical overflow
            double sum_log = 0.0;
            for (double price : path) {
                sum_log += std::log(price);
            }
            avg = std::exp(sum_log / path.size());
        }
        
        if (optType_ == OptionType::Call) {
            return std::max(avg - strike, 0.0);
        } else {
            return std::max(strike - avg, 0.0);
        }
    }
    
    bool isPathDependent() const override { return true; }
    std::string getType() const override {
        std::string typeStr = type_ == Type::Arithmetic ? "ArithmeticAsian" : "GeometricAsian";
        typeStr += optType_ == OptionType::Call ? "Call" : "Put";
        return typeStr;
    }
    OptionType getOptionType() const override { return optType_; }
    
private:
    Type type_;
    OptionType optType_;
};

class LookbackOption : public Option {
public:
    enum class Type { Fixed, Floating };
    
    LookbackOption(Type type, OptionType optType, double K, double T, double S0, double sigma, double r)
        : Option(S0, K, sigma, r, T), type_(type), optType_(optType) {}
    
    double payoff(const std::vector<double>& path) const override {
        double maxPrice = *std::max_element(path.begin(), path.end());
        double minPrice = *std::min_element(path.begin(), path.end());
        
        if (type_ == Type::Fixed) {
            // Fixed strike lookback
            if (optType_ == OptionType::Call) {
                return std::max(maxPrice - strike, 0.0);
            } else {
                return std::max(strike - minPrice, 0.0);
            }
        } else {
            // Floating strike lookback
            if (optType_ == OptionType::Call) {
                return path.back() - minPrice;
            } else {
                return maxPrice - path.back();
            }
        }
    }
    
    bool isPathDependent() const override { return true; }
    
    std::string getType() const override {
        std::string typeStr = type_ == Type::Fixed ? "FixedLookback" : "FloatingLookback";
        typeStr += optType_ == OptionType::Call ? "Call" : "Put";
        return typeStr;
    }
    
    OptionType getOptionType() const override { return optType_; }
    
private:
    Type type_;
    OptionType optType_;
};

} // namespace hybrid_pricer
