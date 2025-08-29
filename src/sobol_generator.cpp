#include "sobol_generator.hpp"
#include <stdexcept>
#include <cmath>

namespace hybrid_pricer {

SobolGenerator::SobolGenerator(size_t dimension) 
    : position_(0), dimension_(dimension) {
    if (dimension == 0 || dimension > MAX_DIM) {
        throw std::invalid_argument("Invalid Sobol sequence dimension");
    }
    
    direction_numbers_.reserve(dimension);
    
    // Initialize first dimension
    direction_numbers_.push_back(init_direction_numbers_1d());
    
    // Initialize remaining dimensions
    for (size_t dim = 1; dim < dimension; ++dim) {
        direction_numbers_.push_back(init_direction_numbers_nd(dim));
    }
}

void SobolGenerator::reset() {
    position_ = 0;
}

void SobolGenerator::skip(size_t n) {
    position_ += n;
}

std::vector<double> SobolGenerator::next() {
    std::vector<double> point(dimension_);
    const double scale = 1.0 / (1ULL << MAX_BITS);
    
    // Increment position first to avoid repeating 0
    ++position_;
    
    // Gray code: position_ xor (position_ >> 1)
    uint32_t gray = position_ ^ (position_ >> 1);
    
    // Generate points for each dimension
    for (size_t dim = 0; dim < dimension_; ++dim) {
        uint32_t value = 0;
        const auto& dir_numbers = direction_numbers_[dim];
        
        // XOR appropriate direction numbers based on gray code
        for (size_t bit = 0; bit < MAX_BITS; ++bit) {
            if ((gray >> bit) & 1) {
                value ^= dir_numbers[bit];
            }
        }
        
        // Scale to [0,1) and ensure uniform distribution
        point[dim] = (value + 0.5) * scale;
    }
    
    return point;
}

std::vector<uint32_t> SobolGenerator::init_direction_numbers_1d() {
    std::vector<uint32_t> dir_nums(MAX_BITS);
    
    // First dimension uses reversed powers of 2 to match Van der Corput sequence
    for (size_t i = 0; i < MAX_BITS; ++i) {
        dir_nums[i] = 1U << i;  // Powers of 2: 1, 2, 4, 8, 16, ...
    }
    
    return dir_nums;
}

std::vector<uint32_t> SobolGenerator::init_direction_numbers_nd(size_t dim) {
    if (dim >= MAX_DIM - 1) {
        throw std::invalid_argument("Dimension too large");
    }
    
    std::vector<uint32_t> dir_nums(MAX_BITS);
    const auto& init_nums = initial_numbers[dim + 1];
    uint32_t poly = primitive_polynomials[dim + 1];
    size_t degree = 0;
    
    // Find degree of primitive polynomial
    for (uint32_t temp = poly; temp > 0; temp >>= 1) {
        ++degree;
    }
    --degree;
    
    // Initialize first values based on the initial numbers
    for (size_t i = 0; i < degree; ++i) {
        dir_nums[i] = init_nums[i] << (MAX_BITS - 1 - i);
    }
    
    // Generate remaining direction numbers using recurrence relation
    for (size_t i = degree; i < MAX_BITS; ++i) {
        uint32_t value = dir_nums[i - degree];
        for (size_t j = 1; j < degree; ++j) {
            if ((poly >> j) & 1) {
                value ^= dir_nums[i - j];
            }
        }
        dir_nums[i] = value >> 1;
    }
    
    return dir_nums;
}

// Static member definitions for SobolSequence
std::array<uint32_t, SobolSequence::BITS> SobolSequence::direction_numbers_;
uint32_t SobolSequence::current_point_ = 0;
bool SobolSequence::initialized_ = false;

void SobolSequence::initialize() {
    if (!initialized_) {
        generateDirectionNumbers();
        initialized_ = true;
    }
}

double SobolSequence::get(uint32_t index) {
    if (!initialized_) {
        initialize();
    }
    current_point_ = index;
    return getNextInteger() / static_cast<double>(1ULL << BITS);
}

double SobolSequence::getNext() {
    if (!initialized_) {
        initialize();
    }
    return getNextInteger() / static_cast<double>(1ULL << BITS);
}

void SobolSequence::reset() {
    current_point_ = 0;
}

void SobolSequence::generateDirectionNumbers() {
    // Initialize direction numbers for first dimension
    // Using powers of 2 as direction numbers
    for (int i = 0; i < BITS; ++i) {
        direction_numbers_[i] = 1U << (BITS - 1 - i);
    }
}

uint32_t SobolSequence::getNextInteger() {
    uint32_t index = current_point_++;
    uint32_t value = 0;
    for (int i = 0; i < BITS; ++i) {
        if (index & 1) {
            value ^= direction_numbers_[i];
        }
        index >>= 1;
        if (!index) break;
    }
    return value;
}

} // namespace hybrid_pricer
