#pragma once

#include <vector>
#include <cstdint>
#include <array>

namespace hybrid_pricer {

class SobolGenerator {
public:
    // Initialize Sobol sequence generator with given dimension
    explicit SobolGenerator(size_t dimension = 1);
    
    // Get next point in sequence
    std::vector<double> next();
    
    // Reset the sequence
    void reset();
    
    // Skip ahead in the sequence
    void skip(size_t n);

private:
    static constexpr size_t MAX_BITS = 32;
    static constexpr size_t MAX_DIM = 32;  // Can be increased if needed
    
    // Direction numbers for each dimension
    std::vector<std::vector<uint32_t>> direction_numbers_;
    
    // Current position in sequence
    uint32_t position_;
    size_t dimension_;
    
    // Initialize direction numbers for first dimension
    static std::vector<uint32_t> init_direction_numbers_1d();
    
    // Initialize direction numbers for higher dimensions
    static std::vector<uint32_t> init_direction_numbers_nd(size_t dim);
    
    // Primitive polynomials in Sobol' sequence
    static constexpr std::array<uint32_t, MAX_DIM> primitive_polynomials = {
        1,     // Special case for dimension 1
        3,     // x + x^2
        7,     // x + x^3
        11,    // x + x^2 + x^3
        13,    // x + x^4
        19,    // x + x^2 + x^4
        25,    // x + x^3 + x^4
        37,    // x + x^5
        59,    // x + x^2 + x^5
        47,    // x + x^3 + x^5
        61,    // x + x^2 + x^3 + x^5
        55,    // x + x^4 + x^5
        41,    // x + x^3 + x^4 + x^5
        67,    // x + x^2 + x^3 + x^4 + x^5
        97,    // x + x^6
        91,    // x + x^2 + x^6
        109,   // x + x^3 + x^6
        103,   // x + x^2 + x^3 + x^6
        115,   // x + x^4 + x^6
        131,   // x + x^5 + x^6
        193,   // x + x^7
        181,   // x + x^2 + x^7
        157,   // x + x^3 + x^7
        185,   // x + x^2 + x^3 + x^7
        167,   // x + x^4 + x^7
        229,   // x + x^5 + x^7
        171,   // x + x^6 + x^7
        213,   // x + x^2 + x^6 + x^7
        191,   // x + x^3 + x^6 + x^7
        253,   // x + x^2 + x^3 + x^6 + x^7
        203,   // x + x^4 + x^6 + x^7
        211    // x + x^5 + x^6 + x^7
    };
    
    // Initial values for direction numbers
    static constexpr std::array<std::array<uint32_t, 8>, MAX_DIM> initial_numbers = {{
        {1, 1, 1, 1, 1, 1, 1, 1},             // Dimension 1 (special case)
        {1, 3, 5, 15, 17, 51, 85, 255},       // Dimension 2
        {1, 1, 7, 11, 13, 61, 67, 79},        // Dimension 3
        {1, 3, 7, 5, 7, 43, 49, 147},         // Dimension 4
        {1, 1, 5, 3, 15, 51, 125, 141},       // Dimension 5
        {1, 3, 1, 1, 9, 59, 25, 89},          // Dimension 6
        {1, 3, 7, 13, 3, 35, 89, 23},         // Dimension 7
        {1, 3, 5, 7, 11, 27, 115, 411},       // Dimension 8
        {1, 1, 3, 13, 7, 35, 188, 155},       // Dimension 9
        {1, 3, 7, 9, 5, 21, 119, 275},        // Dimension 10
        {1, 1, 5, 11, 27, 43, 251, 925},      // Dimension 11
        {1, 3, 1, 7, 11, 45, 169, 539},       // Dimension 12
        {1, 1, 3, 5, 15, 51, 397, 543},       // Dimension 13
        {1, 3, 1, 15, 13, 61, 185, 299},      // Dimension 14
        {1, 1, 7, 11, 13, 29, 225, 767},      // Dimension 15
        {1, 3, 7, 13, 25, 75, 337, 675},      // Dimension 16
        {1, 1, 5, 9, 19, 41, 389, 705},       // Dimension 17
        {1, 3, 1, 15, 21, 49, 377, 915},      // Dimension 18
        {1, 1, 3, 13, 27, 63, 361, 969},      // Dimension 19
        {1, 3, 5, 9, 17, 73, 235, 873},       // Dimension 20
        {1, 1, 7, 13, 25, 79, 329, 621},      // Dimension 21
        {1, 3, 7, 15, 29, 45, 143, 509},      // Dimension 22
        {1, 1, 5, 11, 23, 59, 215, 663},      // Dimension 23
        {1, 3, 1, 7, 31, 91, 411, 879},       // Dimension 24
        {1, 1, 3, 5, 15, 83, 373, 827},       // Dimension 25
        {1, 3, 1, 15, 17, 53, 427, 963},      // Dimension 26
        {1, 1, 7, 11, 21, 67, 339, 711},      // Dimension 27
        {1, 3, 7, 13, 9, 35, 199, 575},       // Dimension 28
        {1, 1, 5, 9, 27, 81, 393, 741},       // Dimension 29
        {1, 3, 1, 15, 29, 87, 205, 821},      // Dimension 30
        {1, 1, 3, 13, 31, 95, 267, 911},      // Dimension 31
        {1, 3, 5, 9, 25, 69, 445, 997}        // Dimension 32
    }};
};

class SobolSequence {
public:
    static void initialize();
    static double get(uint32_t index);
    static double getNext();
    static void reset();

private:
    static constexpr int BITS = 32;
    static constexpr int MAX_DIMENSION = 100;  // Maximum dimension for Sobol sequence
    static std::array<uint32_t, BITS> direction_numbers_;  // Single dimension for now
    static uint32_t current_point_;  // Current point in sequence
    static bool initialized_;

    static void generateDirectionNumbers();
    static uint32_t getNextInteger();  // Renamed from getNext() to avoid overloading conflict
};

} // namespace hybrid_pricer
