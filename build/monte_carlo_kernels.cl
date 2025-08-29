#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Box-Muller transform for normal variates
inline double2 box_muller(const double u1, const double u2) {
    const double eps = 1.0e-10;  // Avoid log(0)
    const double u1_safe = fmax(u1, eps);
    const double log_term = -2.0 * log(u1_safe);
    const double r = sqrt(log_term);
    const double theta = 2.0 * M_PI * u2;
    return (double2)(r * cos(theta), r * sin(theta));
}

// Generate paths for European/Asian options
__kernel void generatePaths(
    __global double* const paths,         // [numPaths][numSteps+1]
    __global const double* const sobol,   // [numPaths][2*numSteps]
    const double spot,
    const double drift,
    const double vol,
    const double dt,
    const ulong numSteps,
    const ulong pathStride)
{
    const size_t gid = get_global_id(0);
    const size_t sobolOffset = gid * 2 * numSteps;
    const size_t pathOffset = gid * pathStride;
    
    // Initial spot price
    paths[pathOffset] = spot;
    
    // Generate path using pre-computed Sobol numbers
    double currentPrice = spot;
    const double sqrtDt = sqrt(dt);
    
    for (size_t i = 0; i < numSteps; ++i) {
        // Generate normal variates using Box-Muller
        const double u1 = fmax(sobol[sobolOffset + 2*i], 1.0e-10);
        const double u2 = sobol[sobolOffset + 2*i + 1];
        const double2 normals = box_muller(u1, u2);
        
        // Use first normal variate
        const double z = normals.x;
        const double dW = vol * sqrtDt * z;
        
        // Update price using log-normal process
        const double logReturn = drift * dt + dW;
        currentPrice *= exp(logReturn);
        
        // Store path value
        paths[pathOffset + i + 1] = currentPrice;
    }
}

// Calculate option payoffs
__kernel void calculatePayoffs(
    __global const double* const paths,   // [numPaths][numSteps+1]
    __global double* const payoffs,       // [numPaths]
    const int optionType,           // 0=European, 1=Asian, 2=Barrier
    const double strike,
    const double barrier,           // For barrier options
    const ulong numSteps,
    const ulong pathStride)
{
    const size_t gid = get_global_id(0);
    const size_t pathOffset = gid * pathStride;
    
    double payoff = 0.0;
    switch (optionType) {
        case 0: // European
            payoff = fmax(paths[pathOffset + numSteps] - strike, 0.0);
            break;
            
        case 1: { // Asian
            // Use Kahan summation for better numerical accuracy
            double sum = 0.0;
            double c = 0.0;  // Running compensation
            for (size_t i = 0; i <= numSteps; ++i) {
                const double y = paths[pathOffset + i] - c;
                const double t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            const double avg = sum / (double)(numSteps + 1);
            payoff = fmax(avg - strike, 0.0);
            break;
        }
            
        case 2: { // Barrier (Up-and-Out)
            bool knocked_out = false;
            for (size_t i = 0; i <= numSteps; ++i) {
                if (paths[pathOffset + i] >= barrier) {
                    knocked_out = true;
                    break;
                }
            }
            if (!knocked_out) {
                payoff = fmax(paths[pathOffset + numSteps] - strike, 0.0);
            }
            break;
        }
    }
    
    payoffs[gid] = payoff;
}
