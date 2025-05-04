#ifndef TIME_H
#define TIME_H

#include <chrono>

// Use a namespace to avoid polluting the global scope
namespace timing {
    // Variable to store the start time point
    std::chrono::high_resolution_clock::time_point start_time;

    // Records the current time point
    inline void tic() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Calculates the elapsed time since the last call to tic()
    // Returns the duration in milliseconds
    inline double toc() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        return duration.count();
    }
} // namespace timing

#endif // TIME_H 