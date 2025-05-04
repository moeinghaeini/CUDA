#include <iostream>
// #include <vector> // Removed
// #include <chrono> // Removed, functionality moved to time.h
// #include <numeric> // Removed
#include <cstdlib> // For malloc, free, std::stoi, rand, srand, RAND_MAX
#include <string>  // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
// #include <random>  // Removed, using rand() instead
#include <ctime>   // For seeding srand with time()

#include "time.h" // Include the new timing header

int main(int argc, char *argv[]) { // Modified main signature
    int n;

    // Check for command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
        return 1;
    }

    // Parse size from command-line argument
    try {
        n = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid number format for vector size." << std::endl;
        return 1;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Vector size out of range." << std::endl;
        return 1;
    }


    if (n <= 0) {
        std::cerr << "Error: Vector size must be positive." << std::endl;
        return 1;
    }

    // Allocate memory using malloc
    double *a = (double*)malloc(n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    double *c = (double*)malloc(n * sizeof(double));

    // Check if allocation succeeded
    if (a == nullptr || b == nullptr || c == nullptr) {
        std::cerr << "Error: Memory allocation failed." << std::endl;
        // Free any memory that was successfully allocated before exiting
        free(a); // free(nullptr) is safe
        free(b);
        free(c);
        return 1;
    }


    // Seed the C random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Initialize arrays a and b with random numbers using rand()
    // std::mt19937 rng(static_cast<unsigned int>(std::time(0))); // Mersenne Twister engine seeded with time // Removed
    // std::uniform_real_distribution<double> dist(0.0, 1.0); // Distribution for doubles between 0.0 and 1.0 // Removed

    for (int i = 0; i < n; ++i) {
        // a[i] = dist(rng); // Replaced with rand()
        // b[i] = dist(rng); // Replaced with rand()
        a[i] = (double)rand() / RAND_MAX; // Generate random double between 0.0 and 1.0
        b[i] = (double)rand() / RAND_MAX; // Generate random double between 0.0 and 1.0
    }


    // Time the vector addition
    // auto start = std::chrono::high_resolution_clock::now(); // Replaced with tic()
    timing::tic();

    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }

    // auto end = std::chrono::high_resolution_clock::now(); // Replaced with toc()
    // std::chrono::duration<double, std::milli> duration = end - start; // Replaced with toc()
    double duration_ms = timing::toc();

    // Print the time taken for addition
    // std::cout << "Vector addition took: " << duration.count() << " ms" << std::endl; // Modified output
    std::cout << "Vector addition took: " << duration_ms << " ms" << std::endl;

    // Optional: Print a few elements of c to verify
    // std::cout << "First few elements of c:" << std::endl;
    // for (int i = 0; i < std::min(n, 10); ++i) {
    //     std::cout << c[i] << " ";
    // }
    // std::cout << std::endl;

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
} 