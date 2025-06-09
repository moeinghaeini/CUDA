#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Array size must be positive." << std::endl;
        return 1;
    }

    // Initialize input data
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    // Vector addition
    {
        Timer timer;
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
        double time = timer.elapsed();
        std::cout << "\nVector addition took: " << time << " ms" << std::endl;
    }

    // Print a few elements to verify
    std::cout << "\nFirst few elements of result:" << std::endl;
    for (int i = 0; i < std::min(n, 5); ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Array reduction (sum)
    float sum;
    {
        Timer timer;
        sum = std::accumulate(a.begin(), a.end(), 0.0f);
        double time = timer.elapsed();
        std::cout << "\nArray reduction (sum) took: " << time << " ms" << std::endl;
        std::cout << "Sum of array: " << sum << std::endl;
    }

    // Array statistics
    float min_val = *std::min_element(a.begin(), a.end());
    float max_val = *std::max_element(a.begin(), a.end());
    float avg_val = sum / n;

    std::cout << "\nArray statistics:" << std::endl;
    std::cout << "  Minimum value: " << min_val << std::endl;
    std::cout << "  Maximum value: " << max_val << std::endl;
    std::cout << "  Average value: " << avg_val << std::endl;

    return 0;
} 