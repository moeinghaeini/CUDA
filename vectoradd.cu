#include <iostream>
#include <cstdlib> // For malloc, free, std::stoi, rand, srand, RAND_MAX
#include <string>  // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <ctime>   // For seeding srand with time()
#include <cuda_runtime.h> // CUDA runtime API
#include <cmath> // For fabs in comparison
#include <limits> // For numeric_limits

#include "time.h" // Include the timing header

// Helper function for checking CUDA errors
inline void checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)
                  << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_ERROR(err) (checkCudaErrors(err, __FILE__, __LINE__))

// CUDA Kernel for vector addition
__global__ void vectorAddKernel(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU Kernel for vector addition
void vectorAddCPU(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
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

    // --- Host Memory Allocation ---
    size_t size = n * sizeof(double);
    double *h_a = (double*)malloc(size);
    double *h_b = (double*)malloc(size);
    double *h_c_gpu = (double*)malloc(size); // Result from GPU
    double *h_c_cpu = (double*)malloc(size); // Result from CPU

    // Check if host allocation succeeded
    if (h_a == nullptr || h_b == nullptr || h_c_gpu == nullptr || h_c_cpu == nullptr) {
        std::cerr << "Error: Host memory allocation failed." << std::endl;
        free(h_a); free(h_b); free(h_c_gpu); free(h_c_cpu);
        return 1;
    }

    // --- Host Data Initialization ---
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        h_a[i] = (double)rand() / RAND_MAX;
        h_b[i] = (double)rand() / RAND_MAX;
    }

    // --- CPU Vector Addition and Timing ---
    timing::tic();
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    double cpu_duration_ms = timing::toc();
    std::cout << "CPU Vector addition took: " << cpu_duration_ms << " ms" << std::endl;

    // --- Device Memory Allocation ---
    double *d_a = nullptr;
    double *d_b = nullptr;
    double *d_c = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, size));

    // --- Copy Data from Host to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // --- Kernel Launch Configuration ---
    int threadsPerBlock = 256;
    // Ensure sufficient blocks, handling cases where n is not a multiple of threadsPerBlock
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // --- Kernel Execution and Timing ---
    timing::tic();
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // Check for errors during kernel launch (optional but recommended)
    CHECK_CUDA_ERROR(cudaGetLastError());
    // Wait for the kernel to complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double gpu_duration_ms = timing::toc();

    // Print the time taken for kernel execution
    std::cout << "CUDA Kernel execution took: " << gpu_duration_ms << " ms" << std::endl;

    // --- Copy Result from Device to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));

    // --- Verify Results ---
    bool match = true;
    double epsilon = std::numeric_limits<double>::epsilon() * n; // Tolerance for floating point comparison
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c_gpu[i] - h_c_cpu[i]) > epsilon) {
            std::cerr << "Mismatch found at index " << i << ": GPU=" << h_c_gpu[i]
                      << ", CPU=" << h_c_cpu[i] << std::endl;
            match = false;
            break; // Stop checking after first mismatch
        }
    }
    if (match) {
        std::cout << "Results match!" << std::endl;
    }

    // --- Free Device Memory ---
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    // --- Free Host Memory ---
    free(h_a);
    free(h_b);
    free(h_c_gpu); // Freed renamed variable
    free(h_c_cpu); // Freed new variable

    return 0;
} 