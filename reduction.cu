#include <iostream>
#include <cstdlib> // For malloc, free, std::stoi, rand, srand, RAND_MAX
#include <string>  // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <ctime>   // For seeding srand with time()
#include <cmath>   // For fabs in comparison
#include <limits>  // For numeric_limits
#include <numeric> // For std::accumulate

#include <cuda_runtime.h> // CUDA runtime API
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

// --- CPU Reduction Function ---
double reduceCPU(const double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// --- GPU Reduction Kernel ---
// This kernel performs reduction within each block using shared memory.
// Each block writes its partial sum to d_partial_sums.
__global__ void reduceKernel(const double *g_idata, double *g_odata, int n) {
    // Shared memory for intermediate results within a block
    extern __shared__ double sdata[];

    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, handling boundary conditions
    sdata[tid] = (i < n) ? g_idata[i] : 0.0;

    __syncthreads(); // Ensure all data is loaded before starting reduction

    // Perform reduction in shared memory
    // Each step halves the number of active threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Write result for this block to global memory (first thread handles this)
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


int main(int argc, char *argv[]) {
    int n;

    // --- Argument Parsing and Size Check ---
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return 1;
    }
    try {
        n = std::stoi(argv[1]);
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid number format for array size." << std::endl;
        return 1;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Array size out of range." << std::endl;
        return 1;
    }
    if (n <= 0) {
        std::cerr << "Error: Array size must be positive." << std::endl;
        return 1;
    }

    // --- Host Memory Allocation ---
    size_t size = n * sizeof(double);
    double *h_data = (double*)malloc(size);
    if (h_data == nullptr) {
        std::cerr << "Error: Host memory allocation failed for input data." << std::endl;
        return 1;
    }
    double h_result_cpu = 0.0;
    double h_result_gpu = 0.0;

    // --- Host Data Initialization ---
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        h_data[i] = (double)rand() / RAND_MAX;
    }

    // --- CPU Reduction and Timing ---
    std::cout << "Starting CPU reduction..." << std::endl;
    timing::tic();
    h_result_cpu = reduceCPU(h_data, n);
    double cpu_duration_ms = timing::toc();
    std::cout << "CPU Reduction took: " << cpu_duration_ms << " ms" << std::endl;
    std::cout << "CPU Result: " << h_result_cpu << std::endl;

    // --- GPU Reduction Setup ---
    std::cout << "\nStarting GPU reduction..." << std::endl;
    int threadsPerBlock = 256; // Must be power of 2 for this reduction kernel
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t partial_sum_size = blocksPerGrid * sizeof(double);
    size_t sharedMemSize = threadsPerBlock * sizeof(double);

    // --- Device Memory Allocation ---
    double *d_data = nullptr;
    double *d_partial_sums = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_partial_sums, partial_sum_size));

    // Allocate host memory for partial sums
    double *h_partial_sums = (double*)malloc(partial_sum_size);
    if (h_partial_sums == nullptr) {
        std::cerr << "Error: Host memory allocation failed for partial sums." << std::endl;
        free(h_data);
        CHECK_CUDA_ERROR(cudaFree(d_data));
        CHECK_CUDA_ERROR(cudaFree(d_partial_sums));
        return 1;
    }

    // --- Copy Data Host to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // --- Kernel Execution and Timing ---
    timing::tic();
    // Kernel launch needs shared memory size specified
    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_data, d_partial_sums, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double gpu_kernel_duration_ms = timing::toc(); // Time only the kernel execution

    // --- Copy Partial Results Device to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_partial_sums, d_partial_sums, partial_sum_size, cudaMemcpyDeviceToHost));

    // --- Final Reduction on Host (summing partial sums) ---
    // Could launch another small kernel, but summing on host is often fast enough for typical block counts
    timing::tic();
    h_result_gpu = reduceCPU(h_partial_sums, blocksPerGrid); // Re-use CPU reduction for final step
    // Alternative using std::accumulate:
    // h_result_gpu = std::accumulate(h_partial_sums, h_partial_sums + blocksPerGrid, 0.0);
    double final_reduction_duration_ms = timing::toc();

    std::cout << "GPU Kernel execution took: " << gpu_kernel_duration_ms << " ms" << std::endl;
    std::cout << "Final reduction on CPU took: " << final_reduction_duration_ms << " ms" << std::endl;
    std::cout << "GPU Result: " << h_result_gpu << std::endl;

    // --- Verification ---
    double tolerance = std::numeric_limits<double>::epsilon() * n *1000;
    double diff = std::fabs(h_result_cpu - h_result_gpu);

    std::cout << "\nVerification:" << std::endl;
    std::cout << "  CPU Result = " << h_result_cpu << std::endl;
    std::cout << "  GPU Result = " << h_result_gpu << std::endl;
    std::cout << "  Difference = " << diff << std::endl;
    std::cout << "  Tolerance  = " << tolerance << std::endl;

    if (diff <= tolerance) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cerr << "Results mismatch!" << std::endl;
    }

    // --- Cleanup ---
    free(h_data);
    free(h_partial_sums);
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_partial_sums));

    return 0;
} 