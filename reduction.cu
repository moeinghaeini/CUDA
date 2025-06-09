#include <iostream>
#include <cstdlib> // For malloc, free, std::stoi, rand, srand, RAND_MAX
#include <string>  // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <ctime>   // For seeding srand with time()
#include <cmath>   // For fabs in comparison
#include <limits>  // For numeric_limits
#include <numeric> // For std::accumulate
#include <vector>
#include <memory>

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

// CPU Reduction Function with OpenMP
double reduceCPU(const double *data, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// Optimized GPU Reduction Kernel
__global__ void reduceKernel(const double *__restrict__ g_idata, 
                            double *__restrict__ g_odata, 
                            int n) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load first element
    double sum = (i < n) ? g_idata[i] : 0.0;
    // Load second element
    if (i + blockDim.x < n) {
        sum += g_idata[i + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll last warp
    if (tid < 32) {
        volatile double *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Write result for this block
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// RAII wrapper for CUDA memory
class CudaMemory {
public:
    CudaMemory(size_t size) : size_(size) {
        CHECK_CUDA_ERROR(cudaMalloc(&ptr_, size));
    }
    
    ~CudaMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    void* get() { return ptr_; }
    const void* get() const { return ptr_; }
    
    // Prevent copying
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
private:
    void* ptr_ = nullptr;
    size_t size_;
};

int main(int argc, char *argv[]) {
    int n;

    // Argument parsing and validation
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

    // Use RAII for host memory management
    std::vector<double> h_data(n);
    double h_result_cpu = 0.0;
    double h_result_gpu = 0.0;

    // Initialize input data
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // CPU Reduction and Timing
    std::cout << "Starting CPU reduction..." << std::endl;
    timing::tic();
    h_result_cpu = reduceCPU(h_data.data(), n);
    double cpu_duration_ms = timing::toc();
    std::cout << "CPU Reduction took: " << cpu_duration_ms << " ms" << std::endl;
    std::cout << "CPU Result: " << h_result_cpu << std::endl;

    // GPU Reduction Setup
    std::cout << "\nStarting GPU reduction..." << std::endl;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    size_t partial_sum_size = blocksPerGrid * sizeof(double);
    size_t sharedMemSize = threadsPerBlock * sizeof(double);

    // Use RAII for device memory management
    CudaMemory d_data(n * sizeof(double));
    CudaMemory d_partial_sums(partial_sum_size);

    // Allocate host memory for partial sums
    std::vector<double> h_partial_sums(blocksPerGrid);

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data.get(), h_data.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Execute kernel and time it
    timing::tic();
    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        static_cast<const double*>(d_data.get()),
        static_cast<double*>(d_partial_sums.get()),
        n
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double gpu_kernel_duration_ms = timing::toc();

    // Copy partial results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_partial_sums.data(), d_partial_sums.get(), partial_sum_size, cudaMemcpyDeviceToHost));

    // Final reduction on host
    timing::tic();
    h_result_gpu = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0);
    double final_reduction_duration_ms = timing::toc();

    std::cout << "GPU Kernel execution took: " << gpu_kernel_duration_ms << " ms" << std::endl;
    std::cout << "Final reduction on CPU took: " << final_reduction_duration_ms << " ms" << std::endl;
    std::cout << "Total GPU time: " << gpu_kernel_duration_ms + final_reduction_duration_ms << " ms" << std::endl;
    std::cout << "GPU Result: " << h_result_gpu << std::endl;
    std::cout << "Speedup: " << cpu_duration_ms / (gpu_kernel_duration_ms + final_reduction_duration_ms) << "x" << std::endl;

    // Verify results
    double tolerance = std::numeric_limits<double>::epsilon() * n * 1000;
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

    return 0;
} 