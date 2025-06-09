#include <iostream>
#include <cstdlib> // For malloc, free, std::stoi, rand, srand, RAND_MAX
#include <string>  // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <ctime>   // For seeding srand with time()
#include <cuda_runtime.h> // CUDA runtime API
#include <cmath> // For fabs in comparison
#include <limits> // For numeric_limits
#include <vector>
#include <memory>

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

// CUDA Kernel for vector addition with improved memory coalescing
__global__ void vectorAddKernel(const double *__restrict__ a, 
                               const double *__restrict__ b, 
                               double *__restrict__ c, 
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU Kernel for vector addition
void vectorAddCPU(const double *a, const double *b, double *c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
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

    // Use RAII for host memory management
    std::vector<double> h_a(n);
    std::vector<double> h_b(n);
    std::vector<double> h_c_gpu(n);
    std::vector<double> h_c_cpu(n);

    // Initialize input data
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<double>(rand()) / RAND_MAX;
        h_b[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // CPU Vector Addition and Timing
    timing::tic();
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    double cpu_duration_ms = timing::toc();
    std::cout << "CPU Vector addition took: " << cpu_duration_ms << " ms" << std::endl;

    // GPU Vector Addition
    size_t size = n * sizeof(double);
    
    // Use RAII for device memory management
    CudaMemory d_a(size);
    CudaMemory d_b(size);
    CudaMemory d_c(size);

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a.get(), h_a.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b.get(), h_b.data(), size, cudaMemcpyHostToDevice));

    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernel and time it
    timing::tic();
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const double*>(d_a.get()),
        static_cast<const double*>(d_b.get()),
        static_cast<double*>(d_c.get()),
        n
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double gpu_duration_ms = timing::toc();

    std::cout << "CUDA Kernel execution took: " << gpu_duration_ms << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_duration_ms / gpu_duration_ms << "x" << std::endl;

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_c_gpu.data(), d_c.get(), size, cudaMemcpyDeviceToHost));

    // Verify results
    bool match = true;
    double epsilon = std::numeric_limits<double>::epsilon() * n;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c_gpu[i] - h_c_cpu[i]) > epsilon) {
            std::cerr << "Mismatch found at index " << i << ": GPU=" << h_c_gpu[i]
                      << ", CPU=" << h_c_cpu[i] << std::endl;
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match!" << std::endl;
    }

    return 0;
} 