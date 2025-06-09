#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <metal_stdlib>
using namespace metal;

// Metal kernel for reduction
const char* kernelSource = R"(
kernel void reduce(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  device atomic_float* result [[buffer(2)]],
                  uint index [[thread_position_in_grid]],
                  uint grid_size [[threads_per_grid]]) {
    // Load input value
    float sum = (index < grid_size) ? input[index] : 0.0f;
    
    // Reduction within threadgroup
    threadgroup float shared_data[256];
    uint tid = thread_position_in_threadgroup;
    uint block_size = threadgroup_size;
    
    // Store in shared memory
    shared_data[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = block_size/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this block
    if (tid == 0) {
        output[threadgroup_position_in_grid] = shared_data[0];
    }
}
)";

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

    // Initialize Metal
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Error: Metal is not supported on this device." << std::endl;
        return 1;
    }

    // Create command queue
    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Create compute pipeline
    NSError* error = nullptr;
    MTL::Library* library = device->newLibrary(kernelSource, nullptr, &error);
    if (!library) {
        std::cerr << "Error: Failed to create library: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    MTL::Function* kernel = library->newFunction("reduce");
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(kernel, &error);
    if (!pipeline) {
        std::cerr << "Error: Failed to create pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Initialize input data
    std::vector<float> h_data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        h_data[i] = dis(gen);
    }

    // CPU reduction
    float cpu_result;
    {
        Timer timer;
        cpu_result = std::accumulate(h_data.begin(), h_data.end(), 0.0f);
        double cpu_time = timer.elapsed();
        std::cout << "CPU Reduction took: " << cpu_time << " ms" << std::endl;
        std::cout << "CPU Result: " << cpu_result << std::endl;
    }

    // GPU reduction setup
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    std::vector<float> h_partial_sums(blocksPerGrid);

    // Create buffers
    MTL::Buffer* bufferInput = device->newBuffer(h_data.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferOutput = device->newBuffer(h_partial_sums.data(), blocksPerGrid * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = device->newBuffer(sizeof(float), MTL::ResourceStorageModeShared);

    // Create command buffer and compute command encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // Set pipeline and buffers
    computeEncoder->setComputePipelineState(pipeline);
    computeEncoder->setBuffer(bufferInput, 0, 0);
    computeEncoder->setBuffer(bufferOutput, 0, 1);
    computeEncoder->setBuffer(bufferResult, 0, 2);

    // Calculate grid and thread size
    MTL::Size gridSize = MTL::Size(n, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(threadsPerBlock, 1, 1);

    // Dispatch threads
    {
        Timer timer;
        computeEncoder->dispatchThreads(gridSize, threadGroupSize);
        computeEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        double gpu_time = timer.elapsed();
        std::cout << "GPU Kernel execution took: " << gpu_time << " ms" << std::endl;
    }

    // Final reduction on CPU
    float gpu_result;
    {
        Timer timer;
        gpu_result = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0f);
        double final_reduction_time = timer.elapsed();
        std::cout << "Final reduction on CPU took: " << final_reduction_time << " ms" << std::endl;
    }

    std::cout << "GPU Result: " << gpu_result << std::endl;

    // Verify results
    float tolerance = std::numeric_limits<float>::epsilon() * n * 1000;
    float diff = std::abs(cpu_result - gpu_result);

    std::cout << "\nVerification:" << std::endl;
    std::cout << "  CPU Result = " << cpu_result << std::endl;
    std::cout << "  GPU Result = " << gpu_result << std::endl;
    std::cout << "  Difference = " << diff << std::endl;
    std::cout << "  Tolerance  = " << tolerance << std::endl;

    if (diff <= tolerance) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cerr << "Results mismatch!" << std::endl;
    }

    // Cleanup
    bufferInput->release();
    bufferOutput->release();
    bufferResult->release();
    pipeline->release();
    kernel->release();
    library->release();
    commandQueue->release();
    device->release();

    return 0;
} 