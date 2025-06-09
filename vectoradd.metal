#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <metal_stdlib>
using namespace metal;

// Metal kernel for vector addition
const char* kernelSource = R"(
kernel void vectorAdd(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* c [[buffer(2)]],
                     uint index [[thread_position_in_grid]]) {
    c[index] = a[index] + b[index];
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
        std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Vector size must be positive." << std::endl;
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

    MTL::Function* kernel = library->newFunction("vectorAdd");
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(kernel, &error);
    if (!pipeline) {
        std::cerr << "Error: Failed to create pipeline: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Initialize input data
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);
    std::vector<float> h_c_cpu(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
    }

    // CPU computation
    {
        Timer timer;
        for (int i = 0; i < n; ++i) {
            h_c_cpu[i] = h_a[i] + h_b[i];
        }
        double cpu_time = timer.elapsed();
        std::cout << "CPU Vector addition took: " << cpu_time << " ms" << std::endl;
    }

    // Create buffers
    MTL::Buffer* bufferA = device->newBuffer(h_a.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(h_b.data(), n * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferC = device->newBuffer(h_c.data(), n * sizeof(float), MTL::ResourceStorageModeShared);

    // Create command buffer and compute command encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // Set pipeline and buffers
    computeEncoder->setComputePipelineState(pipeline);
    computeEncoder->setBuffer(bufferA, 0, 0);
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferC, 0, 2);

    // Calculate grid and thread size
    MTL::Size gridSize = MTL::Size(n, 1, 1);
    MTL::Size threadGroupSize = MTL::Size(256, 1, 1);

    // Dispatch threads
    {
        Timer timer;
        computeEncoder->dispatchThreads(gridSize, threadGroupSize);
        computeEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        double gpu_time = timer.elapsed();
        std::cout << "GPU Vector addition took: " << gpu_time << " ms" << std::endl;
    }

    // Verify results
    bool match = true;
    float epsilon = std::numeric_limits<float>::epsilon() * n;
    for (int i = 0; i < n; ++i) {
        if (std::abs(h_c[i] - h_c_cpu[i]) > epsilon) {
            std::cerr << "Mismatch found at index " << i << ": GPU=" << h_c[i]
                      << ", CPU=" << h_c_cpu[i] << std::endl;
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "Results match!" << std::endl;
    }

    // Cleanup
    bufferA->release();
    bufferB->release();
    bufferC->release();
    pipeline->release();
    kernel->release();
    library->release();
    commandQueue->release();
    device->release();

    return 0;
} 