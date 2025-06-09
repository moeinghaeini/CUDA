# Parallel Computing Examples

This project demonstrates various parallel computing implementations optimized for Apple devices, including both Metal (GPU) and standard C++ (CPU) approaches.

## Project Structure

```
.
├── README.md
├── parallel_example.cpp    # Standard C++ implementation
├── vectoradd.metal        # Metal shader for vector addition
├── vectoradd.cpp         # Metal host code for vector addition
├── reduction.metal       # Metal shader for reduction
└── reduction.cpp         # Metal host code for reduction
```

## Requirements

- macOS (tested on Apple Silicon)
- Xcode Command Line Tools
- C++17 compatible compiler (Clang)

## Building and Running

### Standard C++ Implementation

The `parallel_example.cpp` demonstrates vector operations and array reduction using standard C++:

```bash
# Compile
clang++ -std=c++17 parallel_example.cpp -o parallel_example

# Run (specify array size)
./parallel_example 1000000
```

### Metal Implementation (Optional)

For GPU acceleration using Metal:

1. Compile Metal shaders:
```bash
# Compile vector addition shader
xcrun -sdk macosx metal -c vectoradd.metal -o vectoradd.air
xcrun -sdk macosx metallib vectoradd.air -o vectoradd.metallib

# Compile reduction shader
xcrun -sdk macosx metal -c reduction.metal -o reduction.air
xcrun -sdk macosx metallib reduction.air -o reduction.metallib
```

2. Compile and run the host programs:
```bash
# Vector addition
clang++ -std=c++17 -framework Metal -framework Foundation vectoradd.cpp -o vectoradd
./vectoradd 1000000

# Reduction
clang++ -std=c++17 -framework Metal -framework Foundation reduction.cpp -o reduction
./reduction 1000000
```

## Features

### Standard C++ Implementation
- Vector addition
- Array reduction (sum)
- Array statistics (min, max, average)
- High-resolution timing measurements
- Modern C++ features (RAII, algorithms, etc.)

### Metal Implementation (GPU)
- GPU-accelerated vector addition
- GPU-accelerated reduction
- Efficient memory management
- Error handling and validation
- Performance measurements

## Performance Considerations

- The standard C++ implementation is optimized for CPU execution
- The Metal implementation leverages GPU acceleration for better performance on large datasets
- Both implementations include timing measurements for performance comparison
- Results are verified for correctness

## Notes

- The Metal implementation requires Xcode Command Line Tools
- The standard C++ implementation works on any platform with a C++17 compiler
- Performance may vary depending on your hardware configuration

## License

This project is open source and available under the MIT License.
