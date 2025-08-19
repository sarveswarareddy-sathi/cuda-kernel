/*
 *  CUDA Search Kernel :
 *      --> Prepare data (either random numbers or from a CSV file)
 *      --> Copy it to the GPU
 *      --> Launch a CUDA kernel that searches for a target value
 *      --> Copy the result (index where it was found) back to the CPU
 *      --> Save results to a file
 *
 *  Usage : nvcc -O2 -std=c++14 -o search search.cu
 *      --> ./search                   # default - random data, sorted, random/likely-present target
 *      --> ./search -n 100000 -v 42   # 100000 elements, look for 42
 *      --> ./search -f input.csv      # read data from CSV
 *      --> ./search -s false          # do not sort the input
 *      --> ./search -t 512            # use 512 threads per block
 *      --> ./search -p out            # output file will be : out.txt
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <ctime>
#include <limits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__constant__ int d_v;

__global__ void searchKernel(const int* __restrict__ d_data, int* __restrict__ d_foundIndex, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElements) return;
    int val = d_data[i];
    if (val == d_v) {
        atomicCAS(d_foundIndex, -1, i);
    }
}

int* allocateRandomHostMemory(int numElements) {
    size_t size = static_cast<size_t>(numElements) * sizeof(int);
    int* h_data = (int*)std::malloc(size);
    if(!h_data) {
        std::fprintf(stderr, "Failed to allocate host memory\n");
        std::exit(EXIT_FAILURE);
    }

    // Seed RNG and fill with random ints
    std::srand((unsigned int)std::time(nullptr));
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = std::rand();
    }
    return h_data;
}

std::tuple<int*, int> readCSV(const std::string &filename){
    std::ifstream in(filename);
    if(!in.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::vector<int> temp;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        int val;
        while (ss >> val) {
            temp.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
    }
    in.close();

    int n = static_cast<int>(temp.size());
    int* h_data = (int*)std::malloc(sizeof(int) * n);
    if (!h_data) {
        std::fprintf(stderr, "Failed to allocate host memory for CSV data\n");
        std::exit(EXIT_FAILURE);
    }
    std::copy(temp.begin(), temp.end(), h_data);
    return {h_data, n};
}

std::tuple<int*, int*> allocateDeviceMemory(int numElements) {
    int* d_data = nullptr;
    int* d_foundIndex = nullptr;

    // Allocate array for data on device (global memory)
    CUDA_CHECK(cudaMalloc((void**)&d_data, sizeof(int) * (size_t)numElements));

    // Allocate a single int for the found index result
    CUDA_CHECK(cudaMalloc((void**)&d_foundIndex, sizeof(int)));

    return {d_data, d_foundIndex};
}

void copyFromHostToDevice(int h_target,
                          const int* h_data,
                          int initialFoundIndex,
                          int* d_data,
                          int* d_foundIndex,
                          int numElements)
{
    // Copy data array from Host (CPU) to Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_data, h_data,
                          sizeof(int) * (size_t)numElements,
                          cudaMemcpyHostToDevice));

    // Copy the scalar target value into constant memory on the device
    CUDA_CHECK(cudaMemcpyToSymbol(d_v, &h_target, sizeof(int)));

    // Initialize device result to -1 (meaning "not found yet")
    CUDA_CHECK(cudaMemcpy(d_foundIndex, &initialFoundIndex,
                          sizeof(int), cudaMemcpyHostToDevice));
}

void executeKernel(const int* d_data,
                   int* d_foundIndex,
                   int numElements,
                   int threadsPerBlock)
{
    // Defensive clamp: typical GPUs support up to 1024 threads per block in 1D
    if (threadsPerBlock <= 0) threadsPerBlock = 256;
    if (threadsPerBlock > 1024) threadsPerBlock = 1024;

    // Compute number of blocks to cover all elements:
    // blocks = ceil(numElements / threadsPerBlock)
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel: <<< number_of_blocks, threads_per_block >>>
    searchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_foundIndex, numElements);

    // Check for launch errors, then synchronize to catch runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void copyFromDeviceToHost(const int* d_foundIndex, int& h_foundIndex) {
    CUDA_CHECK(cudaMemcpy(&h_foundIndex, d_foundIndex,
                          sizeof(int), cudaMemcpyDeviceToHost));
}

void outputToFile(const std::string& currentPartId,
                  const int* h_data,
                  int numElements,
                  int searchValue,
                  int foundIndex)
{
    std::string outputFileName = "output-" + currentPartId + ".txt";
    std::ofstream out(outputFileName, std::ios::app);
    if (!out.is_open()) {
        std::fprintf(stderr, "Could not open output file: %s\n", outputFileName.c_str());
        std::exit(EXIT_FAILURE);
    }

    out << "Data: ";
    for (int i = 0; i < numElements; ++i) {
        out << h_data[i] << " ";
    }
    out << "\n";
    out << "Searching for value: " << searchValue << "\n";
    out << "Found Index: " << foundIndex << "\n";

    out.close();
}

std::tuple<int, int, std::string, int, std::string, bool>
parseCommandLineArguments(int argc, char* argv[])
{
    int numElements = 10;
    int h_v = -1;                    // -1 means "choose for me" if not provided
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    bool sortInputData = true;
    std::string inputFilename = "NULL";

    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "-s" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "false") sortInputData = false;
            else                sortInputData = true;
        } else if (opt == "-t" && i + 1 < argc) {
            threadsPerBlock = std::atoi(argv[++i]);
        } else if (opt == "-n" && i + 1 < argc) {
            numElements = std::atoi(argv[++i]);
        } else if (opt == "-v" && i + 1 < argc) {
            h_v = std::atoi(argv[++i]);
        } else if (opt == "-f" && i + 1 < argc) {
            inputFilename = argv[++i];
        } else if (opt == "-p" && i + 1 < argc) {
            currentPartId = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", opt.c_str());
            std::exit(EXIT_FAILURE);
        }
    }

    return {numElements, h_v, currentPartId, threadsPerBlock, inputFilename, sortInputData};
}

std::tuple<int*, int, int>
setUpSearchInput(const std::string& inputFilename,
                 int numElements,
                 int h_v,
                 bool sortInputData)
{
    int* h_data = nullptr;

    if (inputFilename != "NULL") {
        // Load from CSV; this overrides numElements
        auto [csvData, n] = readCSV(inputFilename);
        h_data = csvData;
        numElements = n;
        if (numElements <= 0) {
            std::fprintf(stderr, "CSV input contained no values.\n");
            std::exit(EXIT_FAILURE);
        }
    } else {
        // Generate random data of length numElements
        if (numElements <= 0) {
            std::fprintf(stderr, "numElements must be > 0\n");
            std::exit(EXIT_FAILURE);
        }
        h_data = allocateRandomHostMemory(numElements);
    }

    // Optional: sort the data (useful if you later want to implement binary search)
    if (sortInputData) {
        std::sort(h_data, h_data + numElements);
    }

    // Choose a target if not provided:
    //  - 5/6 chance: pick an existing element so the search likely succeeds
    //  - 1/6 chance: pick a random value (may or may not exist)
    if (h_v == -1) {
        std::srand((unsigned int)std::time(nullptr));
        int diceRoll = std::rand() % 6;
        h_v = (diceRoll < 5) ? h_data[std::rand() % numElements] : std::rand();
    }

    return {h_data, numElements, h_v};
}

void cleanUp(int* h_data, int* d_data, int* d_foundIndex) {
    if (d_data)        CUDA_CHECK(cudaFree(d_data));
    if (d_foundIndex)  CUDA_CHECK(cudaFree(d_foundIndex));
    if (h_data)        std::free(h_data);

    // Reset device (optional but good practice; ensures profilers flush data)
    CUDA_CHECK(cudaDeviceReset());
}

int main(int argc, char* argv[])
{
    // 1) Parse CLI options
    auto [numElements, h_v, currentPartId, threadsPerBlock, inputFilename, sortInputData]
        = parseCommandLineArguments(argc, argv);

    // 2) Prepare input on host (CPU)
    auto [h_data, n, target] = setUpSearchInput(inputFilename, numElements, h_v, sortInputData);
    numElements = n;
    h_v = target;

    // 3) Allocate memory on device (GPU)
    auto [d_data, d_foundIndex] = allocateDeviceMemory(numElements);

    // 4) Copy data + target to device
    int h_foundIndex = -1; // host-side initial value; -1 means "not found yet"
    copyFromHostToDevice(h_v, h_data, h_foundIndex, d_data, d_foundIndex, numElements);

    // 5) Launch the kernel
    executeKernel(d_data, d_foundIndex, numElements, threadsPerBlock);

    // 6) Retrieve result back on host
    copyFromDeviceToHost(d_foundIndex, h_foundIndex);

    // 7) Persist results to a file
    outputToFile(currentPartId, h_data, numElements, h_v, h_foundIndex);

    // 8) Cleanup
    cleanUp(h_data, d_data, d_foundIndex);

    // Also print to console for quick feedback
    std::cout << "Search value: " << h_v << "\n";
    std::cout << "Found Index:  " << h_foundIndex << "\n";
    return 0;
}