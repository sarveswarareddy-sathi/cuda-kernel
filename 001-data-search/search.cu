/*
 *  CUDA Search Kernel : 1) Kernel function, 2) Host memory allocation, 3) Device memory allocation, 4) Data transfers, 5) Kernel launch, 6) Result retrieval, & 7) Clean-UP
 *      > Prepare some data (either random numbers or from a CSV file.
 *      > Copy it to the GPU 
 *      > Launch a CUDA kernel that searches for a target value
 *      > Copy the result (index where it was found) back to the CPU.
 *      > Save results to a file.
 */

 __device__ __const__ int d_v;

 __global__ void searchKernel(const int *d_data, int *d_index, int numEl)