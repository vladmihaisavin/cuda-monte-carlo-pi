
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_BLOCKS 2048
#define NUM_THREADS_PER_BLOCK 32
#define NUM_SAMPLES_PER_THREAD 1e3 * 2
#define KERNEL_ITERATIONS 1e1

float getPiValue(int numSamplesInCircle) {
    return 4.0 * (float) numSamplesInCircle / (NUM_SAMPLES_PER_THREAD * NUM_THREADS_PER_BLOCK * NUM_BLOCKS * KERNEL_ITERATIONS);
}

cudaError_t countSamplesInCircleWithCuda(int iterationNo, int* host_sampleCountPerBlock);

__global__ void countSamplesInCircleKernel(int* device_sampleCountPerBlock, curandState* states)
{
    __shared__ int sampleCountPerThreads[NUM_THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float device_x, device_y;

    curand_init(clock64(), idx, 0, &states[idx]);

    int samplesInCircle = 0;

    for (int i = 0; i < NUM_SAMPLES_PER_THREAD; ++i) {
        device_x = curand_uniform(&states[idx]); // Random x position in [0,1]
        device_y = curand_uniform(&states[idx]); // Random y position in [0,1]
        if (device_x * device_x + device_y * device_y <= 1.0f) {
            samplesInCircle++;
        }
    }

    sampleCountPerThreads[threadIdx.x] = samplesInCircle;

    if (threadIdx.x == 0) {
        int totalSamplesInCircle = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            totalSamplesInCircle += sampleCountPerThreads[i];
        }
        device_sampleCountPerBlock[blockIdx.x] = totalSamplesInCircle;
    }
}

int main()
{
    std::clock_t c_start = std::clock();

    int* host_sampleCountPerBlock = new int[NUM_BLOCKS];
    int numSamplesInCircle = 0;
    cudaError_t cudaStatus;
    int memorySize = int(NUM_SAMPLES_PER_THREAD * NUM_THREADS_PER_BLOCK * NUM_BLOCKS) / 1024 / 1024 * 4;

    for (int i = 0; i < KERNEL_ITERATIONS; ++i) {
        // Call kernel    
        cudaStatus = countSamplesInCircleWithCuda(i, host_sampleCountPerBlock);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "countSamplesInCircleWithCuda failed!");
            return 1;
        }

        for (int i = 0; i < NUM_BLOCKS; ++i) {
            numSamplesInCircle += host_sampleCountPerBlock[i];
        }

        printf("Device memory used: %dMb\n", memorySize);
    }
   

    float piValue = getPiValue(numSamplesInCircle);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    printf("Total processing time: %lf ms\n", time_elapsed_ms);
    printf("Number of samples: %d\n", int(NUM_SAMPLES_PER_THREAD * NUM_THREADS_PER_BLOCK * NUM_BLOCKS * KERNEL_ITERATIONS));
    printf("pi = %f\n", piValue);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t countSamplesInCircleWithCuda(int iterationNo, int* host_sampleCountPerBlock)
{
    cudaError_t cudaStatus;
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    const size_t COUNT_SIZE = NUM_BLOCKS * sizeof(int);
    float elapsedTime;

    cudaEventRecord(start2, 0);
    printf("\nDevice malloc sampleCountPerBlock...\n");
    int* device_sampleCountPerBlock;
    cudaStatus = cudaMalloc(&device_sampleCountPerBlock, COUNT_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (device_sampleCountPerBlock)\n");
        goto Error;
    }
    curandState* devStates;
    cudaMalloc((void**)&devStates, NUM_BLOCKS * NUM_THREADS_PER_BLOCK * sizeof(curandState));
    // Launch a kernel on the GPU with one thread for each element.
    printf("Launching kernel (%d)...\n", iterationNo);
    countSamplesInCircleKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(device_sampleCountPerBlock, devStates);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime, start2, stop2);
    printf("Kernel execution time: %f\n", elapsedTime);

    cudaEventRecord(start3, 0);
    printf("Copying sampleCountPerBlock from device to host...\n");
    cudaStatus = cudaMemcpy(host_sampleCountPerBlock, device_sampleCountPerBlock, COUNT_SIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (device_sampleCountPerBlock)\n");
        goto Error;
    }
    cudaEventRecord(stop3, 0);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&elapsedTime, start3, stop3);
    printf("Copy from --graphics card-- to --proc-- time: %f\n", elapsedTime);

Error:
    cudaFree(device_sampleCountPerBlock);
    
    return cudaStatus;
}
