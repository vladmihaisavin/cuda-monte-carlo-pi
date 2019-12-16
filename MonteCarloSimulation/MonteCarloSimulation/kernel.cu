
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>

#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024
#define NUM_SAMPLES 1e6

float* generateSamples(int size) {
    float* array = new float[size];
    srand(time(NULL));

    for (int i = 0; i < size; ++i) {
        array[i] = (float) rand() / RAND_MAX;
    }
    return array;
}

float getPiValue(int numSamplesInCircle) {
    return 4.0 * (float) numSamplesInCircle / NUM_SAMPLES;
}

cudaError_t countSamplesInCircleWithCuda(float* host_randX, float* host_randY, int* host_sampleCountPerBlock);

__global__ void countSamplesInCircleKernel(float* device_randX, float* device_randY, int* device_sampleCountPerBlock)
{
    __shared__ int sampleCountPerThreads[NUM_THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int samplesInCircle = 0;

    for (int i = idx; i < NUM_SAMPLES; i += blockDim.x * NUM_BLOCKS) {
        if (device_randX[i] * device_randX[i] + device_randY[i] * device_randY[i] <= 1.0f) {
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
    printf("Generating host X samples...\n");
    float* host_randX = generateSamples(NUM_SAMPLES);

    printf("Generating host Y samples...\n");
    float* host_randY = generateSamples(NUM_SAMPLES);
    std::clock_t c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    printf("Generated vectors in: %lf ms\n", time_elapsed_ms);

    int* host_sampleCountPerBlock = new int[NUM_BLOCKS];

    // Call kernel    
    cudaError_t cudaStatus = countSamplesInCircleWithCuda(host_randX, host_randY, host_sampleCountPerBlock);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "countSamplesInCircleWithCuda failed!");
        return 1;
    }

    int numSamplesInCircle = 0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        numSamplesInCircle += host_sampleCountPerBlock[i];
    }

    float piValue = getPiValue(numSamplesInCircle);
    printf("pi = %f\n", piValue);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t countSamplesInCircleWithCuda(float* host_randX, float* host_randY, int* host_sampleCountPerBlock)
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

    const size_t SAMPLE_SIZE = NUM_SAMPLES * sizeof(float);
    float* device_randX;
    float* device_randY;

    printf("Device malloc randX...\n");
    cudaEventRecord(start1, 0);
    cudaStatus = cudaMalloc(&device_randX, SAMPLE_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (device_randX)\n");
        goto Error;
    }

    printf("Device malloc randY...\n");
    cudaStatus = cudaMalloc(&device_randY, SAMPLE_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (device_randY)\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    printf("Copying randX from host to device...\n");
    cudaStatus = cudaMemcpy(device_randX, host_randX, SAMPLE_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (device_randX)\n");
        goto Error;
    }

    printf("Copying randY from host to device...\n");
    cudaStatus = cudaMemcpy(device_randY, host_randY, SAMPLE_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! (device_randY)\n");
        goto Error;
    }
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start1, stop1);
    printf("Malloc and copy from --proc-- to --graphics card-- time: %f\n", elapsedTime);

    const size_t COUNT_SIZE = NUM_BLOCKS * sizeof(int);

    cudaEventRecord(start2, 0);
    printf("Device malloc sampleCountPerBlock...\n");
    int* device_sampleCountPerBlock;
    cudaStatus = cudaMalloc(&device_sampleCountPerBlock, COUNT_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! (device_sampleCountPerBlock)\n");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    printf("Launching kernel...\n");
    countSamplesInCircleKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(device_randX, device_randY, device_sampleCountPerBlock);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsedTime, start2, stop2);
    printf("Kernel execution time: %f\n", elapsedTime);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    
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
    cudaFree(device_randX);
    cudaFree(device_randY);
    cudaFree(device_sampleCountPerBlock);
    
    return cudaStatus;
}
