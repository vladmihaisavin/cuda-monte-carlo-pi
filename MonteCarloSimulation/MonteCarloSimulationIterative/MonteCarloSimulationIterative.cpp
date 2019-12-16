
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>

#define NUM_SAMPLES 1e8

float* generateSamples(int size) {
    float* array = new float[size];
    srand(time(NULL));

    for (int i = 0; i < size; ++i) {
        array[i] = (float)rand() / RAND_MAX;
    }
    return array;
}

float getPiValue(int numSamplesInCircle) {
    return 4.0 * (float)numSamplesInCircle / NUM_SAMPLES;
}

int main()
{
    std::clock_t c_start = std::clock();
    printf("Generating host X samples...\n");
    float* host_randX = generateSamples(NUM_SAMPLES);

    printf("Generating host Y samples...\n");
    float* host_randY = generateSamples(NUM_SAMPLES);
    std::clock_t c_end1 = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end1 - c_start) / CLOCKS_PER_SEC;
    printf("Generated vectors in %lf ms\n", time_elapsed_ms);

    int samplesInCircle = 0;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        if (host_randX[i] * host_randX[i] + host_randY[i] * host_randY[i] <= 1.0f) {
            samplesInCircle++;
        }
    }

    float piValue = getPiValue(samplesInCircle);
    printf("pi = %f\n", piValue);

    std::clock_t c_end2 = std::clock();

    time_elapsed_ms = 1000.0 * (c_end2 - c_end1) / CLOCKS_PER_SEC;
    printf("Pi calculus took %lf ms\n", time_elapsed_ms);
    time_elapsed_ms = 1000.0 * (c_end2 - c_start) / CLOCKS_PER_SEC;
    printf("Whole process took %lf ms\n", time_elapsed_ms);

    return 0;
}
