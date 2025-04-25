// ha
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>  // OpenMP library for parallel processing
#include "hawkZip_compressor.h"

/**
 * @brief Compresses floating-point data using the hawkZip algorithm
 *
 * This function compresses floating-point data while maintaining error bounds.
 * It uses parallel processing via OpenMP to improve performance.
 *
 * @param oriData      Original floating-point data array to be compressed
 * @param cmpData      Output buffer to store compressed data
 * @param nbEle        Number of elements in the original data array
 * @param cmpSize      Pointer to store the size of compressed data (in bytes)
 * @param errorBound   Maximum allowed error for lossy compression
 */
 void hawkZip_compress(float* oriData, unsigned char* cmpData, size_t nbEle, size_t* cmpSize, float errorBound)
{
    // Optimize block calculations
    const int blockNum = (nbEle + NUM_THREADS*32 - 1) / (NUM_THREADS*32) * NUM_THREADS;
    
    // Allocate memory with proper alignment for SIMD operations (64-byte alignment for cache lines)
    int* absQuant = (int*)_mm_malloc(sizeof(int)*nbEle, 64);
    unsigned int* signFlag = (unsigned int*)_mm_malloc(sizeof(unsigned int)*blockNum, 64);
    int* fixedRate = (int*)_mm_malloc(sizeof(int)*blockNum, 64);
    unsigned int* threadOfs = (unsigned int*)_mm_malloc(sizeof(unsigned int)*NUM_THREADS, 64);
    
    // Initialize arrays in parallel
    #pragma omp parallel sections
    {
        #pragma omp section
        { memset(absQuant, 0, sizeof(int)*nbEle); }
        
        #pragma omp section
        { memset(signFlag, 0, sizeof(unsigned int)*blockNum); }
        
        #pragma omp section
        { memset(fixedRate, 0, sizeof(int)*blockNum); }
        
        #pragma omp section
        { memset(threadOfs, 0, sizeof(unsigned int)*NUM_THREADS); }
    }
    
    // Prefetch first chunk of data
    for(size_t i = 0; i < 1024 && i < nbEle; i += 16) {
        _mm_prefetch((const char*)&oriData[i], _MM_HINT_T0);
    }
    
    // Start timing
    const double timerCMP_start = omp_get_wtime();
    
    // Call compression kernel
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate, threadOfs, nbEle, cmpSize, errorBound);
    
    // End timing
    const double timerCMP_end = omp_get_wtime();
    const double compress_time = timerCMP_end - timerCMP_start;
    
    // Calculate metrics with proper precision
    const double compression_ratio = (double)(sizeof(float)*nbEle) / (double)(sizeof(unsigned char)*(*cmpSize));
    const double throughput_gb_s = (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/compress_time;
    
    // Print metrics with formatted output
    printf("hawkZip   compression ratio:      %.6f\n", compression_ratio);
    printf("hawkZip   compression throughput: %.6f GB/s\n", throughput_gb_s);
    
    // Free aligned memory
    _mm_free(absQuant);
    _mm_free(signFlag);
    _mm_free(fixedRate);
    _mm_free(threadOfs);
}

/**
 * @brief Decompresses data previously compressed with hawkZip_compress
 *
 * This function reconstructs the original floating-point data from compressed data
 * while maintaining the error bounds specified during compression.
 *
 * @param decData      Output buffer to store decompressed floating-point data
 * @param cmpData      Input buffer containing compressed data
 * @param nbEle        Number of elements in the original/decompressed data
 * @param errorBound   Error bound used during compression
 */
void hawkZip_decompress(float* decData, unsigned char* cmpData, size_t nbEle, float errorBound)
{
    // Calculate the number of blocks, just as in compression
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS + 31) / 32 * NUM_THREADS;
    
    // Allocate memory for decompression data structures
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);             // Array to store reconstructed quantization values
    memset(absQuant, 0, sizeof(int)*nbEle);                      // Initialize with zeros
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);         // Array to store compression rates for each block
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);  // Offsets for each thread
    
    // Initialize the output buffer with zeros
    memset(decData, 0, sizeof(float)*nbEle);
    
    // Variables for performance timing
    double timerDEC_start, timerDEC_end;
    
    // Start timing the decompression operation
    timerDEC_start = omp_get_wtime();
    
    // Call the internal decompression kernel function that does the actual work
    hawkZip_decompress_kernel(decData, cmpData, absQuant, fixedRate, threadOfs, nbEle, errorBound);
    
    // End timing and calculate performance metrics
    timerDEC_end = omp_get_wtime();
    
    // Print decompression throughput in gigabytes per second
    printf("hawkZip decompression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerDEC_end-timerDEC_start));
    
    // Free temporary memory used during decompression
    free(absQuant);
    free(fixedRate);
    free(threadOfs);
}