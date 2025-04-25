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
    // Calculate the number of blocks needed for parallel processing
    // Each block contains NUM_THREADS elements, rounded up to a multiple of 32
    int blockNum = ((nbEle + NUM_THREADS - 1) / NUM_THREADS + 31) / 32 * NUM_THREADS;
    
    // Allocate memory for quantization and compression data structures
    int* absQuant = (int*)malloc(sizeof(int)*nbEle);             // Array to store absolute quantization values
    unsigned int* signFlag = (unsigned int*)malloc(sizeof(unsigned int)*blockNum);  // Bitflags for storing signs
    int* fixedRate = (int*)malloc(sizeof(int)*blockNum);         // Array to store compression rates for each block
    unsigned int* threadOfs = (unsigned int*)malloc(sizeof(unsigned int)*NUM_THREADS);  // Offsets for each thread
    
    // Initialize the output buffer with zeros
    memset(cmpData, 0, sizeof(float)*nbEle);
    
    // Variables for performance timing
    double timerCMP_start, timerCMP_end;
    
    // Start timing the compression operation
    timerCMP_start = omp_get_wtime();
    
    // Call the internal compression kernel function that does the actual work
    hawkZip_compress_kernel(oriData, cmpData, absQuant, signFlag, fixedRate, threadOfs, nbEle, cmpSize, errorBound);
    
    // End timing and calculate performance metrics
    timerCMP_end = omp_get_wtime();
    
    // Print compression ratio (original size / compressed size)
    printf("hawkZip compression ratio: %f\n", (float)(sizeof(float)*nbEle) / (float)(sizeof(unsigned char)*(*cmpSize)));
    
    // Print throughput in gigabytes per second
    printf("hawkZip compression throughput: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0/1024.0)/(timerCMP_end-timerCMP_start));
    
    // Resize the compressed data buffer to exactly fit the compressed data
    // Note: This line is ineffective as it doesn't update the pointer in the caller's scope
    cmpData = (unsigned char*)realloc(cmpData, sizeof(unsigned char)*(*cmpSize));
    
    // Free temporary memory used during compression
    free(absQuant);
    free(signFlag);
    free(fixedRate);
    free(threadOfs);
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