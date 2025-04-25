/**
* @file hawkZip_compressor.h
* @brief Implements compression and decompression kernels for the hawkZip algorithm
* 
* This file contains the core functions that handle lossy compression of floating-point data
* while maintaining error bounds specified by the user. The implementation uses OpenMP
* for parallel processing to improve performance.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>  // For SSE2 intrinsics - though not currently used in the code
#include <omp.h>        // For OpenMP parallel processing

#define NUM_THREADS 32   // Number of threads to use for parallel processing

/**
* @brief Core compression kernel function for hawkZip
*
* This function performs lossy compression on floating-point data using a block-based
* quantization approach with fixed-length encoding. The compression is done in parallel
* using OpenMP.
*
* @param oriData       Original floating-point data to compress
* @param cmpData       Output buffer to store compressed data
* @param absQuant      Temporary array to store absolute quantization values
* @param signFlag      Array to store sign flags for each data block
* @param fixedRate     Array to store compression rate for each data block
* @param threadOfs     Array to store byte offsets for each thread
* @param nbEle         Number of elements in the original data array
* @param cmpSize       Pointer to store the final compressed size in bytes
* @param errorBound    Maximum allowed error for lossy compression
*/
void hawkZip_compress_kernel(float* oriData, unsigned char* cmpData, int* absQuant, unsigned int* signFlag, int* fixedRate, unsigned int* threadOfs, size_t nbEle, size_t* cmpSize, float errorBound)
{
   // Calculate chunk size for dividing work among threads
   int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
   omp_set_num_threads(NUM_THREADS);
   
   // Begin parallel compression region
   #pragma omp parallel
   {
       // Divide data into chunks for each thread
       int thread_id = omp_get_thread_num();
       int start = thread_id * chunk_size;          // Starting index for this thread
       int end = start + chunk_size;                // Ending index for this thread
       if(end > nbEle) end = nbEle;                 // Boundary check
       int block_num = (chunk_size+31)/32;          // Number of blocks per thread
       int start_block = thread_id * block_num;     // Starting block index
       int block_start, block_end;                  // Block boundaries
       const float recip_precision = 0.5f/errorBound; // Reciprocal of precision for quantization
       int sign_ofs;                                // Offset for sign bit
       unsigned int thread_ofs = 0;                 // Byte offset for this thread's compressed data

       // Process each block assigned to this thread
       for(int i=0; i<block_num; i++)
       {
           // Calculate the boundaries of the current block
           block_start = start + i * 32;            // 32 elements per block
           block_end = (block_start+32) > end ? end : block_start+32; // Handle partial blocks
           float data_recip;                        // Temporary storage for scaled data
           int s;                                   // Sign indicator
           int curr_quant, max_quant=0;             // Current and maximum quantization values
           int curr_block = start_block + i;        // Current block index
           unsigned int sign_flag = 0;              // Sign flag for current block
           int temp_fixed_rate;                     // Bit length for fixed-rate encoding
           
           // Perform quantization on each element in the block
            for(int j=block_start; j<block_end; j++)
           {
               // Step 1: Quantize the float value based on error bound
               data_recip = oriData[j] * recip_precision;
               s = data_recip >= -0.5f ? 0 : 1;     // Handle rounding for negative values
               curr_quant = (int)(data_recip + 0.5f) - s;
               
               // Step 2: Store sign information in a bit flag
               sign_ofs = j % 32;                   // Position within the block
               sign_flag |= (curr_quant < 0) << (31 - sign_ofs); // Set sign bit if negative
               
               // Step 3: Store absolute value and track maximum
               max_quant = max_quant > abs(curr_quant) ? max_quant : abs(curr_quant);
               absQuant[j] = abs(curr_quant);       // Store absolute value
           }

           // Store metadata for this block
           signFlag[curr_block] = sign_flag;        // Store sign flag
           
           // Calculate minimum number of bits needed to represent the maximum value
           temp_fixed_rate = max_quant==0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_quant);
           fixedRate[curr_block] = temp_fixed_rate; // Store bit length
           cmpData[curr_block] = (unsigned char)temp_fixed_rate; // Store in output buffer

           // Calculate bytes needed for this block (metadata + data)
           thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
       }

       // Store total bytes needed for this thread's compressed data
       threadOfs[thread_id] = thread_ofs;
       #pragma omp barrier  // Wait for all threads to complete this phase

       // Calculate global offset for this thread in the output buffer
       unsigned int global_ofs = 0;
       for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
       unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

       // Encode and write compressed data to output buffer

       for(int i=0; i<block_num; i++)
       {
           // Get block boundaries and metadata
           block_start = start + i * 32;
           block_end = (block_start+32) > end ? end : block_start+32;
           int curr_block = start_block + i;
           int temp_fixed_rate = fixedRate[curr_block];
           unsigned int sign_flag = signFlag[curr_block];

           // Skip empty blocks (all zeros)
           if(temp_fixed_rate)
           {
               // Write sign flag (4 bytes)
               cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 24); // Most significant byte
               cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 16);
               cmpData[cmp_byte_ofs++] = 0xff & (sign_flag >> 8);
               cmpData[cmp_byte_ofs++] = 0xff & sign_flag;         // Least significant byte

               // Write quantized data bits
               unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3; // Temp storage for each byte
               int mask = 1;  // Bit mask for extracting bits
               
               // Process each bit position (fixed-rate encoding)
               for(int j=0; j<temp_fixed_rate; j++)
               {
                    // Clear temporary byte buffers
                    tmp_char0 = 0;
                    tmp_char1 = 0;
                    tmp_char2 = 0;
                    tmp_char3 = 0;

                // Single statement for each byte
                tmp_char0 = ((absQuant[block_start]   & mask) ? 0x80 : 0) |
                            ((absQuant[block_start+1] & mask) ? 0x40 : 0) |
                            ((absQuant[block_start+2] & mask) ? 0x20 : 0) |
                            ((absQuant[block_start+3] & mask) ? 0x10 : 0) |
                            ((absQuant[block_start+4] & mask) ? 0x08 : 0) |
                            ((absQuant[block_start+5] & mask) ? 0x04 : 0) |
                            ((absQuant[block_start+6] & mask) ? 0x02 : 0) |
                            ((absQuant[block_start+7] & mask) ? 0x01 : 0);

                tmp_char1 = ((absQuant[block_start+8]  & mask) ? 0x80 : 0) |
                            ((absQuant[block_start+9]  & mask) ? 0x40 : 0) |
                            ((absQuant[block_start+10] & mask) ? 0x20 : 0) |
                            ((absQuant[block_start+11] & mask) ? 0x10 : 0) |
                            ((absQuant[block_start+12] & mask) ? 0x08 : 0) |
                            ((absQuant[block_start+13] & mask) ? 0x04 : 0) |
                            ((absQuant[block_start+14] & mask) ? 0x02 : 0) |
                            ((absQuant[block_start+15] & mask) ? 0x01 : 0);

                tmp_char2 = ((absQuant[block_start+16] & mask) ? 0x80 : 0) |
                            ((absQuant[block_start+17] & mask) ? 0x40 : 0) |
                            ((absQuant[block_start+18] & mask) ? 0x20 : 0) |
                            ((absQuant[block_start+19] & mask) ? 0x10 : 0) |
                            ((absQuant[block_start+20] & mask) ? 0x08 : 0) |
                            ((absQuant[block_start+21] & mask) ? 0x04 : 0) |
                            ((absQuant[block_start+22] & mask) ? 0x02 : 0) |
                            ((absQuant[block_start+23] & mask) ? 0x01 : 0);

                // For the last byte, use conditional expressions for boundary checking
                tmp_char3 = ((block_start+24 < block_end && (absQuant[block_start+24] & mask)) ? 0x80 : 0) |
                            ((block_start+25 < block_end && (absQuant[block_start+25] & mask)) ? 0x40 : 0) |
                            ((block_start+26 < block_end && (absQuant[block_start+26] & mask)) ? 0x20 : 0) |
                            ((block_start+27 < block_end && (absQuant[block_start+27] & mask)) ? 0x10 : 0) |
                            ((block_start+28 < block_end && (absQuant[block_start+28] & mask)) ? 0x08 : 0) |
                            ((block_start+29 < block_end && (absQuant[block_start+29] & mask)) ? 0x04 : 0) |
                            ((block_start+30 < block_end && (absQuant[block_start+30] & mask)) ? 0x02 : 0) |
                            ((block_start+31 < block_end && (absQuant[block_start+31] & mask)) ? 0x01 : 0);

                    // Write the packed bytes to output buffer
                    cmpData[cmp_byte_ofs++] = tmp_char0;
                    cmpData[cmp_byte_ofs++] = tmp_char1;
                    cmpData[cmp_byte_ofs++] = tmp_char2;
                    cmpData[cmp_byte_ofs++] = tmp_char3;

                    // Shift mask to next bit position
                    mask <<= 1;
               }
           }
       }
       
       // Calculate and store total compressed size (only in last thread)
       if(thread_id == NUM_THREADS - 1)
       {
           unsigned int cmpBlockInBytes = 0;
           for(int i=0; i<=thread_id; i++) cmpBlockInBytes += threadOfs[i];
           *cmpSize = (size_t)(cmpBlockInBytes + block_num * NUM_THREADS);
       }
   }
}

/**
* @brief Core decompression kernel function for hawkZip
*
* This function performs decompression of data previously compressed with hawkZip_compress_kernel.
* It restores the original floating-point data within the specified error bounds.
*
* @param decData       Output buffer to store decompressed data
* @param cmpData       Compressed data to decompress
* @param absQuant      Temporary array to store absolute quantization values
* @param fixedRate     Array to store compression rate for each data block
* @param threadOfs     Array to store byte offsets for each thread
* @param nbEle         Number of elements in the original/decompressed data
* @param errorBound    Error bound used during compression
*/
void hawkZip_decompress_kernel(float* decData, unsigned char* cmpData, int* absQuant, int* fixedRate, unsigned int* threadOfs, size_t nbEle, float errorBound)
{
   // Calculate chunk size for dividing work among threads
   int chunk_size = (nbEle + NUM_THREADS - 1) / NUM_THREADS;
   omp_set_num_threads(NUM_THREADS);
   
   // Begin parallel decompression region
   #pragma omp parallel
   {
       // Divide data into chunks for each thread
       int thread_id = omp_get_thread_num();
       int start = thread_id * chunk_size;          // Starting index for this thread
       int end = start + chunk_size;                // Ending index for this thread
       if(end > nbEle) end = nbEle;                 // Boundary check
       int block_num = (chunk_size+31)/32;          // Number of blocks per thread
       int block_start, block_end;                  // Block boundaries
       int start_block = thread_id * block_num;     // Starting block index
       unsigned int thread_ofs = 0;                 // Byte offset for this thread's compressed data

       // First pass: read block metadata and calculate offsets
       for(int i=0; i<block_num; i++)
       {
           // Read fixed-rate (bits per value) for this block
           int curr_block = start_block + i;
           int temp_fixed_rate = (int)cmpData[curr_block];
           fixedRate[curr_block] = temp_fixed_rate;

           // Calculate bytes needed for this block (metadata + data)
           thread_ofs += temp_fixed_rate ? (32+temp_fixed_rate*32)/8 : 0;
       }

       // Store total bytes needed for this thread's compressed data
       threadOfs[thread_id] = thread_ofs;
       #pragma omp barrier  // Wait for all threads to complete this phase

       // Calculate global offset for this thread in the input buffer
       unsigned int global_ofs = 0;
       for(int i=0; i<thread_id; i++) global_ofs += threadOfs[i];
       unsigned int cmp_byte_ofs = global_ofs + block_num * NUM_THREADS;

       // Process each block to reconstruct data
       for(int i=0; i<block_num; i++)
       {
           // Get block boundaries and metadata
           block_start = start + i * 32;
           block_end = (block_start+32) > end ? end : block_start+32;
           int curr_block = start_block + i;
           int temp_fixed_rate = fixedRate[curr_block];
           unsigned int sign_flag = 0;
           int sign_ofs;

           // Skip empty blocks (all zeros)
           if(temp_fixed_rate)
           {
               // Read sign flag (4 bytes)
               sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                           (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                           (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                           (0x000000ff & cmpData[cmp_byte_ofs++]);

               // Read quantized data bits
               unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3; // Temp storage for each byte
               
               // Process each bit position (fixed-rate encoding)
               for(int j=0; j<temp_fixed_rate; j++)
               {
                   // Read packed bytes
                   tmp_char0 = cmpData[cmp_byte_ofs++];
                   tmp_char1 = cmpData[cmp_byte_ofs++];
                   tmp_char2 = cmpData[cmp_byte_ofs++];
                   tmp_char3 = cmpData[cmp_byte_ofs++];

                    // Unpack bits for elements 0-7 from first byte
                    absQuant[block_start]   |= ((tmp_char0 >> 7) & 0x00000001) << j;
                    absQuant[block_start+1] |= ((tmp_char0 >> 6) & 0x00000001) << j;
                    absQuant[block_start+2] |= ((tmp_char0 >> 5) & 0x00000001) << j;
                    absQuant[block_start+3] |= ((tmp_char0 >> 4) & 0x00000001) << j;
                    absQuant[block_start+4] |= ((tmp_char0 >> 3) & 0x00000001) << j;
                    absQuant[block_start+5] |= ((tmp_char0 >> 2) & 0x00000001) << j;
                    absQuant[block_start+6] |= ((tmp_char0 >> 1) & 0x00000001) << j;
                    absQuant[block_start+7] |= ((tmp_char0 >> 0) & 0x00000001) << j;

                    // Unpack bits for elements 8-15 from second byte
                    absQuant[block_start+8]  |= ((tmp_char1 >> 7) & 0x00000001) << j;
                    absQuant[block_start+9]  |= ((tmp_char1 >> 6) & 0x00000001) << j;
                    absQuant[block_start+10] |= ((tmp_char1 >> 5) & 0x00000001) << j;
                    absQuant[block_start+11] |= ((tmp_char1 >> 4) & 0x00000001) << j;
                    absQuant[block_start+12] |= ((tmp_char1 >> 3) & 0x00000001) << j;
                    absQuant[block_start+13] |= ((tmp_char1 >> 2) & 0x00000001) << j;
                    absQuant[block_start+14] |= ((tmp_char1 >> 1) & 0x00000001) << j;
                    absQuant[block_start+15] |= ((tmp_char1 >> 0) & 0x00000001) << j;

                    // Unpack bits for elements 16-23 from third byte
                    absQuant[block_start+16] |= ((tmp_char2 >> 7) & 0x00000001) << j;
                    absQuant[block_start+17] |= ((tmp_char2 >> 6) & 0x00000001) << j;
                    absQuant[block_start+18] |= ((tmp_char2 >> 5) & 0x00000001) << j;
                    absQuant[block_start+19] |= ((tmp_char2 >> 4) & 0x00000001) << j;
                    absQuant[block_start+20] |= ((tmp_char2 >> 3) & 0x00000001) << j;
                    absQuant[block_start+21] |= ((tmp_char2 >> 2) & 0x00000001) << j;
                    absQuant[block_start+22] |= ((tmp_char2 >> 1) & 0x00000001) << j;
                    absQuant[block_start+23] |= ((tmp_char2 >> 0) & 0x00000001) << j;

                    // Unpack bits for elements 24-31 from fourth byte
                    if (block_start+24 < block_end) absQuant[block_start+24] |= ((tmp_char3 >> 7) & 0x00000001) << j;
                    if (block_start+25 < block_end) absQuant[block_start+25] |= ((tmp_char3 >> 6) & 0x00000001) << j;
                    if (block_start+26 < block_end) absQuant[block_start+26] |= ((tmp_char3 >> 5) & 0x00000001) << j;
                    if (block_start+27 < block_end) absQuant[block_start+27] |= ((tmp_char3 >> 4) & 0x00000001) << j;
                    if (block_start+28 < block_end) absQuant[block_start+28] |= ((tmp_char3 >> 3) & 0x00000001) << j;
                    if (block_start+29 < block_end) absQuant[block_start+29] |= ((tmp_char3 >> 2) & 0x00000001) << j;
                    if (block_start+30 < block_end) absQuant[block_start+30] |= ((tmp_char3 >> 1) & 0x00000001) << j;
                    if (block_start+31 < block_end) absQuant[block_start+31] |= ((tmp_char3 >> 0) & 0x00000001) << j;
               }

               // Convert quantized values back to floating-point
               int currQuant;
               for(int i=block_start; i<block_end; i++)
               {
                   // Apply sign based on sign flag
                   sign_ofs = i % 32;
                   if(sign_flag & (1 << (31 - sign_ofs)))
                       currQuant = absQuant[i] * -1;  // Negative value
                   else
                       currQuant = absQuant[i];       // Positive value
                   
                   // Convert quantized value back to float using error bound
                   decData[i] = currQuant * errorBound * 2;
               }
           }
       }
   }
}