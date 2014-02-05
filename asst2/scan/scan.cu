#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <math.h>

#include "CycleTimer.h"

#define BLOCK_SIZE 128

extern float toBW(int bytes, float sec);

/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void scan_warp(int* array, int threadIndex){
	int lane = threadIndex & 31; //index of thread in wrap (0...31)
	
	if(lane >= 1){
		array[threadIndex] += array[threadIndex - 1];
	}
	if(lane >= 2){
		array[threadIndex] += array[threadIndex - 2];
	}
	if(lane >= 4){
		array[threadIndex] += array[threadIndex - 4];
	}
	if(lane >= 8){
		array[threadIndex] += array[threadIndex - 8];
	}
	if(lane >= 16){
		array[threadIndex] += array[threadIndex - 16];
	}
	//return (lane>0) ? array[threadIndex - 1] : 0;
}

__global__ void scan_block(int* input, int* array, int warp_width, int threads_per_block, int* device_debug, int N){
	int globalIndex = blockIdx.x*threads_per_block + threadIdx.x;
	int lane = threadIdx.x & (warp_width-1);
	int warpid = threadIdx.x / warp_width;
	int globalWarpId = blockIdx.x*threads_per_block + warpid;
	
	for(int i=1;i<warp_width;i*=2){
		if(lane >= i){
			array[globalIndex] += array[globalIndex - i];
		}
	}
	
	int val = lane>0 ? array[globalIndex - 1] : 0; //Step 1: per-warp partial scan
	if(blockIdx.x>0){
		val += input[blockIdx.x*threads_per_block-1];
	}
	device_debug[globalIndex] = val;
	
	__syncthreads();
	
	if(lane == (warp_width-1)){
		array[globalWarpId] = array[globalIndex]; //Step 2: copy partial-scan bases
	}
	
	__syncthreads();
	
	if(warpid == 0 && threadIdx.x<(threads_per_block/warp_width)){ //Step 3: scan to accumulate bases
		for(int i=1;i<warp_width;i*=2){
			if(lane >= i){
				array[globalIndex] += array[globalIndex - i];
			}
		}
	}
	__syncthreads();
	
	if(warpid > 0){
		val += array[globalWarpId - 1]; //Step 4: apply bases to all elements
	}
	__syncthreads();
	
	array[globalIndex] = val;
	
	/*
	__syncthreads();
	if(blockIdx.x==0 && threadIdx.x==0){
		printf("Array_block_0:");
		for(int i=0;i<N;i++){
			printf("%d, ", array[i]);
		}
		printf("\n");
	}
	__syncthreads();
	*/
}

__global__ void scan_combine(int* array, int scaned_width, int threads_num){
	int globalStartIndex = (blockIdx.x*threads_num + threadIdx.x)*scaned_width;
	
	int sumBase = 0;
	int partialSumIndexBase = (blockIdx.x*threads_num + 1)*scaned_width - 1;
	for(int i=0;i<(threadIdx.x);i++){
		sumBase += array[partialSumIndexBase + i*scaned_width];
	}
	
	__syncthreads();
	
	for(int j=0;j<scaned_width;j++){
		array[globalStartIndex+j] += sumBase;
	}
	__syncthreads();
}

void exclusive_scan(int* device_start, int length, int* device_result, int* device_debug)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
	
	
	int roundlen = nextPow2(length);
	
	//Maximum threads per block is 512
	int threads_per_block = 128;
	int block_width = 32;
	
	int blocks = 1;
	int threads = roundlen;
	if(roundlen>threads_per_block){
		blocks = roundlen/threads_per_block;
		threads = threads_per_block;
	}
	
	double startTime = CycleTimer::currentSeconds();
	scan_block<<<blocks, threads>>>(device_start, device_result, block_width, threads, device_debug, length);
	cudaThreadSynchronize();
	double endTime = CycleTimer::currentSeconds();
	//printf("Time for scan_block:%.3f ms\n", 1000.f * (endTime-startTime));
	
	/*
	printf("Value:");
	for(int i=0;i<length;i++){
		printf("%d, ", device_result[i]);
	}
	printf("\n");
	*/
	
	startTime = CycleTimer::currentSeconds();
	if(blocks > 1){
		threads = blocks;
		blocks = 1;
		int scaned_width = threads_per_block;
		
		while(threads > threads_per_block){
			blocks = threads/threads_per_block;
			threads = threads_per_block;
			
			scan_combine<<<blocks, threads>>>(device_result, scaned_width, threads);
			
			threads = blocks;
			blocks = 1;
			scaned_width *= threads_per_block;
		}
		
		scan_combine<<<blocks, threads>>>(device_result, scaned_width, threads);
	}
	cudaThreadSynchronize();
	endTime = CycleTimer::currentSeconds();

/*int* scanhost = (int*)malloc(sizeof(int)*roundlen);
cudaMemcpy(scanhost, device_result, roundlen*sizeof(int), cudaMemcpyDeviceToHost);
printf("SCANN: ");
for(int i=0;i<length;i++){
    printf("%d, ", scanhost[i]);
}
printf("\n");*/
	//printf("Time for scan_combine:%.3f ms\n", 1000.f * (endTime-startTime));
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray, bool debug)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    printf("Made it to cudaScan\n");	
	int* device_debug;
	cudaMalloc((void **)&device_debug, sizeof(int) * rounded_length);
    exclusive_scan(device_input, end - inarray, device_result, device_debug);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void process_input(int* input, int length, int* tmp){
	int index = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(index < length-1)
		tmp[index] = (input[index+1]-input[index] == 0) ? 1 : 0;
/*__syncthreads();
if(blockIdx.x==0 && threadIdx.x==0){
    printf("PROCE: ");
    for(int i=0;i<length;i++){
        printf("%d, ", tmp[i]);
    }
    printf("\n");
}
__syncthreads();*/
}

__global__ void repeats(int* deduct, int* scan, int* result, int length){
	int index = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(index<length && deduct[index]==1)
		result[scan[index]] = index;
        __syncthreads();
        if(blockIdx.x==0 && threadIdx.x==0){
            scan[0] = scan[length-1];
        }
/*__syncthreads();
if(blockIdx.x==0 && threadIdx.x==0){
    printf("SCANN: ");
    for(int i=0;i<length;i++){
        printf("%d, ", scan[i]);
    }
    printf("\n");
}
__syncthreads();*/

}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */
/*printf("INPUT: ");
for(int i=0;i<length;i++){
    printf("%d, ", device_input[i]);
}
printf("\n");*/
	int roundlen = nextPow2(length);

	int *device_debug;
	cudaMalloc((void **)&device_debug, roundlen * sizeof(int));
	
	int *tmp_deduct;
	cudaMalloc((void **)&tmp_deduct, roundlen * sizeof(int));
	
	int *tmp_scan;
	cudaMalloc((void **)&tmp_scan, roundlen * sizeof(int));
	
        int blocks = roundlen/BLOCK_SIZE;
        if(blocks==0)blocks=1;
        printf("call process_input \n"); 
	process_input<<<blocks, BLOCK_SIZE>>>(device_input, length, tmp_deduct);
        cudaThreadSynchronize();

        cudaMemcpy(tmp_scan, tmp_deduct, roundlen*sizeof(int), cudaMemcpyDeviceToDevice);

	exclusive_scan(tmp_deduct, roundlen, tmp_scan, device_debug);
        cudaThreadSynchronize();

        repeats<<<blocks, BLOCK_SIZE>>>(tmp_deduct, tmp_scan, device_output, length);         
        cudaThreadSynchronize();
        int *host_scan = (int*)malloc(sizeof(int));
        cudaMemcpy(host_scan, tmp_scan, sizeof(int), cudaMemcpyDeviceToHost);

        //printf("Host scan is %d\n", *host_scan);

    return *host_scan;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    printf("---------CALL FIND_REPEATS\n");    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
