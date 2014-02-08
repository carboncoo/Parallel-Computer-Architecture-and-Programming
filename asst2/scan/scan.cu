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
#define WARP_WIDTH 32

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

__global__ void scan_block(int* input, int* array, int threads_per_block, int N){
	int globalIndex = blockIdx.x*threads_per_block + threadIdx.x;
	
	if(globalIndex>(N-1))return;
	
	int lane = threadIdx.x & (WARP_WIDTH-1);
	int warpid = threadIdx.x / WARP_WIDTH;
	int globalWarpId = blockIdx.x*threads_per_block + warpid;
	
	for(int i=1;i<WARP_WIDTH;i*=2){
		if(lane >= i){
			array[globalIndex] += array[globalIndex - i];
		}
	}
	
	int val = lane>0 ? array[globalIndex - 1] : 0; //Step 1: per-warp partial scan
	if(blockIdx.x>0){
		val += input[blockIdx.x*threads_per_block-1];
	}
	
	__syncthreads();
	
	if(lane == (WARP_WIDTH-1)){
		array[globalWarpId] = array[globalIndex]; //Step 2: copy partial-scan bases
	}
	
	__syncthreads();
	
	if(warpid == 0 && threadIdx.x<(threads_per_block/WARP_WIDTH)){ //Step 3: scan to accumulate bases
		for(int i=1;i<WARP_WIDTH;i*=2){
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
}

__global__ void thread_add(int* array, int scaned_width, int length, int* val){
	int index = blockIdx.x*BLOCK_SIZE + scaned_width + threadIdx.x;
	if(index<length)array[index] += val[index/scaned_width];
}

__global__ void scan_combine(int* array, int scaned_width, int threads_num, int N, int* vals){
	__shared__ int partialSum[BLOCK_SIZE];
	
	int globalStartIndex = (blockIdx.x*threads_num + threadIdx.x)*scaned_width;
	if(globalStartIndex>N-1)return;
	int globalEndIndex = globalStartIndex + scaned_width;
	if(globalEndIndex>N)globalEndIndex=N;
	
	partialSum[threadIdx.x] = array[globalEndIndex-1];
	
	__syncthreads();
	
	int lane = threadIdx.x & (WARP_WIDTH-1);
	int warpid = threadIdx.x / WARP_WIDTH;
	
	for(int i=1;i<WARP_WIDTH;i*=2){
		if(lane >= i){
			partialSum[threadIdx.x] += partialSum[threadIdx.x - i];
		}
	}

	int val = lane>0 ? partialSum[threadIdx.x - 1] : 0; //Step 1: per-warp partial scan
	
	__syncthreads();
	
	if(lane == (WARP_WIDTH-1)){
		partialSum[warpid] = partialSum[threadIdx.x]; //Step 2: copy partial-scan bases
	}
	
	__syncthreads();
	
	if(warpid == 0){ //Step 3: scan to accumulate bases
		for(int i=1;i<WARP_WIDTH;i*=2){
			if(lane >= i){
				partialSum[threadIdx.x] += partialSum[threadIdx.x - i];
			}
		}
	}
	__syncthreads();
	
	if(warpid > 0){
		val += partialSum[warpid - 1]; //Step 4: apply bases to all elements
	}
	
	vals[blockIdx.x*BLOCK_SIZE + threadIdx.x] = val;
}

void exclusive_scan(int* device_start, int length, int* device_result)
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
		
	//Maximum threads per block is 512
	
	int blocks = 1;
	int threads = length;
	if(length>BLOCK_SIZE){
		blocks = length/BLOCK_SIZE;
		threads = BLOCK_SIZE;
	}
	
	if(length%BLOCK_SIZE!=0){
		blocks++;
	}
	
	int roundblock = blocks;
	
	scan_block<<<blocks, threads>>>(device_start, device_result, threads, length);
	cudaThreadSynchronize();
	
	if(blocks > 1){
		int* device_val;
		cudaMalloc((void **)&device_val, sizeof(int) * blocks);
		
		threads = blocks;
		blocks = 1;
		int scaned_width = BLOCK_SIZE;
		
		while(threads > BLOCK_SIZE){
			blocks = threads/BLOCK_SIZE;
			if(threads%BLOCK_SIZE!=0)blocks++;
			threads = BLOCK_SIZE;
			
			scan_combine<<<blocks, threads>>>(device_result, scaned_width, threads, length, device_val);
			cudaThreadSynchronize();
			
			roundblock = (length-scaned_width)/BLOCK_SIZE;
			roundblock = (length-scaned_width)%BLOCK_SIZE!=0 ? roundblock+1 : roundblock;
			
			thread_add<<<roundblock, BLOCK_SIZE>>>(device_result, scaned_width, length, device_val);
			cudaThreadSynchronize();
			
			threads = blocks;
			blocks = 1;
			scaned_width *= BLOCK_SIZE;
		}
		
		scan_combine<<<blocks, threads>>>(device_result, scaned_width, threads, length, device_val);
		cudaThreadSynchronize();
		
		roundblock = (length-scaned_width)/BLOCK_SIZE;
		roundblock = (length-scaned_width)%BLOCK_SIZE!=0 ? roundblock+1 : roundblock;
		thread_add<<<roundblock, BLOCK_SIZE>>>(device_result, scaned_width, length, device_val);
		cudaThreadSynchronize();
	}
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
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
	
    exclusive_scan(device_input, end - inarray, device_result);

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
}

__global__ void repeats(int* deduct, int* scan, int* result, int length){
	int index = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	if(index<length && deduct[index]==1){
		result[scan[index]] = index;
	}
	__syncthreads();
	if(blockIdx.x==0 && threadIdx.x==0){
		scan[0] = scan[length-1];
	}
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
	int roundlen = nextPow2(length);
	
	int *tmp_deduct;
	cudaMalloc((void **)&tmp_deduct, length * sizeof(int));
	
	int *tmp_scan;
	cudaMalloc((void **)&tmp_scan, length * sizeof(int));
	
	int blocks = length/BLOCK_SIZE;
	if(length%BLOCK_SIZE!=0)blocks++;
	
	process_input<<<blocks, BLOCK_SIZE>>>(device_input, length, tmp_deduct);
    cudaThreadSynchronize();

    cudaMemcpy(tmp_scan, tmp_deduct, length*sizeof(int), cudaMemcpyDeviceToDevice);

	exclusive_scan(tmp_deduct, length, tmp_scan);
    cudaThreadSynchronize();

	repeats<<<blocks, BLOCK_SIZE>>>(tmp_deduct, tmp_scan, device_output, length);         
	cudaThreadSynchronize();
	int *host_scan = (int*)malloc(sizeof(int));
	cudaMemcpy(host_scan, tmp_scan, sizeof(int), cudaMemcpyDeviceToHost);

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
