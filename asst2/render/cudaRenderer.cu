#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include "cycleTimer.h"

#define BUNDLESIZE		32
#define PREFIXBLOCKSIZE	128
#define WARP_WIDTH		32

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

struct RenderData {
	//int id;
	
	//added
	//int valid;
	int circleIndex;
	float3 p;
	
	//struct RenderData *next;
	//added end
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles(RenderData* bundles, int* valids, int bundlesWidth, int numCircles) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;
		
    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // for all pixels in the bonding box
	int bundleIndex, pixelX, pixelY;
    for (pixelY=screenMinY; pixelY<screenMaxY; pixelY+=BUNDLESIZE) {
        for (pixelX=screenMinX; pixelX<screenMaxX; pixelX+=BUNDLESIZE) {
			bundleIndex = (pixelY / BUNDLESIZE) * bundlesWidth + (pixelX / BUNDLESIZE);
			bundleIndex = bundleIndex * numCircles + index;
			
			valids[bundleIndex] = 1;
			bundles[bundleIndex].circleIndex = index;
			bundles[bundleIndex].p = p;
        }
		if((pixelX-BUNDLESIZE)!=(screenMaxX-1)){
			bundleIndex = (pixelY / BUNDLESIZE) * bundlesWidth + ((screenMaxX-1) / BUNDLESIZE);
			bundleIndex = bundleIndex * numCircles + index;
			
			valids[bundleIndex] = 1;
			bundles[bundleIndex].circleIndex = index;
			bundles[bundleIndex].p = p;
		}
    }
	if((pixelY-BUNDLESIZE)!=(screenMaxY-1)){
		for (pixelX=screenMinX; pixelX<screenMaxX; pixelX+=BUNDLESIZE) {
			bundleIndex = ((screenMaxY-1) / BUNDLESIZE) * bundlesWidth + (pixelX / BUNDLESIZE);
			bundleIndex = bundleIndex * numCircles + index;
			
			valids[bundleIndex] = 1;
			bundles[bundleIndex].circleIndex = index;
			bundles[bundleIndex].p = p;
        }
		if((pixelX-BUNDLESIZE)!=(screenMaxX-1)){
			bundleIndex = ((screenMaxY-1) / BUNDLESIZE) * bundlesWidth + ((screenMaxX-1) / BUNDLESIZE);
			bundleIndex = bundleIndex * numCircles + index;
			
			valids[bundleIndex] = 1;
			bundles[bundleIndex].circleIndex = index;
			bundles[bundleIndex].p = p;
		}
	}
}

__global__ void renderPixels(int blockSize, int imgWidth, int imgHeight, int totalPixels, int bundlesWidth, int numCircles, RenderData* bundles, int* valids){
	int pixel1D = blockIdx.x * blockSize + threadIdx.x;
	if(pixel1D >= totalPixels){
		return;
	}
	
	int pixelY = pixel1D / imgWidth;
	int pixelX = pixel1D % imgWidth;
	
	int bundleIndex = ((pixelY/BUNDLESIZE)*bundlesWidth + (pixelX/BUNDLESIZE)) * numCircles;
	float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imgWidth + pixelX)]);
	float2 pixelCenterNorm = make_float2((static_cast<float>(pixelX) + 0.5f)/imgWidth, (static_cast<float>(pixelY) + 0.5f)/imgHeight);
	
	float4 color = *imgPtr;
	
	RenderData data;
	int i=0;
	for(;i<numCircles&&valids[i+bundleIndex]!=0;i++){
		data = bundles[i+bundleIndex];
		
		float3 p = data.p;
		float diffX = p.x - pixelCenterNorm.x;
		float diffY = p.y - pixelCenterNorm.y;
		float pixelDist = diffX * diffX + diffY * diffY;

		float rad = cuConstRendererParams.radius[data.circleIndex];;
		float maxDist = rad * rad;

		// circle does not contribute to the image
		if (pixelDist > maxDist)
			continue;

		float3 rgb;
		float alpha;

		if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

			const float kCircleMaxAlpha = .5f;
			const float falloffScale = 4.f;

			float normPixelDist = sqrt(pixelDist) / rad;
			rgb = lookupColor(normPixelDist);

			float maxAlpha = .6f + .4f * (1.f-p.z);
			maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
			alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

		} else {
			// simple: each circle has an assigned color
			int index3 = 3 * data.circleIndex;
			rgb = *(float3*)&(cuConstRendererParams.color[index3]);
			alpha = .5f;
		}

		float oneMinusAlpha = 1.f - alpha;

		// global memory read
		color.x *= oneMinusAlpha;
		color.x += alpha * rgb.x;
		
		color.y *= oneMinusAlpha;
		color.y += alpha * rgb.y;
		
		color.z *= oneMinusAlpha;
		color.z += alpha * rgb.z;
		
		color.w += alpha;
	}	
	// global memory write
	*imgPtr = color;
}

__global__ void scan_block(int* array, int bundleWidth){ //bundleWidth is number of circles
	int inBundleIndex = blockIdx.y*PREFIXBLOCKSIZE + threadIdx.x;
	
	if(inBundleIndex>(bundleWidth-1))return;
	
	__shared__ int warp[PREFIXBLOCKSIZE/WARP_WIDTH];
	
	int globalBundleBase = blockIdx.x*bundleWidth;
	int globalBundleIndex = globalBundleBase + inBundleIndex;
	
	int lane = threadIdx.x & (WARP_WIDTH-1);
	int warpid = threadIdx.x / WARP_WIDTH;
		
	for(int i=1;i<WARP_WIDTH;i*=2){
		if(lane >= i){
			array[globalBundleIndex] += array[globalBundleIndex - i];
		}
	}
	
	int val = array[globalBundleIndex]; //Step 1: per-warp partial scan
	
	__syncthreads();
	
	if(lane == (WARP_WIDTH-1)){
		warp[warpid] = val;
	}
	
	__syncthreads();
	
	if(warpid == 0 && threadIdx.x<(PREFIXBLOCKSIZE/WARP_WIDTH)){ //Step 3: scan to accumulate bases
		for(int i=1;i<PREFIXBLOCKSIZE/WARP_WIDTH;i*=2){
			if(lane >= i){
				warp[threadIdx.x] += warp[threadIdx.x-i];
			}
		}
	}
	__syncthreads();
	
	if(warpid > 0){
		val += warp[warpid-1];
	}
	__syncthreads();
	
	array[globalBundleIndex] = val;
}

__global__ void thread_add(int* array, int scaned_width, int bundleWidth, int combineValGridY, int* val, int boundary, int bundlesTotal){
	int index = blockIdx.y*PREFIXBLOCKSIZE + scaned_width + threadIdx.x;
	int i=1;
	while(index>=i*boundary)i++;
	i--;
	if(index>=boundary*i+scaned_width && index<(boundary*(i+1))){
		int globalIndex = index+blockIdx.x*bundleWidth;
		if(globalIndex<((blockIdx.x+1))*bundleWidth){
			array[globalIndex] += val[blockIdx.x*combineValGridY + index/scaned_width - 1];			
		}
	}
}

__global__ void scan_combine(int* array, int scaned_width, int threads_num, int bundleWidth, int combineValGridY, int* vals){
	__shared__ int partialSum[PREFIXBLOCKSIZE];
	
	int bundleStartIndex = (blockIdx.y*threads_num + threadIdx.x)*scaned_width;
	if(bundleStartIndex>bundleWidth-1){
		return;
	}
	int bundleEndIndex = bundleStartIndex + scaned_width;
	if(bundleEndIndex>bundleWidth){
		bundleEndIndex=bundleWidth;
	}
	
	int gloableBase = blockIdx.x*bundleWidth;
	
	partialSum[threadIdx.x] = array[gloableBase + bundleEndIndex - 1];
	
	__syncthreads();
	
	int lane = threadIdx.x & (WARP_WIDTH-1);
	int warpid = threadIdx.x / WARP_WIDTH;
	
	for(int i=1;i<WARP_WIDTH;i*=2){
		if(lane >= i){
			partialSum[threadIdx.x] += partialSum[threadIdx.x - i];
		}
	}

	int val = partialSum[threadIdx.x]; //Step 1: per-warp partial scan
	
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
	
	vals[combineValGridY*blockIdx.x + blockIdx.y*threads_num + threadIdx.x] = val;
}

__global__ void move_bundle(int* scaned, int* valid, RenderData* bundles, int bundleWidth){
	int bundleIndex = blockIdx.y*PREFIXBLOCKSIZE + threadIdx.x;
	if(bundleIndex<bundleWidth){
		int base = blockIdx.x*bundleWidth;
		int globalIndex = bundleIndex + base;
		
		if(valid[globalIndex]==1){
			int moveToIndex = base+scaned[globalIndex]-1;
			//bundles[moveToIndex].circleIndex = bundles[globalIndex].circleIndex;
			//bundles[moveToIndex].p = bundles[globalIndex].p;
			bundles[moveToIndex] = bundles[globalIndex];
			valid[moveToIndex] = 1;
		}
		if(bundleIndex==bundleWidth-1 && scaned[globalIndex]<bundleWidth){
			valid[scaned[globalIndex]+base] = 0;
		}
	}
}

__global__ void renderLittleCircle(int imgDim, float invWidth, float invHeight, int circleNum){
	int globalIndex = blockIdx.x * 256 + threadIdx.x;
	int pixelY = globalIndex / imgDim;
	int pixelX = globalIndex % imgDim;
	
	float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imgDim + pixelX)]);
	
	float pixelCenterX = invWidth * (static_cast<float>(pixelX) + 0.5f);
	float pixelCenterY = invHeight * (static_cast<float>(pixelY) + 0.5f);
	
	float4 color = *imgPtr;
	float3 rgb;
	for(int i=0;i<circleNum;i++){
		float3 p = *(float3*)(&cuConstRendererParams.position[i*3]);
		float diffX = p.x - pixelCenterX;
		float diffY = p.y - pixelCenterY;
		float pixelDist = diffX * diffX + diffY * diffY;

		float rad = cuConstRendererParams.radius[i];
		float maxDist = rad * rad;

		// circle does not contribute to the image
		if (pixelDist > maxDist)
			continue;

		rgb = *(float3*)&(cuConstRendererParams.color[i*3]);

		color.x = (color.x+rgb.x)*.5f;
		color.y = (color.y+rgb.y)*.5f;
		color.z = (color.z+rgb.z)*.5f;
		color.w += .5f;
	}

    *imgPtr = color;
}
////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaThreadSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {

        // 256 threads per block is a healthy number
        dim3 blockDim(256, 1);
        dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
        cudaThreadSynchronize();
    }
}

void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
	
	int imgWidth = image->width;
	int imgHeight = image->height;
	
	if(numCircles<10){ //For rgb scene
		int totalPixel = imgWidth*imgHeight;
		int blocks = (totalPixel + 255)/256;
		float invWidth = 1.f / imgWidth;
		float invHeight = 1.f / imgHeight;
		renderLittleCircle<<<blocks, 256>>>(imgWidth, invWidth, invHeight, numCircles);
		return;
	}
	
	int bundlesWidth = (imgWidth + BUNDLESIZE - 1) / BUNDLESIZE;
	int bundlesHeight = (imgHeight + BUNDLESIZE - 1) / BUNDLESIZE;
	
	int bundlesTotal = bundlesWidth * bundlesHeight * numCircles;
	
	RenderData* bundles;
	cudaMalloc((void **)&bundles, sizeof(RenderData) * bundlesTotal);
	
	int* valids;
	cudaMalloc((void **)&valids, sizeof(int) * bundlesTotal);

	int* hostValids = (int*)calloc(sizeof(int), bundlesTotal);
	cudaMemcpy(valids, hostValids, bundlesTotal * sizeof(int), cudaMemcpyHostToDevice); //init valids to zero
	
	int* valids_copy;
	cudaMalloc((void **)&valids_copy, sizeof(int) * bundlesTotal);
	
    kernelRenderCircles<<<gridDim, blockDim>>>(bundles, valids, bundlesWidth, numCircles);
    cudaThreadSynchronize();

	cudaMemcpy(valids_copy, valids, bundlesTotal * sizeof(int), cudaMemcpyDeviceToDevice);
	
	int gridDimX = bundlesWidth*bundlesHeight; //fixed
	int gridDimY = (numCircles + PREFIXBLOCKSIZE - 1) / PREFIXBLOCKSIZE;
	dim3 scanBlockGridDim(gridDimX, gridDimY, 1);

	scan_block<<<scanBlockGridDim, PREFIXBLOCKSIZE>>>(valids, numCircles); //correct
	cudaThreadSynchronize();
	
	if(gridDimY > 1){
		int* device_val;
		cudaMalloc((void **)&device_val, sizeof(int) * gridDimY * gridDimX);
		
		int threads = gridDimY;
		int combineValGridY = gridDimY;
		gridDimY = 1;
		int scaned_width = PREFIXBLOCKSIZE;
		int roundblock;
		
		while(threads > PREFIXBLOCKSIZE){
			gridDimY = (threads + PREFIXBLOCKSIZE - 1) / PREFIXBLOCKSIZE;
			threads = PREFIXBLOCKSIZE;
			
			dim3 scanCombineGridDim(gridDimX, gridDimY, 1);
			scan_combine<<<scanCombineGridDim, threads>>>(valids, scaned_width, threads, numCircles, combineValGridY, device_val);
			cudaThreadSynchronize();
			
			roundblock = (numCircles - scaned_width + PREFIXBLOCKSIZE - 1) / PREFIXBLOCKSIZE;
			
			dim3 threadAddGridDim(gridDimX, roundblock, 1);
			thread_add<<<threadAddGridDim, PREFIXBLOCKSIZE>>>(valids, scaned_width, numCircles, combineValGridY, device_val, scaned_width*PREFIXBLOCKSIZE, bundlesTotal);
			cudaThreadSynchronize();
			
			threads = gridDimY;
			gridDimY = 1;
			scaned_width *= PREFIXBLOCKSIZE;
		}
		
		dim3 scanCombineOuterGridDim(gridDimX, gridDimY, 1);
		scan_combine<<<scanCombineOuterGridDim, threads>>>(valids, scaned_width, threads, numCircles, combineValGridY, device_val);
		cudaThreadSynchronize();
		
		roundblock = (numCircles - scaned_width + PREFIXBLOCKSIZE - 1) / PREFIXBLOCKSIZE;
			
		dim3 threadAddOuterGridDim(gridDimX, roundblock, 1);
		thread_add<<<threadAddOuterGridDim, PREFIXBLOCKSIZE>>>(valids, scaned_width, numCircles, combineValGridY, device_val, scaned_width*PREFIXBLOCKSIZE, bundlesTotal);
		cudaThreadSynchronize();
	}
	
	dim3 moveBundleGridDim(gridDimX, (numCircles + PREFIXBLOCKSIZE - 1) / PREFIXBLOCKSIZE, 1);
	move_bundle<<<moveBundleGridDim, PREFIXBLOCKSIZE>>>(valids, valids_copy, bundles, numCircles);
	cudaThreadSynchronize();

	int totalPixels = imgWidth*imgHeight;
	int pixelBlocks = (totalPixels + blockDim.x - 1) / blockDim.x;

	renderPixels<<<pixelBlocks, blockDim.x>>>(blockDim.x, imgWidth, imgHeight, totalPixels, bundlesWidth, numCircles, bundles, valids_copy);
	cudaThreadSynchronize();
	
	//cudaFree(bundles);
}
