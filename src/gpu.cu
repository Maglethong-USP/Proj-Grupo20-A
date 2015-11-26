extern "C"
{
#include "gpu.h"
}


extern "C"
__global__ void kernel_smooth(	unsigned char *img_in, unsigned char *img_out, 
								unsigned int width, unsigned int height 		)
{
	__shared__ int smem[BLOCK_WIDTH *BLOCK_HEIGHT];
	int x = blockIdx.x*TILE_W + threadIdx.x - 1;
	int y = blockIdx.y*TILE_H + threadIdx.y - 1;

	// clamp to edge of image
	x = max(0, x);
	x = min(x, width -1);
	y = max(y, 0);
	y = min(y, height -1);

	unsigned int index = y *width +x;
	unsigned int bindex = threadIdx.y *blockDim.y +threadIdx.x;

	// each thread copies its pixel of the block to shared memory
	smem[bindex] = img_in[index];
	__syncthreads();

	// only threads inside the apron will write results
	if( (threadIdx.x >= 1) && (threadIdx.x < (BLOCK_WIDTH -1)) && 
		(threadIdx.y >= 1) && (threadIdx.y < (BLOCK_HEIGHT -1))	) 
	{
		float sum = 0;

		sum += smem[bindex -blockDim.x -1];
		sum += smem[bindex -blockDim.x   ];
		sum += smem[bindex -blockDim.x +1];

		sum += smem[bindex             -1];
		sum += smem[bindex               ];
		sum += smem[bindex             +1];

		sum += smem[bindex +blockDim.x -1];
		sum += smem[bindex +blockDim.x   ];
		sum += smem[bindex +blockDim.x +1];

		img_out[index] = sum / 9;
	}
}


void smooth(	unsigned char *host_input, unsigned char *host_output, 
				unsigned int width, unsigned int height 				)
{
	int size = width *height;

	// Declare device pointer
	unsigned char *device_input, *device_output;

	// Allocating device memory
	cudaMalloc(&device_input, size);
	cudaMalloc(&device_output, size);

	// Copy from host to device
	cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);

	// Declaring block size
	dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);

	 // Declaring grid size to fit whole image
	dim3 grid_size;
	grid_size.x = (width + TILE_W - 1)/TILE_W;
	grid_size.y = (height + TILE_H - 1)/TILE_H;

	// Call Kernel
	kernel_smooth<<<grid_size, block_size>>>(device_input, device_output, width, height);

	// Copy from device to host
	cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(device_input);
	cudaFree(device_output);
}
