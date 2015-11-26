extern "C"
{
#include "gpu.h"
}


extern "C"
__global__ void kernel_smooth(	unsigned char *img_in, unsigned char *img_out, 
								unsigned int width, unsigned int height 		)
{
	__shared__ int smem[BLOCK_WIDTH *BLOCK_HEIGHT];
	int x = blockIdx.x*TILE_W + threadIdx.x - R;
	int y = blockIdx.y*TILE_H + threadIdx.y - R;

	// clamp to edge of image
	x = max(0, x);
	x = min(x, width-1);
	y = max(y, 0);
	y = min(y, height-1);

	unsigned int index = y*width + x;
	unsigned int bindex = threadIdx.y *blockDim.y +threadIdx.x;

	// each thread copies its pixel of the block to shared memory
	smem[bindex] = img_in[index];
	__syncthreads();

	// only threads inside the apron will write results
	if( (threadIdx.x >= R) && (threadIdx.x < (BLOCK_WIDTH-R)) && 
		(threadIdx.y >= R) && (threadIdx.y < (BLOCK_HEIGHT-R))	) 
	{
		float sum = 0;
		for(int dy=-R; dy<=R; dy++)
		{
			for(int dx=-R; dx<=R; dx++) 
			{
				float i = smem[bindex + (dy*blockDim.x) + dx];
				sum += i;
			}
		}
		img_out[index] = sum / S;
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
	grid_size.x = (width + block_size.x -2 - 1)/(block_size.x -2);
	grid_size.y = (height + block_size.y -2 - 1)/(block_size.y -2);

	// Call Kernel
	kernel_smooth<<<grid_size, block_size>>>(device_input, device_output, width, height);

	// Copy from device to host
	cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(device_input);
	cudaFree(device_output);
}
