extern "C"
{
#include "gpu.h"
}

extern "C"

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void kernel_smooth_c3(	unsigned char *img_in_r, unsigned char *img_out_r, 
									unsigned char *img_in_g, unsigned char *img_out_g,
									unsigned char *img_in_b, unsigned char *img_out_b,
									unsigned int width, unsigned int height 		)
{
	__shared__ int smem_r[BLOCK_WIDTH *BLOCK_HEIGHT];
	__shared__ int smem_g[BLOCK_WIDTH *BLOCK_HEIGHT];
	__shared__ int smem_b[BLOCK_WIDTH *BLOCK_HEIGHT];

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
	smem_r[bindex] = img_in_r[index];
	smem_g[bindex] = img_in_g[index];
	smem_b[bindex] = img_in_b[index];
	__syncthreads();

	// only threads inside the apron will write results
	if( (threadIdx.x >= 1) && (threadIdx.x < (BLOCK_WIDTH -1)) && 
		(threadIdx.y >= 1) && (threadIdx.y < (BLOCK_HEIGHT -1))	) 
	{
		float sum[] = {0, 0, 0};

		// R
		sum[0] += smem_r[bindex -blockDim.x -1];
		sum[0] += smem_r[bindex -blockDim.x   ];
		sum[0] += smem_r[bindex -blockDim.x +1];

		sum[0] += smem_r[bindex             -1];
		sum[0] += smem_r[bindex               ];
		sum[0] += smem_r[bindex             +1];

		sum[0] += smem_r[bindex +blockDim.x -1];
		sum[0] += smem_r[bindex +blockDim.x   ];
		sum[0] += smem_r[bindex +blockDim.x +1];

		// G
		sum[1] += smem_g[bindex -blockDim.x -1];
		sum[1] += smem_g[bindex -blockDim.x   ];
		sum[1] += smem_g[bindex -blockDim.x +1];

		sum[1] += smem_g[bindex             -1];
		sum[1] += smem_g[bindex               ];
		sum[1] += smem_g[bindex             +1];

		sum[1] += smem_g[bindex +blockDim.x -1];
		sum[1] += smem_g[bindex +blockDim.x   ];
		sum[1] += smem_g[bindex +blockDim.x +1];

		// B
		sum[2] += smem_b[bindex -blockDim.x -1];
		sum[2] += smem_b[bindex -blockDim.x   ];
		sum[2] += smem_b[bindex -blockDim.x +1];

		sum[2] += smem_b[bindex             -1];
		sum[2] += smem_b[bindex               ];
		sum[2] += smem_b[bindex             +1];

		sum[2] += smem_b[bindex +blockDim.x -1];
		sum[2] += smem_b[bindex +blockDim.x   ];
		sum[2] += smem_b[bindex +blockDim.x +1];

		// Write
		img_out_r[index] = sum[0] / 9;
		img_out_g[index] = sum[1] / 9;
		img_out_b[index] = sum[2] / 9;
	}
}


__global__ void kernel_smooth_c1(	unsigned char *img_in, unsigned char *img_out,
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

void smooth(Image *host_input, Image *host_output)
{
	int size = host_input->width *host_input->height;

	// Declare device pointer
	unsigned char *device_input_r, *device_output_r;
	unsigned char *device_input_g, *device_output_g;
	unsigned char *device_input_b, *device_output_b;

	// Declaring block size
	dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);

	 // Declaring grid size to fit whole image
	dim3 grid_size;
	grid_size.x = (host_input->width + TILE_W - 1)/TILE_W;
	grid_size.y = (host_input->height + TILE_H - 1)/TILE_H;

	// 3 Channel Image
	if(host_input->channel == 3)
	{
		// Allocating device memory
		gpuErrchk( cudaMalloc(&device_input_r, size) );
		gpuErrchk( cudaMalloc(&device_output_r, size) );
		gpuErrchk( cudaMalloc(&device_input_g, size) );
		gpuErrchk( cudaMalloc(&device_output_g, size) );
		gpuErrchk( cudaMalloc(&device_input_b, size) );
		gpuErrchk( cudaMalloc(&device_output_b, size) );

		// Copy from host to device
		gpuErrchk( cudaMemcpy(device_input_r, host_input->array[0], size, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(device_input_g, host_input->array[1], size, cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(device_input_b, host_input->array[2], size, cudaMemcpyHostToDevice) );

		// Call Kernel
		kernel_smooth_c3<<<grid_size, block_size>>>(	device_input_r, device_output_r,
														device_input_g, device_output_g,
														device_input_b, device_output_b,
														host_input->width, host_input->height);
	
		// Error check
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// Copy from device to host
		gpuErrchk( cudaMemcpy(host_output->array[0], device_output_r, size, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(host_output->array[1], device_output_g, size, cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(host_output->array[2], device_output_b, size, cudaMemcpyDeviceToHost) );

		// Free device memory
		cudaFree(device_input_r);
		cudaFree(device_output_r);
		cudaFree(device_input_g);
		cudaFree(device_output_g);
		cudaFree(device_input_b);
		cudaFree(device_output_b);

	}
	// 1 Channel Image
	else if(host_input->channel == 1)
	{
		// Allocating device memory
		gpuErrchk( cudaMalloc(&device_input_r, size) );
		gpuErrchk( cudaMalloc(&device_output_r, size) );

		// Copy from host to device
		gpuErrchk( cudaMemcpy(device_input_r, host_input->array[0], size, cudaMemcpyHostToDevice) );

		// Call Kernel
		kernel_smooth_c1<<<grid_size, block_size>>>(	device_input_r, device_output_r,
														host_input->width, host_input->height);

		// Error check
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );

		// Copy from device to host
		gpuErrchk( cudaMemcpy(host_output->array[0], device_output_r, size, cudaMemcpyDeviceToHost) );

		// Free device memory
		cudaFree(device_input_r);
		cudaFree(device_output_r);
	}
}