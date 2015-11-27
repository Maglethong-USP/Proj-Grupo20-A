#ifndef __GPU_H__
#define __GPU_H__

#include "image.h"

/*
	Definition of image tile size
	Each CUDA Block will process one tile
*/
#define TILE_W   16
#define TILE_H   16

/*
	Definition of a block size [threads per block]
	 - 256 (16 *16) is good
	 - Old GPU max: 512
	 - New GPU max: 1024
	We will make the block larger than the tile by 2 units
	so it can contain the required border for processing
	the filter
 */
#define BLOCK_WIDTH 	(TILE_W +2)
#define BLOCK_HEIGHT 	(TILE_H +2)
void smooth_c3(Image *host_input, Image *host_output);
void smooth_c1(Image *host_input, Image *host_output);

#endif