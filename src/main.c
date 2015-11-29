

#include "image.h"
#include "gpu.h"

#include <stdio.h>




int main(int argc, char *argv[])
{
	Image *img1;
	Image *img2;
	int ret;

	// Create and read
	img1 = image_Create();
	ret = image_Read(img1, argv[1]);
	if(ret)
		printf("%d\n", ret);

	img2 = image_Create_Alloc(img1->width, img1->height, img1->depth, img1->channel);


	// Smooth
	smooth(img1, img2);


	// Write
	ret = image_Write(img2, argv[2]);
	if(ret)
		printf("%d\n", ret);

	// Destroy
	image_Destroy(&img1);
	image_Destroy(&img2);


	return 0;
}
