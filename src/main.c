

#include "image.h"

#include <stdio.h>




int main(int argc, char *argv[])
{
	Image *img;
	int ret;

	img = image_Create();

	ret = image_Read(img, argv[1]);

	ret = image_Write(img, argv[2]);

	image_Destroy(&img);


	return 0;
}