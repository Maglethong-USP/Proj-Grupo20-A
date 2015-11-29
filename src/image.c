#include "image.h"


#include <stdlib.h>
#include <stdio.h>


Image *image_Create()
{
	Image *img = (Image *) malloc( sizeof(Image) );
	if( img == NULL )
		return NULL;
	// Init
	img->width = 0;
	img->height = 0;
	img->depth = 0;
	img->channel = 0;
	img->array = NULL;

	return img;
}

Image *image_Create_Alloc(	const unsigned int width, 
							const unsigned int height,
							const unsigned int depth,
							const unsigned short int channel	)
{
	int i;

	Image *img = (Image *) malloc( sizeof(Image) );
	if( img == NULL )
		return NULL;
	// Init
	img->width = width;
	img->height = height;
	img->depth = depth;
	img->channel = channel;

	// Allocate Image
	img->array = (p_type **) malloc(sizeof(p_type *) *channel);
	if( img->array == NULL )
	{
		free(img);
		return NULL;
	}
	for(i=0; i<channel; i++)
	{
		img->array[i] = (p_type *) malloc(sizeof(p_type) *width *height);
		if(img->array[i] == NULL)
		{
			for(i--; i>=0; i--)
				free(img->array[i]);
			free(img->array);
			free(img);
			return NULL;
		}
	}

	return img;
}

void image_Destroy(Image **image)
{
	int i;

	if( (*image)->array != NULL )
	{
		for(i=0; i<(*image)->channel; i++)
			free( (*image)->array[i] );
		free( (*image)->array );
	}

	(*image)->array = NULL;
	free( *image );
	*image = NULL;
}

int image_Read(Image *image, const char *fileName)
{
	FILE *fp;

	int mode;	// File mode [colored? binary?]
	int width;	
	int height;
	int depth;	// Pixel depth
	int channel;

	int i, j;
	char tmp[2];
	p_type *tmpAr;

	// Open
	if( (fp = fopen(fileName, "r+b")) == NULL )
		return 1;

	// Read header
	fscanf(fp, "%c%c", tmp, tmp +1);
	fscanf(fp, "%d", &width);
	fscanf(fp, "%d", &height);
	fscanf(fp, "%d", &depth);

	// Discover mode
	if(tmp[0] == 'P' && tmp[1] == '6')
		mode = COLOR_BINARY;
	else if(tmp[0] == 'P' && tmp[1] == '3')
		mode = COLOR_TEXT;
	else if(tmp[0] == 'P' && tmp[1] == '5')
		mode = BW_BINARY;
	else if(tmp[0] == 'P' && tmp[1] == '2')
		mode = BW_TEXT;
	else
	{
		fclose(fp);
		return 2;
	}

	if(mode & COLOR)
		channel = 3;
	else
		channel = 1;

	// Reading whitespace
	fgetc(fp);

	// Allocate Image
	image->array = (p_type **) malloc(sizeof(p_type *) *channel);
	if( image->array == NULL )
	{
		fclose(fp);
		return 3;
	}
	for(i=0; i<channel; i++)
	{
		image->array[i] = (p_type *) malloc(sizeof(p_type) *width *height);
		if(image->array[i] == NULL)
		{
			for(i--; i>=0; i--)
				free(image->array[i]);
			free(image->array);
			fclose(fp);
			return 3;
		}

	}

	// Allocate temporary
	tmpAr = (p_type *) malloc(sizeof(p_type) *width *height *channel);
	if(tmpAr == NULL)
	{
		for(i=0; i<channel; i++)
			free(image->array[i]);
		free(image->array);
		fclose(fp);
		return 3;
	}

	// Read image from file
	i = fread(tmpAr, sizeof(p_type), width *height *channel, fp);
	if(i < sizeof(p_type) *width *height *channel)
	{
		free(tmpAr);
		for(i=0; i<channel; i++)
			free(image->array[i]);
		free(image->array);
		fclose(fp);
		return 4;
	}

	// Copy to correct channel array
	for(i=0; i < width *height; i++)
		for(j=0; j<channel; j++)
			image->array[j][i] = tmpAr[i *channel +j];

	// Write image header to structure
	image->width = width;
	image->height = height;
	image->depth = depth;
	image->channel = channel;

	// End
	free(tmpAr);
	fclose(fp);
	return 0;
}

int image_Write(Image *image, const char *fileName)
{
	FILE *fp;

	p_type *tmpAr;
	int i, j;

	// Open
	if( (fp = fopen(fileName, "w+b")) == NULL )
		return 1;

	// Allocate temporary
	tmpAr = (p_type *) malloc(	sizeof(p_type) 
								*image->width *image->height *image->channel);
	if(tmpAr == NULL)
	{
		fclose(fp);
		return 3;
	}

	// Copy to correct channel array
	for(i=0; i < image->width *image->height; i++)
		for(j=0; j<image->channel; j++)
			tmpAr[i *image->channel +j] = image->array[j][i];

	// Write header
	if(image->channel == 1)
		fprintf(fp, "P5\n");
	else
		fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", image->width, image->height);
	fprintf(fp, "%d\n", image->depth);

	// Write image
	i = fwrite(tmpAr, sizeof(p_type), 
				image->width *image->height *image->channel, fp);
	if(i < sizeof(p_type) *image->width *image->height *image->channel)
	{
		free(tmpAr);
		fclose(fp);
		return 5;
	}

	// End
	free(tmpAr);
	fclose(fp);
	return 0;
}