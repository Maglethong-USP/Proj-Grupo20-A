#ifndef __IMAGE_H__
#define __IMAGE_H__

// Defining pixel type
typedef unsigned char p_type;

// Defining image structure
typedef struct _Image
{
	unsigned int width;
	unsigned int height;
	unsigned int depth;
	unsigned short int channel;
	p_type **array;
} Image;

//! Channel color enumeration
/*!
	First bit  -> Color/Black&Wite
	Second bit -> Text/Binary
*/
enum Image_File_Mode 
{ 
	// Modes
	BW_TEXT			= 0, 	// 00
	BW_BINARY		= 1, 	// 01
	COLOR_TEXT		= 2, 	// 10
	COLOR_BINARY	= 3, 	// 11
	// Masks
	BINARY 			= 1,	// 01
	COLOR 			= 2 	// 10
};

//! Create
/*!	
	@return		The created image structure
				NULL if allocation error occurred
*/
Image *image_Create();

//! Destroy
/*!	
	@param		image 		The image structure delete
*/
void image_Destroy(Image **image);

//! Read image
/*!	
	@param		image 		The image structure to read into
	@param		fileName 	The file name to read from
	@return		0 for success
				1 for file opening error
				2 for bad file header
				3 for allocation error
				4 for read error
*/
int image_Read(Image *image, const char *fileName);

//! Write image
/*!	
	@param		image 		The image structure to read from
	@param		fileName 	The file name to write to
	@return		0 for success
				1 for file opening error
				3 for allocation error
				5 for write error
*/
int image_Write(Image *image, const char *fileName);
#endif