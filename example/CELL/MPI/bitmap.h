#ifndef BITMAP_H_INCLUDED_
#define BITMAP_H_INCLUDED_

#define COPY_COLOR(dest, source) \
	dest.red = source.red; \
	dest.green = source.green; \
	dest.blue = source.blue;

struct rgb_t
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

void load_image (const char *image, int *width, int *height, struct rgb_t **data);

void save_image (const char *image, int width, int height, struct rgb_t *data);

#endif
