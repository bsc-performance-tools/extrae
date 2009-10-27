/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/example/CELL/SEQ/bitmap.c,v $
 | 
 | @last_commit: $Date: 2008/01/26 11:18:22 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
static char rcsid[] = "$Id: bitmap.c,v 1.2 2008/01/26 11:18:22 harald Exp $";

#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include "bitmap.h"

#define CHECK_ERROR(val, call) { \
	if (val < 0) { \
		perror (#call); \
		exit (-1); \
	} \
}

void load_image (const char *image, int *width, int *height, struct rgb_t **data)
{
	struct rgb_t *pixels;
	int size;
	char lines[4][80];
	int i;
	int fd;
	int res;
	FILE *f;

	fd = open (image, O_RDONLY);
	CHECK_ERROR (fd, open);

	f = fdopen (fd, "r");
	CHECK_ERROR ( (f == NULL) ? -1 : 0, "fdopen");

	for (i = 0; i < 4; i++)
		fgets (lines[i], 80, f);

	sscanf (lines[2], "%d %d", width, height);

	size = *width * *height;

	res = lseek (fd, -(size*sizeof(struct rgb_t)), SEEK_END);
	CHECK_ERROR (res, "lseek");

	pixels = (struct rgb_t *) valloc (size*sizeof(struct rgb_t));
	CHECK_ERROR( (pixels == NULL) ? -1 : 0, "valloc");

	res = read (fd, pixels, size*sizeof(struct rgb_t));
	CHECK_ERROR(res, "read");
	*data = pixels;

	fclose (f);
	close (fd);
}

void save_image (const char *image, int width, int height, struct rgb_t *data)
{
	char buffer[2048];
	int fd;
	int res;

	sprintf (buffer,
		"P6\n"
		"# CREATOR: The GIMP's PNM Filter Version 1.0\n"
		"%d %d\n"
		"255\n", width, height);

	fd = open (image, O_WRONLY|O_CREAT|O_TRUNC, 0600);
	CHECK_ERROR (fd, open);

	res = write (fd, buffer, strlen(buffer));
	CHECK_ERROR (res, write);

	res = write (fd, data, width*height*sizeof(struct rgb_t));
	CHECK_ERROR (res, write);

	res = close (fd);
	CHECK_ERROR (res, close);
}

