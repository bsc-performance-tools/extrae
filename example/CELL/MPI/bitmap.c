/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
static char rcsid[] = "$Id$";

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

