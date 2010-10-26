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

#include <spu_intrinsics.h>
#include <spu_internals.h>

#include "bitmap.h"
#include "spu_trace.h"
#include <spu_mfcio.h>

void __inline__ cell_asynch_get (void *ls, void *ea, int size, int tag)
{
	/* DMA transfers must be x16 bytes */
	if ((size & 0x0f) != 0x00)
		return;

	while (spu_readchcnt (MFC_Cmd) < 1);
	spu_mfcdma32 (ls, (unsigned int) ea, size, tag, 0x0 | 0x40);
}

void __inline__ cell_asynch_put (void *ls, void *ea, int size, int tag)
{
	/* DMA transfers must be x16 bytes */
	if ((size & 0x0f) != 0x00)
		return;

	while (spu_readchcnt (MFC_Cmd) < 1);
	spu_mfcdma32 (ls, (unsigned int) ea, size, tag, 0x0 | 0x20);
}

static void cell_wait (int tag)
{
	spu_writech (MFC_WrTagMask, 1 << tag);
	spu_mfcstat (2);
}

static unsigned int get_mail (void)
{
	while (spu_stat_in_mbox () < 1);
	return spu_read_in_mbox();
}

static void get_arguments (unsigned int *ID, struct rgb_t **image1,
	struct rgb_t **image2, unsigned int *count, struct rgb_t **out)
{
	*ID = get_mail ();
	*image1 = (struct rgb_t*) get_mail ();
	*image2 = (struct rgb_t*) get_mail ();
	*count = get_mail ();
	*out = (struct rgb_t*) get_mail ();
}

static void cell_work (struct rgb_t *chroma, struct rgb_t *image, unsigned int count,
	struct rgb_t *out)
{
	unsigned int i;

	for (i = 0; i < count; i++)
	{
		if (chroma[i].green==255&&chroma[i].red==60&&chroma[i].blue==0)
		{
			COPY_COLOR (out[i],image[i]);
		}
		else
		{
			COPY_COLOR (out[i],chroma[i]);
		}
	}
}

static void cell_get_pixels (struct rgb_t *PUimage, struct rgb_t *SPUimage,
	unsigned int npixels)
{
	cell_asynch_get (SPUimage, PUimage, npixels*sizeof(struct rgb_t), 0);
	cell_wait (0);
}

static void cell_put_pixels (struct rgb_t *PUimage, struct rgb_t *SPUimage,
	unsigned int npixels)
{
	cell_asynch_put (SPUimage, PUimage, npixels*sizeof(struct rgb_t), 2);
	cell_wait (2);
}

#define MAX_PIXELS 640 
struct 	rgb_t chroma[MAX_PIXELS] __attribute ((aligned(128))),
				image[MAX_PIXELS] __attribute ((aligned(128))), 
				local_out[MAX_PIXELS] __attribute ((aligned(128)));

int main (int argc, char *argv[])
{
	struct rgb_t *image1, *image2, *global_out;
	unsigned int count, ID;

	get_arguments (&ID, &image1, &image2, &count, &global_out);

	Extrae_init ();

	while (count > 0)
	{
		Extre_event (1000, count);

		/* GET nLINES FROM DMA */
		cell_get_pixels (image1, chroma, MAX_PIXELS);
		cell_get_pixels (image2, image, MAX_PIXELS);

		/* Do the work! */
		cell_work (chroma, image, MAX_PIXELS, local_out);

		/* PUT nLINES TO DMA */
		cell_put_pixels (global_out, local_out, MAX_PIXELS);

		image1     += MAX_PIXELS;
		image2     += MAX_PIXELS;
		global_out += MAX_PIXELS;
		count      -= MAX_PIXELS;
	}

	Extrae_fini ();

	return 0;
}

