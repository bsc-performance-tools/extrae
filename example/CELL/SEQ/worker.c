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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/example/CELL/SEQ/worker.c,v $
 | 
 | @last_commit: $Date: 2008/01/26 11:18:22 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
static char rcsid[] = "$Id: worker.c,v 1.3 2008/01/26 11:18:22 harald Exp $";

#include <spu_intrinsics.h>
#include <spu_internals.h>

#include "spu.h"
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

	SPUtrace_init ();

	while (count > 0)
	{
		SPUtrace_event (1000, count);

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

	SPUtrace_fini ();

	return 0;
}

