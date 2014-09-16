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
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <libspe2.h>
#include <sys/mman.h>
#include "ppu_trace.h"

#include "bitmap.h"

#define CHECK_NULL(val, call) { \
	if (val == 0) { \
		perror (#call); \
		exit (-1); \
	}  \
}

#define CHECK_ERROR(val, call) { \
	if (val < 0) { \
		perror (#call); \
		exit (-1); \
	} \
}

void send_mail (spe_context_ptr_t id, unsigned int data)
{
	while (spe_in_mbox_status(id) <= 0);
	spe_in_mbox_write (id, &data, 1, SPE_MBOX_ANY_NONBLOCKING);
}

void *ppu_pthread_function (void *arg)
{
	spe_context_ptr_t context = *(spe_context_ptr_t*) arg;
	unsigned int entry = SPE_DEFAULT_ENTRY;
	spe_stop_info_t stop_info;

	spe_context_run (context, &entry, 0, NULL, NULL, &stop_info);
	pthread_exit (NULL);
}

void create_threads (int numthreads, const char *program, spe_context_ptr_t **contexts, 
	pthread_t **pthreads, int *works, struct rgb_t *image1, struct rgb_t *image2,
	struct rgb_t *out, int width, int height)
{
	int i, j;
	spe_program_handle_t *handle;
	spe_context_ptr_t *conts;
	pthread_t *thids;
	unsigned int args[8];

	thids = (pthread_t*) malloc (numthreads * sizeof(pthread_t));
	conts = (spe_context_ptr_t*) malloc (numthreads * sizeof(spe_context_ptr_t));

	handle = spe_image_open (program);
	CHECK_NULL (handle, spe_open_image);

	/* Prepare some params! */

#if defined(__powerpc64__)
	args[2] = (unsigned int) (((unsigned long long) image1) >> 32); /* hi bits of @ of image 1 */
	args[4] = (unsigned int) (((unsigned long long) image2) >> 32); /* hi bits of @ of image 2 */
	args[6] = (unsigned int) (((unsigned long long) out) >> 32); /* hi bits of @ of out */
#else
	args[2] = 0; /* hi bits of @ of image 1 */
	args[4] = 0; /* hi bits of @ of image 2 */
	args[6] = 0; /* hi bits of @ of out */
#endif
	args[3] = (unsigned int) (((unsigned long long) image1) & 0xFFFFFFFF); /* lo bits of @ of image 1 */
	args[5] = (unsigned int) (((unsigned long long) image2) & 0xFFFFFFFF); /* lo bits of @ of image 2 */
	args[7] = (unsigned int) (((unsigned long long) out) & 0xFFFFFFFF); /* lo bits of @ of out */

	for (i = 0; i < numthreads; i++)
	{
		/* Configure params! */
		args[0] = i; /* ID */
		args[1] = works[i] * width;  /* Number of PIXELS */

		/* PARAMS ARE:
			--
			param(0) = ID of the thread
		        param(1) = number of pixels of this chunk of work
			param(2)+param(3) = @ of the first image (chroma)
			param(4)+param(5) = @ of the second image (normal)
			param(6)+param(7) = @ of the output image
			-- */

		conts[i] = spe_context_create (0, NULL);
		spe_program_load (conts[i], handle);
		pthread_create (&thids[i], NULL, ppu_pthread_function, &conts[i]);
		CHECK_NULL (thids[i], pthread_create);

		for (j = 0; j < 8; j++)
			send_mail (conts[i], args[j]);

		/* Set the next params */
		args[3] += (works[i]*width)*sizeof(struct rgb_t); 
		args[5] += (works[i]*width)*sizeof(struct rgb_t); 
		args[7] += (works[i]*width)*sizeof(struct rgb_t); 
	}
	*pthreads = thids;
	*contexts = conts;
}

void wait_threads (int count, pthread_t *threads)
{
	int i;
	for (i = 0; i < count; i++)
		pthread_join (threads[i], NULL);
}

void distribute_work (int numthreads, int work, int **distributed)
{
	int remain_work = work;
	int remain_threads, i;
	int *distribution = (int *) malloc (sizeof(int)*numthreads);

  /* Search for an equitative work distribution */ 	
	for (i = 0, remain_threads = numthreads;
		remain_threads > 0;
		remain_threads--, i++)
	{
		int thread_work = remain_work / remain_threads;
		remain_work = remain_work - thread_work;

		distribution[i] = thread_work;
	}

	printf ("\nWork distribution:\n------------------\n");
	for (i = 0; i < numthreads; i++)
		printf ("\tThread %d: number of lines = %d\n", i, distribution[i]);
	printf ("\n");

	*distributed = distribution;
}

int main (int argc, char *argv[])
{
	int i, w1, w2, h1, h2, numthreads, *works;
	struct rgb_t *image1, *image2, *out;
	pthread_t *pthreads;
	spe_context_ptr_t *contexts;

	Extrae_init ();

	if (argc != 5)
	{
		printf ("Usage:\n%s <image1.pnm> <image2.pnm> <out.pnm> <numSPUs>\n",
			argv[0]);
		exit (1);
	}
	numthreads = atoi (argv[4]);
	if (numthreads == 0)
	{
		printf ("Invalid value for number of threads (%s)\n", argv[4]);
		exit (1);
	}
	if (numthreads < 0 || numthreads > 16)
	{
		printf ("Number of threads must be between 0 and 16\n");
		exit (1);
	}

	fprintf (stdout, "Using %d threads -- hopefully 1thread-2-1spu\n", numthreads);

	Extrae_event (1000, 1);
	load_image (argv[1], &w1, &h1, &image1);
	load_image (argv[2], &w2, &h2, &image2);
	Extrae_event (1000, 0);

	if (w1 == w2 && h1 == h2)
	{
		distribute_work (numthreads, h1, &works);

		out = (struct rgb_t*) valloc (w1*h1*sizeof(struct rgb_t));

		create_threads (numthreads, "./worker", &contexts, &pthreads, works,
			image1, image2, out, w1, h1);

		Extrae_CELL_init (numthreads, contexts);

		Extrae_event (1000, 2);
		wait_threads (numthreads, pthreads);
		for (i = 0; i < numthreads; i++)
			spe_context_destroy(contexts[i]);
		Extrae_event (1000, 0);

		Extrae_event (1000, 3);
		save_image (argv[3], w1, h1, out);
		Extrae_event (1000, 0);

		Extrae_CELL_fini ();
	}
	else
	{
		printf ("Both images must be equally sized\nExiting\n");
	}

	Extrae_fini();

	return 0;
}

