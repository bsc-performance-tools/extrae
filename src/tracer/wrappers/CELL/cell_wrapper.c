/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "cell_wrapper.h"
#include "wrapper.h"
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#define CHECK_ERROR(val, call) { \
	if (val < 0) { \
		perror (#call); \
		exit (-1); \
	} \
}

unsigned int cell_tracing_enabled = TRUE;
unsigned int spu_dma_channel      = DEFAULT_DMA_CHANNEL;
unsigned int spu_buffer_size      = DEFAULT_SPU_BUFFER_SIZE;
unsigned int spu_file_size        = DEFAULT_SPU_FILE_SIZE;

static unsigned int *** spu_buffer;
static unsigned int *** spu_counter;
static unsigned int * number_of_spus;
static unsigned int * threads_prepared;
static int CELLtrace_init_prepared = FALSE;

/*
	prepare_CELLTrace_init (int threads)

	prepares the CELL instrumentation to allow multiple (non SPU) threads run
	multiple SPU threads.
*/

int prepare_CELLTrace_init (int nthreads)
{
	int i;

	spu_buffer = (unsigned int ***) malloc (nthreads*sizeof(unsigned int**));
	if (NULL == spu_buffer)
	{
		fprintf (stderr, PACKAGE_NAME": Could not allocate spu_buffer\n");
		return FALSE;
	}
	for (i = 0; i < nthreads; i++)
		spu_buffer[i] = NULL;

	spu_counter = (unsigned int ***) malloc (nthreads*sizeof(unsigned int**));
	if (NULL == spu_counter)
	{
		fprintf (stderr, PACKAGE_NAME": Could not allocate spu_counter\n");
		return FALSE;
	}
	for (i = 0; i < nthreads; i++)
		spu_counter[i] = NULL;

	number_of_spus = (unsigned int *) malloc (nthreads*sizeof(unsigned int));
	if (NULL == number_of_spus)
	{
		fprintf (stderr, PACKAGE_NAME": Could not allocate number_of_spus\n");
		return FALSE;
	}

	threads_prepared = (unsigned int *) malloc (nthreads*sizeof(unsigned int));
	if (NULL == threads_prepared)
	{
		fprintf (stderr, PACKAGE_NAME": Could not allocate memory for threads_prepared\n");
		return FALSE;
	}
	else
		CELLtrace_init_prepared = TRUE;

	for (i = 0; i < nthreads; i++)
		number_of_spus[i] = threads_prepared[i] = 0;

	return CELLtrace_init_prepared;
}

#if CELL_SDK == 1
static inline void send_mail (speid_t id, unsigned int data)
{
	while (spe_write_in_mbox (id, data));
}
#elif CELL_SDK == 2
static inline void send_mail (spe_context_ptr_t id, unsigned int data)
{
	while (spe_in_mbox_status(id) <= 0);
	spe_in_mbox_write (id, &data, 1, SPE_MBOX_ANY_NONBLOCKING);
}
#endif

#if CELL_SDK == 1
static inline int read_mail (speid_t id)
{
	return spe_read_out_mbox (id);
}
#elif CELL_SDK == 2
static inline int read_mail (spe_context_ptr_t id)
{
	unsigned int data;
	while (spe_out_mbox_status(id) <= 0);
	spe_out_mbox_read (id, &data, 1);
	return data;
}
#endif

static void flush_spu_buffers (unsigned THREAD, int nthreads, unsigned **prvbuffer, unsigned int **prvcount)
{
#ifdef SPU_USES_WRITE

	char trace_tmp[TRACE_FILE];
	char trace[TRACE_FILE];
	int linear_thread = get_maximum_NumOfThreads();
	int i;
	
	/*
	   linear_thread allows converting SPU thread id into simple threads in
	   the Paraver trace.  It is based on:
		   the total number of threads existing + the other created SPU threads
	*/
	for (i = 0; i < THREAD; i++)
		linear_thread += number_of_spus[i];

	for (i = 0; i < nthreads; i++)
	{
		FileName_PTT (trace_tmp, final_dir, appl_name, getpid(), TASKID, i+linear_thread, EXT_TMP_MPIT);
		FileName_PTT (trace, final_dir, appl_name, getpid(), TASKID, i+linear_thread, EXT_MPIT);

		rename_or_copy (trace_tmp, trace);

		fprintf (stdout, PACKAGE_NAME": Intermediate raw trace file created for SPU %d (in thread %d): %s\n", i+1, THREAD, trace);
	}

#else
	char trace[TRACE_FILE];
	int fd, res, i;
	int linear_thread = get_maximum_NumOfThreads();

	/*
	   linear_thread allows converting SPU thread id into simple threads in
	   the Paraver trace.  It is based on:
		   the total number of threads existing + the other created SPU threads
	*/
	for (i = 0; i < THREAD; i++)
		linear_thread += number_of_spus[i];

	fprintf (stdout, "\n");
	for (i = 0; i < nthreads; i++)
	{
		FileName_PTT (trace, final_dir, appl_name, getpid(), TASKID, i+linear_thread, EXT_MPIT);
		fprintf (stdout, PACKAGE_NAME": Intermediate raw trace file created for SPU %d (in thread %d): %s\n", i+1, THREAD, trace);

		fd = open (trace, O_WRONLY|O_CREAT|O_TRUNC, 0644);
		CHECK_ERROR (fd, open);

		res = write (fd, prvbuffer[i], *prvcount[i]);
		CHECK_ERROR (res, write);

		res = close (fd);
		CHECK_ERROR (res, close);
	}
#endif
}

/* HSG this shouldn't be here! */
extern unsigned long long proc_timebase();

#if CELL_SDK == 1
int CELLtrace_init (int spus, speid_t * spe_id) __attribute__ ((alias ("Extrae_CELL_init")));
int Extrae_CELL_init (int spus, speid_t * spe_id)
#elif CELL_SDK == 2
int CELLtrace_init (int spus, spe_context_ptr_t * spe_id) __attribute__ ((alias ("Extrae_CELL_init")));
int Extrae_CELL_init (int spus, spe_context_ptr_t * spe_id)
#endif
{
#ifdef SPU_USES_WRITE
	unsigned int linear_thread;
	static int warning_message_shown = FALSE;
#endif
	unsigned int i, TB_high, TB_low, all_spus_ok;
	unsigned long long TB, spu_creation_time[spus];
	unsigned THREAD = get_trace_thread_number();

#ifdef SPU_USES_WRITE
	if (!warning_message_shown)
	{
		fprintf (stdout, PACKAGE_NAME": WARNING!\n"
		                 PACKAGE_NAME": WARNING! The SPUs will write directly their buffers into disk!\n"
		                 PACKAGE_NAME": WARNING! Such behavior makes flushes very costly!\n"
		                 PACKAGE_NAME": WARNNIG!\n");
	}
#endif

	if (!CELLtrace_init_prepared)
	{
		fprintf (stderr, PACKAGE_NAME": CELLtrace_init was called but never prepared!\n");
		exit (-1);
	}

	/* Broadcast if the tracing is enabled */
	for (i = 0; i < spus; i++)
		send_mail (spe_id[i], mpitrace_on && cell_tracing_enabled);

	if (!(mpitrace_on && cell_tracing_enabled))
		return 0;

	spu_buffer[THREAD] = (unsigned int**) malloc (spus*sizeof(unsigned int*));
	if (spu_buffer[THREAD] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Unable to allocate spu_buffer[%d]. Exiting!\n", THREAD);
		exit (-1);
	}
	spu_counter[THREAD] = (unsigned int**) malloc (spus*sizeof(unsigned int*));
	if (spu_counter[THREAD] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Unable to allocate spu_counter[%d]. Exiting!\n", THREAD);
		exit (-1);
	}
	TB = proc_timebase();
	TB_high = TB >> 32;
	TB_low  = TB;
	number_of_spus[THREAD] = spus;

#ifdef SPU_USES_WRITE
	linear_thread = get_maximum_NumOfThreads();
	for (i = 0; i < THREAD; i++)
	{
		if (!threads_prepared[i])
		{
			fprintf (stderr, PACKAGE_NAME": Error! Requires that threads are initialized in order\n");
			exit (-1);
		}
		linear_thread += number_of_spus[i];
	}
#endif

	/* Initialize the tracing on the SPU side */
	for (i = 0; i < spus; i++)
	{
		char trace[TRACE_FILE];
		int descriptor;
		unsigned long long addr_buffer = 0, addr_counter = 0;

		send_mail (spe_id[i], (unsigned int) TB_high);
		send_mail (spe_id[i], (unsigned int) TB_low);

		spu_creation_time[i] = TIME;
		send_mail (spe_id[i], (unsigned int) (spu_creation_time[i] >> 32));
		send_mail (spe_id[i], (unsigned int) spu_creation_time[i]);

#ifdef SPU_USES_WRITE

		/* Create a temporal file for the SPU */
		FileName_PTT (trace, final_dir, appl_name, getpid(), TASKID, i+linear_thread, EXT_TMP_MPIT);
		descriptor = open (trace, O_WRONLY|O_CREAT|O_TRUNC, 0644);

		/* If using 'write', just can ignore buffer limits */
		spu_buffer[THREAD][i] = 0xdeadbeef; /* Useless pointer! */
		spu_counter[THREAD][i] = 0xdeadbeef; /* Useless pointer! */

#else

		/* Create buffer and touch it */
		spu_buffer[THREAD][i] = (unsigned int*) valloc (spu_file_size * 1024 * 1024);
		if (spu_buffer[THREAD][i] == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Unable to allocate spu_buffer[%d][%d]. Exiting!\n", THREAD, i);
			exit (-1);
		}
		memset (spu_buffer[THREAD][i], 0, spu_file_size * 1024 * 1024);

		/* Create buffer and touch it */
		spu_counter[THREAD][i] = (unsigned int*) valloc (16);
		if (spu_counter[THREAD][i] == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Unable to allocate spu_counter[%d][%d]. Exiting!\n", THREAD, i);
			exit (-1);
		}
		memset (spu_buffer[THREAD][i], 0, 16);

		/* If not using 'write', just ignore descriptor */
		descriptor = -1; /* Useless descriptor */

#endif

		send_mail (spe_id[i], (unsigned int) i);
		send_mail (spe_id[i], spu_file_size * 1024 * 1024);
		send_mail (spe_id[i], descriptor);

		addr_buffer = (unsigned long)spu_buffer[THREAD][i];
		send_mail (spe_id[i], (unsigned int) (addr_buffer >> 32));
		send_mail (spe_id[i], (unsigned int) addr_buffer);
		addr_counter = (unsigned long)spu_counter[THREAD][i];
		send_mail (spe_id[i], (unsigned int) (addr_counter >> 32));
		send_mail (spe_id[i], (unsigned int) addr_counter);

		send_mail (spe_id[i], (unsigned int) spu_dma_channel);
		send_mail (spe_id[i], (unsigned int) spu_buffer_size);
	}
	if (TASKID == 0)
		fprintf (stdout, PACKAGE_NAME": PPU initialized\n");

	if (TASKID == 0)
	{
		fprintf (stdout, PACKAGE_NAME": SPU initialized { ");
		fflush (stdout);
	}
	all_spus_ok = TRUE;
	for (i = 0; i < spus; i++)
	{
		int info = read_mail (spe_id[i]);

		if (TASKID == 0)
		{
			fprintf (stdout, "%s SPU %d has %s", (i!=0)?",":"", i+1, (info==0)?"failed":"succeeded");
			fflush (stdout);
		}

		all_spus_ok = all_spus_ok && info;
	}
	fprintf (stdout, " }\n");

	if (!all_spus_ok)
	{
		if (TASKID == 0)
			fprintf (stderr, PACKAGE_NAME": Some of the SPUs failed in the initialization. Check the messages given. Exiting!\n");
		exit (-1);
	}

	threads_prepared[THREAD] = TRUE;

	return 1;	
}

int CELLtrace_fini (void) __attribute__ ((alias ("Extrae_CELL_fini")));
int Extrae_CELL_fini (void)
{
	unsigned THREAD = get_trace_thread_number();

	if (mpitrace_on)
	{
 		/* Dump SPU buffers */
		flush_spu_buffers (THREAD, number_of_spus[THREAD], spu_buffer[THREAD], spu_counter[THREAD]);
#if defined(MPI_SUPPORT)
		generate_spu_file_list (number_of_spus[THREAD]);
#endif

		if (TASKID == 0)
			fprintf (stdout, PACKAGE_NAME": Application has ended. Tracing has been terminated.\n");
	}

  return 0;
}
