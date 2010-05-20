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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdarg.h>
#include <fcntl.h>

#include <spu_intrinsics.h>
#include <spu_internals.h>
#include <spu_mfcio.h>

#include "wrapper.h"
#include "threadid.h"
#include "defaults.h"
#include "record.h"
#include "events.h"
#include "clock.h"

#if !defined(TRUE)
# define TRUE (1==1)
#endif

void flush_buffer (int mark_on_trace, int thread);

/******************************************************************************
 **********************         V A R I A B L E S        **********************
 ******************************************************************************/

int tracejant = TRUE;
int mpitrace_on = TRUE;

struct trace_prda *PRDAUSR;
event_t *buffer;
event_t **buffers = &buffer;
struct trace_prda estructura;

static unsigned long long out_trace;
static unsigned long long out_count;
static unsigned int max_file_size;

void advance_current(int thread)
{
	CUREVT(thread)++;
	if (CUREVT(thread) >= LASTEVT(thread))
		flush_buffer(1, thread);
}

/******************************************************************************
 *****************     I N I  and   F I N I  functions      *******************
 ******************************************************************************/

/******************************************************************************
 ***  allocate_buffers
 ******************************************************************************/
#ifdef SPU_DYNAMIC_BUFFER
int EVENT_BUFFER_SIZE = DEFAULT_SPU_BUFFER_SIZE;

static event_t * vector_events;

int allocate_buffers (int rank)
{
	vector_events = (event_t *)malloc(EVENT_BUFFER_SIZE * sizeof(event_t));
	if (vector_events == NULL)
		return 0;
	buffers[0] = vector_events;
	return 1;
}
#else
#define EVENT_BUFFER_SIZE DEFAULT_SPU_BUFFER_SIZE

static event_t vector_events[EVENT_BUFFER_SIZE] __attribute ((aligned(128)));

int allocate_buffers ()
{
	buffers[0] = vector_events;
	return 1;
}
#endif


int spu_init_backend (int me, unsigned long long trace_ptr, unsigned long long count_trace_ptr, unsigned int file_size, int fd)
{
	max_file_size = file_size;
	out_trace = trace_ptr;
	out_count = count_trace_ptr;

#if defined(SPU_DYNAMIC_BUFFER)
	if (!allocate_buffers (me))
		return 0;
#else
	if (!allocate_buffers ())
		return 0;
#endif

	PRDAUSR = &estructura;

 	FIRSTEVT(0) = CUREVT(0) = (event_t *) buffers[0];
 	LASTEVT(0) = FIRSTEVT(0) + EVENT_BUFFER_SIZE - 1;
 	FD(0) = fd;
 	FLUSHED(0) = 0;
 	PRDAVPID(0) = 0;

	CELLTRACE_EVENT (TIME, APPL_EV, EVT_BEGIN);

	return 1;
}

/******************************************************************************
 * flush_buffer
 ******************************************************************************/

static void __inline__ mpitrace_cell_asynch_put (void *ls, unsigned long long ea, int size, int tag)
{
	/* DMA transfers must be x16 bytes */
	if ((size & 0x0f) != 0x00)
		return;

	while (spu_readchcnt (MFC_Cmd) < 1);
	mfc_put (ls, ea, size, tag, 0x0, 0x0);
}

static void mpitrace_cell_wait (int tag)
{
	spu_writech (MFC_WrTagMask, 1 << tag);
	spu_mfcstat (2);
}

unsigned int dma_channel = DEFAULT_DMA_CHANNEL;

void flush_buffer (int mark_on_trace, int thread)
{
	unsigned long long init_time, fini_time;
	static unsigned int previous = 0;

#ifdef SPU_USES_WRITE
	unsigned int size;

	init_time = TIME;

	size = (CUREVT(0) - FIRSTEVT(0))*sizeof(event_t);

	if (size + previous >= max_file_size)
	{
		tracejant = 0;
	}
	else
	{
		write (FD(0), vector_events, size);
		previous += size;

		fini_time = TIME;

		CUREVT(0) = FIRSTEVT(0);
	}

#else
	unsigned int size[4] __attribute ((aligned(128)));


	init_time = TIME;
	
	size[0] = (CUREVT(0) - FIRSTEVT(0))*sizeof(event_t) + previous;
	
	if (size[0] >= max_file_size)
	{
		tracejant = 0;
	}
	else
	{
		mpitrace_cell_asynch_put (vector_events, out_trace+previous,
			EVENT_BUFFER_SIZE*sizeof(event_t), dma_channel);
		mpitrace_cell_wait (dma_channel);
		mpitrace_cell_asynch_put (size, out_count, 16, dma_channel);
		mpitrace_cell_wait (dma_channel);

		previous = size[0];
		CUREVT(0) = FIRSTEVT(0);

		fini_time = TIME;
	}
#endif

	if (mark_on_trace)
	{
		CELLTRACE_EVENT (init_time, FLUSH_EV, EVT_BEGIN);
		CELLTRACE_EVENT (fini_time, FLUSH_EV, EVT_END);
	}
}

#ifndef SPU_USES_WRITE
void Touch_PPU_Buffer (void)
{
	unsigned int size[4] __attribute ((aligned(128)));
	unsigned i;

	for (i = 0; i < 4; i++)
		size[i] = 0;

	mpitrace_cell_asynch_put (vector_events, out_trace, EVENT_BUFFER_SIZE*sizeof(event_t), dma_channel);
	mpitrace_cell_wait (dma_channel);
	mpitrace_cell_asynch_put (size, out_count, 16, dma_channel);
	mpitrace_cell_wait (dma_channel);
}
#endif


void Thread_Finalization ()
{
	CELLTRACE_EVENT (TIME, APPL_EV, EVT_END);
	flush_buffer (0,0); /* This flush won't make the FLUSH_EV into the final buffer */
}

