/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/spu/wrapper.c,v $
 | 
 | @last_commit: $Date: 2009/06/19 14:30:26 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: wrapper.c,v 1.7 2009/06/19 14:30:26 harald Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>

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
   if (CUREVT(thread) >= LASTEVT(thread)) flush_buffer(1, thread);
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
  {
#if 0
    printf ("CELLtrace: <SPU %d> Not enough memory for the SPU tracing buffer (size: %d events)\n", rank, EVENT_BUFFER_SIZE);
#endif
    return 1;
  }
  buffers[0] = vector_events;
  return 0;
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


int spu_init_backend (int me, unsigned long long trace_ptr, unsigned long long count_trace_ptr, unsigned int file_size)
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
 	FD(0) = 0;
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
	unsigned int tamany[4] __attribute ((aligned(128)));
	unsigned long long temps_inicial, temps_final;

	static unsigned int previ = 0;

	temps_inicial = TIME;
	
	tamany[0] = (CUREVT(0) - FIRSTEVT(0))*sizeof(event_t) + previ;
	
	if (tamany[0] >= max_file_size)
	{
#if 0
		printf ("CELLtrace: SPU has reached the maximum space for its temporal file. Increase MPTRACE_SPU_FILE_SIZE!\n");
#endif
		tracejant = 0;
	}
	else
	{
		mpitrace_cell_asynch_put (vector_events, out_trace+previ,
			EVENT_BUFFER_SIZE*sizeof(event_t), dma_channel);
		mpitrace_cell_wait (dma_channel);
		mpitrace_cell_asynch_put (tamany, out_count, 16, dma_channel);
		mpitrace_cell_wait (dma_channel);

		previ = tamany[0];
		CUREVT(0) = FIRSTEVT(0);

		temps_final = TIME;

		/* Buffer must hold > 2 events!! */ 
		CELLTRACE_EVENT (temps_inicial, FLUSH_EV, EVT_BEGIN);
		CELLTRACE_EVENT (temps_final, FLUSH_EV, EVT_END);
	}
}

void Touch_PPU_Buffer (void)
{
	unsigned int tamany[4] __attribute ((aligned(128)));
	unsigned i;

	for (i = 0; i < 4; i++)
		tamany[i] = 0;

	mpitrace_cell_asynch_put (vector_events, out_trace, EVENT_BUFFER_SIZE*sizeof(event_t), dma_channel);
	mpitrace_cell_wait (dma_channel);
	mpitrace_cell_asynch_put (tamany, out_count, 16, dma_channel);
	mpitrace_cell_wait (dma_channel);
}


void Thread_Finalization ()
{
  CELLTRACE_EVENT (TIME, APPL_EV, EVT_END);
	flush_buffer (0,0);
}

