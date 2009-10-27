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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/spu/spu_wrapper.c,v $
 | 
 | @last_commit: $Date: 2009/01/12 16:13:21 $
 | @version:     $Revision: 1.10 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: spu_wrapper.c,v 1.10 2009/01/12 16:13:21 gllort Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>

#include "wrapper.h"
#include "threadid.h"
#include "events.h"
#include "spu_wrapper.h"
#include "clock.h"

extern int tracejant;
extern int *TracingBitmap;

unsigned int TaskID = 0;

/******************************************************************************
 ***  mpitrace_shutdown
 ******************************************************************************/

void mpitrace_shutdown_Wrapper ()
{
  /*
   *   event     : TRACING_EV                  value : EVT_END
   *   parameter : ---                         size  : ---
   */
  CELLTRACE_MISCEVENT (TIME, TRACING_EV, EVT_END, EMPTY);
  tracejant = FALSE;
#if USE_HARDWARE_COUNTERS
  HARDWARE_COUNTERS_STOP ();
#endif
}


/******************************************************************************
 ***  mpitrace_restart
 ******************************************************************************/

void mpitrace_restart_Wrapper ()
{
  tracejant = TRUE;
#if USE_HARDWARE_COUNTERS
  HARDWARE_COUNTERS_START ();
#endif
  /*
   *   event     : TRACING_EV                  value : EVT_BEGIN
   *   parameter : ---                         size  : ---
   */
  CELLTRACE_MISCEVENT (TIME, TRACING_EV, EVT_BEGIN, EMPTY);
}

/******************************************************************************
 ***  Trace_Event_C_Wrapper
 ******************************************************************************/

void Trace_Event_C_Wrapper (unsigned int tipus, unsigned int valor)
{
  /*
   *   event     : USER_EV                  value : TIPUS 
   *   parameter : VALOR                    size  : ---
   */
  CELLTRACE_MISCEVENT (TIME, USER_EV, tipus, valor); 
}

/******************************************************************************
 ***  Trace_Event_C_Wrapper
 ******************************************************************************/

void Trace_MultipleEvent_C_Wrapper (int count, unsigned int *tipus, unsigned int *valor)
{
	unsigned long long temps = TIME;
	int i;
	int events_remaining_for_flush = (CUREVT(0) - FIRSTEVT(0));

  /*
   *   event     : USER_EV                  value : TIPUS 
   *   parameter : VALOR                    size  : ---
   */

	if (count > events_remaining_for_flush)
		flush_buffer (0, 0);

	for (i = 0; i < count; i++)
	  CELLTRACE_MISCEVENT (temps, USER_EV, tipus[i], valor[i]); 
}

