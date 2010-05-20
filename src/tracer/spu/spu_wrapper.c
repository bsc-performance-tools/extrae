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

