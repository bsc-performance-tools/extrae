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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "dimemas_generator.h"
#include "misc_trf_semantics.h"
#include "trace_to_trf.h"
#include "semantics.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#include "events.h"

/******************************************************************************
 ***  Appl_Event
 ******************************************************************************/
static int Appl_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset)
{
	Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
	Dimemas_User_Event (fset->output_file, task-1, thread-1,
		Get_EvEvent (current_event), Get_EvValue (current_event));

	return 0;
}

/******************************************************************************
 ***  User_Event
 ******************************************************************************/
static int User_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	unsigned int EvType, EvValue;

	EvType  = Get_EvValue (current_event);     /* Value is the user event type.  */
	EvValue = Get_EvMiscParam (current_event); /* Param is the user event value. */

	Dimemas_User_Event (fset->output_file, task-1, thread-1, EvType, EvValue);

  return 0;
}

/******************************************************************************
 **      Function name : Set_Overflow_Event
 **      Description :
 ******************************************************************************/
static int Set_Overflow_Event (event_t * current,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t *fset )
{
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	HardwareCounters_SetOverflow (ptask, task, thread, current);
#endif

	return 0;
}

#if USE_HARDWARE_COUNTERS
/******************************************************************************
 **      Function name : ResetCounters
 **      Description :
 ******************************************************************************/
static void ResetCounters (int ptask, int task)
{
	int thread, cnt;

	obj_table[ptask].tasks[task].tracing_disabled = FALSE;

	for (thread = 0; thread < obj_table[ptask].tasks[task].nthreads; thread++)
		for (cnt = 0; cnt < MAX_HWC; cnt++)
			obj_table[ptask].tasks[task].threads[thread].counters[cnt] = 0LL;
}
#endif

/******************************************************************************
 **      Function name : Evt_SetCounters
 **      Description :
 ******************************************************************************/
static int Evt_SetCounters ( event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset )
{
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	int i;
	unsigned int hwctype[MAX_HWC+1];
	unsigned long long hwcvalue[MAX_HWC+1];
	unsigned int newSet = Get_EvValue(current_event);

	Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
	ResetCounters (ptask-1, task-1);
	HardwareCounters_Change (ptask, task, thread, newSet, hwctype, hwcvalue);

	for (i = 0; i < MAX_HWC+1; i++)
		if (NO_COUNTER != hwctype[i])
			Dimemas_User_Event (fset->output_file, task-1, thread-1, hwctype[i], hwcvalue[i]);
#endif

	return 0;
}

SingleEv_Handler_t TRF_MISC_Event_Handlers[] = {
	{ FLUSH_EV, SkipHandler },
	{ READ_EV, SkipHandler },
	{ WRITE_EV, SkipHandler },
	{ APPL_EV, Appl_Event },
	{ USER_EV, User_Event },
	{ HWC_EV, SkipHandler }, /* hardware counters will be emitted at the main loop */
	{ HWC_CHANGE_EV, Evt_SetCounters },
	{ TRACING_EV, SkipHandler },
	{ SET_TRACE_EV, SkipHandler },
	{ CPU_BURST_EV, SkipHandler },
	{ RUSAGE_EV, SkipHandler },
	{ MEMUSAGE_EV, SkipHandler },
	{ MPI_STATS_EV, SkipHandler },
	{ USRFUNC_EV, SkipHandler },
	{ SAMPLING_EV, SkipHandler },
	{ HWC_SET_OVERFLOW_EV, Set_Overflow_Event },
	{ TRACING_MODE_EV, SkipHandler },
	{ NULL_EV, NULL }
};

