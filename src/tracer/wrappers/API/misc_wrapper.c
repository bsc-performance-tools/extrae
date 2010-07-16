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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
# include <sys/resource.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_MATH_H
# include <math.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "wrapper.h"
#include "clock.h"
#include "hwc.h"
#include "extrae_user_events.h"
#include "misc_wrapper.h"
#include "common_hwc.h"

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

void MPItrace_shutdown_Wrapper (void)
{
	TRACE_MISCEVENTANDCOUNTERS (TIME, TRACING_EV, EVT_END, EMPTY);
	tracejant = FALSE;
}

void MPItrace_restart_Wrapper (void)
{
	tracejant = TRUE;
	TRACE_MISCEVENTANDCOUNTERS (TIME, TRACING_EV, EVT_BEGIN, EMPTY);
}

void MPItrace_Event_Wrapper (unsigned int *tipus, unsigned int *valor)
{
	TRACE_MISCEVENT (TIME, USER_EV, *tipus, *valor);
}


void MPItrace_N_Event_Wrapper (unsigned int *count, unsigned int *types, unsigned int *values)
{
	iotimer_t temps;
	unsigned i;
	int events_id[MAX_MULTIPLE_EVENTS];

	if (*count > 0)
	{
		for (i = 0; i < *count; i++)
			events_id[i] = USER_EV;
		temps = TIME;
		TRACE_N_MISCEVENT(temps, *count, events_id, types, values);
	}
}

void MPItrace_Eventandcounters_Wrapper (int *tipus, int *valor)
{
#if USE_HARDWARE_COUNTERS
	if (tracejant)
		TRACE_MISCEVENTANDCOUNTERS (TIME, USER_EV, *tipus, *valor);
#else
	MPItrace_Event_Wrapper (tipus, valor);
#endif
}


void MPItrace_N_Eventsandcounters_Wrapper (unsigned int *count, unsigned int *types, unsigned int *values)
{
	iotimer_t temps;
	unsigned i;
	int events_id[MAX_MULTIPLE_EVENTS];

	if (*count > 0)
	{
		for (i = 0; i < *count; i++)
			events_id[i] = USER_EV;
		temps = TIME;
		TRACE_N_MISCEVENTANDCOUNTERS(temps, *count, events_id, types, values);
	}
}

void MPItrace_counters_Wrapper (void)
{
#if USE_HARDWARE_COUNTERS
	TRACE_EVENTANDCOUNTERS (TIME, HWC_EV, 0, TRUE);
#endif
}

void MPItrace_next_hwc_set_Wrapper (void)
{
#if USE_HARDWARE_COUNTERS
	HWC_Start_Next_Set (TIME, THREADID);
#endif
}

void MPItrace_previous_hwc_set_Wrapper (void)
{
#if USE_HARDWARE_COUNTERS
	HWC_Start_Previous_Set (TIME, THREADID);
#endif
}

void MPItrace_set_options_Wrapper (int options)
{
	Trace_Caller_Enabled[CALLER_MPI] = (options & EXTRAE_CALLER_OPTION);
	Trace_HWC_Enabled = (options & EXTRAE_HWC_OPTION);
	tracejant_hwc_mpi = (options & EXTRAE_MPI_HWC_OPTION);
	tracejant_mpi     = (options & EXTRAE_MPI_OPTION);   
	tracejant_omp     = (options & EXTRAE_OMP_OPTION);   
	tracejant_hwc_omp = (options & EXTRAE_OMP_HWC_OPTION);   
	tracejant_hwc_uf  = (options & EXTRAE_UF_HWC_OPTION);
	setSamplingEnabled (options & EXTRAE_SAMPLING_OPTION);
}

void MPItrace_getrusage_Wrapper (iotimer_t timestamp)
{
	int err;
	struct rusage current_usage;
	static int init_pending = TRUE;
	static int getrusage_running = FALSE;
	static struct rusage accum_usage;

	if (TRACING_RUSAGE)
	{
		if (getrusage_running)
			return;

		getrusage_running = TRUE;

		err = getrusage(RUSAGE_SELF, &current_usage);

		if (init_pending) 
		{
			memset(&accum_usage, 0, sizeof(accum_usage));
			init_pending = FALSE;
		}

		if (!err) 
		{
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_UTIME_EV, ((current_usage.ru_utime.tv_sec * 1000000) + current_usage.ru_utime.tv_usec) - ((accum_usage.ru_utime.tv_sec * 1000000) + accum_usage.ru_utime.tv_usec))
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_STIME_EV, ((current_usage.ru_stime.tv_sec * 1000000) + current_usage.ru_stime.tv_usec) - ((accum_usage.ru_stime.tv_sec * 1000000) + accum_usage.ru_stime.tv_usec))
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_MINFLT_EV, current_usage.ru_minflt - accum_usage.ru_minflt);
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_MAJFLT_EV, current_usage.ru_majflt - accum_usage.ru_majflt);
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_NVCSW_EV,  current_usage.ru_nvcsw - accum_usage.ru_nvcsw);
			TRACE_MISCEVENT(timestamp, RUSAGE_EV, RUSAGE_NIVCSW_EV, current_usage.ru_nivcsw - accum_usage.ru_nivcsw);
		}

		accum_usage.ru_utime.tv_sec = current_usage.ru_utime.tv_sec;
		accum_usage.ru_utime.tv_usec = current_usage.ru_utime.tv_usec;
		accum_usage.ru_stime.tv_sec = current_usage.ru_stime.tv_sec;
		accum_usage.ru_stime.tv_usec = current_usage.ru_stime.tv_usec;
		accum_usage.ru_minflt = current_usage.ru_minflt;
		accum_usage.ru_majflt = current_usage.ru_majflt;
		accum_usage.ru_nvcsw  = current_usage.ru_nvcsw;
		accum_usage.ru_nivcsw = current_usage.ru_nivcsw;
  
		getrusage_running = FALSE;
	}
}

void MPItrace_memusage_Wrapper (iotimer_t timestamp)
{
	if (TRACING_MEMUSAGE)
	{
		struct mallinfo mi = mallinfo();
		int inuse = mi.arena + mi.hblkhd - mi.fordblks;

		TRACE_MISCEVENT(timestamp, MEMUSAGE_EV, MEMUSAGE_ARENA_EV,    mi.arena);
		TRACE_MISCEVENT(timestamp, MEMUSAGE_EV, MEMUSAGE_HBLKHD_EV,   mi.hblkhd);
		TRACE_MISCEVENT(timestamp, MEMUSAGE_EV, MEMUSAGE_UORDBLKS_EV, mi.uordblks);
		TRACE_MISCEVENT(timestamp, MEMUSAGE_EV, MEMUSAGE_FORDBLKS_EV, mi.fordblks);
		TRACE_MISCEVENT(timestamp, MEMUSAGE_EV, MEMUSAGE_INUSE_EV,    inuse);
	}
}

void MPItrace_user_function_Wrapper (int enter)
{
	UINT64 ip = (enter)?get_caller(4):EMPTY;
	TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, ip, tracejant_hwc_uf);
}

void MPItrace_function_from_address_Wrapper (int type, void *address)
{
	if (type == USRFUNC_EV || type == OMPFUNC_EV)
	{
#if USE_HARDWARE_COUNTERS
		int filter = (type==USRFUNC_EV)?tracejant_hwc_uf:tracejant_hwc_omp;
		TRACE_EVENTANDCOUNTERS (TIME, type, (UINT64) address, filter);
#else
		TRACE_EVENT(TIME, type, (UINT64) address);
#endif
	}
}

static void Generate_Task_File_List (void)
{
	int filedes;
	unsigned thid;
	ssize_t ret;
	char tmpname[1024];
	char hostname[1024];
	char tmp_line[1024];
	
	sprintf (tmpname, "%s/%s.mpits", final_dir, appl_name);

	filedes = open (tmpname, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (filedes < 0)
		return;

	if (gethostname (hostname, 1024 - 1) != 0)
		sprintf (tmpname, "localhost");

	for (thid = 0; thid < Backend_getNumberOfThreads(); thid++)
	{
		FileName_PTT(tmpname, Get_FinalDir(0), appl_name, getpid(), 0, thid, EXT_MPIT);

		sprintf (tmp_line, "%s on %s\n", tmpname, hostname);
		ret = write (filedes, tmp_line, strlen (tmp_line));
		if (ret != (ssize_t) strlen (tmp_line))
		{
			close (filedes);
			return;
		}
	}

	close (filedes);
	return;
}


void MPItrace_init_Wrapper (void)
{
	iotimer_t temps;

	mptrace_IsMPI = FALSE;
	NumOfTasks = 1;

	/* Initialize the backend */
	if (!Backend_preInitialize (TASKID, NumOfTasks, getenv("EXTRAE_CONFIG_FILE")))
		return;

	Generate_Task_File_List();

	/* Take the time */
	temps = TIME;

	/* End initialization of the backend */
	if (!Backend_postInitialize (TASKID, NumOfTasks, temps, temps, NULL))
		return;

}

void MPItrace_fini_Wrapper (void)
{
	if (!mpitrace_on)
		return;

	/* Es tanca la llibreria de traceig */
	Thread_Finalization ();
}

void Extrae_init_UserCommunication_Wrapper (struct extrae_UserCommunication *ptr)
{
	memset (ptr, 0, sizeof(struct extrae_UserCommunication));
}

void Extrae_init_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr)
{
	memset (ptr, 0, sizeof(struct extrae_CombinedEvents));
	ptr->UserFunction = EXTRAE_USER_FUNCTION_NONE;
}

void Extrae_emit_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr)
{
	iotimer_t myTime;
	unsigned i;
	int events_id[MAX_MULTIPLE_EVENTS];

	myTime = TIME;

	/* Emit events first */
	if (ptr->nEvents > 0)
	{
		for (i = 0; i < ptr->nEvents; i++)
			events_id[i] = USER_EV;
		if (ptr->HardwareCounters)
		{
			TRACE_N_MISCEVENTANDCOUNTERS(myTime, ptr->nEvents, events_id, ptr->Types, ptr->Values);
		}
		else
		{
			TRACE_N_MISCEVENT(myTime, ptr->nEvents, events_id, ptr->Types, ptr->Values);
		}
	}

	/* Emit user function. If hwc were emitted before, don't emit now because they
	   will share the same timestamp and Paraver won't handle that well. Otherwise,
	   honor tracejant_hwc_uf
	*/
	if (ptr->UserFunction != EXTRAE_USER_FUNCTION_NONE)
	{
		UINT64 ip = (ptr->UserFunction == EXTRAE_USER_FUNCTION_ENTER)?get_caller(4):EMPTY;
#if USE_HARDWARE_COUNTERS
		int EmitHWC = (!ptr->HardwareCounters && tracejant_hwc_uf);
		TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, ip, EmitHWC);
#else
		TRACE_EVENT (TIME, USRFUNC_EV, ip);
#endif
	}

	/* Now emit the callers */
	if (ptr->Callers)
	{
		trace_callers (myTime, 4, CALLER_MPI);
	}

	/* Finally emit user communications */
	for (i = 0; i < ptr->nCommunications ; i++)
	{
		TRACE_USER_COMMUNICATION_EVENT(myTime,
		  (ptr->Communications[i].type==EXTRAE_USER_SEND)?USER_SEND_EV:USER_RECV_EV,
		  ptr->Communications[i].partner, ptr->Communications[i].size,
		  ptr->Communications[i].tag, ptr->Communications[i].id) 
	}
}
