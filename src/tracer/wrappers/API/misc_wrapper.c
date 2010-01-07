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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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
#include "mpitrace_user_events.h"
#include "misc_wrapper.h"

#if 0
#if defined(MPI_SUPPORT)
# include "mpi_wrapper.h"
# include "myrinet_hwc.h"
#endif
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
	int i;
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
	int i;
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
	Trace_Caller_Enabled[CALLER_MPI] = (options & MPITRACE_CALLER_OPTION);
	Trace_HWC_Enabled = (options & MPITRACE_HWC_OPTION);
	tracejant_hwc_mpi = (options & MPITRACE_MPI_HWC_OPTION);
	tracejant_mpi     = (options & MPITRACE_MPI_OPTION);   
	tracejant_omp     = (options & MPITRACE_OMP_OPTION);   
	tracejant_hwc_omp = (options & MPITRACE_OMP_HWC_OPTION);   
	tracejant_hwc_uf  = (options & MPITRACE_UF_HWC_OPTION);
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

void MPItrace_user_function_Wrapper (int enter)
{
	UINT64 ip = (enter)?get_caller(4):EMPTY;
	TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, ip, tracejant_hwc_uf);
}

void MPItrace_function_from_address_Wrapper (int type, void *address)
{
	if (type == USRFUNC_EV || type == OMPFUNC_EV)
	{
		int filter = (type==USRFUNC_EV)?tracejant_hwc_uf:tracejant_hwc_omp;
		TRACE_EVENTANDCOUNTERS (TIME, type, (UINT64) address, filter);
	}
}

static void Generate_Task_File_List (void)
{
	int filedes;
	int thid;
	int ret;
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
		/* Tracefile_Name (tmpname, final_dir, appl_name, getpid(), 0, thid); */
		FileName_PTT(tmpname, final_dir, appl_name, getpid(), 0, thid, EXT_MPIT);

		sprintf (tmp_line, "%s on %s\n", tmpname, hostname);
		ret = write (filedes, tmp_line, strlen (tmp_line));
		if (ret != strlen (tmp_line))
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
	if (!Backend_preInitialize (TASKID, NumOfTasks, getenv("MPTRACE_CONFIG_FILE")))
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

