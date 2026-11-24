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

#include "common.h"

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif

#include "wrapper.h"
#include "hwc.h"
#include "utils.h"
#include "xalloc.h"
#include "change_mode.h"
#if defined(HAVE_BURST)
# include "burst_mode.h"
#endif

/* Default configuration variables */
int Starting_Trace_Mode = TRACE_MODE_DETAIL;

/* Bursts mode specific configuration variables */
unsigned long long BurstMode_Threshold = 10000000; /* 10ms */
int BurstMode_MPI_Stats = DISABLED;
int BurstMode_OMP_Stats = DISABLED;
int BurstMode_OMP_Summarization = DISABLED;

static int is_ValidMode (int mode)
{
	switch(mode)
	{
		case TRACE_MODE_DETAIL:
		case TRACE_MODE_BURST:
			return TRUE;
		default:
			return FALSE;
	}
}

int Trace_Mode_reInitialize (int old_num_threads, int new_num_threads)
{
	int i;

	/* Blocks already exist; only initialize the newly added threads. */
	for (i=old_num_threads; i<new_num_threads; i++)
	{
		Extrae_thread_trace_data_t *data = get_thread_trace_data(i);
		data->mpi_deepness = 0;
		data->current_trace_mode = Starting_Trace_Mode;
		data->future_trace_mode = Starting_Trace_Mode;
		data->pending_trace_mode_change = FALSE;
		data->first_trace_mode = TRUE;
	}

	return TRUE;
}

int Trace_Mode_FirstMode (unsigned thread)
{
	return get_thread_trace_data(thread)->first_trace_mode;
}

int Trace_Mode_Initialize (int num_threads)
{
	int res = Trace_Mode_reInitialize (0, num_threads);

	/* Show configuration */
	if (res && TASKID == 0)
	{
		fprintf(stdout, PACKAGE_NAME": Tracing mode is set to: ");
		switch(Starting_Trace_Mode)
		{
			case TRACE_MODE_DETAIL:
				fprintf(stdout, "Detail.\n");
				break;
			case TRACE_MODE_BURST:
				fprintf(stdout, "CPU Bursts.\n");
				fprintf(stdout, PACKAGE_NAME": Minimum burst threshold is %llu ns.\n", BurstMode_Threshold);
				fprintf(stdout, PACKAGE_NAME": MPI statistics are %s.\n", (BurstMode_MPI_Stats ? "enabled" : "disabled"));
				fprintf(stdout, PACKAGE_NAME": OpenMP statistics are %s.\n", (BurstMode_OMP_Stats ? "enabled" : "disabled"));
				fprintf(stdout, PACKAGE_NAME": OpenMP summarization is %s.\n", (BurstMode_OMP_Summarization ? "enabled" : "disabled"));
				break;
			default:
				fprintf(stdout, "Unknown.\n");
				break;
		}
	}

	return res;
}

void Trace_Mode_Change (int tid, iotimer_t time)
{
	Extrae_thread_trace_data_t *data = get_thread_trace_data(tid);
	if (data->pending_trace_mode_change || data->first_trace_mode)
	{
		if (data->future_trace_mode != data->current_trace_mode || data->first_trace_mode)
		{
			switch(data->future_trace_mode)
			{
				case TRACE_MODE_DETAIL:
#if defined(HAVE_BURST)
					xtr_burst_emit_statistics();
#endif
					break;
				case TRACE_MODE_BURST:
					ACCUMULATED_COUNTERS_RESET(tid);
					break;
				default:
					break;
			}
			data->current_trace_mode = data->future_trace_mode;
			TRACE_EVENT (time, TRACING_MODE_EV, data->current_trace_mode);
		}
		data->pending_trace_mode_change = FALSE;
		data->first_trace_mode = FALSE;
	}
}

void
Trace_mode_switch(void)
{
	unsigned i;

	/*
	 * XXX Should this be Backend_getMaximumOfThreads()? If we decrease the
	 * number of threads, switch tracing mode and then increase again the the
	 * number of threads, only the "old" threads will use burst mode, while the
	 * "new" ones will continue in detail.
	 */
	for (i=0; i<Backend_getNumberOfThreads(); i++)
	{
		Extrae_thread_trace_data_t *data = get_thread_trace_data(i);
		data->pending_trace_mode_change = TRUE;
		data->future_trace_mode = (data->current_trace_mode == TRACE_MODE_DETAIL)?TRACE_MODE_BURST:TRACE_MODE_DETAIL;
	}
}

/* Configure options */

void TMODE_setInitial (int mode)
{
	if (is_ValidMode (mode))
	{
		Starting_Trace_Mode = mode;
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setInitial: Invalid mode '%d'.\n", mode);
	}
}

/* Change between detail/burst mode */
void TMODE_setCurrent (unsigned long long burst_threshold)
{
#if defined(HAVE_BURST)
	int new_mode = TRACE_MODE_DETAIL;

	if (burst_threshold > 0)
	{
		new_mode = TRACE_MODE_BURST;
		TMODE_setBurstThreshold(burst_threshold);

	}

	if (new_mode == TRACE_MODE_BURST) xtr_burst_init();

	/*
	 * XXX Should this be Backend_getMaximumOfThreads()? If we decrease the
	 * number of threads, switch tracing mode and then increase again the the
	 * number of threads, only the "old" threads will use burst mode, while the
	 * "new" ones will continue in detail.
	 */
	for (int i=0; i<Backend_getNumberOfThreads(); i++)
	{
		Extrae_thread_trace_data_t *data = get_thread_trace_data(i);
		data->future_trace_mode = new_mode;
		if (data->current_trace_mode != data->future_trace_mode)
		{
			data->pending_trace_mode_change = TRUE;
		}
	}
#endif /* HAVE_BURST */
}

/* Burst mode specific */

void TMODE_setBurstThreshold (unsigned long long threshold)
{
	if (threshold > 0)
	{
		BurstMode_Threshold = threshold;
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setBurstThreshold: Invalid minimum threshold '%llu'.\n", threshold);
	}
}

void TMODE_setBurstStatistics (int type, int status)
{
	if ((status != TRUE) && (status != FALSE))
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setBurstStatistics: Invalid argument '%d'.\n", status); 
		return;
	}
	switch (type)
	{
	case TM_BURST_OMP_STATISTICS:
		BurstMode_OMP_Stats = status;
		if (TASKID == 0 )
			fprintf(stdout, PACKAGE_NAME": Tracing OMP runtime statistics \n");
		break;

	case TM_BURST_MPI_STATISTICS:
		BurstMode_MPI_Stats = status;
		if (TASKID == 0 )
			fprintf(stdout, PACKAGE_NAME": Tracing mpi runtime statistics \n");
		break;

	default:
		break;
	}
}

void TMODE_setBurstOMPSummarization (int status)
{
	if ((status == TRUE) || (status == FALSE))
	{
		BurstMode_OMP_Summarization = status;
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setBurstOMPSummarization: Invalid argument '%d'.\n", status);
	}
}

