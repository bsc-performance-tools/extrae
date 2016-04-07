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
#include "mode.h"
#include "utils.h"

int *MPI_Deepness              = NULL;
int *Current_Trace_Mode        = NULL;
static int *Future_Trace_Mode         = NULL;
int *Pending_Trace_Mode_Change = NULL;
static int *First_Trace_Mode          = NULL;

/* Default configuration variables */
int Starting_Trace_Mode = TRACE_MODE_DETAIL;

/* Bursts mode specific configuration variables */
unsigned long long BurstsMode_Threshold = 10000000; /* 10ms */
int                BurstsMode_MPI_Stats = ENABLED; 

static int is_ValidMode (int mode)
{
	switch(mode)
	{
		case TRACE_MODE_DETAIL:
		case TRACE_MODE_BURSTS:
			return TRUE;
		default:
			return FALSE;
	}
}

void Trace_Mode_CleanUp (void)
{
	xfree (MPI_Deepness);
	xfree (Current_Trace_Mode);
	xfree (Future_Trace_Mode);
	xfree (Pending_Trace_Mode_Change);
	xfree (First_Trace_Mode);
}

int Trace_Mode_reInitialize (int old_num_threads, int new_num_threads)
{
	int i, size;

	size = sizeof(int) * new_num_threads;

	MPI_Deepness = (int *)realloc(MPI_Deepness, size);
	if (MPI_Deepness == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for 'MPI_Deepness'\n");
		return FALSE; 
	}

	Current_Trace_Mode = (int *)realloc(Current_Trace_Mode, size);
	if (Current_Trace_Mode == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for 'Current_Trace_Mode'\n");
		return FALSE;
	}

	Future_Trace_Mode = (int *)realloc(Future_Trace_Mode,size);
	if (Future_Trace_Mode == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for 'Future_Trace_Mode'\n");
		return FALSE;
	}

	Pending_Trace_Mode_Change = (int *)realloc(Pending_Trace_Mode_Change, size);
	if (Pending_Trace_Mode_Change == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for 'Pending_Trace_Mode_Change'\n");
		return FALSE;
	}

	First_Trace_Mode = (int *)realloc(First_Trace_Mode, size);
	if (First_Trace_Mode == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for 'First_Trace_Mode'\n");
		return FALSE;
	}

	for (i=old_num_threads; i<new_num_threads; i++)
	{
		MPI_Deepness[i] = 0;
		Current_Trace_Mode[i] = Starting_Trace_Mode;
		Future_Trace_Mode[i] = Starting_Trace_Mode;
		Pending_Trace_Mode_Change[i] = FALSE;
		First_Trace_Mode[i] = TRUE;
	}

	return TRUE;
}

int Trace_Mode_FirstMode (unsigned thread)
{
	return First_Trace_Mode[thread];
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
			case TRACE_MODE_BURSTS:
				fprintf(stdout, "CPU Bursts.\n");
				fprintf(stdout, PACKAGE_NAME": Minimum burst threshold is %llu ns.\n", BurstsMode_Threshold);
				fprintf(stdout, PACKAGE_NAME": MPI statistics are %s.\n", (BurstsMode_MPI_Stats ? "enabled" : "disabled"));
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
	if (Pending_Trace_Mode_Change[tid] || First_Trace_Mode[tid])
	{
		if (Future_Trace_Mode[tid] != Current_Trace_Mode[tid] || First_Trace_Mode[tid])
		{
			switch(Future_Trace_Mode[tid])
			{
				case TRACE_MODE_DETAIL:
					break;
				case TRACE_MODE_BURSTS:
					ACCUMULATED_COUNTERS_RESET(tid);
					break;
				default:
					break;
			}
			Current_Trace_Mode[tid] = Future_Trace_Mode[tid];
			TRACE_EVENT (time, TRACING_MODE_EV, Current_Trace_Mode[tid]);
		}
		Pending_Trace_Mode_Change[tid] = FALSE;
		First_Trace_Mode[tid] = FALSE;
	}
}

void Trace_mode_switch (void)
{   
	unsigned i;

	for (i=0; i<Backend_getNumberOfThreads(); i++)
	{
		Pending_Trace_Mode_Change[i] = TRUE;
		Future_Trace_Mode[i] = (Current_Trace_Mode[i] == TRACE_MODE_DETAIL)?TRACE_MODE_BURSTS:TRACE_MODE_DETAIL;
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

/* Bursts mode specific */

void TMODE_setBurstsThreshold (unsigned long long threshold)
{
	if (threshold > 0)
	{
		BurstsMode_Threshold = threshold;
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setBurstsThreshold: Invalid minimum threshold '%llu'.\n", threshold);
	}
}

void TMODE_setBurstsStatistics (int status)
{
	if ((status == TRUE) || (status == FALSE))
	{
		BurstsMode_MPI_Stats = status;
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME": TMODE_setBurstsStatistics: Invalid argument '%d'.\n", status);
	}
}

