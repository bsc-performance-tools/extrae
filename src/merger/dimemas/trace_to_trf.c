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

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
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
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#if defined (PARALLEL_MERGE)
# include <mpi.h>
# include "mpi-tags.h"
# include "mpi-aux.h"
# include "parallel_merge_aux.h"
#endif

#include "utils.h"
#include "file_set.h"
#include "events.h"
#include "record.h"
#include "object_tree.h"
#include "mpi2out.h"
#include "trace_to_trf.h"
#include "mpi_prv_semantics.h"
#include "misc_prv_semantics.h"
#include "omp_prv_semantics.h"
#include "pthread_prv_semantics.h"
#include "mpi_comunicadors.h"
#include "labels.h"
#include "trace_mode.h"
#include "semantics.h"
#include "dump.h"
#include "communicators.h"
#include "cpunode.h"
#include "checkoptions.h"
#include "options.h"

#include "paraver_state.h"
#include "paraver_generator.h"
#include "dimemas_generator.h"

#include "mpi_prv_events.h"
#include "pthread_prv_events.h"
#include "omp_prv_events.h"
#include "misc_prv_events.h"
#include "addr2info.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

UINT64 InitTracingTime;

char dimemas_tmp[PATH_MAX];

unsigned long long Dimemas_hr_to_relative (UINT64 iotimer)
{
	return iotimer - InitTracingTime;
}

static void Dimemas_GenerateOffsets (unsigned num_appl, unsigned long long **ptr,
	unsigned int *count)
{
	unsigned long long *offsets;
	unsigned i = 0;
	unsigned ptask;
	unsigned task;
	unsigned thread;

	*ptr = NULL;
	*count = 0;

	/* Count total number of running threads of the application */
	for (ptask = 0; ptask < num_appl; ptask++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ptask+1);
		for (task = 0; task < ptask_info->ntasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1,task+1);
			i += task_info->nthreads;
		}
	}

	/* Allocate memory for the offsets of those threads */
	offsets = (unsigned long long*) malloc (sizeof(unsigned long long)*i);
	if (NULL == offsets)
	{
		fprintf (stderr, "mpi2dim: Unable to allocate memory for %d offsets\n", i);
		fflush (stderr);
		exit (-1);
	}

	/* Put the dimemas_size field on the offsets */
	i = 0;
	for (ptask = 0; ptask < num_appl; ptask++)
	{
		ptask_t *ptask_info = GET_PTASK_INFO(ptask+1);
		for (task = 0; task < ptask_info->ntasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1,task+1);
			for (thread = 0; thread < task_info->nthreads; thread++, i++)
			{
				thread_t *thread_info = GET_THREAD_INFO(ptask+1,task+1,thread+1);
				offsets[i] = thread_info->dimemas_size;
			}
		}
	}

	/* Put the returning values */
	*ptr = offsets;
	*count = i;
}

/******************************************************************************
 ***  Dimemas_ProcessTraceFiles
 ******************************************************************************/

int Dimemas_ProcessTraceFiles (char *outName, unsigned long nfiles,
	struct input_t *files, unsigned int num_appl,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int taskid)
{
	FileSet_t * fset;
	event_t * current_event;
	char envName[PATH_MAX], *tmp;
	unsigned int cpu, ptask, task, thread, error;
	unsigned int Type, EvType, current_file, count;
	unsigned long long current_time = 0;
	unsigned long long num_of_events, parsed_events, tmp_nevents;
	unsigned long long trace_size;
	unsigned long long *offsets;
	long long options;
	double pct, last_pct;

	if (1 != num_appl)
	{
		fprintf (stderr, "mpi2dim: Error! This merger does not support merging more than 1 parallel application\n");
		fflush (stderr);
		exit (-1);
	}

	InitializeObjectTable (num_appl, files, nfiles);

#if defined(PARALLEL_MERGE)
	ParallelMerge_InitCommunicators();
#endif
	Semantics_Initialize (TRF_SEMANTICS);

	fset = Create_FS (nfiles, files, taskid, TRF_SEMANTICS);
	error = (fset == NULL);

	/* If no actual filename is given, use the binary name if possible */
	if (!get_merge_GivenTraceName())
	{
		char *FirstBinaryName = ObjectTable_GetBinaryObjectName (1, 1);
		if (FirstBinaryName != NULL)
		{
			char prvfile[strlen(FirstBinaryName) + 5];
			sprintf (prvfile, "%s.dim", FirstBinaryName);
			set_merge_OutputTraceName (prvfile);
			set_merge_GivenTraceName (TRUE);
		}
	}

	if (file_exists(get_merge_OutputTraceName()) &&
	    !get_option_merge_TraceOverwrite())
	{
		unsigned lastid = 0;
		char tmp[1024];
		do
		{
			lastid++;
			if (lastid >= 10000)
			{
				fprintf (stderr, "Error! Automatically given ID for the tracefile surpasses 10000!\n");
				exit (-1);
			}

			strncpy (tmp, get_merge_OutputTraceName(), sizeof(tmp));
			if (strcmp (&tmp[strlen(tmp)-strlen(".dim")], ".dim") == 0)
			{
				char extra[1+4+1+3+1];
				sprintf (extra, ".%04d.dim", lastid);
				strncpy (&tmp[strlen(tmp)-strlen(".dim")], extra, strlen(extra));
			}
		} while (file_exists (tmp));
		set_merge_OutputTraceName (tmp);
		set_merge_GivenTraceName (TRUE);
	}

#if defined(PARALLEL_MERGE)
	if (numtasks > 1)
	{
		int res;
		unsigned int temp;

		res = MPI_Allreduce (&error, &temp, 1, MPI_UNSIGNED, MPI_LOR, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Allreduce, "Failed to share translation completion!");

		error = temp;
	}
#endif

	if (error)
	{
		if (0 == taskid)
		{
			fprintf (stderr, "mpi2dim: Error! Some of the processors failed create trace descriptors\n");
			fflush (stderr);
		}
		return -1;
	}

	if (get_option_merge_dump())
		make_dump (fset);

#if defined(HETEROGENEOUS_SUPPORT)
	/*	Must we accomodate all the intermediate files into this machine endianes? */
	EndianCorrection (fset, numtasks, taskid);
#endif

	/*
	 * Initialize the communicators management
	 */
	initialize_comunicadors (num_appl);

	options = GetTraceOptions (fset, numtasks, taskid);

	CheckHWCcontrol (taskid, options);
	CheckClockType (taskid, options, TRF_SEMANTICS, get_option_merge_ForceFormat());

	CheckCircularBufferWhenTracing (fset, numtasks, taskid);

	/*
	 * Take the first event like the minimum time. 
	 */
	current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);

	InitTracingTime = Get_EvTime (current_event);

#if defined(PARALLEL_MERGE)
	if (numtasks > 1)
	{
		int res;
		UINT64 temp;

		res = MPI_Allreduce (&InitTracingTime, &temp, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Allreduce, "Failed to share init tracing time!");

		InitTracingTime = temp;
	}
#endif

	error = FALSE;

	/*
	   First step, search and generate communicators.
	*/
	if (0 == taskid)
	{
		fprintf (stdout, "mpi2dim: Parsing intermediate files. Generating communicators.\n");
		if (1 == numtasks)
			fprintf (stdout, "mpi2dim: Progress 1 of 2 ... ");
		fflush (stdout);
	}
	Rewind_FS (fset);
	parsed_events = 0;
	last_pct = 0.0;
	num_of_events = EventsInFS (fset);
	current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);

	do
	{
		tmp_nevents = 1;
		EvType = Get_EvEvent (current_event);
		
		if (getEventType (EvType, &Type))
			if (MPI_COMM_ALIAS_TYPE == Type)
			{
				error = GenerateAliesComunicator (current_event, current_time, cpu, ptask, task, thread, fset, &tmp_nevents, TRF_SEMANTICS);
				Enable_MPI_Operation (EvType);
			}

		/* Right now, progress bar is only shown when numtasks is 1 */
		if (1 == numtasks)
		{
			parsed_events += tmp_nevents;
			pct = ((double) parsed_events)/((double) num_of_events)*100.0f;
			if (pct > last_pct + 5.0 && pct <= 100.0f)
			{
				fprintf (stdout, "%.0lf%% ", pct);
				fflush (stdout);
				while (last_pct + 5.0 < pct)
					last_pct += 5.0;
			}
		}
		current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);
	} while ((current_event != NULL) && !error);

#if defined(PARALLEL_MERGE)
	ParallelMerge_BuildCommunicators (numtasks, taskid);
#endif

	if (1 == numtasks)
	{
		fprintf (stdout, "\n");
		fflush (stdout);
	}

	/*
	   Second step, build the final trace
	*/
	if (0 == taskid)
	{
		fprintf (stdout, "mpi2dim: Parsing intermediate files. Generating trace.\n");
		if (numtasks > 1)
			fprintf (stdout, "mpi2dim: Progress ... ");
		else
			fprintf (stdout, "mpi2dim: Progress 2 of 2 ... ");
		fflush (stdout);
	}
	Rewind_FS (fset);
	parsed_events = 0;
	last_pct = 0.0;
	num_of_events = EventsInFS (fset);
	current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);

	if (0 == taskid)
	{
#if HAVE_FOPEN64
		fset->output_file = fopen64 (outName, "w+");
#else
		fset->output_file = fopen (outName, "w+");
#endif
		if (NULL == fset->output_file)
		{
			fprintf (stderr, "\nmpi2dim ERROR: Creating Dimemas tracefile : %s on processor %d\n", outName, taskid);
			exit (-1);
		}
	} /* taskid == 0 */
	else
	{

		if (getenv ("MPI2DIM_TMP_DIR") == NULL)
			if (getenv ("TMPDIR") == NULL)
				sprintf (dimemas_tmp, "TmpFileXXXXXX");
			else
				sprintf (dimemas_tmp, "%s/TmpFileXXXXXX", getenv ("TMPDIR"));
		else
			sprintf (dimemas_tmp, "%s/TmpFileXXXXXX", getenv ("MPI2DIM_TMP_DIR"));

		if (mkstemp (dimemas_tmp) == -1)
		{
			perror ("mkstemp");
			fprintf (stderr, "mpi2dim: Unable to create temporal file using mkstemp\n");
			fflush (stderr);
			exit (-1);
		}

#if HAVE_FOPEN64
		fset->output_file = fopen64 (dimemas_tmp, "w+");
#else
		fset->output_file = fopen (dimemas_tmp, "w+");
#endif

		if (NULL == fset->output_file)
		{
			fprintf (stderr, "mpi2dim ERROR: Creating Dimemas temporal tracefile : %s on processor %d\n", outName, taskid);
			exit (-1);
		}

		remove (dimemas_tmp);
	}

	if (0 == taskid)
		Dimemas_WriteHeader (num_appl, fset->output_file, NodeCPUinfo, outName);

	current_file = -1;

	do
	{
		if (current_file != GetActiveFile(fset))
		{
#if !defined(HAVE_FTELL64) && !defined(HAVE_FTELLO64)
			(GET_THREAD_INFO(ptask,task,thread))->dimemas_size = ftell (fset->output_file);
#elif defined(HAVE_FTELL64)
			(GET_THREAD_INFO(ptask,task,thread))->dimemas_size = ftell64 (fset->output_file);
#elif defined(HAVE_FTELLO64)
			(GET_THREAD_INFO(ptask,task,thread))->dimemas_size = ftello64 (fset->output_file);
#endif
			InitTracingTime = Get_EvTime (current_event);
			current_file = GetActiveFile (fset);
		}

		tmp_nevents = 1;

		EvType = Get_EvEvent (current_event);

		if (getEventType (EvType, &Type))
		{
			current_time = Dimemas_hr_to_relative (Get_EvTime (current_event));

			if (Type == PTHREAD_TYPE || Type == OPENMP_TYPE ||
			    Type == MISC_TYPE || Type == MPI_TYPE)
			{
				Ev_Handler_t *handler = Semantics_getEventHandler (EvType);
				if (handler != NULL)
				{
					handler (current_event, current_time, cpu, ptask, task, thread, fset);
					
					if (PTHREAD_TYPE == Type)
						Enable_pthread_Operation (EvType);
					else if (OPENMP_TYPE == Type)
						Enable_OMP_Operation (EvType);
					else if (MPI_TYPE == Type)
						Enable_MPI_Operation (EvType);
				}
				else
					fprintf (stderr, "mpi2dim: Error! unregistered event type %d in %s+%d\n", EvType, __func__, __LINE__);

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				if (Get_EvHWCRead (current_event))
				{
					int i;
					unsigned int hwctype[2*MAX_HWC];
					unsigned long long hwcvalue[2*MAX_HWC];

					HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue, FALSE);
					for (i = 0; i < MAX_HWC; i++)
						if (NO_COUNTER != hwctype[i])
							Dimemas_User_Event (fset->output_file, task-1, thread-1, hwctype[i], hwcvalue[i]);
					if (get_option_merge_AbsoluteCounters())
					{
						HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue, TRUE);
						for (i = 0; i < MAX_HWC; i++)
							if (NO_COUNTER != hwctype[i])
								Dimemas_User_Event (fset->output_file, task-1, thread-1, hwctype[i], hwcvalue[i]);
					}
				}
#endif
			}
		}

		(GET_THREAD_INFO(ptask,task,thread))->First_Event = FALSE;
		(GET_THREAD_INFO(ptask,task,thread))->Previous_Event_Time = current_time;

		/* Right now, progress bar is only shown when numtasks is 1 */
		if (1 == numtasks)
		{
			parsed_events += tmp_nevents;
			pct = ((double) parsed_events)/((double) num_of_events)*100.0f;
			if (pct > last_pct + 5.0 && pct <= 100.0f)
			{
				fprintf (stdout, "%d%% ", (int) pct);
				fflush (stdout);
				while (last_pct + 5.0 < pct)
					last_pct += 5.0;
			}
		}
		current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);
	} while ((current_event != NULL) && !error);

	fflush (fset->output_file);

#if !defined(HAVE_FTELL64) && !defined(HAVE_FTELLO64)
	trace_size = ftell (fset->output_file);
#elif defined(HAVE_FTELL64)
	trace_size = ftell64 (fset->output_file);
#elif defined(HAVE_FTELLO64)
	trace_size = ftello64 (fset->output_file);
#endif

	if (1 == numtasks)
	{
		fprintf (stdout, "\n");
		fflush (stdout);
	}

	fprintf (stdout, "mpi2dim: Processor %d %s to translate its assigned files\n", taskid, error?"failed":"succeeded");
	fflush (stdout);

	Dimemas_GenerateOffsets (num_appl, &offsets, &count);

#if defined(PARALLEL_MERGE)
	if (numtasks > 1)
	{
		Gather_Dimemas_Traces (numtasks, taskid, fset->output_file, 1024*1024*get_option_merge_MaxMem());
		Gather_Dimemas_Offsets (numtasks, taskid, count, offsets, &offsets, trace_size, fset);

		if (0 == taskid)
		{
#if !defined(HAVE_FSEEK64) && !defined(HAVE_FSEEKO64)
			fseek (fset->output_file, 0L, SEEK_END);
#elif defined(HAVE_FSEEK64)
			fseek64 (fset->output_file, 0L, SEEK_END);
#elif defined(HAVE_FSEEKO64)
			fseeko64 (fset->output_file, 0L, SEEK_END);
#endif
#if !defined(HAVE_FTELL64) && !defined(HAVE_FTELLO64)
			trace_size = ftell (fset->output_file);
#elif defined(HAVE_FTELL64)
			trace_size = ftell64 (fset->output_file);
#elif defined(HAVE_FTELLO64)
			trace_size = ftello64 (fset->output_file);
#endif
		}
	}
#endif

#if defined(PARALLEL_MERGE)
	/* In the parallel merge we have to */
	if (numtasks > 1)
		ShareTraceInformation (numtasks, taskid);
#endif


	if (0 == taskid)
	{
		int somefailed = FALSE;

		Dimemas_WriteOffsets (num_appl, fset->output_file, outName, trace_size, count, offsets);

		fclose (fset->output_file);

		strcpy (envName, outName);
		tmp = &(envName[strlen (envName) - 4]);

		strcpy (tmp, ".pcf");
		if (Labels_GeneratePCFfile (envName, options) == -1)
		{
			fprintf (stderr, "mpi2dim: WARNING! Unable to create PCF file!\n");
			somefailed = TRUE;
		}

		strcpy (tmp, ".row");
		if (GenerateROWfile (envName, NodeCPUinfo, nfiles, files) == -1)
		{
			fprintf (stderr, "mpi2dim: WARNING! Unable to create ROW file!\n");
			somefailed = TRUE;
		}

		fprintf (stdout, "mpi2dim: Congratulations! %s has been generated %s\n", outName,
		  somefailed?" (although some files were not created).":".");
		fflush (stdout);
	}
	return 0;
}
