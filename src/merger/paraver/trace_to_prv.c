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
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
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
#include "trace_to_prv.h"
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
#include "openshmem_prv_events.h"
#include "omp_prv_events.h"
#include "misc_prv_events.h"
#include "opencl_prv_events.h"
#include "cuda_prv_events.h"
#include "java_prv_events.h"
#include "addr2info.h"
#include "timesync.h"
#include "vector.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

int **EnabledTasks = NULL;
unsigned long long **EnabledTasks_time = NULL;
struct address_collector_t CollectedAddresses;

mpi2prv_vector_t *RegisteredStackValues = NULL;
Extrae_Vector_t RegisteredCodeLocationTypes;

static void InitializeEnabledTasks (int numberoftasks, int numberofapplications)
{
  int i, j;

  EnabledTasks = (int **) malloc (sizeof (int *) * numberofapplications);
  if (EnabledTasks == NULL)
  {
    fprintf (stderr, "mpi2prv: Error: Unable to allocate memory for 'EnabledTasks'\n");
    fflush (stderr);
    exit (-1);
  }
  for (i = 0; i < numberofapplications; i++)
  {
    EnabledTasks[i] = (int *) malloc (sizeof (int) * numberoftasks);
    if (EnabledTasks[i] == NULL)
    {
      fprintf (stderr, "mpi2prv: Error: Unable to allocate memory for 'EnabledTasks[%d]'\n", i);
      fflush (stderr);
      exit (-1);
    }
    for (j = 0; j < numberoftasks; j++)
      EnabledTasks[i][j] = TRUE;
  }

  EnabledTasks_time =
    (unsigned long long **) malloc (sizeof (unsigned long long *) *
                                    numberofapplications);
  if (EnabledTasks_time == NULL)
  {
    fprintf (stderr, "mpi2prv: Error Unable to allocate memory for 'EnabledTasks_time'\n");
    fflush (stderr);
    exit (-1);
  }
  for (i = 0; i < numberofapplications; i++)
  {
    EnabledTasks_time[i] =
      (unsigned long long *) malloc (sizeof (unsigned long long) *
                                     numberoftasks);
    if (EnabledTasks_time[i] == NULL)
    {
      fprintf (stderr, "mpi2prv: Error: Unable to allocate memory for 'EnabledTasks_time[%d]'\n", i);
      fflush (stderr);
      exit (-1);
    }
    for (j = 0; j < numberoftasks; j++)
      EnabledTasks_time[i][j] = 0LL;
  }
}

static void HandleStackedType (unsigned ptask, unsigned task, unsigned thread,
	unsigned EvType, event_t *current_event)
{
	if (Get_EvEvent(current_event) == USER_EV &&
	    Vector_Search (RegisteredStackValues, EvType))
	{
		unsigned pos;
		unsigned u;
		int found;
		task_t *task_info = GET_TASK_INFO(ptask, task);
		thread_t *thread_info = GET_THREAD_INFO(ptask, task, thread);
		active_task_thread_t *att = &(task_info->active_task_threads[thread_info->active_task_thread-1]);

		/* Look for existing stacked_types for the current task/thread */
		for (found = FALSE, u = 0; u < att->num_stacks && !found; u++)
			if (att->stacked_type[u].type == EvType)
			{
				pos = u;
				found = TRUE;
			}
		if (!found)
		{
			/* If wasn't find, this is the first event for such type in
			   this task/thread, create it */
			pos = att->num_stacks;

			att->stacked_type = (active_task_thread_stack_type_t*) realloc
			  (att->stacked_type, sizeof(active_task_thread_stack_type_t)*(pos+1));
			if (att->stacked_type == NULL)
			{
				fprintf (stderr, "mpi2prv: Fatal error! Cannot reallocate stacked_type for the task/thread\n");
				exit (0);
			}
			att->stacked_type[pos].stack = Stack_Init();
			att->stacked_type[pos].type = EvType;
			att->num_stacks++;
		}

		if (Get_EvMiscParam(current_event) > 0)
			Stack_Push (att->stacked_type[pos].stack, Get_EvMiscParam(current_event));
		else if (Get_EvMiscParam (current_event) == 0)
			Stack_Pop (att->stacked_type[pos].stack);
	}
}


/******************************************************************************
 ***  Paraver_ProcessTraceFiles
 ******************************************************************************/

int Paraver_ProcessTraceFiles (unsigned long nfiles,
	struct input_t *files, unsigned int num_appl,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int taskid)
{
	struct timeval time_begin, time_end;
	FileSet_t * fset;
	unsigned int cpu, ptask, task, thread, error;
	event_t * current_event;
	char envName[PATH_MAX], *tmp;
	unsigned int Type, EvType;
	unsigned long long current_time = 0;
	unsigned long long num_of_events, parsed_events, tmp_nevents;
	unsigned long long records_per_task;
	double pct, last_pct;
	UINT64 *StartingTimes, *SynchronizationTimes;
	unsigned i;
	long long options;
	int num_appl_tasks[num_appl];

	records_per_task = 1024*1024/sizeof(paraver_rec_t); /* num of events in 1 Mbytes */
	records_per_task *= get_option_merge_MaxMem();            /* let's use this memory */
#if defined(PARALLEL_MERGE)
	records_per_task /= get_option_merge_TreeFanOut();        /* divide by the tree fan out */
#endif

	InitializeObjectTable (num_appl, files, nfiles);
	for (i = 0; i < num_appl; i++)
		num_appl_tasks[i] = (GET_PTASK_INFO(i+1))->ntasks;

#if defined(PARALLEL_MERGE)
	ParallelMerge_InitCommunicators();
	InitPendingCommunication ();
	InitForeignRecvs (numtasks);
#endif
	Semantics_Initialize (PRV_SEMANTICS);

	fset = Create_FS (nfiles, files, taskid, PRV_SEMANTICS);
	error = (fset == NULL);

	if (taskid == 0)
		Labels_loadLocalSymbols (taskid, nfiles, files);

	/* If no actual filename is given, use the binary name if possible */
	if (!get_merge_GivenTraceName())
	{
		char *tmp = ObjectTable_GetBinaryObjectName (1, 1);

		/* Duplicate the string as basename may modify it */
		char *FirstBinaryName = NULL;
		if (tmp != NULL)
			FirstBinaryName = strdup (tmp);

		if (FirstBinaryName != NULL)
		{
			char prvfile[strlen(FirstBinaryName) + 5];
			sprintf (prvfile, "%s.prv", basename(FirstBinaryName));
			set_merge_OutputTraceName (prvfile);
			set_merge_GivenTraceName (TRUE);
			free (FirstBinaryName);
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
			if (strcmp (&tmp[strlen(tmp)-strlen(".prv")], ".prv") == 0)
			{
				char extra[1+4+1+3+1];
				sprintf (extra, ".%04d.prv", lastid);
				strncpy (&tmp[strlen(tmp)-strlen(".prv")], extra, strlen(extra));
			}
			else if (strcmp (&tmp[strlen(tmp)-strlen(".prv.gz")], ".prv.gz") == 0)
			{
				char extra[1+4+1+3+1+2+1];
				sprintf (extra, ".%04d.prv.gz", lastid);
				strncpy (&tmp[strlen(tmp)-strlen(".prv.gz")], extra, strlen(extra));
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
			fprintf (stderr, "mpi2prv: Error! Some of the processors failed create trace descriptors\n");
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

	/* CheckHWCcontrol (taskid, options); */
	CheckClockType (taskid, options, PRV_SEMANTICS, get_option_merge_ForceFormat());

/**************************************************************************************/

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Searching synchronization points...");
		fflush (stdout);
	}
	Search_Synchronization_Times (taskid, numtasks, fset, &StartingTimes,
	  &SynchronizationTimes);
	if (0 == taskid)
		fprintf (stdout, " done\n");

#if defined(DEBUG)
	for (i = 0; i < nfiles; i++)
	{
		fprintf (stderr, "[DEBUG] SynchronizationTimes[%d] = %llu "
		                 " files[%d].SpawnOffset = %llu\n",
		  i, SynchronizationTimes[i], i, files[i].SpawnOffset);
	}
#endif

	TimeSync_Initialize (num_appl, num_appl_tasks);
	for (i = 0; i < nfiles; i++)
		if (files[i].thread-1 == 0)
			TimeSync_SetInitialTime (files[i].ptask-1,
			  files[i].task-1,
			  StartingTimes[i],
			  SynchronizationTimes[i] - files[i].SpawnOffset,
			  files[i].node);

	if (get_option_merge_SincronitzaTasks_byNode())
	{
		if (0 == taskid)
			fprintf (stdout, "mpi2prv: Enabling Time Synchronization (Node).\n");
		TimeSync_CalculateLatencies (TS_NODE);
	}
	else if (get_option_merge_SincronitzaTasks())
	{
		if (0 == taskid)
			fprintf (stdout, "mpi2prv: Enabling Time Synchronization (Task).\n");
		TimeSync_CalculateLatencies (TS_TASK);
	}
	else 
	{
		if (0 == taskid)
			fprintf(stdout, "mpi2prv: Time Synchronization disabled.\n");
		TimeSync_CalculateLatencies (TS_NOSYNC);
	}
	
/**************************************************************************************/

	CheckCircularBufferWhenTracing (fset, numtasks, taskid);

#if defined(MPI_PHYSICAL_COMM)
	BuscaComunicacionsFisiques (fset);
#endif

	InitializeEnabledTasks (nfiles, num_appl);

	error = FALSE;

	Initialize_States (fset);
	AddressCollector_Initialize (&CollectedAddresses);

	RegisteredStackValues = Vector_Init();
	Extrae_Vector_Init (&RegisteredCodeLocationTypes);

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Parsing intermediate files\n");
		if (1 == numtasks)
			fprintf (stdout, "mpi2prv: Progress 1 of 2 ... ");
		fflush (stdout);
	}

	Rewind_FS (fset);

	parsed_events = 0;
	last_pct = 0.0;
	num_of_events = EventsInFS (fset);

	current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);

	if (taskid == 0)
	{
		gettimeofday (&time_begin, NULL);
	}

	do
	{
		tmp_nevents = 0;

		EvType = Get_EvEvent (current_event);

		if (getEventType (EvType, &Type))
		{
			current_time = TIMESYNC(ptask-1, task-1, Get_EvTime (current_event));

#if defined(DEBUG)
			fprintf (stderr, "mpi2prv: Parsing event %u:%u:%u <%u,%llu(%llu)> @%llu|%llu\n", ptask, task, thread, Get_EvEvent (current_event), Get_EvValue (current_event), Get_EvMiscParam(current_event), current_time, Get_EvTime(current_event));
#endif

			if (Type == PTHREAD_TYPE || Type == OPENMP_TYPE || Type == MISC_TYPE ||
			    Type == MPI_TYPE || Type == CUDA_TYPE || Type == OPENCL_TYPE ||
			    Type == OPENSHMEM_TYPE || Type == JAVA_TYPE)
			{
				task_t *task_info = GET_TASK_INFO(ptask, task);
				Ev_Handler_t *handler = Semantics_getEventHandler (EvType);
				if (handler != NULL)
				{
					handler (current_event, current_time, cpu, ptask, task, thread, fset);
					tmp_nevents = 1;

					if (MISC_TYPE == Type)
						Enable_MISC_Operation (EvType);
					else if (PTHREAD_TYPE == Type)
						Enable_pthread_Operation (EvType);
					else if (OPENMP_TYPE == Type)
						Enable_OMP_Operation (EvType);
					else if (MPI_TYPE == Type)
						Enable_MPI_Operation (EvType);
					else if (CUDA_TYPE == Type)
						Enable_CUDA_Operation (EvType);
					else if (OPENCL_TYPE == Type)
						Enable_OpenCL_Operation (EvType);
					else if (OPENSHMEM_TYPE == Type)
						Enable_OPENSHMEM_Operation (EvType);
					else if (JAVA_TYPE == Type)
						Enable_Java_Operation (EvType);
				}
				else	
				{
					fprintf (stderr, "mpi2prv: Error! unregistered event type %d in %s\n", EvType, __func__);
				}

				/* Deal with Nanos Task View if we have registered stacked values
				   and if we have seen a resume/suspend thread */
				if (!get_option_merge_NanosTaskView() &&
				    Vector_Count(RegisteredStackValues) > 0 &&
				    task_info->num_active_task_threads > 0)
				{
					int found;
					unsigned u;
					Extrae_Addr2Type_t *addr2types = NULL;

					HandleStackedType (ptask, task, thread, Get_EvValue(current_event), current_event);

					for (found = FALSE, u = 0; u < Extrae_Vector_Count (&RegisteredCodeLocationTypes) && !found; u++)
					{
						addr2types = Extrae_Vector_Get (&RegisteredCodeLocationTypes, u);
						found = addr2types->LineType == Get_EvValue(current_event);
					}
					if (found)
						HandleStackedType (ptask, task, thread, addr2types->FunctionType, current_event);
				}

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				if (Get_EvHWCRead (current_event))
				{
					thread_t *Sthread = GET_THREAD_INFO(ptask, task, thread);
 
					if (Sthread->last_hw_group_change != current_time)
					{
						int i;
						int hwctype[MAX_HWC];
						unsigned long long hwcvalue[MAX_HWC];

						if (HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue, FALSE))
							for (i = 0; i < MAX_HWC; i++)
								if (NO_COUNTER != hwctype[i])
									trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], hwcvalue[i]);
						if (get_option_merge_AbsoluteCounters())
						{
							if (HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue, TRUE))
								for (i = 0; i < MAX_HWC; i++)
									if (NO_COUNTER != hwctype[i])
										trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], hwcvalue[i]);
						}
					}
				}
#endif
			}
			else if (Type == MPI_COMM_ALIAS_TYPE)
			{
				error = GenerateAliesComunicator (current_event, current_time, cpu, ptask, task, thread, fset, &tmp_nevents, PRV_SEMANTICS);
				Enable_MPI_Operation (EvType);
			}
		}
		else
		{
			tmp_nevents = 1;
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

	if (1 == numtasks)
	{
		fprintf (stdout, "done\n");
		fflush (stdout);
	}

	fprintf (stdout, "mpi2prv: Processor %d %s to translate its assigned files\n", taskid, error?"failed":"succeeded");
	fflush (stdout);

#if defined(PARALLEL_MERGE)
	if (numtasks > 1)
	{
		int res;
		unsigned int temp;
		UINT64 temp2;

		res = MPI_Allreduce (&error, &temp, 1, MPI_UNSIGNED, MPI_LOR, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Allreduce, "Failed to share translation completion!");
		error = temp;

		if (!error)
		{
			res = MPI_Allreduce (&current_time, &temp2, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Allreduce, "Failed to share end time!");
			current_time = temp2;
		}
	}
#endif

	if (taskid == 0)
	{
		time_t delta;
		gettimeofday (&time_end, NULL);
		delta = time_end.tv_sec - time_begin.tv_sec;
		fprintf (stdout, "mpi2prv: Elapsed time translating files: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));
	}

	if (error)
	{
		if (0 == taskid)
		{
			fprintf (stderr, "mpi2prv: Error! Some of the processors failed to translate its files\n");
			fflush (stderr);
		}
		return -1;
	}

	/* Finalize states */
	Finalize_States (fset, current_time);

	/* Dump the Write File Buffer structure. As Finalize_States creates a new
	   paraver_rec_t for a new state, remove it (so TRUE in 2nd param) */
	Flush_FS (fset, FALSE);

	if (get_option_merge_SortAddresses())
	{
		gettimeofday (&time_begin, NULL);

#if defined(PARALLEL_MERGE)
		AddressCollector_GatherAddresses (numtasks, taskid, &CollectedAddresses);
#endif

		/* Address translation and address sorting is only done by the master */
		if (taskid == 0)
		{
			UINT64 *buffer_addresses = AddressCollector_GetAllAddresses (&CollectedAddresses);
			int *buffer_types = AddressCollector_GetAllTypes (&CollectedAddresses);
			unsigned *buffer_ptasks = AddressCollector_GetAllPtasks (&CollectedAddresses);
			unsigned *buffer_tasks = AddressCollector_GetAllTasks (&CollectedAddresses);

			for (i = 0; i < AddressCollector_Count(&CollectedAddresses); i++)
				Address2Info_Translate (buffer_ptasks[i], buffer_tasks[i],
				  buffer_addresses[i], buffer_types[i], get_option_merge_UniqueCallerID());

			Address2Info_Sort (get_option_merge_UniqueCallerID());
		}

		if (taskid == 0)
		{
			time_t delta;
			gettimeofday (&time_end, NULL);
			delta = time_end.tv_sec - time_begin.tv_sec;
#if !defined(PARALLEL_MERGE)
			fprintf (stdout, "mpi2prv: Elapsed time sorting addresses: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));
#else
			fprintf (stdout, "mpi2prv: Elapsed time broadcasting and sorting addresses: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));
#endif
		}
	}

#if defined(PARALLEL_MERGE)
	if (taskid == 0)
		gettimeofday (&time_begin, NULL);

	ParallelMerge_BuildCommunicators (numtasks, taskid);

	/* In the parallel merge we have to */
	if (numtasks > 1)
	{
		NewDistributePendingComms (numtasks, taskid, get_option_merge_UseDiskForComms());
		ShareTraceInformation (numtasks, taskid);
	}

	if (taskid == 0)
	{
		time_t delta;
		gettimeofday (&time_end, NULL);
		delta = time_end.tv_sec - time_begin.tv_sec;
		fprintf (stdout, "mpi2prv: Elapsed time sharing communications: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));
	}
#endif

	error = Paraver_JoinFiles (num_appl, get_merge_OutputTraceName(),
	  fset, current_time, NodeCPUinfo, numtasks,
	  taskid, records_per_task, get_option_merge_TreeFanOut());

	strcpy (envName, get_merge_OutputTraceName());
#ifdef HAVE_ZLIB
	if (strlen (envName) >= 7 && strncmp (&(envName[strlen (envName) - 7]), ".prv.gz", 7) == 0)
	{
		tmp = &(envName[strlen (envName) - 7]);
	}
	else
	{
		tmp = &(envName[strlen (envName) - 4]);
	}
#else
	tmp = &(envName[strlen (envName) - 4]);
#endif

	if (0 == taskid)
	{
		strcpy (tmp, ".pcf");
		if (Labels_GeneratePCFfile (envName, options) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create PCF file!\n");

		strcpy (tmp, ".row");
		if (GenerateROWfile (envName, NodeCPUinfo, nfiles, files) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create ROW file!\n");
	}

	if (0 == taskid)
	{
		if (error == 0)
		{
			fprintf (stdout, "mpi2prv: Congratulations! %s has been generated.\n",
			    get_merge_OutputTraceName());
		} else
		{
			fprintf (stdout, "mpi2prv: WARNING! Merge process finished with errors. Trace file may be incomplete.\n");
		}
		fflush (stdout);
	}

	return 0;
}
