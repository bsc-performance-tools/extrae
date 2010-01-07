/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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

#include "paraver_state.h"
#include "paraver_generator.h"
#include "dimemas_generator.h"

#include "mpi_prv_events.h"
#include "pacx_prv_events.h"
#include "pthread_prv_events.h"
#include "omp_prv_events.h"
#include "misc_prv_events.h"
#include "trt_prv_events.h"
#include "addr2info.h"
#include "timesync.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

struct ptask_t *obj_table;
unsigned int num_ptasks;

int **EnabledTasks = NULL;
unsigned long long **EnabledTasks_time = NULL;

#if defined(IS_BG_MACHINE)
struct QuadCoord
{
  int X, Y, Z, T;
};
static struct QuadCoord *coords = NULL;

void AnotaBGPersonality (unsigned int event, unsigned long long valor, int task)
{
  switch (event)
  {
    case BG_PERSONALITY_TORUS_X:
      coords[task - 1].X = valor;
      break;
    case BG_PERSONALITY_TORUS_Y:
      coords[task - 1].Y = valor;
      break;
    case BG_PERSONALITY_TORUS_Z:
      coords[task - 1].Z = valor;
      break;
    case BG_PERSONALITY_PROCESSOR_ID:
      coords[task - 1].T = valor;
      break;
    default:
      break;
  }
}
#endif

/******************************************************************************
 ***  InitializeObjectTable
 ******************************************************************************/
static void InitializeObjectTable (int num_appl, struct input_t * files,
	unsigned long nfiles)
{
	unsigned int ptask, task, thread, i, j;
	unsigned int maxtasks = 0, maxthreads = 0;

	/* This is not the perfect way to allocate everything, but it's
	  good enough for runs where all the ptasks (usually 1), have the
	  same number of threads */
	num_ptasks = num_appl;
	for (i = 0; i < nfiles; i++)
	{
		maxtasks = MAX(files[i].task, maxtasks);
		maxthreads = MAX(files[i].thread, maxthreads);
	}

	obj_table = (struct ptask_t*) malloc (sizeof(struct ptask_t)*num_ptasks);
	if (NULL == obj_table)
	{
		fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d ptasks!\n", num_ptasks);
		fflush (stderr);
		exit (-1);
	}
	for (i = 0; i < num_ptasks; i++)
	{
		obj_table[i].tasks = (struct task_t*) malloc (sizeof(struct task_t)*maxtasks);
		if (NULL == obj_table[i].tasks)
		{
			fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d tasks (ptask = %d)\n", maxtasks, i+1);
			fflush (stderr);
			exit (-1);
		}
		for (j = 0; j < maxtasks; j++)
		{
			obj_table[i].tasks[j].threads = (struct thread_t*) malloc (sizeof(struct thread_t)*maxthreads);
			if (NULL == obj_table[i].tasks[j].threads)
			{
				fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d threads (ptask = %d / task = %d)\n", maxthreads, i+1, j+1);
				fflush (stderr);
				exit (-1);
			}
		}
	}

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	INIT_QUEUE (&CountersTraced);
#endif

	for (ptask = 0; ptask < num_ptasks; ptask++)
	{
		obj_table[ptask].ntasks = 0;
		for (task = 0; task < maxtasks; task++)
		{
			obj_table[ptask].tasks[task].tracing_disabled = FALSE;
			obj_table[ptask].tasks[task].nthreads = 0;
			for (thread = 0; thread < maxthreads; thread++)
			{
				obj_table[ptask].tasks[task].threads[thread].nStates = 0;
				obj_table[ptask].tasks[task].threads[thread].First_Event = TRUE;
				obj_table[ptask].tasks[task].threads[thread].First_HWCChange = TRUE;
				obj_table[ptask].tasks[task].threads[thread].MatchingComms = TRUE;

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
			    obj_table[ptask].tasks[task].threads[thread].HWCSets = NULL;
			    obj_table[ptask].tasks[task].threads[thread].num_HWCSets = 0;
			    obj_table[ptask].tasks[task].threads[thread].current_HWCSet = 0;
#endif
			}
		}
	}

	for (i = 0; i < nfiles; i++)
	{
		ptask = files[i].ptask;
		task = files[i].task;
		thread = files[i].thread;

		obj_table[ptask-1].tasks[task-1].nodeid = files[i].nodeid;
		obj_table[ptask-1].tasks[task-1].threads[thread-1].cpu = files[i].cpu;
		obj_table[ptask-1].tasks[task-1].threads[thread-1].dimemas_size = 0;
		obj_table[ptask-1].ntasks = MAX (obj_table[ptask-1].ntasks, task);
		obj_table[ptask-1].tasks[task-1].nthreads = MAX (obj_table[ptask-1].tasks[task-1].nthreads, thread);
	}
}


void InitializeEnabledTasks (int numberoftasks, int numberofapplications)
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

/******************************************************************************
 ***  Paraver_ProcessTraceFiles
 ******************************************************************************/

int Paraver_ProcessTraceFiles (char *outName, unsigned long nfiles,
	struct input_t *files, unsigned int num_appl, char *sym_file,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int taskid,
	int MBytesPerAllSegments, int forceformat)
{
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
	int i, total_tasks;
	long long options;

	records_per_task = 1024*1024/sizeof(paraver_rec_t);  /* num of events in 1 Mbytes */
	records_per_task *= (MBytesPerAllSegments - 8);      /* let's use this memory less 8 Mbytes */
	records_per_task /= numtasks;                        /* divide by all the tasks */

	InitializeObjectTable (num_appl, files, nfiles);
#if defined(PARALLEL_MERGE)
	InitCommunicators();
	InitPendingCommunication ();
	InitForeignRecvs (numtasks);
#endif
	Semantics_Initialize (PRV_SEMANTICS);

	fset = Create_FS (nfiles, files, taskid, PRV_SEMANTICS);
	error = (fset == NULL);

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

	if (dump)
		make_dump (fset);

#if defined(HETEROGENEOUS_SUPPORT)
	/*	Must we accomodate all the intermediate files into this machine endianes? */
	EndianCorrection (fset, numtasks, taskid);
#endif

	/*
	 * Initialize the communicators management
	 */
	initialize_comunicadors (num_ptasks);

	options = GetTraceOptions (fset, numtasks, taskid);

	CheckHWCcontrol (taskid, options);
	CheckClockType (taskid, options, PRV_SEMANTICS, forceformat);

/**************************************************************************************/

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Searching synchronization points...");
		fflush (stdout);
	}
	total_tasks = Search_Synchronization_Times(fset, &StartingTimes, &SynchronizationTimes);
	if (0 == taskid)
		fprintf (stdout, " done\n");

	TimeSync_Initialize (total_tasks);
	for (i=0; i<total_tasks; i++)
	{
		TimeSync_SetInitialTime (i, StartingTimes[i], SynchronizationTimes[i], files[i].node);
	}

	if (SincronitzaTasks_byNode)
	{
		if (0 == taskid)
			fprintf (stdout, "mpi2prv: Enabling Time Synchronization (Node).\n");
		TimeSync_CalculateLatencies (TS_NODE);
	}
	else if (SincronitzaTasks)
	{
		if (0 == taskid)
			fprintf (stdout, "mpi2prv: Enabling Time Synchronization (Task).\n");
		TimeSync_CalculateLatencies (TS_TASK);
	}
	else 
	{
		if (0 == taskid) fprintf(stderr, "mpi2prv: Time Synchronization disabled.\n");
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

	do
	{
		tmp_nevents = 0;

		EvType = Get_EvEvent (current_event);

		if (getEventType (EvType, &Type))
		{
			current_time = TIMESYNC(task-1, Get_EvTime (current_event));

			if (Type == PTHREAD_TYPE || Type == OPENMP_TYPE || Type == MISC_TYPE ||
			    Type == MPI_TYPE || Type == PACX_TYPE)
			{
				Ev_Handler_t *handler = Semantics_getEventHandler (EvType);
				if (handler != NULL)
				{
					handler (current_event, current_time, cpu, ptask, task, thread, fset);
					tmp_nevents = 1;
					
					if (PTHREAD_TYPE == Type)
						Enable_pthread_Operation (EvType);
					else if (OPENMP_TYPE == Type)
						Enable_OMP_Operation (EvType);
					else if (MPI_TYPE == Type)
						Enable_MPI_Operation (EvType);
					else if (PACX_TYPE == Type)
						Enable_PACX_Operation (EvType);
					else if (TRT_TYPE == Type)
						Enable_TRT_Operation (EvType);
				}
				else
					fprintf (stderr, "mpi2prv: Error! unregistered event type %d in %s\n", EvType, __func__);

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				if (Get_EvHWCRead (current_event))
				{
					int i;
					unsigned int hwctype[MAX_HWC];
					unsigned long long hwcvalue[MAX_HWC];

					if (Get_EvHWCSet(current_event) != HardwareCounters_GetCurrentSet(ptask, task, thread))
					{
						/* The HWC_CHANGE_EV was missing (probably due to a circular buffer) */
						HWC_Change_Ev (Get_EvHWCSet(current_event), current_time, cpu, ptask, task, thread);
					}

#warning "Aixo es horrible, caldra retocar-ho"
					HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue);
					for (i = 0; i < MAX_HWC; i++)
						if (NO_COUNTER != hwctype[i])
							trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], hwcvalue[i]);
				}
#endif
			}
			else if (Type == MPI_COMM_ALIAS_TYPE)
			{
				error = GenerateAliesComunicator (current_event, current_time, cpu, ptask, task, thread, fset, &tmp_nevents, PRV_SEMANTICS);
				Enable_MPI_Operation (EvType);
			}
			else if (Type == PACX_COMM_ALIAS_TYPE)
			{
				error = GenerateAliesComunicator (current_event, current_time, cpu, ptask, task, thread, fset, &tmp_nevents, PRV_SEMANTICS);
				Enable_PACX_Operation (EvType);
			}
		}
		else
		{
			tmp_nevents = 1;
		}

		obj_table[ptask-1].tasks[task-1].threads[thread-1].First_Event = FALSE;
		obj_table[ptask-1].tasks[task-1].threads[thread-1].Previous_Event_Time = current_time;

		/* Right now, progress bar is only shown when numtasks is 1 */
		if (1 == numtasks)
		{
			parsed_events += tmp_nevents;
			pct = ((double) parsed_events)/((double) num_of_events)*100.0f;
			if (pct > last_pct + 5.0 && pct <= 100.0f)
			{
				fprintf (stdout, "%.1lf%% ", pct);
				fflush (stdout);
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


#if defined(PARALLEL_MERGE)
	BuildCommunicators (numtasks, taskid);

	/* In the parallel merge we have to */
	if (numtasks > 1)
	{
		NewDistributePendingComms (numtasks, taskid, option_UseDiskForComms);
		ShareTraceInformation (numtasks, taskid);
	}
#endif

	Paraver_JoinFiles (outName, fset, current_time, nfiles, NodeCPUinfo, numtasks, taskid, records_per_task);

	strcpy (envName, outName);
#ifdef HAVE_ZLIB
	if (strlen (outName) >= 7 && strncmp (&(outName[strlen (outName) - 7]), ".prv.gz", 7) == 0)
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
		if (GeneratePCFfile (envName, options) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create PCF file!\n");

		strcpy (tmp, ".row");
		if (GenerateROWfile (envName, NodeCPUinfo) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create ROW file!\n");
	}

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Congratulations! %s has been generated.\n", outName);
		fflush (stdout);
	}

	return 0;
}
