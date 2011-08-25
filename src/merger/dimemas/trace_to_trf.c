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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
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

struct ptask_t *obj_table;
unsigned int num_ptasks;
UINT64 InitTracingTime;

char dimemas_tmp[PATH_MAX];

#if defined(DEAD_CODE)
int **EnabledTasks = NULL;
unsigned long long **EnabledTasks_time = NULL;
#endif

#if defined(DEAD_CODE)
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

	if (1 != num_ptasks)
	{
		fprintf (stderr, "mpi2trf: Error! This merger does not support merging more than 1 parallel application\n");
		fflush (stderr);
		exit (-1);
	}

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

unsigned long long Dimemas_hr_to_relative (UINT64 iotimer)
{
	return iotimer - InitTracingTime;
}

#if defined(DEAD_CODE)
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
#endif

static void Dimemas_GenerateOffsets (struct ptask_t *table,
	unsigned long long **ptr, unsigned int *count)
{
	unsigned long long *offsets;
	unsigned i = 0;
	unsigned ptask;
	unsigned task;
	unsigned thread;

	*ptr = NULL;
	*count = 0;

	/* Count total number of running threads of the application */
	for (ptask = 0; ptask < num_ptasks; ptask++)
		for (task = 0; task < table[ptask].ntasks; task++)
			i += table[ptask].tasks[task].nthreads;

	/* Allocate memory for the offsets of those threads */
	offsets = (unsigned long long*) malloc (sizeof(unsigned long long)*i);
	if (NULL == offsets)
	{
		fprintf (stderr, "mpi2trf: Unable to allocate memory for %d offsets\n", i);
		fflush (stderr);
		exit (-1);
	}

	/* Put the dimemas_size field on the offsets */
	i = 0;
	for (ptask = 0; ptask < num_ptasks; ptask++)
		for (task = 0; task < table[ptask].ntasks; task++)
			for (thread = 0; thread < table[ptask].tasks[task].nthreads; thread++, i++)
				offsets[i] = table[ptask].tasks[task].threads[thread].dimemas_size;

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

	InitializeObjectTable (num_appl, files, nfiles);
#if defined(PARALLEL_MERGE)
	InitCommunicators();
#endif
	Semantics_Initialize (TRF_SEMANTICS);

	fset = Create_FS (nfiles, files, taskid, TRF_SEMANTICS);
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

	if (get_option_merge_dump())
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

#if defined(DEAD_CODE)
	InitializeEnabledTasks (nfiles, num_appl);
#endif

	error = FALSE;

	/*
	   First step, search and generate communicators.
	*/
	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Parsing intermediate files. Generating communicators.\n");
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
				fprintf (stdout, "%.1lf%% ", pct);
				fflush (stdout);
				last_pct += 5.0;
			}
		}
		current_event = GetNextEvent_FS (fset, &cpu, &ptask, &task, &thread);
	} while ((current_event != NULL) && !error);

#if defined(PARALLEL_MERGE)
	BuildCommunicators (numtasks, taskid);
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
		fprintf (stdout, "mpi2prv: Parsing intermediate files. Generating trace.\n");
		if (numtasks > 1)
			fprintf (stdout, "mpi2prv: Progress ... ");
		else
			fprintf (stdout, "mpi2prv: Progress 2 of 2 ... ");
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
			fprintf (stderr, "mpi2prv ERROR: Creating Dimemas tracefile : %s on processor %d\n", outName, taskid);
			return -1;
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
			fprintf (stderr, "mpi2trf: Unable to create temporal file using mkstemp\n");
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
			fprintf (stderr, "mpi2prv ERROR: Creating Diememas temporal tracefile : %s on processor %d\n", outName, taskid);
			return -1;
		}

		remove (dimemas_tmp);
	}

	if (0 == taskid)
		Dimemas_WriteHeader (fset->output_file, NodeCPUinfo, outName);

	current_file = -1;

	do
	{
		if (current_file != GetActiveFile(fset))
		{
#if !defined(HAVE_FTELL64) && !defined(HAVE_FTELLO64)
			obj_table[ptask-1].tasks[task-1].threads[thread-1].dimemas_size = ftell (fset->output_file);
#elif defined(HAVE_FTELL64)
			obj_table[ptask-1].tasks[task-1].threads[thread-1].dimemas_size = ftell64 (fset->output_file);
#elif defined(HAVE_FTELLO64)
			obj_table[ptask-1].tasks[task-1].threads[thread-1].dimemas_size = ftello64 (fset->output_file);
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
					fprintf (stderr, "mpi2prv: Error! unregistered event type %d in %s+%d\n", EvType, __func__, __LINE__);

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				if (Get_EvHWCRead (current_event))
				{
					int i;
					unsigned int hwctype[MAX_HWC];
					unsigned long long hwcvalue[MAX_HWC];

#warning "Aixo es horrible, caldra retocar-ho"
					HardwareCounters_Emit (ptask, task, thread, current_time, current_event, hwctype, hwcvalue);
					for (i = 0; i < MAX_HWC; i++)
						if (NO_COUNTER != hwctype[i])
							Dimemas_User_Event (fset->output_file, task-1, thread-1, hwctype[i], hwcvalue[i]);
				}
#endif
			}
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

	fprintf (stdout, "mpi2prv: Processor %d %s to translate its assigned files\n", taskid, error?"failed":"succeeded");
	fflush (stdout);

	Dimemas_GenerateOffsets (obj_table, &offsets, &count);

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
		Dimemas_WriteOffsets (fset->output_file, outName, trace_size, count, offsets);

		fclose (fset->output_file);

		strcpy (envName, outName);
		tmp = &(envName[strlen (envName) - 4]);

		strcpy (tmp, ".pcf");
		if (GeneratePCFfile (envName, options) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create PCF file!\n");

		strcpy (tmp, ".row");
		if (GenerateROWfile (envName, NodeCPUinfo) == -1)
			fprintf (stderr, "mpi2prv: WARNING! Unable to create ROW file!\n");

		fprintf (stdout, "mpi2prv: Congratulations! %s has been generated.\n", outName);
		fflush (stdout);
	}
	return 0;
}
