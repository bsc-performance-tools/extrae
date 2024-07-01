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
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif

#if defined(PARALLEL_MERGE)
# include <mpi.h>
#endif

#include "events.h"
#include "labels.h"
#include "mpi_prv_events.h"
#include "omp_prv_events.h"
#include "java_prv_events.h"
#include "cuda_prv_events.h"
#include "opencl_prv_events.h"
#include "pthread_prv_events.h"
#include "misc_prv_events.h"
#include "misc_prv_semantics.h"
#include "openshmem_prv_events.h"
#include "openacc_prv_events.h"
#include "gaspi_prv_events.h"
#include "trace_mode.h"
#include "addr2info.h" 
#include "options.h"
#include "object_tree.h"
#include "utils.h"
#include "xalloc.h"
#include "online_events.h"
#include "HardwareCounters.h"
#include "queue.h"

static codelocation_label_t *labels_codelocation = NULL;
static unsigned num_labels_codelocation = 0;

typedef struct event_type_t 
{
     evttype_t event_type;
     Extrae_Vector_t event_values;
}
event_type_t;

static Extrae_Vector_t defined_user_event_types;

static Extrae_Vector_t defined_basic_block_labels;

static void Labels_Add_CodeLocation_Label (int eventcode, codelocation_type_t type, char *description)
{
	unsigned u;

	/* Check first if this label is already included */
	for (u = 0; u < num_labels_codelocation; u++)
	{
		/* If already exists, produce a warning if labels are different */
		if (labels_codelocation[u].eventcode == eventcode &&
		    labels_codelocation[u].type == type)
		{
			if (strcmp (labels_codelocation[u].description, description))
			{
				fprintf (stderr, PACKAGE_NAME": mpi2prv Warning! Already existing definition for event %d with a different description\n", eventcode);
			}

			return;
		}
	}

	labels_codelocation = (codelocation_label_t*) xrealloc (labels_codelocation,
	  (num_labels_codelocation+1)*sizeof(codelocation_label_t));

	labels_codelocation[num_labels_codelocation].eventcode = eventcode;
	labels_codelocation[num_labels_codelocation].type = type;
	labels_codelocation[num_labels_codelocation].description = strdup (description);
	if (labels_codelocation[num_labels_codelocation].description == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": mpi2prv Error! Cannot allocate memory to duplicate a code location label\n");
		exit (-1);
	}

	num_labels_codelocation++;
}

struct color_t states_inf[STATES_NUMBER] = {
  {STATE_0, STATE0_LBL, STATE0_COLOR},
  {STATE_1, STATE1_LBL, STATE1_COLOR},
  {STATE_2, STATE2_LBL, STATE2_COLOR},
  {STATE_3, STATE3_LBL, STATE3_COLOR},
  {STATE_4, STATE4_LBL, STATE4_COLOR},
  {STATE_5, STATE5_LBL, STATE5_COLOR},
  {STATE_6, STATE6_LBL, STATE6_COLOR},
  {STATE_7, STATE7_LBL, STATE7_COLOR},
  {STATE_8, STATE8_LBL, STATE8_COLOR},
  {STATE_9, STATE9_LBL, STATE9_COLOR},
  {STATE_10, STATE10_LBL, STATE10_COLOR},
  {STATE_11, STATE11_LBL, STATE11_COLOR},
  {STATE_12, STATE12_LBL, STATE12_COLOR},
  {STATE_13, STATE13_LBL, STATE13_COLOR},
  {STATE_14, STATE14_LBL, STATE14_COLOR},
  {STATE_15, STATE15_LBL, STATE15_COLOR},
  {STATE_16, STATE16_LBL, STATE16_COLOR},
  {STATE_17, STATE17_LBL, STATE17_COLOR},
  {STATE_18, STATE18_LBL, STATE18_COLOR},
  {STATE_19, STATE19_LBL, STATE19_COLOR},
  {STATE_20, STATE20_LBL, STATE20_COLOR},
  {STATE_21, STATE21_LBL, STATE21_COLOR},
  {STATE_22, STATE22_LBL, STATE22_COLOR},
  {STATE_23, STATE23_LBL, STATE23_COLOR},
  {STATE_24, STATE_24_LBL, STATE_24_COLOR},
  {STATE_25, STATE_25_LBL, STATE_25_COLOR},
  {STATE_26, STATE_26_LBL, STATE_26_COLOR},
  {STATE_27, STATE_27_LBL, STATE_27_COLOR},
  {STATE_28, STATE_28_LBL, STATE_28_COLOR},
  {STATE_29, STATE_29_LBL, STATE_29_COLOR},
  {STATE_30, STATE_30_LBL, STATE_30_COLOR},
  {STATE_31, STATE_31_LBL, STATE_31_COLOR},
  {STATE_32, STATE_32_LBL, STATE_32_COLOR}
};

struct color_t gradient_inf[GRADIENT_NUMBER] = {
  {GRADIENT_0, GRADIENT0_LBL, GRADIENT0_COLOR},
  {GRADIENT_1, GRADIENT1_LBL, GRADIENT1_COLOR},
  {GRADIENT_2, GRADIENT2_LBL, GRADIENT2_COLOR},
  {GRADIENT_3, GRADIENT3_LBL, GRADIENT3_COLOR},
  {GRADIENT_4, GRADIENT4_LBL, GRADIENT4_COLOR},
  {GRADIENT_5, GRADIENT5_LBL, GRADIENT5_COLOR},
  {GRADIENT_6, GRADIENT6_LBL, GRADIENT6_COLOR},
  {GRADIENT_7, GRADIENT7_LBL, GRADIENT7_COLOR},
  {GRADIENT_8, GRADIENT8_LBL, GRADIENT8_COLOR},
  {GRADIENT_9, GRADIENT9_LBL, GRADIENT9_COLOR},
  {GRADIENT_10, GRADIENT10_LBL, GRADIENT10_COLOR},
  {GRADIENT_11, GRADIENT11_LBL, GRADIENT11_COLOR},
  {GRADIENT_12, GRADIENT12_LBL, GRADIENT12_COLOR},
  {GRADIENT_13, GRADIENT13_LBL, GRADIENT13_COLOR},
  {GRADIENT_14, GRADIENT14_LBL, GRADIENT14_COLOR}
};

struct rusage_evt_t rusage_evt_labels[RUSAGE_EVENTS_COUNT] = {
   { RUSAGE_UTIME_EV, RUSAGE_UTIME_LBL },
   { RUSAGE_STIME_EV, RUSAGE_STIME_LBL },
   { RUSAGE_MAXRSS_EV,   RUSAGE_MAXRSS_LBL },
   { RUSAGE_IXRSS_EV,    RUSAGE_IXRSS_LBL },
   { RUSAGE_IDRSS_EV,    RUSAGE_IDRSS_LBL },
   { RUSAGE_ISRSS_EV,    RUSAGE_ISRSS_LBL },
   { RUSAGE_MINFLT_EV,   RUSAGE_MINFLT_LBL },
   { RUSAGE_MAJFLT_EV,   RUSAGE_MAJFLT_LBL },
   { RUSAGE_NSWAP_EV,    RUSAGE_NSWAP_LBL },
   { RUSAGE_INBLOCK_EV,  RUSAGE_INBLOCK_LBL },
   { RUSAGE_OUBLOCK_EV,  RUSAGE_OUBLOCK_LBL },
   { RUSAGE_MSGSND_EV,   RUSAGE_MSGSND_LBL },
   { RUSAGE_MSGRCV_EV,   RUSAGE_MSGRCV_LBL },
   { RUSAGE_NSIGNALS_EV, RUSAGE_NSIGNALS_LBL },
   { RUSAGE_NVCSW_EV,    RUSAGE_NVCSW_LBL },
   { RUSAGE_NIVCSW_EV,   RUSAGE_NIVCSW_LBL }
};

struct memusage_evt_t memusage_evt_labels[MEMUSAGE_EVENTS_COUNT] = {
   { MEMUSAGE_ARENA_EV, MEMUSAGE_ARENA_LBL },
   { MEMUSAGE_HBLKHD_EV, MEMUSAGE_HBLKHD_LBL },
   { MEMUSAGE_UORDBLKS_EV, MEMUSAGE_UORDBLKS_LBL },
   { MEMUSAGE_FORDBLKS_EV, MEMUSAGE_FORDBLKS_LBL },
   { MEMUSAGE_INUSE_EV, MEMUSAGE_INUSE_LBL }
};

struct syscall_evt_t syscall_evt_labels[SYSCALL_EVENTS_COUNT] = {
   { SYSCALL_SCHED_YIELD_EV, SYSCALL_SCHED_YIELD_LBL }
};

/******************************************************************************
 ***  state_labels
 ******************************************************************************/
static void Paraver_state_labels (FILE * fd)
{
  int i;

  fprintf (fd, "%s\n", STATES_LBL);
  for (i = 0; i < STATES_NUMBER; i++)
  {
    fprintf (fd, "%d    %s\n", states_inf[i].value, states_inf[i].label);
  }

  LET_SPACES (fd);
}


/******************************************************************************
 ***  state_colors
 ******************************************************************************/
static void Paraver_state_colors (FILE * fd)
{
  int i;

  fprintf (fd, "%s\n", STATES_COLOR_LBL);
  for (i = 0; i < STATES_NUMBER; i++)
  {
    fprintf (fd, "%d    {%d,%d,%d}\n", states_inf[i].value,
             states_inf[i].rgb[0], states_inf[i].rgb[1],
             states_inf[i].rgb[2]);
  }

  LET_SPACES (fd);
}

/******************************************************************************
 ***  gradient_colors
 ******************************************************************************/
static void Paraver_gradient_colors (FILE * fd)
{
  int i;

  fprintf (fd, "%s\n", GRADIENT_COLOR_LBL);
  for (i = 0; i < GRADIENT_NUMBER; i++)
  {
    fprintf (fd, "%d    {%d,%d,%d}\n", gradient_inf[i].value,
             gradient_inf[i].rgb[0],
             gradient_inf[i].rgb[1], gradient_inf[i].rgb[2]);
  }

  LET_SPACES (fd);
}

/******************************************************************************
 ***  gradient_names
 ******************************************************************************/
static void Paraver_gradient_names (FILE * fd)
{
  int i;

  fprintf (fd, "%s\n", GRADIENT_LBL);
  for (i = 0; i < GRADIENT_NUMBER; i++)
    fprintf (fd, "%d    %s\n", gradient_inf[i].value, gradient_inf[i].label);

  LET_SPACES (fd);
}

/******************************************************************************
 *** concat_user_labels
 ******************************************************************************/
static void Concat_User_Labels (FILE * fd)
{
	char *str;
	char line[1024];
	FILE *labels;

	if ((str = getenv ("EXTRAE_LABELS")) != NULL)
	{
		labels = fopen (str, "r");
		if (labels == NULL)
		{
			fprintf (stderr, "mpi2prv: Cannot open file pointed by EXTRAE_LABELS (%s)\n",str);
			return;
		}

		fprintf (fd, "\n");
		while (fscanf (labels, "%[^\n]\n", line) != EOF)
		{
			if (strlen (line) == 0)
			{
				line[0] = fgetc (labels);
				fprintf (fd, "%s\n", line);
				continue;
			}
			fprintf (fd, "%s\n", line);
		}
		fclose (labels);
		fprintf (fd, "\n");
	}
}

/******************************************************************************
 *** PARAVER_default_options
 ******************************************************************************/
static void Paraver_default_options (FILE * fd)
{
	fprintf (fd, "DEFAULT_OPTIONS\n\n");
	fprintf (fd, "LEVEL               %s\n", DEFAULT_LEVEL);
	fprintf (fd, "UNITS               %s\n", DEFAULT_UNITS);
	fprintf (fd, "LOOK_BACK           %d\n", DEFAULT_LOOK_BACK);
	fprintf (fd, "SPEED               %d\n", DEFAULT_SPEED);
	fprintf (fd, "FLAG_ICONS          %s\n", DEFAULT_FLAG_ICONS);
	fprintf (fd, "NUM_OF_STATE_COLORS %d\n", DEFAULT_NUM_OF_STATE_COLORS);
	fprintf (fd, "YMAX_SCALE          %d\n", DEFAULT_YMAX_SCALE);

	LET_SPACES (fd);

	fprintf (fd, "DEFAULT_SEMANTIC\n\n");
	fprintf (fd, "THREAD_FUNC          %s\n", DEFAULT_THREAD_FUNC);

	LET_SPACES (fd);
}

#if USE_HARDWARE_COUNTERS

/******************************************************************************
 *** HWC_PARAVER_Labels
******************************************************************************/

static void HWC_PARAVER_Labels (FILE *pcfFD)
{
	int i = 0;
	hwc_info_t **used_counters_info;
	int num_used_counters = HardwareCounters_GetUsed(&used_counters_info);

	if (num_used_counters > 0)
	{
		fprintf (pcfFD, "%s\n", TYPE_LABEL);

		for (i = 0; i < num_used_counters; i ++)
		{
			fprintf (pcfFD, "%d  %d %s (%s)\n", 7, 
				used_counters_info[i]->global_id, 
				used_counters_info[i]->name, 
				used_counters_info[i]->description);

			if (get_option_merge_AbsoluteCounters())
			{
				fprintf (pcfFD, "%d  %d Absolute %s (%s)\n", 7, 
					used_counters_info[i]->global_id + HWC_DELTA_ABSOLUTE,
					used_counters_info[i]->name,
					used_counters_info[i]->description);
			}
		}

		fprintf (pcfFD, "%d  %d %s\n", 7, HWC_GROUP_ID, "Active hardware counter set");

		LET_SPACES (pcfFD);

		xfree(used_counters_info);
	}
}

#endif /* USE_HARDWARE_COUNTERS */


static char * Rusage_Event_Label (int rusage_evt) {
   int i;

   for (i=0; i<RUSAGE_EVENTS_COUNT; i++) {
      if (rusage_evt_labels[i].evt_type == rusage_evt) {
         return rusage_evt_labels[i].label;
      }
   }
   return "Unknown getrusage event";
}

static void Write_rusage_Labels (FILE * pcf_fd)
{
   int i;

   if (Rusage_Events_Found) {
      fprintf (pcf_fd, "%s\n", TYPE_LABEL);

      for (i=0; i<RUSAGE_EVENTS_COUNT; i++) {
         if (GetRusage_Labels_Used[i]) {
            fprintf(pcf_fd, "0    %d    %s\n", RUSAGE_BASE+i, Rusage_Event_Label(i));
         }
      }
      LET_SPACES (pcf_fd);
   }
}

static char * Memusage_Event_Label (int memusage_evt) {
	int i;
	
	for (i=0; i<MEMUSAGE_EVENTS_COUNT; i++) {
		if (memusage_evt_labels[i].evt_type == memusage_evt) {
			return memusage_evt_labels[i].label;
		}
	}
	return "Unknown memusage event";
}

static void Write_memusage_Labels (FILE * pcf_fd)
{
   int i;

   if (Memusage_Events_Found) {
      fprintf (pcf_fd, "%s\n", TYPE_LABEL);

      for (i=0; i<MEMUSAGE_EVENTS_COUNT; i++) {
         if (Memusage_Labels_Used[i]) {
            fprintf(pcf_fd, "0    %d    %s\n", MEMUSAGE_BASE+i, Memusage_Event_Label(i));
         }
      }
      LET_SPACES (pcf_fd);
   }
}

static void Write_Trace_Mode_Labels (FILE * pcf_fd)
{
	fprintf (pcf_fd, "%s\n", TYPE_LABEL);
	fprintf (pcf_fd, "9    %d    %s\n", TRACING_MODE_EV, "Tracing mode:");
	fprintf (pcf_fd, "%s\n", VALUES_LABEL);
	fprintf (pcf_fd, "%d      %s\n", TRACE_MODE_DETAIL, "Detailed");
	fprintf (pcf_fd, "%d      %s\n", TRACE_MODE_BURST, "CPU Bursts");
	LET_SPACES (pcf_fd);
}

static void Write_Clustering_Labels (FILE * pcf_fd)
{
	if (MaxClusterId > 0)
	{
		unsigned i;

		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "9    %d    %s\n", CLUSTER_ID_EV, CLUSTER_ID_LABEL);
		fprintf (pcf_fd, "%s\n", VALUES_LABEL);
		fprintf (pcf_fd, "0   End\n");
		fprintf (pcf_fd, "1   Missing Data\n");
		fprintf (pcf_fd, "2   Duration Filtered\n");
		fprintf (pcf_fd, "3   Range Filtered\n");
		fprintf (pcf_fd, "4   Threshold Filtered\n");
		fprintf (pcf_fd, "5   Noise\n");
		for (i=6; i<=MaxClusterId; i++)
			fprintf (pcf_fd, "%d   Cluster %d\n", i, i-5);
		LET_SPACES (pcf_fd);
	}
}

static void Write_Spectral_Labels (FILE * pcf_fd)
{
        if (HaveSpectralEvents)
        {
		unsigned i;

                fprintf (pcf_fd, "%s\n", TYPE_LABEL);
                fprintf (pcf_fd, "9    %d    %s\n", PERIODICITY_EV, PERIODICITY_LABEL);
                fprintf (pcf_fd, "%s\n", VALUES_LABEL);
                fprintf (pcf_fd, "0   Non-periodic zone\n");
		for (i=1; i<=MaxRepresentativePeriod; i++)
			fprintf (pcf_fd, "%d   Period #%d\n", i, i);
                LET_SPACES (pcf_fd);

                fprintf (pcf_fd, "%s\n", TYPE_LABEL);
                fprintf (pcf_fd, "9    %d    %s\n", DETAIL_LEVEL_EV, DETAIL_LEVEL_LABEL);
                fprintf (pcf_fd, "%s\n", VALUES_LABEL);
                fprintf (pcf_fd, "0   Not tracing\n");
                fprintf (pcf_fd, "1   Profiling\n");
                fprintf (pcf_fd, "2   Burst mode\n");
                fprintf (pcf_fd, "3   Detail mode\n");
                LET_SPACES (pcf_fd);

                fprintf (pcf_fd, "%s\n", TYPE_LABEL);
                fprintf (pcf_fd, "9    %d    %s\n", RAW_PERIODICITY_EV, RAW_PERIODICITY_LABEL);
                fprintf (pcf_fd, "%s\n", VALUES_LABEL);
                fprintf (pcf_fd, "0   Non-periodic zone\n");
                for (i=1; i<=MaxRepresentativePeriod; i++)
                        fprintf (pcf_fd, "%d   Raw period #%d\n", i, i);
                LET_SPACES (pcf_fd);

                fprintf (pcf_fd, "%s\n", TYPE_LABEL);
                fprintf (pcf_fd, "9    %d    %s\n", RAW_BEST_ITERS_EV, RAW_BEST_ITERS_LABEL);
                fprintf (pcf_fd, "%s\n", VALUES_LABEL);
                for (i=1; i<=MaxRepresentativePeriod; i++)
                        fprintf (pcf_fd, "%d   Selected iterations from period #%d\n", i, i);
                LET_SPACES (pcf_fd);
        }
}

/**
 * Translation of the local identifiers of each task's open
 * files into a global identifier that is unique for all tasks 
 * in the application
 */
typedef struct
{
  unsigned ptask;
  unsigned task;
  int      local_file_id;
  int      global_file_id;
} open_file_t;

open_file_t *OpenFilesPerTask = NULL; /* List of all open files    */
int NumberOfOpenFiles = 0;            /* Counter of all open files */

char **GlobalFiles = NULL;            /* Vector of unique file names, indexed by their global identifier (see open_file_t) */
int NumberOfGlobalFiles = 0;          /* Counter of all file names (no repetitions) */

/**
 * Unify_File_Id 
 * 
 * Transforms a task's local file id into a global id 
 *
 * \param ptask   The application identifier
 * \param task    The task identifier
 * \param file_id The local file identifier in specified task 
 * \return A global file identifier that is unique for each file pathname 
 */
int Unify_File_Id(unsigned ptask, unsigned task, int file_id)
{
  int i = 0;

  /* Search in the list of open files */
  for (i=0; i<NumberOfOpenFiles; i++)
  {
    /* Check for a matching application/task/id */
    if ((OpenFilesPerTask[i].ptask == ptask) && (OpenFilesPerTask[i].task == task) && (OpenFilesPerTask[i].local_file_id == file_id))
    {
      /* If found, return the previously assigned global id */
      return OpenFilesPerTask[i].global_file_id;
    }
  }
  /* Not found? This should not happen */
  return 0;
}

/**
 * Assign_File_Global_Id
 * 
 * Search in the list of unique filenames if we've seen this file before and return its global identifer; 
 * otherwise, store this file and assign a new id
 * 
 * \param file_name The name of a file opened by this application
 * \return A unique identifier for the given file
 */
int Assign_File_Global_Id(char *file_name)
{
  int i = 0;

  /* Search in the list of unique filenames for the given file */
  for (i=0; i<NumberOfGlobalFiles; i++)
  {
    if (strcmp(GlobalFiles[i], file_name) == 0)
    {
      /* If found, return the index of the vector as the global id (from 1 to N) */
      return i+1;
    }
  }
  
  /* If not found, store this file in the array and return the new index (from 1 to N) */
  GlobalFiles = (char **)xrealloc(GlobalFiles, sizeof(char *) * (NumberOfGlobalFiles + 1));
  GlobalFiles[NumberOfGlobalFiles] = strdup(file_name);
 
  NumberOfGlobalFiles ++;

  return NumberOfGlobalFiles;
}



/******************************************************************************
 *** Labels_loadSYMfile
 ******************************************************************************/
void Labels_loadSYMfile (int taskid, int allobjects, unsigned ptask,
	unsigned task, char *name, int report, UINT64 *io_TaskStartTime, UINT64 *io_TaskSyncTime)
{
	static int Labels_loadSYMfile_init = FALSE;
	FILE *FD;
	char LINE[1024], Type;
	unsigned function_count = 0, hwc_count = 0, other_count = 0;
	int sync_points_seen = 0;
 	unsigned long long first_sync_point = 0;
	unsigned long long last_sync_point = 0;

	if (!Labels_loadSYMfile_init)
	{
		Extrae_Vector_Init (&defined_user_event_types);
		Extrae_Vector_Init (&defined_basic_block_labels);
		Labels_loadSYMfile_init = TRUE;
	}
	event_type_t * last_event_type_used = NULL;

	if (!name)
		return;

	if (strlen(name) == 0)
		return;

	if (!__Extrae_Utils_file_exists(name))
		return;

	FD = (FILE *) fopen (name, "r");
	if (FD == NULL)
	{
		fprintf (stderr, "mpi2prv: WARNING: Task %d Can\'t open symbols file %s\n", taskid, name);
		return;
	}

	while (!feof (FD))
	{
		int args_assigned;

		if (fgets (LINE, 1024, FD) == NULL) break;

		args_assigned = sscanf (LINE, "%c %[^\n]", &Type, LINE);

		if (args_assigned == 2)
		{
			switch (Type)
			{
				case 'B':
					{
						unsigned long start, end, offset;
						char module[1024];
						int res = sscanf (LINE, "0 \"%lx-%lx %lx %[^\n\"]\"", &start, &end, &offset, module);
						if (res == 4)
						{
							ObjectTable_AddBinaryObject (allobjects, ptask, task, start, end, offset, module);
						}
						else fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);
					}
					break;

				case 'O':
				case 'U':
				case 'P':
					{
#ifdef HAVE_BFD
						/* Example of line: U 0x100016d4 fA mpi_test.c 0 */
						char fname[1024], modname[1024];
						int line;
						int type;
						int res;
						UINT64 address;

						res = sscanf (LINE, "%lx \"%[^\"]\" \"%[^\"]\" %d", &address, fname, modname, &line);
						if (res != 4) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);

						if (!get_option_merge_UniqueCallerID())
						{
							if (Type == 'O') type = OTHER_FUNCTION_TYPE;
							else if (Type == 'U') type = USER_FUNCTION_TYPE;
							else if (Type == 'P') type = OUTLINED_OPENMP_TYPE;
						}
						else type = UNIQUE_TYPE;

						Address2Info_AddSymbol (address, type, fname, modname, line);
						function_count++;
#endif /* HAVE_BFD */
					}
					break;
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				case 'H':
					{
						int res, eventcode;
						char hwc_description[1024];

						res = sscanf (LINE, "%d \"%[^\"]\"", &eventcode, hwc_description);
						if (res != 2) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);

						HardwareCounters_AssignGlobalID (ptask, eventcode, hwc_description);
						hwc_count++;
					}
					break;
#endif
				case 'c':
				case 'C':
					{
						int res, eventcode;
						char code_description[1024];

						res = sscanf (LINE, "%d \"%[^\"]\"", &eventcode, code_description);
						if (res != 2) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);

						Labels_Add_CodeLocation_Label (eventcode,
							(Type == 'C') ? CODELOCATION_FUNCTION : CODELOCATION_FILELINE,
							code_description);
						other_count++;
					}
					break;

				case 'd':
				{
					int res, eventvalue;
					char value_description[1024];
					value_t * evt_value = NULL;
					unsigned i, max = Extrae_Vector_Count (&last_event_type_used->event_values);

					res = sscanf (LINE, "%d \"%[^\"]\"", &eventvalue, value_description);
					if (res != 2) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);
                        
					for (i = 0; i < max; i++)
					{
						value_t * evt = Extrae_Vector_Get (&last_event_type_used->event_values, i);
						if(evt->value == eventvalue)
						{
							if(strcmp(evt->label, value_description))
							{
								fprintf(stderr, PACKAGE_NAME"(%s,%d): Warning! Ignoring duplicate definition \"%s\" for value type %d,%d!\n",__FILE__, __LINE__, value_description,last_event_type_used->event_type.type, eventvalue);
							}
							evt_value = evt;
							break;
						}
					}
					if (!evt_value)
					{
						evt_value = (value_t*) xmalloc (sizeof (value_t));
						evt_value->value = eventvalue;

						strncpy(evt_value->label, value_description, VALUE_LBL);
                        if (strlen(value_description) > VALUE_LBL - 1)
                        {
                            fprintf(stderr, PACKAGE_NAME": Warning! Value description exceeds the maximum length of %d, the label will be truncated.\n", VALUE_LBL);
                            evt_value->label[VALUE_LBL - 1] = '\0';
                        }

						Extrae_Vector_Append (&last_event_type_used->event_values, evt_value);
						other_count++;
					}
				}
                    break;

				case 'D':
				{
					int res, eventcode;
					char code_description[1024];
					unsigned i, max = Extrae_Vector_Count (&defined_user_event_types);
					event_type_t * evt_type = NULL;

					res = sscanf (LINE, "%d \"%[^\"]\"", &eventcode, code_description);
					if (res != 2) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);

					for (i = 0; i < max; i++)
					{
						event_type_t * evt = Extrae_Vector_Get (&defined_user_event_types, i);
						if (evt->event_type.type == eventcode)
						{
							if(strcmp(evt->event_type.label, code_description))
							{
								fprintf(stderr, PACKAGE_NAME"(%s,%d): Warning! Ignoring duplicate definition \"%s\" for type %d!\n", __FILE__, __LINE__, code_description, eventcode);
							}
							evt_type = evt;
							break;
						}
					}

					if (!evt_type)
					{
						evt_type = (event_type_t*) xmalloc (sizeof (event_type_t));
						evt_type->event_type.type = eventcode;

						strncpy(evt_type->event_type.label, code_description, EVENT_LBL);
                        if (strlen(code_description) > EVENT_LBL - 1)
                        {
                            fprintf(stderr, PACKAGE_NAME": Warning! Event description exceeds the maximum length of %d, the label will be truncated.\n", TYPE_LBL);
                            evt_type->event_type.label[EVENT_LBL - 1] = '\0';
                        }

						Extrae_Vector_Init(&evt_type->event_values);
   
						Extrae_Vector_Append(&defined_user_event_types, evt_type);
						other_count++;
					}
					last_event_type_used = evt_type;
				}
				break;

				case 'b': // BasicBlocks symbol
				{
					int res, eventvalue;
					char bb_description[1024];
					unsigned i, max = Extrae_Vector_Count (&defined_basic_block_labels);
					event_type_t * evt_type = NULL;
					value_t * evt_value = NULL;

					res = sscanf (LINE, "%d \"%[^\"]\"", &eventvalue, bb_description);
					if (res != 2) fprintf (stderr, PACKAGE_NAME": Error! Invalid line ('%s') in %s\n", LINE, name);
					if (max==0)
					{
						evt_type = (event_type_t*) xmalloc (sizeof (event_type_t));
						evt_type->event_type.type = USRFUNC_EV_BB;
						strcpy(evt_type->event_type.label, "BASIC_BLOCKS");
						Extrae_Vector_Init(&evt_type->event_values);
						Extrae_Vector_Append(&defined_basic_block_labels, evt_type);
					} 
					else 
					{
						evt_type = Extrae_Vector_Get (&defined_basic_block_labels, 0); // There is only one event type in the vector
					}
					max = Extrae_Vector_Count (&evt_type->event_values);

					for (i = 0; i < max; i++)
					{
						value_t * evt = Extrae_Vector_Get (&evt_type->event_values, i);
						if (evt->value == eventvalue)
						{
							if (strcmp(evt->label, bb_description))
							{
								fprintf(stderr, "Extrae (%s,%d): Warning! Ignoring duplicate definition \"%s\" for value type %d,%d!\n",__FILE__, __LINE__, bb_description,evt_type->event_type.type, eventvalue);
							}
							evt_value = evt;
							break;
						}
					}

					if (!evt_value)
					{
						evt_value = (value_t*) xmalloc (sizeof (value_t));
						evt_value->value = eventvalue;
						strcpy(evt_value->label, bb_description);
						Extrae_Vector_Append (&evt_type->event_values, evt_value);
						other_count++;
					}
				}
				break;

				/* The 'F' entries in the *.SYM represent open files */
				case 'F':
				{
					int open_counter = 0;
					char pathname[4096];
					int res = sscanf (LINE, "%d \"%[^\n\"]\"\"", &open_counter, pathname);
					if (res == 2)
					{
						/* Store this entry in the list of open files per task */
						OpenFilesPerTask = (open_file_t *)realloc(OpenFilesPerTask, sizeof(open_file_t) * (NumberOfOpenFiles + 1));

						OpenFilesPerTask[NumberOfOpenFiles].ptask          = ptask;
						OpenFilesPerTask[NumberOfOpenFiles].task           = task;
						OpenFilesPerTask[NumberOfOpenFiles].local_file_id  = open_counter; // The local file identifier 
						OpenFilesPerTask[NumberOfOpenFiles].global_file_id = Assign_File_Global_Id(pathname); // Assign the global identifier for this file

						NumberOfOpenFiles ++;
					}
				}
				break;

				/* 'S' represents synchronization points.
				 * Currently, these are 3 synchronization points:
				 * APPL_EV (1st event of the trace)
				 * TRACE_INIT_EV (emitted at the beginning) and 
				 * MPI_INIT_EV (or START_PES_EV, somewhat later)
				 * We need to retrieve the 1st one (starting times), and 
				 * the last one, so if an app is MPI we use the time of the MPI_Init
				 * as synchronization point. Otherwise, the time of the sequential initialization is used
				 */ 
				case 'S':
				{
					unsigned long long sync_point = atoll(LINE);
					if (!sync_points_seen) first_sync_point = sync_point;
					last_sync_point = sync_point;
					sync_points_seen ++;
					break;
				}

				default:
				{
					fprintf (stderr, PACKAGE_NAME" mpi2prv: Error! Task %d found unexpected line in symbol file '%s'\n", taskid, LINE);
					break;
				}
			}
		}
	}

	if (taskid == 0 && report)
	{
		fprintf (stdout, "mpi2prv: A total of %u symbols were imported from %s file\n", function_count+hwc_count+other_count, name);
		fprintf (stdout, "mpi2prv: %u function symbols imported\n", function_count);
		fprintf (stdout, "mpi2prv: %u HWC counter descriptions imported\n", hwc_count);
	}

	if (io_TaskStartTime != NULL) *io_TaskStartTime = first_sync_point;
	if (io_TaskSyncTime != NULL) *io_TaskSyncTime = last_sync_point;

	fclose (FD);
}

void Write_OpenFiles_Labels(FILE * pcf_fd)
{
  int i = 0;

  if (NumberOfGlobalFiles > 0)
  {
    fprintf (pcf_fd, "%s\n", TYPE_LABEL);
    fprintf (pcf_fd, "0    %d    %s\n", FILE_NAME_EV, FILE_NAME_LBL);
    fprintf (pcf_fd, "%s\n", VALUES_LABEL);
    fprintf (pcf_fd, "%d      %s\n", 0, "Unknown");

    for (i = 0; i < NumberOfGlobalFiles; i ++)
    {
      fprintf (pcf_fd, "%d      %s\n", i+1, GlobalFiles[i]);
    }
    LET_SPACES (pcf_fd);
  }
}

static void Write_syscall_Labels (FILE * pcf_fd)
{
   int i;

	 if (Syscall_Events_Found) {
		 fprintf (pcf_fd, "%s\n", TYPE_LABEL);
	   fprintf (pcf_fd, "9    %d    %s\n", SYSCALL_EV, "System call");
	   fprintf (pcf_fd, "%s\n", VALUES_LABEL);

     fprintf(pcf_fd, "%d     %s\n", 0, "End");
		 for (i=0; i<SYSCALL_EVENTS_COUNT; i++) {
			 if (Syscall_Labels_Used[i])
			 {
				 fprintf(pcf_fd, "%d     %s\n", i+1, syscall_evt_labels[i].label);
			 }
		 }
     LET_SPACES (pcf_fd);
	 }
}

void Write_UserDefined_Labels(FILE * pcf_fd)
{
    unsigned i, j, max_types = Extrae_Vector_Count (&defined_user_event_types);
    for (i = 0; i < max_types; i++)
    {
        event_type_t * evt = Extrae_Vector_Get (&defined_user_event_types, i);
        unsigned max_values = Extrae_Vector_Count (&evt->event_values);
        fprintf (pcf_fd, "%s\n", TYPE_LABEL);
        fprintf (pcf_fd, "0    %d    %s\n", evt->event_type.type, evt->event_type.label);
        if (max_values>0)
        {
            fprintf (pcf_fd, "%s\n", VALUES_LABEL);
            for (j = 0; j < max_values; j++)
            {
                value_t * values = Extrae_Vector_Get (&evt->event_values, j);
                fprintf (pcf_fd, "%d      %s\n", values->value, values->label);
            }
        }
        LET_SPACES (pcf_fd);
    }
}

void Write_BasickBlock_Labels(FILE * pcf_fd)
{
     unsigned i, j, max_types = Extrae_Vector_Count (&defined_basic_block_labels);
    for (i = 0; i < max_types; i++)
    {
        event_type_t * evt = Extrae_Vector_Get (&defined_basic_block_labels, i);
        unsigned max_values = Extrae_Vector_Count (&evt->event_values);
        fprintf (pcf_fd, "%s\n", TYPE_LABEL);
        fprintf (pcf_fd, "0    %d    %s\n", evt->event_type.type, evt->event_type.label);
        if (max_values>0)
        {
            fprintf (pcf_fd, "%s\n", VALUES_LABEL);
            for (j = 0; j < max_values; j++)
            {
                value_t * values = Extrae_Vector_Get (&evt->event_values, j);
                fprintf (pcf_fd, "%d      %s\n", values->value, values->label);
            }
        }
        LET_SPACES (pcf_fd);
    }
   
}

/******************************************************************************
 *** generatePCFfile
 ******************************************************************************/

int Labels_GeneratePCFfile (char *name, long long options)
{
	FILE *fd;

	fd = fopen (name, "w");
	if (fd == NULL)
		return -1;

	Paraver_default_options (fd);

	Paraver_state_labels (fd);
	Paraver_state_colors (fd);

	MPITEvent_WriteEnabled_MPI_Operations (fd);
	SoftCountersEvent_WriteEnabled_MPI_Operations (fd);
	OMPEvent_WriteEnabledOperations (fd);
	WriteEnabled_pthread_Operations (fd);
	MISCEvent_WriteEnabledOperations (fd, options);
	CUDAEvent_WriteEnabledOperations (fd);
	JavaEvent_WriteEnabledOperations (fd);

#if USE_HARDWARE_COUNTERS
	HWC_PARAVER_Labels (fd);
#endif

	Paraver_gradient_colors (fd);
	Paraver_gradient_names (fd);

#ifdef HAVE_BFD
	Address2Info_Write_LibraryIDs (fd);
	Address2Info_Write_MPI_Labels (fd, get_option_merge_UniqueCallerID());
	Address2Info_Write_UF_Labels (fd, get_option_merge_UniqueCallerID());
	Address2Info_Write_Sample_Labels (fd, get_option_merge_UniqueCallerID());
	Address2Info_Write_CUDA_Labels (fd, get_option_merge_UniqueCallerID());
	Address2Info_Write_OTHERS_Labels (fd, get_option_merge_UniqueCallerID(),
		num_labels_codelocation, labels_codelocation);
# if defined(BFD_MANAGER_GENERATE_ADDRESSES)
	if (get_option_dump_Addresses())
		ObjectTable_dumpAddresses (fd, ADDRESSES_FOR_BINARY_EV);
# endif
#endif

	Write_rusage_Labels (fd);
	Write_memusage_Labels (fd);
	Write_Trace_Mode_Labels (fd);
	Write_Clustering_Labels (fd);
	Write_Spectral_Labels (fd);
	WriteEnabled_OpenCL_Operations (fd);
	WriteEnabled_OPENSHMEM_Operations (fd);
	WriteEnabled_OPENACC_Operations(fd);
	WriteEnabled_GASPI_Operations(fd);

	Write_UserDefined_Labels(fd);

	Write_BasickBlock_Labels(fd);

	Write_OpenFiles_Labels(fd);

	Write_syscall_Labels(fd);
    
	Concat_User_Labels (fd);

	fclose(fd);
    
	return 0;
}

void Labels_loadLocalSymbols (int taskid, unsigned long nfiles,
	struct input_t * IFiles, UINT64 **io_StartingTimes, UINT64 **io_SynchronizationTimes)
{
	unsigned long file;

	UINT64 *StartingTimes = NULL;
	UINT64 *SynchronizationTimes = NULL;

	/* Allocate space for the synchronization times of each task */
	StartingTimes = xmalloc_and_zero (nfiles * sizeof(UINT64));
	SynchronizationTimes = xmalloc_and_zero (nfiles * sizeof(UINT64));

	if (taskid == 0)
	{
		for (file = 0; file < nfiles; file++)
		{
			char symbol_file_name[PATH_MAX];
			UINT64 TaskStartTime = 0, TaskSyncTime = 0;

			strcpy (symbol_file_name, IFiles[file].name);
			symbol_file_name[strlen(symbol_file_name)-strlen(EXT_MPIT)] = (char) 0; /* remove ".mpit" extension */
			strcat (symbol_file_name, EXT_SYM); /* add ".sym" */

			if (__Extrae_Utils_file_exists(symbol_file_name))
			{
				Labels_loadSYMfile (taskid, FALSE, IFiles[file].ptask, IFiles[file].task, symbol_file_name, FALSE, &TaskStartTime, &TaskSyncTime);
	
				StartingTimes[file] = TaskStartTime;
				SynchronizationTimes[file] = TaskSyncTime;
			}
		}
	}

#if defined(PARALLEL_MERGE)
	MPI_Bcast( StartingTimes, nfiles, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD );
	MPI_Bcast( SynchronizationTimes, nfiles, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD );
#endif

	*io_StartingTimes = StartingTimes;
	*io_SynchronizationTimes = SynchronizationTimes;
}

#if defined(PARALLEL_MERGE)

/**
 * Share_File_Names
 * 
 * Broadcast the list of open files from merger's task 0 (this is the only that parses the *.SYM files)
 * to all other merger tasks so that all tasks know how to translate the local file ids into the global
 * ids. 
 *
 * \param taskid The merger's task rank
 */
void Share_File_Names(int taskid)
{
  int i = 0;
  unsigned *ptask_array     = NULL;
  unsigned *task_array      = NULL;
  int *local_file_id_array  = NULL;
  int *global_file_id_array = NULL;

  /* Send the number of open files */
  MPI_Bcast ( &NumberOfOpenFiles, 1, MPI_INT, 0, MPI_COMM_WORLD );

  /* Allocate arrays to serialize the translation table (well, it's a list not a table, see open_file_t) */
  ptask_array          = (unsigned *)xmalloc(sizeof(unsigned) * NumberOfOpenFiles);
  task_array           = (unsigned *)xmalloc(sizeof(unsigned) * NumberOfOpenFiles);
  local_file_id_array  = (int *)xmalloc(sizeof(int) * NumberOfOpenFiles);
  global_file_id_array = (int *)xmalloc(sizeof(int) * NumberOfOpenFiles);

  if (taskid == 0)
  {
    /* Merger's master task serializes the table (this is the only task that has the information loaded from the *.SYM) */
    for (i=0; i<NumberOfOpenFiles; i++)
    {
      ptask_array[i]          = OpenFilesPerTask[i].ptask;
      task_array[i]           = OpenFilesPerTask[i].task;
      local_file_id_array[i]  = OpenFilesPerTask[i].local_file_id;
      global_file_id_array[i] = OpenFilesPerTask[i].global_file_id;
    }
  }

  /* Broadcast the serialized arrays to all merger's tasks */
  MPI_Bcast ( ptask_array, NumberOfOpenFiles, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
  MPI_Bcast ( task_array, NumberOfOpenFiles, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
  MPI_Bcast ( local_file_id_array, NumberOfOpenFiles, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast ( global_file_id_array, NumberOfOpenFiles, MPI_INT, 0, MPI_COMM_WORLD );

  if (taskid > 0)
  {
    /* All the other tasks reconstruct their local translation table */
    OpenFilesPerTask = (open_file_t *)xmalloc(sizeof(open_file_t) * NumberOfOpenFiles);

    for (i=0; i<NumberOfOpenFiles; i++)
    { 
      OpenFilesPerTask[i].ptask          = ptask_array[i];
      OpenFilesPerTask[i].task           = task_array[i];
      OpenFilesPerTask[i].local_file_id  = local_file_id_array[i];
      OpenFilesPerTask[i].global_file_id = global_file_id_array[i];
    }
  }

  /* Free resources */
  xfree(ptask_array);
  xfree(task_array);
  xfree(local_file_id_array);
  xfree(global_file_id_array);
}

#endif
