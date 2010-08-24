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
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
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
#ifdef HAVE_CTYPE_H
# include <ctype.h>
#endif

#include "utils.h"
#include "semantics.h"
#include "dump.h"
#include "file_set.h"
#include "object_tree.h"
#include "mpi2out.h"
#include "trace_to_prv.h"
#include "trace_to_trf.h"
#include "labels.h"
#include "addr2info_hashcache.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

#ifdef HAVE_BFD
# include "addr2info.h" 
#endif

#define DEFAULT_PRV_OUTPUT_NAME "MPITRACE_Paraver_Trace.prv"
#define DEFAULT_DIM_OUTPUT_NAME "MPITRACE_Dimemas_Trace.dim"

typedef enum {FileOpen_Default, FileOpen_Absolute, FileOpen_Relative} FileOpen_t;
typedef enum {Block, Cyclic, Size, ConsecutiveSize} WorkDistribution_t;

char OutTrace[PATH_MAX];
char callback_file[PATH_MAX] = "";
char symbol_file[PATH_MAX] = "";
char executable_file[PATH_MAX] = ""; 
struct input_t *InputTraces;
unsigned nTraces = 0;
unsigned num_applications;
int dump = FALSE;
int SincronitzaTasks = FALSE;
int SincronitzaTasks_byNode = FALSE;
static int AutoSincronitzaTasks = TRUE;
static WorkDistribution_t WorkDistribution= Block;
int MBytesPerAllSegments = 512;
int option_UseDiskForComms = FALSE;
int option_SkipSendRecvComms = FALSE;
int option_UniqueCallerID = TRUE;
int option_VerboseLevel = 0;
int option_TreeFanOut = 0;

#if defined(IS_BG_MACHINE)
int option_XYZT = 0;
#endif

/******************************************************************************
 ***  Help
 ******************************************************************************/

void Help (const char *ProgName)
{
  printf ("Usage: %s inputfile1 ... [--] inputfileN [-o <OutputFile>] [otheroptions]\n"
          "       %s -f file.mpits [-o <OutputFile>] [otheroptions]\n"
          "       %s -h\n"
          "Options:\n"
          "    -h        Get this help.\n"
          "    -v        Increase verbosity.\n"
          "    -o file   Output trace file name.\n"
          "    -e file   Uses the executable file to obtain some information.\n"
          "    -f file   MpitFILE File with the names of the \".mpit\" input files.\n"
          "    -syn      Synchronize traces at the end of MPI_Init.\n"
          "    -syn-node Synchronize traces using node information.\n"
          "    -no-syn   Do not synchronize traces at the end of MPI_Init.\n"
          "    -maxmem M Uses up to M megabytes of memory at the last step of merging process.\n"
          "    -dimemas  Force the generation of a Dimemas trace.\n"
          "    -paraver  Force the generation of a Paraver trace.\n"
#if defined(IS_BG_MACHINE)
          "    -xyzt     Generates additional output file with BG/L torus coordinates.\n"
#endif
#if defined(PARALLEL_MERGE)
					"    -tree-fan-out N   Orders the parallel merge to distribute its work in a N-order tree.\n"
          "    -cyclic   Distributes MPIT files cyclically among tasks.\n"
          "    -block    Distributes MPIT files in a block fashion among tasks.\n"
          "    -size     Distributes MPIT trying to build groups of equal size.\n"
          "    -consecutive-size Distributes MPIT files in a block fashion considering file size.\n"
          "    -use-disk-for-comms Uses the disk instead of memory to match foreign communications.\n"
#endif
          "    -s file   Indicates the symbol file attached to the *.mpit files.\n"
          "    -d        Sequentially dumps the contents of every *.mpit file.\n"
          "    -extended-glop-info\n"
          "              Each global operation adds additional information records.\n"
          "    -split-states\n"
          "              Do not merge consecutives states that are the same.\n"
          "    -skip-sendrecv\n"
          "              Do not emit communication for SendReceive operations.\n"
          "    -[no-]unique-caller-id\n"
          "              Choose whether use a unique value identifier for different callers.\n"  
          "    --        Take the next trace files as a diferent parallel task.\n"
          "\n",
          ProgName, ProgName, ProgName);
}

/******************************************************************************
 ***  Process_MPIT_File
 ***  Adds an MPIT file into the required strctures.
 ******************************************************************************/

void Process_MPIT_File (char *file, char *node, int *cptask, int *cfiles)
{
	int fd;
	int name_length;
	int task;
	int thread;
	int i;
	int cur_ptask = *cptask, cur_files = *cfiles;
	char *tmp_name;

	InputTraces[nTraces].InputForWorker = -1;
	InputTraces[nTraces].name = (char *) malloc (strlen (file) + 1);
	if (InputTraces[nTraces].name == NULL)
	{
		fprintf (stderr, "mpi2prv: Error cannot obtain memory for namefile\n");
		fflush (stderr);
		exit (1);
	}
	strcpy (InputTraces[nTraces].name, file);

	if (node != NULL)
	{
		InputTraces[nTraces].node = (char *) malloc (strlen (node) + 1);
		if (InputTraces[nTraces].node == NULL)
		{
			fprintf (stderr, "mpi2prv: Error cannot obtain memory for NODE information!\n");
			fflush (stderr);
			exit (1);
		}
		else
			strcpy (InputTraces[nTraces].node, node);
	}
	else 
		InputTraces[nTraces].node = "(unknown)";

	name_length = strlen (InputTraces[nTraces].name);
	tmp_name = InputTraces[nTraces].name;
	tmp_name = &(tmp_name[name_length - strlen(EXT_MPIT)]);
	if (strcmp (tmp_name, EXT_MPIT))
	{
		fprintf (stderr, "mpi2prv: Error! File %s does not contain a valid extension!. Skipping.\n", InputTraces[nTraces].name);
		return;
	}

	InputTraces[nTraces].filesize = 0;
	fd = open (InputTraces[nTraces].name, O_RDONLY);
	if (-1 != fd)
	{
		InputTraces[nTraces].filesize = lseek (fd, 0, SEEK_END);
		close (fd);
	}

	tmp_name = InputTraces[nTraces].name;
	tmp_name = &(tmp_name[name_length - strlen(EXT_MPIT) - DIGITS_TASK - DIGITS_THREAD]);

	task = 0;
	for (i = 0; i < DIGITS_TASK; i++)
	{
		task = task * 10 + ((int) tmp_name[0] - ((int) '0'));
		tmp_name++;
	}
	InputTraces[nTraces].task = task;

	thread = 0;
	for (i = 0; i < DIGITS_THREAD; i++)
	{
		thread = thread * 10 + (tmp_name[0] - ((int) '0'));
		tmp_name++;
	}
	InputTraces[nTraces].thread = thread;

	InputTraces[nTraces].task++;
	InputTraces[nTraces].thread++;
	InputTraces[nTraces].ptask = cur_ptask;
	InputTraces[nTraces].order = nTraces;

	nTraces++;
	cur_files++;

	*cfiles = cur_files;
}

/******************************************************************************
 **      Function name : strip
 **      Author : HSG
 **      Description : Removes spaces and control characters from a string.
 ******************************************************************************/

static char *strip (char *buffer)
{
	int l = strlen (buffer);
	int min = 0, max = l - 1;

	if (min == max)
		return NULL;

	while (isspace (buffer[min]) || iscntrl (buffer[min]))
		min++;

	while (isspace (buffer[max]) || iscntrl (buffer[min]))
		max--;

	buffer[max + 1] = (char) 0;

	return &(buffer[min]);
}


/******************************************************************************
 ***  Read_MPITS_file
 ***  Insertis into trace tables the contents of a ascii file!
 ******************************************************************************/

void Read_MPITS_file (const char *file, int *cptask, int *cfiles, FileOpen_t opentype)
{
	int info;
	FILE *fd = fopen (file, "r");
	char mybuffer[4096];
	char host[2048];
	char path[2048];

	if (fd == NULL)
	{
		fprintf (stderr, "mpi2prv: Unable to open %s file.\n", file);
		return;
	}

	do
	{
		fgets (mybuffer, sizeof(mybuffer), fd);
		if (!feof(fd))
		{
			char *stripped;

			info = sscanf (mybuffer, "%s on %s", path, host);
			stripped = strip (path);

			/* If mode is not forced, check first if the absolute path exists,
			   if not, try to open in the curren directory */
			if (opentype == FileOpen_Default)
			{
				if (!file_exists(stripped))
					Process_MPIT_File (basename(stripped), (info==2)?host:NULL, cptask, cfiles);
				else
					Process_MPIT_File (stripped, (info==2)?host:NULL, cptask, cfiles);
			}
			else if (opentype == FileOpen_Absolute)
			{
				Process_MPIT_File (stripped, (info==2)?host:NULL, cptask, cfiles);
			}
			else if (opentype == FileOpen_Relative)
			{
				Process_MPIT_File (basename(stripped), (info==2)?host:NULL, cptask, cfiles);
			}
		}
	}
	while (!feof(fd));

	fclose (fd);
}

/******************************************************************************
 ***  ProcessArgs
 ******************************************************************************/

void ProcessArgs (int numtasks, int rank, int argc, char *argv[],
	int *traceformat,  int *forceformat)
{
	char *BinaryName;
	int CurArg;
	unsigned int cur_ptask = 1;   /* Ptask counter. Each -- the ptask number is
	                               * incremented. */
	unsigned int cur_files = 1;   /* File counter within a Ptask. */

	if (argc == 1)                /* No params? */
	{
		Help (argv[0]);
		exit (0);
	}

	BinaryName = strdup (argv[0]);
	if (NULL == BinaryName)
	{
		fprintf (stderr, "merger: Error! Unable to duplicate binary name!\n");
		exit (-1);
	}
	BinaryName = basename (BinaryName);

	if ((strncmp (BinaryName, "mpi2prv", 7) == 0)
	    || (strncmp (BinaryName, "mpimpi2prv", 10) == 0))
	{
		*traceformat = PRV_SEMANTICS;
		*forceformat = FALSE;
		strcpy (OutTrace, DEFAULT_PRV_OUTPUT_NAME);
	}
	else if ((strncmp (BinaryName, "mpi2dim", 7) == 0)
	    || (strncmp (BinaryName, "mpimpi2dim", 10) == 0))
	{
		*traceformat = TRF_SEMANTICS;
		*forceformat = FALSE;
		strcpy (OutTrace, DEFAULT_DIM_OUTPUT_NAME);
	}
	else
	{
		*traceformat = PRV_SEMANTICS;
		*forceformat = FALSE;
		strcpy (OutTrace, DEFAULT_PRV_OUTPUT_NAME);
	}

	for (CurArg = 1; CurArg < argc; CurArg++)
	{
		if (!strcmp (argv[CurArg], "-h"))
		{
			Help (argv[0]);
			exit (0);
		}
		if (!strcmp (argv[CurArg], "-v"))
		{
			option_VerboseLevel++;
			continue;
		}
		if (!strcmp (argv[CurArg], "-o"))
		{
			CurArg++;
			if (CurArg < argc)
				strcpy (OutTrace, argv[CurArg]);
			continue;
		}
		if (!strcmp (argv[CurArg], "-s"))
		{
			CurArg++;
			if (CurArg < argc)
				strcpy (symbol_file, argv[CurArg]);
			continue;
		}
		if (!strcmp (argv[CurArg], "-f"))
		{
			CurArg++;
			if (CurArg < argc)
				Read_MPITS_file (argv[CurArg], &cur_ptask, &cur_files, FileOpen_Default);
			continue;
		}
		if (!strcmp (argv[CurArg], "-f-relative"))
		{
			CurArg++;
			if (CurArg < argc)
				Read_MPITS_file (argv[CurArg], &cur_ptask, &cur_files, FileOpen_Relative);
			continue;
		}
		if (!strcmp (argv[CurArg], "-f-absolute"))
		{
		  CurArg++;
		  if (CurArg < argc)
		    Read_MPITS_file (argv[CurArg], &cur_ptask, &cur_files, FileOpen_Absolute);
		  continue;
		}
#if defined(IS_BG_MACHINE)
		if (!strcmp (argv[CurArg], "-xyzt"))
		{
			option_XYZT = TRUE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-xyzt"))
		{
			option_XYZT = FALSE;
			continue;
		}
#endif
		if (!strcmp (argv[CurArg], "-unique-caller-id"))
		{
			option_UniqueCallerID = TRUE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-unique-caller-id"))
		{
			option_UniqueCallerID = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-split-states"))
		{
			Joint_States = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-use-disk-for-comms"))
		{
			option_UseDiskForComms = TRUE;
			continue;
		}
#if defined(PARALLEL_MERGE)
		if (!strcmp (argv[CurArg], "-cyclic"))
		{
			WorkDistribution = Cyclic;
			continue;
		}
		if (!strcmp (argv[CurArg], "-block"))
		{
			WorkDistribution = Block;
			continue;
		}
		if (!strcmp (argv[CurArg], "-consecutive-size"))
		{
			WorkDistribution = ConsecutiveSize;
			continue;
		}
		if (!strcmp (argv[CurArg], "-size"))
		{
			WorkDistribution = Size;
			continue;
		}
		if (!strcmp (argv[CurArg], "-tree-fan-out"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				if (atoi(argv[CurArg]) > 0)
				{
					option_TreeFanOut = atoi(argv[CurArg]);
				}
				else
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: WARNING: Invalid value for -tree-fan-out parameter\n");
				}
			}
			continue;
		}
#endif
		if (!strcmp (argv[CurArg], "-evtnum"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				if (atoi(argv[CurArg]) > 0)
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: Using %d events for thread\n", atoi(argv[CurArg]));
					setLimitOfEvents (atoi(argv[CurArg]));
				}
				else
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: WARNING: Invalid value for -evtnum parameter\n");
				}
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-c"))
		{
			CurArg++;
			if (CurArg < argc)
				strcpy (callback_file, argv[CurArg]);
			continue;
		}
		if (!strcmp (argv[CurArg], "-d"))
		{
			dump = TRUE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-maxmem"))
		{
			unsigned long long records_per_task;

			CurArg++;
			if (CurArg < argc)
			{
				MBytesPerAllSegments = atoi(argv[CurArg]);
				if (MBytesPerAllSegments == 0)
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: Error! Invalid parameter for -maxmem option. Using 512 Mbytes\n");
					MBytesPerAllSegments = 512;
				}
				else if (MBytesPerAllSegments < 16)
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: Error! Cannot use less than 16 MBytes for the merge step\n");
					MBytesPerAllSegments = 16;
				}
			}

			records_per_task = 1024*1024/sizeof(paraver_rec_t);  /* num of events in 1 Mbytes */
			records_per_task *= MBytesPerAllSegments;            /* let's use this memory */
			records_per_task /= numtasks;                        /* divide by all the tasks */

			if (0 == records_per_task)
			{
				if (0 == rank)
					fprintf (stderr, "mpi2prv: Error! Assigned memory by -maxmem is insufficient for this number of tasks\n");
				exit (-1);
			}

			continue;
		}
		if (!strcmp (argv[CurArg], "-dimemas"))
		{
			*forceformat = TRUE;
			*traceformat = TRF_SEMANTICS;
			continue;
		}
		if (!strcmp (argv[CurArg], "-paraver"))
		{
			*forceformat = TRUE;
			*traceformat = PRV_SEMANTICS;
			continue;
		}
		if (!strcmp (argv[CurArg], "-skip-sendrecv"))
		{
			option_SkipSendRecvComms = TRUE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-syn"))
		{
			SincronitzaTasks = TRUE;
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-syn-node"))
		{
			SincronitzaTasks = TRUE;
			SincronitzaTasks_byNode = TRUE;
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-syn"))
		{
			SincronitzaTasks = FALSE;
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "--"))
		{
			if (cur_files != 1)
			{
				cur_ptask++;
				cur_files = 1;
			}
			continue;
		}
		if (!strcmp(argv[CurArg], "-e"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				strcpy(executable_file, argv[CurArg]);
#if !defined(HAVE_BFD)
				if (0 == rank)
					fprintf (stdout, PACKAGE_NAME": WARNING! This mpi2prv does not support -e flag!\n");
#endif
				continue;
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -e: You must specify the path of the executable file.\n");
				Help(argv[0]);
				exit(0);
			}
		}
		else
			Process_MPIT_File ((char *) (argv[CurArg]), NULL, &cur_ptask, &cur_files);
	}
	num_applications = cur_ptask;

	/* Specific things to be applied per format */
	if (rank == 0)
	{
		if (TRF_SEMANTICS == *traceformat)
		{
			/* Dimemas traces doesn't know about synchronization */
			SincronitzaTasks = FALSE;
			SincronitzaTasks_byNode = FALSE;
			AutoSincronitzaTasks = FALSE;

			fprintf (stdout, "merger: Output trace format is: Dimemas\n");

#if defined(PARALLEL_MERGE)
			if (WorkDistribution != Block)
			{
				fprintf (stdout, "merger: Other work distribution than 'block' are not supporting when generating Dimemas traces\n");
				WorkDistribution = Block;
			}
#endif

		}
		else if (PRV_SEMANTICS == *traceformat)
		{
			fprintf (stdout, "merger: Output trace format is: Paraver\n");
		}
	}
}

static void PrintNodeNames (int numtasks, int processor_id, char **nodenames)
{
	int i;

	if (processor_id == 0)
	{
		fprintf (stdout, "mpi2prv: Assigned nodes <");
		for (i = 0; i < numtasks; i++)
			fprintf (stdout, " %s%c", nodenames[i], (i!=numtasks-1)?',':' ');
		fprintf (stdout, ">\n");
	}
}

static void DistributeWork (unsigned num_processors, unsigned processor_id)
{
	unsigned num_operative_tasks = num_processors;
	unsigned mpits_per_task = (nTraces+num_operative_tasks-1)/num_operative_tasks;
	unsigned remaining_files = nTraces;
	unsigned index;

	if (WorkDistribution == Block)
	{
		/* Files will be distributed in blocks */
		unsigned ub = MIN(mpits_per_task-1, nTraces);
		unsigned lb = 0;
		for (index = 0; index < num_processors; index++)
		{
			unsigned index2;

			for (index2 = lb; index2 <= ub; index2++)
				InputTraces[index2].InputForWorker = index;
  
			/* compute how many files remain to get distributed */
			remaining_files = remaining_files - mpits_per_task;
			if (remaining_files > 0)
			{
				num_operative_tasks--;
				mpits_per_task = (remaining_files+num_operative_tasks-1)/(num_operative_tasks);
				lb = MIN(ub+1,nTraces);
				ub = MIN(ub+mpits_per_task, nTraces);
			}
		}
	}
	else if (WorkDistribution == Cyclic)
	{
		/* Files will be distributed in cycles */
		for (index = 0; index < remaining_files; index++)
			InputTraces[index].InputForWorker = index%num_operative_tasks;
	}
	else if (WorkDistribution == Size || WorkDistribution == ConsecutiveSize)
	{
		off_t remaining_size;
		off_t average_size_per_task;
		off_t assigned_size[num_processors];
		char assigned_files[nTraces];
		unsigned file;

		if (WorkDistribution == Size)
			qsort (InputTraces, nTraces, sizeof(input_t), SortBySize);

		for (index = 0; index < num_processors; index++)
			assigned_size[index] = 0;

		for (remaining_size = 0, index = 0; index < remaining_files; index++)
			remaining_size += InputTraces[index].filesize;
		average_size_per_task = remaining_size / num_processors;

		for (index = 0; index < nTraces; index++)
			assigned_files[index] = FALSE;

		for (index = 0; index < num_processors; index++)
		{
			file = 0;
			while (assigned_size[index] < average_size_per_task)
			{
				if (!assigned_files[file])
					if (assigned_size[index]+InputTraces[file].filesize <= average_size_per_task)
					{
						assigned_files[file] = TRUE;
						assigned_size[index] += InputTraces[file].filesize;
						InputTraces[file].InputForWorker = index;
						remaining_size -= InputTraces[file].filesize;
					}
				if (++file >= nTraces)
					break;
			}
			average_size_per_task = remaining_size / (num_processors-index-1);
		}

		if (WorkDistribution == Size)
			qsort (InputTraces, nTraces, sizeof(input_t), SortByOrder);
	}

	/* Check assigned traces... */
	for (index = 0; index < nTraces; index++)
		if (InputTraces[index].InputForWorker >= num_operative_tasks &&
		    InputTraces[index].InputForWorker < 0)
		{
			fprintf (stderr, "mpi2prv: FATAL ERROR! Bad input assignament into processor namespace.\n");
			fprintf (stderr, "mpi2prv: FATAL ERROR! Input %d assigned to processor %d.\n", index, InputTraces[index].InputForWorker);
			exit (-1);
		}

	/* Show information of sizes */
	if (processor_id == 0)
	{
		fprintf (stdout, "mpi2prv: Assigned size per processor <");
		for (index = 0; index < num_processors; index++)
		{
			unsigned file;
			off_t size_assigned_to_task;

			size_assigned_to_task = 0;
			for (file = 0; file < nTraces; file++)
				if (InputTraces[file].InputForWorker == index)
					size_assigned_to_task += InputTraces[file].filesize;

			if (size_assigned_to_task != 0)
			{
				if (size_assigned_to_task < 1024*1024)
					fprintf (stdout, " <1 Mbyte");
				else
#if SIZEOF_OFF_T == 8 && SIZEOF_LONG == 8
					fprintf (stdout, " %ld Mbytes", size_assigned_to_task/(1024*1024));
#elif SIZEOF_OFF_T == 8 && SIZEOF_LONG == 4
					fprintf (stdout, " %lld Mbytes", size_assigned_to_task/(1024*1024));
#elif SIZEOF_OFF_T == 4
					fprintf (stdout, " %d Mbytes", size_assigned_to_task/(1024*1024));
#endif
			}
			else
				fprintf (stdout, " 0 bytes");
			fprintf (stdout, "%c", (index!=num_processors-1)?',':' ');
		}
		fprintf (stdout, ">\n");

		for (index = 0 ; index < nTraces; index++)
			fprintf (stdout,"mpi2prv: File %s is object %d.%d.%d on node %s assigned to processor %d\n",
				InputTraces[index].name, InputTraces[index].ptask,
				InputTraces[index].task, InputTraces[index].thread,
				InputTraces[index].node==NULL?"unknown":InputTraces[index].node,
				InputTraces[index].InputForWorker);
		fflush (stdout);
	}
}

/******************************************************************************
 ***  main entry point
 ******************************************************************************/

#if defined (LICENSE) && defined (LICENSE_IN_MERGE)
# include "license.c"
#endif

int merger (int numtasks, int idtask, int argc, char *argv[])
{
#if defined(PARALLEL_MERGE)
	char **nodenames;
#else
	char nodename[1024];
	char *nodenames[1];
#endif
	int error;
	int traceformat;
	int forceformat;
	struct Pair_NodeCPU *NodeCPUinfo;

#if defined(PARALLEL_MERGE)
	if (numtasks <= 1)
	{
		fprintf (stderr, "mpi2prv: The parallel version of the mpi2prv is not suited for 1 processor! Dying...\n");
		exit (1);
	}
#endif

	InputTraces = (struct input_t *) malloc (sizeof(struct input_t)*MAX_FILES);
	if (InputTraces == NULL)
	{
	  perror ("malloc");
	  fprintf (stderr, "mpi2prv: Cannot allocate InputTraces memory. Dying...\n");
	  exit (1);
	}

	ProcessArgs (numtasks, idtask, argc, argv, &traceformat, &forceformat);

#if defined(PARALLEL_MERGE)
	if (option_TreeFanOut == 0)
	{
		if (idtask == 0)
			fprintf (stdout, "mpi2prv: Tree order is not set. Setting automatically to %d\n", numtasks);
		option_TreeFanOut = numtasks;
	}
	else if (option_TreeFanOut > numtasks)
	{
		if (idtask == 0)
			fprintf (stdout, "mpi2prv: Tree order is set to %d but is larger that numtasks. Setting to tree order to %d\n", option_TreeFanOut, numtasks);
		option_TreeFanOut = numtasks;
	}
	else if (option_TreeFanOut <= numtasks)
	{
		if (idtask == 0)
			fprintf (stdout, "mpi2prv: Tree order is set to %d\n", option_TreeFanOut);
	}
#endif

	if (0 == nTraces)
	{
	  fprintf (stderr, "mpi2prv: No intermediate trace files given.\n");
		fflush (stderr);
	  return 0;
	}

#if defined(PARALLEL_MERGE)
	ShareNodeNames (numtasks, &nodenames);
#else
	gethostname (nodename, sizeof(nodename));
	nodenames[0] = nodename;
#endif

	PrintNodeNames (numtasks, idtask, nodenames);
	DistributeWork (numtasks, idtask);
	NodeCPUinfo = AssignCPUNode (nTraces, InputTraces);

	if (AutoSincronitzaTasks)
	{
		unsigned i;
		unsigned all_nodes_are_equal = TRUE;
		unsigned first_node = InputTraces[0].nodeid;
		for (i = 1; i < nTraces && all_nodes_are_equal; i++)
			all_nodes_are_equal = (first_node == InputTraces[i].nodeid);
		SincronitzaTasks = !all_nodes_are_equal;

		if (0 == idtask)
		{
			fprintf (stdout, "mpi2prv: Time synchronization has been turned %s\n", SincronitzaTasks?"on":"off");
			fflush (stdout);
		}
	}

#if defined (LICENSE) && defined (LICENSE_IN_MERGE)
	if (0 == idtask)
		if (!verify_execution())
			exit (0);
#endif

#ifdef HAVE_BFD
	Address2Info_Initialize (executable_file);
	loadSYMfile (symbol_file);
#endif

	if (PRV_SEMANTICS == traceformat)
		error = Paraver_ProcessTraceFiles (OutTrace, nTraces, InputTraces,
		  num_applications, NodeCPUinfo, numtasks, idtask,
		  MBytesPerAllSegments, forceformat, option_TreeFanOut);
	else if (TRF_SEMANTICS == traceformat)
		error = Dimemas_ProcessTraceFiles (OutTrace, nTraces, InputTraces,
		  num_applications, NodeCPUinfo, numtasks, idtask,
		  MBytesPerAllSegments, forceformat);
	else
		error = FALSE;

	if (error)
		fprintf (stderr, "mpi2prv: An error has been encountered when generating the tracefile. Dying...\n");

#ifdef HAVE_BFD
	if (option_VerboseLevel > 0)
		Addr2Info_HashCache_ShowStatistics();
#endif

	return 0;
}
