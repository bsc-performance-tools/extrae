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
#ifdef HAVE_UNISTD_H
# include <unistd.h>
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
#include "paraver_state.h"
#include "options.h"
#include "addresses.h"
#include "intercommunicators.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
# include "mpi-aux.h"
# include <mpi.h>
#endif

#if defined(HAVE_BFD)
# include "addr2info.h" 
#endif

typedef enum {Block, Cyclic, Size, ConsecutiveSize} WorkDistribution_t;

static struct input_t *InputTraces = NULL;
unsigned nTraces = 0;
static int AutoSincronitzaTasks = TRUE;
static WorkDistribution_t WorkDistribution= Block;
static char **MPITS_Files = NULL;
static unsigned Num_MPITS_Files = 0;

/******************************************************************************
 ***  Help
 ******************************************************************************/

void Help (const char *ProgName)
{
  printf ("Usage: %s inputfile1 ... [--] inputfileN [-o <OutputFile>] [otheroptions]\n"
		  "       %s -f file.mpits [-o <OutputFile>] [otheroptions]\n"
		  "       %s -h\n"
		  "Options:\n"
		  "    -h                   Get this help.\n"
		  "    -v                   Increase verbosity.\n"
		  "    -absolute-counters   Emit hardware counters in absolute form in addition to relative form.\n"
		  "    -o file              Output trace file name.\n"
		  "    -e file              Uses the executable file to obtain some information.\n"
		  "    -f file              MpitFILE File with the names of the \".mpit\" input files.\n"
		  "    -syn                 Synchronize traces at the MPI task-level using the MPI_Init information.\n"
		  "    -syn-node            Synchronize traces at the MPI node-level using the MPI_Init information.\n"
		  "    -no-syn              Do not synchronize traces at the end of MPI_Init.\n"
		  "    -maxmem M            Uses up to M megabytes of memory at the last step of merging process.\n"
		  "    -dimemas             Force the generation of a Dimemas trace.\n"
		  "    -paraver             Force the generation of a Paraver trace.\n"
		  "    -keep-mpits          Keeps MPIT files after trace generation (default)\n"
		  "    -no-keep-mpits       Removes MPIT files after trace generation.\n"
		  "    -trace-overwrite     Overwrites the tracefile.\n"
		  "    -no-trace-overwrite  Do not overwrite the tracefile, renaming the new one.\n"
#if defined(IS_BG_MACHINE)
		  "    -xyzt                Generates additional output file with BG/L torus coordinates.\n"
#endif
#if defined(PARALLEL_MERGE)
		  "    -tree-fan-out N      Orders the parallel merge to distribute its work in a N-order tree.\n"
		  "    -cyclic              Distributes MPIT files cyclically among tasks.\n"
		  "    -block               Distributes MPIT files in a block fashion among tasks.\n"
		  "    -size                Distributes MPIT trying to build groups of equal size.\n"
		  "    -consecutive-size    Distributes MPIT files in a block fashion considering file size.\n"
		  "    -use-disk-for-comms  Uses the disk instead of memory to match foreign communications.\n"
#endif
		  "    -s file              Indicates the symbol (*.sym) file attached to the *.mpit files.\n"
		  "    -d/-dump             Sequentially dumps the contents of every *.mpit file.\n"
		  "    -dump-without-time   Do not show event time in when dumping events (valuable for test purposes).\n"
		  "    -remove-files        Remove intermediate files after processing them.\n"
		  "    -split-states        Do not merge consecutives states that are the same.\n"
		  "    -skip-sendrecv       Do not emit communication for SendReceive operations.\n"
		  "    -unique-caller-id    Choose whether use a unique value identifier for different callers.\n"
		  "    -translate-addresses Translate code addresses into code references if available.\n"
		  "    -no-translate-addresses Do not translate code addresses into code references if available.\n"
		  "    -emit-library-events Emit library information for unknown references if possible.\n"
		  "    -sort-addresses      Sort file name, line events in information linked with source code.\n"
		  "    -task-view           Swap the thread level in Paraver timeline to show Nanos Tasks.\n"
          "    -without-addresses   Do not emit address information into PCF (valuable for test purposes).\n"
		  "    --                   Take the next trace files as a diferent parallel task.\n"
		  "\n",
          ProgName, ProgName, ProgName);
}

/******************************************************************************
 ***  Process_MPIT_File
 ***  Adds an MPIT file into the required structures.
 ******************************************************************************/

static void Process_MPIT_File (char *file, char *thdname, int *cptask,
	int taskid)
{
	int name_length;
	int task;
	int thread;
	int i;
	int cur_ptask = *cptask;
	char *tmp_name;
	size_t pos;
	int has_node_separator;
	int hostname_len;

	xrealloc(InputTraces, InputTraces, sizeof(struct input_t) * (nTraces + 1));
	if (InputTraces == NULL)
	{
		perror ("realloc");
		fprintf (stderr, "mpi2prv: Cannot allocate InputTraces memory for MPIT %d. Dying...\n", nTraces + 1);
		exit (1);
	}

	InputTraces[nTraces].InputForWorker = -1;
	InputTraces[nTraces].name = (char *) malloc (strlen (file) + 1);
	if (InputTraces[nTraces].name == NULL)
	{
		fprintf (stderr, "mpi2prv: Error cannot obtain memory for namefile\n");
		fflush (stderr);
		exit (1);
	}
	strcpy (InputTraces[nTraces].name, file);

	pos = strlen(file)-strlen(EXT_MPIT)-DIGITS_PID-DIGITS_TASK
	  -DIGITS_THREAD-1; // Last -1 is for extra .
	has_node_separator = FALSE;
	hostname_len = 0;
	while (!has_node_separator)
	{
		has_node_separator = file[pos] == TEMPLATE_NODE_SEPARATOR_CHAR;
		if (has_node_separator)
		{
			InputTraces[nTraces].node = (char*) malloc (
			  (hostname_len+1)*sizeof(char));
			if (InputTraces[nTraces].node == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot obtain memory for NODE information!\n");
				fflush (stderr);
				exit (1);
			}
			snprintf (InputTraces[nTraces].node, hostname_len, "%s", &file[pos+1]);
			break;
		}
		else
		{
			if (pos == 0)
			{
				fprintf (stderr, "merger: Could not find node separator in file '%s'\n", file);
				InputTraces[nTraces].node = "(unknown)";
				break;
			}
			else
			{
				hostname_len++;
				pos--;
			}
		}
	}

	name_length = strlen (InputTraces[nTraces].name);
	tmp_name = InputTraces[nTraces].name;
	tmp_name = &(tmp_name[name_length - strlen(EXT_MPIT)]);
	if (strcmp (tmp_name, EXT_MPIT))
	{
		fprintf (stderr, "mpi2prv: Error! File %s does not contain a valid extension!. Skipping.\n", InputTraces[nTraces].name);
		return;
	}

	InputTraces[nTraces].filesize = 0;

	/* this will be shared afterwards at merger_post_share_file_sizes */
	if (taskid == 0) 
	{
		int fd = open (InputTraces[nTraces].name, O_RDONLY);
		if (-1 != fd)
		{
			InputTraces[nTraces].filesize = lseek (fd, 0, SEEK_END);
			close (fd);
		}
	}

	tmp_name = InputTraces[nTraces].name;
	tmp_name = &(tmp_name[name_length - strlen(EXT_MPIT) - DIGITS_TASK - DIGITS_THREAD]);

	/* Extract the information from the filename */
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
	/* This will be changed latter if Read_SPAWN_file is applied */
	InputTraces[nTraces].SpawnOffset = 0;

	if (thdname != NULL)
	{
		InputTraces[nTraces].threadname = strdup (thdname);
		if (InputTraces[nTraces].threadname == NULL)
		{
			fprintf (stderr, "mpi2prv: Error cannot obtain memory for THREAD NAME information!\n");
			fflush (stderr);
			exit (1);
		}
	}
	else
	{
		int res;

		/* 7+4 for THREAD + (ptask + three dots) THREAD 1.1.1 */
		InputTraces[nTraces].threadname = malloc (sizeof(char)*(10+DIGITS_TASK+DIGITS_THREAD+1));
		if (InputTraces[nTraces].threadname == NULL)
		{
			fprintf (stderr, "mpi2prv: Error cannot obtain memory for THREAD NAME information!\n");
			fflush (stderr);
			exit (1);
		}
		res = sprintf (InputTraces[nTraces].threadname, "THREAD %d.%d.%d",
		  InputTraces[nTraces].ptask, InputTraces[nTraces].task,
		  InputTraces[nTraces].thread);
		if (res >= 10+DIGITS_TASK+DIGITS_THREAD+1)
		{
			fprintf (stderr, "mpi2prv: Error! Thread name exceeds buffer size!\n");
			fflush (stderr);
			exit (1);
		}
	}

	nTraces++;
}

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
void Read_SPAWN_file (char *mpit_file, int current_ptask)
{
  char spawn_file_name[PATH_MAX];
  strcpy (spawn_file_name, mpit_file);
  spawn_file_name[strlen(spawn_file_name)-strlen(EXT_MPITS)] = (char) 0; /* remove ".mpit" extension */
  strcat (spawn_file_name, EXT_SPAWN);

  if (file_exists(spawn_file_name))
  {
    /* Read the synchronization latency */
    unsigned i;
    FILE *fd;
    char line[256];
    unsigned long long SpawnSyncLatency = 0;

    fd = fopen(spawn_file_name, "r");
	if (fd == NULL)
	{
		fprintf (stderr, "mpi2prv: Fatal error! Cannot load spawn file '%s'\n", spawn_file_name);
		exit (-1);
	}
    fgets(line, sizeof(line), fd);
    sscanf(line, "%llu", &SpawnSyncLatency);
    fclose(fd);

    for (i=0; i<nTraces; i++)
    {
      if (InputTraces[i].ptask == current_ptask)
      {
        InputTraces[i].SpawnOffset = SpawnSyncLatency;
      }
    }
    
    /* Load the intercommunicators table */
    intercommunicators_load (spawn_file_name, current_ptask);
  }
}
#endif /* MPI_SUPPORTS_MPI_COMM_SPAWN */



/******************************************************************************
 ***  Read_MPITS_file
 ***  Inserts into trace tables the contents of a ascii file!
 ******************************************************************************/

static char *last_mpits_file = NULL;

void Read_MPITS_file (const char *file, int *cptask, FileOpen_t opentype, int taskid)
{
	int info;
	char mybuffer[4096];
	char thdname[2048];
	char path[2048];
	FILE *fd = fopen (file, "r");

	if (fd == NULL)
	{
		fprintf (stderr, "mpi2prv: Unable to open %s file.\n", file);
		return;
	}

	MPITS_Files = (char**) realloc (MPITS_Files, sizeof(char*)*(Num_MPITS_Files+1));
	if (MPITS_Files == NULL)
	{
		fprintf (stderr, "mpi2prv: Unable to allocate memory for MPITS file: %s\n", file);
		exit (-1);
	}
	MPITS_Files[Num_MPITS_Files] = strdup (file);
	Num_MPITS_Files++;

	last_mpits_file = (char*) file;

	do
	{
		char * res = fgets (mybuffer, sizeof(mybuffer), fd);
		if (!feof(fd) && res != NULL)
		{
			char *stripped;

			path[0] = thdname[0] = (char) 0;

			info = sscanf (mybuffer, "%s named %s", path, thdname);
			stripped = trim (path);

			if (strncmp (mybuffer, "--", 2) == 0)
			{
				/* If we find --, advance to the next ptask */
				(*cptask)++;
			}
			else if (info >= 1 && opentype == FileOpen_Default)
			{
				/* If mode is not forced, check first if the absolute path exists,
				   if not, try to open in the current directory */

				if (!file_exists(stripped))
				{
					/* Look for /set- in string, and then use set- (thus +1) */
					char * stripped_basename = strstr (stripped, "/set-");
					if (stripped_basename != NULL)
					{
						/* Look in current directory, if not use list file directory */
						if (!file_exists(&stripped_basename[1]))
						{
							char dir_file[2048];
							char *duplicate = strdup (file);
							char *directory = dirname (duplicate);

							sprintf (dir_file, "%s%s", directory, stripped_basename);
							Process_MPIT_File (dir_file, (info==2)?thdname:NULL, cptask, taskid);

							free (duplicate);
						}
						else
							Process_MPIT_File (&stripped_basename[1], (info==2)?thdname:NULL, cptask, taskid);
					}
					else
						fprintf (stderr, "merger: Error cannot find 'set-' signature in filename %s\n", stripped);
				}
				else
					Process_MPIT_File (stripped, (info==2)?thdname:NULL, cptask, taskid);
			}
			else if (info >= 1 && opentype == FileOpen_Absolute)
			{
				Process_MPIT_File (stripped, (info==2)?thdname:NULL, cptask, taskid);
			}
			else if (info >= 1 && opentype == FileOpen_Relative)
			{
				/* Look for /set- in string, and then use set- (thus +1) */
				char * stripped_basename = strstr (stripped, "/set-");
				if (stripped_basename != NULL)
				{
					/* Look in current directory, if not use list file directory */
					if (!file_exists(&stripped_basename[1]))
					{
						char dir_file[2048];
						char *duplicate = strdup (file);
						char *directory = dirname (duplicate);

						sprintf (dir_file, "%s%s", directory, stripped_basename);
						Process_MPIT_File (dir_file, (info==2)?thdname:NULL, cptask, taskid);

						free (duplicate);
					}
					else
						Process_MPIT_File (&stripped_basename[1], (info==2)?thdname:NULL, cptask, taskid);
				}
				else
					fprintf (stderr, "merger: Error cannot find 'set-' signature in filename %s\n", stripped);
			}
		}
	}
	while (!feof(fd));

	fclose (fd);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	Read_SPAWN_file (file, *cptask);
#endif
}

/******************************************************************************
 ***  ProcessArgs
 ******************************************************************************/

void ProcessArgs (int rank, int argc, char *argv[])
{
	char *BinaryName, *bBinaryName;
	int CurArg;
	unsigned int cur_ptask = 1;   /* Ptask counter. Each -- the ptask number is
	                               * incremented. */
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
	bBinaryName = basename (BinaryName);

	if ((strncmp (bBinaryName, "mpi2prv", 7) == 0)
	    || (strncmp (bBinaryName, "mpimpi2prv", 10) == 0))
	{
		set_option_merge_ParaverFormat (TRUE);
		set_option_merge_ForceFormat (FALSE);
		set_merge_OutputTraceName (DEFAULT_PRV_OUTPUT_NAME);
	}
	else if ((strncmp (bBinaryName, "mpi2dim", 7) == 0)
	    || (strncmp (bBinaryName, "mpimpi2dim", 10) == 0))
	{
		set_option_merge_ParaverFormat (FALSE);
		set_option_merge_ForceFormat (FALSE);
		set_merge_OutputTraceName (DEFAULT_DIM_OUTPUT_NAME);
	}
	else
	{
		set_option_merge_ParaverFormat (TRUE);
		set_option_merge_ForceFormat (FALSE);
		set_merge_OutputTraceName (DEFAULT_PRV_OUTPUT_NAME);
	}
	free (BinaryName);

	for (CurArg = 1; CurArg < argc; CurArg++)
	{
		if (!strcmp (argv[CurArg], "-h"))
		{
			Help (argv[0]);
			exit (0);
		}
		if (!strcmp (argv[CurArg], "-keep-mpits"))
		{
			set_option_merge_RemoveFiles (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-keep-mpits"))
		{
			set_option_merge_RemoveFiles (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-trace-overwrite"))
		{
			set_option_merge_TraceOverwrite (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-trace-overwrite"))
		{
			set_option_merge_TraceOverwrite (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-v"))
		{
			set_option_merge_VerboseLevel (get_option_merge_VerboseLevel()+1);
			continue;
		}
		if (!strcmp (argv[CurArg], "-translate-addresses"))
		{
			set_option_merge_TranslateAddresses (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-translate-addresses"))
		{
			set_option_merge_TranslateAddresses (FALSE);
			set_option_merge_SortAddresses (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-emit-library-events"))
		{
			set_option_merge_EmitLibraryEvents (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-absolute-counters"))
		{
			set_option_merge_AbsoluteCounters (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-o"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				set_merge_OutputTraceName (argv[CurArg]);
				set_merge_GivenTraceName (TRUE);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -o: You must specify the output trace name.\n");
				Help(argv[0]);
				exit(0);
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-s"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				set_merge_SymbolFileName (argv[CurArg]);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -s: You must specify the path of the symbol file.\n");
				Help(argv[0]);
				exit(0);
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-c"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				set_merge_CallbackFileName (argv[CurArg]);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -c: You must specify the path of the callback file.\n");
				Help(argv[0]);
				exit(0);
			}
			continue;
		}
		if (!strcmp(argv[CurArg], "-e"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				set_merge_ExecutableFileName (argv[CurArg]);
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
		if (!strcmp (argv[CurArg], "-f"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				Read_MPITS_file (argv[CurArg], &cur_ptask, FileOpen_Default, rank);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -f: You must specify the path of the list file.\n");
				Help(argv[0]);
				exit(0);
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-f-relative"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				Read_MPITS_file (argv[CurArg], &cur_ptask, FileOpen_Relative, rank);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -f-relative: You must specify the path of the list file.\n");
				Help(argv[0]);
				exit(0);
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-f-absolute"))
		{
			CurArg++;
			if (CurArg < argc)
			{
				Read_MPITS_file (argv[CurArg], &cur_ptask, FileOpen_Absolute, rank);
			}
			else 
			{
				if (0 == rank)
					fprintf (stderr, PACKAGE_NAME": Option -f-absolute: You must specify the path of the list file.\n");
				Help(argv[0]);
				exit(0);
			}
		  continue;
		}
#if defined(IS_BG_MACHINE)
		if (!strcmp (argv[CurArg], "-xyzt"))
		{
			set_option_merge_BG_XYZT (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-xyzt"))
		{
			set_option_merge_BG_XYZT (FALSE);
			continue;
		}
#endif
		if (!strcmp (argv[CurArg], "-unique-caller-id"))
		{
			set_option_merge_UniqueCallerID (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-unique-caller-id"))
		{
			set_option_merge_UniqueCallerID (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-split-states"))
		{
			set_option_merge_JointStates (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-split-states"))
		{
			set_option_merge_JointStates (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-use-disk-for-comms"))
		{
			set_option_merge_UseDiskForComms (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-use-disk-for-comms"))
		{
			set_option_merge_UseDiskForComms (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-sort-addresses"))
		{
			set_option_merge_SortAddresses (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-sort-addresses"))
		{
			set_option_merge_SortAddresses (FALSE);
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
					set_option_merge_TreeFanOut (atoi(argv[CurArg]));
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
		if (!strcmp (argv[CurArg], "-d") || !strcmp(argv[CurArg], "-dump"))
		{
			set_option_merge_dump (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-dump-without-time"))
		{
			set_option_dump_Time (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-dump-with-time"))
		{
			set_option_dump_Time (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-with-addresses"))
		{
			set_option_dump_Addresses (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-without-addresses"))
		{
			set_option_dump_Addresses (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-maxmem"))
		{

			CurArg++;
			if (CurArg < argc)
			{
				int tmp = atoi(argv[CurArg]);
				if (tmp == 0)
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: Error! Invalid parameter for -maxmem option. Using 512 Mbytes\n");
					tmp = 512;
				}
				else if (tmp < 16)
				{
					if (0 == rank)
						fprintf (stderr, "mpi2prv: Error! Cannot use less than 16 MBytes for the merge step\n");
					tmp = 16;
				}
				set_option_merge_MaxMem (tmp);
			}
			else
			{	
				if (0 == rank)
					fprintf (stderr, "mpi2prv: WARNING: Invalid value for -maxmem parameter\n");
			}
			continue;
		}
		if (!strcmp (argv[CurArg], "-dimemas"))
		{
			set_option_merge_ForceFormat (TRUE);
			set_option_merge_ParaverFormat (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-paraver"))
		{
			set_option_merge_ForceFormat (TRUE);
			set_option_merge_ParaverFormat (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-skip-sendrecv"))
		{
			set_option_merge_SkipSendRecvComms (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-skip-sendrecv"))
		{
			set_option_merge_SkipSendRecvComms (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-syn"))
		{
			set_option_merge_SincronitzaTasks (TRUE);
			set_option_merge_SincronitzaTasks_byNode (FALSE);
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-syn-node"))
		{
			set_option_merge_SincronitzaTasks (TRUE);
			set_option_merge_SincronitzaTasks_byNode (TRUE);
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-syn"))
		{
			set_option_merge_SincronitzaTasks (FALSE);
			set_option_merge_SincronitzaTasks_byNode (FALSE);
			AutoSincronitzaTasks = FALSE;
			continue;
		}
		if (!strcmp (argv[CurArg], "-task-view"))
		{
			set_option_merge_NanosTaskView (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-task-view"))
		{
			set_option_merge_NanosTaskView (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-remove-files"))
		{
			set_option_merge_RemoveFiles (TRUE);
			continue;
		}
		if (!strcmp (argv[CurArg], "-no-remove-files"))
		{
			set_option_merge_RemoveFiles (FALSE);
			continue;
		}
		if (!strcmp (argv[CurArg], "--"))
		{
			cur_ptask++;
			continue;
		}
		else
			Process_MPIT_File (argv[CurArg], NULL, &cur_ptask, rank);
	}
	set_option_merge_NumApplications (cur_ptask);

	/* Specific things to be applied per format */
	if (rank == 0)
	{
		if (!get_option_merge_ParaverFormat())
		{
			/* Dimemas traces doesn't know about synchronization */
			set_option_merge_SincronitzaTasks (FALSE);
			set_option_merge_SincronitzaTasks_byNode (FALSE);
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
		else
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

typedef struct 
{
	unsigned ptask;
	unsigned task;
	off_t task_size;
} all_tasks_ids_t;

static void AssignFilesToWorker( unsigned merger_worker_id, all_tasks_ids_t task )
{
	unsigned i = 0;

	for (i=0; i<nTraces; i++)
	{
		if ((InputTraces[i].ptask == task.ptask) &&
		    (InputTraces[i].task  == task.task))
		{
			InputTraces[i].InputForWorker = merger_worker_id;
		}
	}
}

int SortTasksBySize (const void *t1, const void *t2)
{
        all_tasks_ids_t *task1 = (all_tasks_ids_t *)t1;
        all_tasks_ids_t *task2 = (all_tasks_ids_t *)t2;

        if (task1->task_size < task2->task_size)
                return -1;
        else if (task1->task_size > task2->task_size)
                return 1;
        else
                return 0;
}

static void DistributeWork (unsigned num_processors, unsigned processor_id)
{
	unsigned num_apps = 0;
	unsigned *num_tasks_per_app = NULL;
	unsigned **task_sizes_per_app = NULL;
	unsigned i = 0;
	unsigned j = 0;
	unsigned index = 0;
	unsigned all_tasks = 0;
	all_tasks_ids_t *all_tasks_ids = NULL;

#if defined(DEBUG)
	for (i = 0; i < nTraces; i ++)
	{
		fprintf(stderr, "[DEBUG] InputTraces[%d] ptask=%u task=%u\n", i, InputTraces[i].ptask, InputTraces[i].task);
	}
#endif

	for (i = 0; i < nTraces; i ++)
	{
		num_apps = MAX(num_apps, InputTraces[i].ptask);
	}	

#if defined(DEBUG)
	fprintf(stderr, "[DEBUG] num_apps = %d\n", num_apps);
#endif

	xmalloc(num_tasks_per_app, num_apps * sizeof(unsigned));
	xmalloc(task_sizes_per_app, num_apps * sizeof(unsigned *));

	for (i = 0; i < num_apps; i++)
	{
		num_tasks_per_app[i] = 0;
	}

	for (i = 0; i < nTraces; i++)
	{
		num_tasks_per_app[ InputTraces[i].ptask - 1 ] = MAX( num_tasks_per_app[ InputTraces[i].ptask - 1 ], InputTraces[i].task );
	}

	for (i = 0; i < num_apps; i++)
	{
#if defined(DEBUG)
		fprintf(stderr, "[DEBUG] num_tasks_per_app[%d]=%d\n", i, num_tasks_per_app[i]);
#endif

		xmalloc(task_sizes_per_app[i], num_tasks_per_app[i] * sizeof(unsigned));
		for (j = 0; j < num_tasks_per_app[i]; j++)
		{
			task_sizes_per_app[i][j] = 0;
		}

		all_tasks += num_tasks_per_app[i];
	}

	if (all_tasks < num_processors)
	{
		fprintf (stderr, "mpi2prv: FATAL ERROR! You are using more tasks for merging than tasks were traced! Please use less than %d tasks to merge.\n", all_tasks);
		exit(-1);
	}

	for (i = 0; i < nTraces; i++)
	{
		task_sizes_per_app[ InputTraces[i].ptask - 1 ][ InputTraces[i].task - 1 ] += InputTraces[i].filesize;
	}

	xmalloc(all_tasks_ids, all_tasks * sizeof(all_tasks_ids_t));
	index = 0;
	for (i = 0; i < num_apps; i ++)
	{
		for (j = 0; j < num_tasks_per_app[i]; j ++)
		{
			all_tasks_ids[index].ptask = i+1;
			all_tasks_ids[index].task  = j+1;
			all_tasks_ids[index].task_size = task_sizes_per_app[i][j];
			index ++;
		}
	}

#if defined(DEBUG)
	fprintf(stderr, "[DEBUG] all_tasks=%d\n", all_tasks);
	for (i = 0; i < all_tasks; i++)
	{
		fprintf(stderr, "[DEBUG] all_tasks[%d] ptask=%d task=%d task_size=%d\n", 
		  i+1, all_tasks_ids[i].ptask, all_tasks_ids[i].task, (int)all_tasks_ids[i].task_size);
	}
#endif

	if (WorkDistribution == Block)
	{
		unsigned tasks_per_merger = (all_tasks + num_processors - 1) / num_processors;
		for (i=0; i<num_processors; i++)
		{
			for (j=0; j<tasks_per_merger; j++)
			{
				unsigned task_to_assign = (i * tasks_per_merger) + j;
				if (task_to_assign < all_tasks)
					AssignFilesToWorker( i, all_tasks_ids[task_to_assign] );
			}
		}
	}
	else if (WorkDistribution == Cyclic)
	{
		/* Files will be distributed in cycles */
		for (i=0; i < all_tasks; i++)
			AssignFilesToWorker(i % num_processors, all_tasks_ids[i]);
	}
	else if (WorkDistribution == Size || WorkDistribution == ConsecutiveSize)
	{
		off_t average_size_per_worker;
		off_t remaining_size = 0;
		off_t assigned_size[num_processors];
		char assigned_files[all_tasks];
		unsigned current_task;

		if (WorkDistribution == Size)
			qsort (all_tasks_ids, all_tasks, sizeof(all_tasks_ids_t), SortTasksBySize);

		for (i=0; i < num_processors; i++)
		{
			assigned_size[i] = 0;
		}

		for (i=0; i < all_tasks; i++)
		{
			remaining_size += all_tasks_ids[i].task_size;
			assigned_files[i] = FALSE;
		}
		
		average_size_per_worker = remaining_size / num_processors;

		for (i=0; i<num_processors; i++)
		{
			current_task = 0;
			while (assigned_size[i] < average_size_per_worker)
			{
				if (!assigned_files[current_task])
				{
					if (assigned_size[i]+all_tasks_ids[current_task].task_size <= average_size_per_worker)
					{
						assigned_files[current_task] = TRUE;
						assigned_size[i] += all_tasks_ids[current_task].task_size;
						AssignFilesToWorker(i, all_tasks_ids[current_task]);
						remaining_size -= all_tasks_ids[current_task].task_size;
					}
				}
				if (++current_task >= all_tasks)
				{
					break;
				}
			}
			average_size_per_worker = remaining_size / (num_processors - i - 1);
		}
	}

	/* Check assigned traces... */
	for (index = 0; index < nTraces; index++)
		if (InputTraces[index].InputForWorker >= (int)num_processors ||
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
			if (InputTraces[file].InputForWorker == (int)index)
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

	if (task_sizes_per_app)
	{
		for (i = 0; i < num_apps; i++)
			if (task_sizes_per_app[i])
				free (task_sizes_per_app[i]);
		free (task_sizes_per_app);
	}
	if (num_tasks_per_app)
		free (num_tasks_per_app);
	if (all_tasks_ids)
		free (all_tasks_ids);
}


/******************************************************************************
 ***  main entry point
 ******************************************************************************/

/* To be called before ProcessArgs */

void merger_pre (int numtasks)
{
#if !defined(PARALLEL_MERGE)
	UNREFERENCED_PARAMETER(numtasks);
#endif

#if defined(PARALLEL_MERGE)
	if (numtasks <= 1)
	{
		fprintf (stderr, "mpi2prv: The parallel version of the mpi2prv is not suited for 1 processor! Dying...\n");
		exit (1);
	}
#endif
}


/* To be called after ProcessArgs */

#if defined(PARALLEL_MERGE)
static void merger_post_share_file_sizes (int taskid)
{
	int res;
	unsigned i;
	unsigned long long *sizes;

	sizes = malloc (sizeof(unsigned long long)*nTraces);
	if (sizes == NULL)
	{
		fprintf (stderr, "mpi2prv: Cannot allocate memory to share trace file sizes\n");
		perror ("malloc");
		exit (1);
	}

	if (taskid == 0)
		for (i = 0; i < nTraces; i++)
			sizes[i] = InputTraces[i].filesize; 

	res = MPI_Bcast (sizes, nTraces, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Cannot share trace file sizes");

	if (taskid != 0)
		for (i = 0; i < nTraces; i++)
			InputTraces[i].filesize = sizes[i]; 

	free (sizes);
}
#endif

int merger_post (int numtasks, int taskid)
{
	unsigned long long records_per_task;
#if defined(PARALLEL_MERGE)
	char **nodenames;
#else
	char nodename[1024];
	char *nodenames[1];
#endif
	int error;
	struct Pair_NodeCPU *NodeCPUinfo;

	if (taskid == 0)
		fprintf (stdout, "merger: "PACKAGE_STRING" (revision %d based on %s)\n",
		  EXTRAE_SVN_REVISION, EXTRAE_SVN_BRANCH);

	if (0 == nTraces)
	{
	  fprintf (stderr, "mpi2prv: No intermediate trace files given.\n");
	  return 0;
	}

#if defined(PARALLEL_MERGE)
	merger_post_share_file_sizes (taskid);

	if (get_option_merge_TreeFanOut() == 0)
	{
		if (taskid == 0)
			fprintf (stdout, "mpi2prv: Tree order is not set. Setting automatically to %d\n", numtasks);
		set_option_merge_TreeFanOut (numtasks);
	}
	else if (get_option_merge_TreeFanOut() > numtasks)
	{
		if (taskid == 0)
			fprintf (stdout, "mpi2prv: Tree order is set to %d but is larger that numtasks. Setting tree order to %d\n", get_option_merge_TreeFanOut(), numtasks);
		set_option_merge_TreeFanOut (numtasks);
	}
	else if (get_option_merge_TreeFanOut() <= numtasks)
	{
		if (taskid == 0)
			fprintf (stdout, "mpi2prv: Tree order is set to %d\n", get_option_merge_TreeFanOut());
	}

	if (numtasks > nTraces)
	{
		if (taskid == 0)
			fprintf (stderr, "mpi2prv: FATAL ERROR! The tree fan out (%d) is larger than the number of MPITs (%d)\n", numtasks, nTraces);
		exit (0);
	}
#endif

	records_per_task = 1024*1024/sizeof(paraver_rec_t);  /* num of events in 1 Mbytes */
	records_per_task *= get_option_merge_MaxMem();       /* let's use this memory */
#if defined(PARALLEL_MERGE)
	records_per_task /= get_option_merge_TreeFanOut();   /* divide by the tree fan out */

	if (0 == records_per_task)
	{
		if (0 == taskid)
			fprintf (stderr, "mpi2prv: Error! Assigned memory by -maxmem is insufficient for this tree fan out\n");
		exit (-1);
	}
#endif

#if defined(PARALLEL_MERGE)
	ShareNodeNames (numtasks, &nodenames);
#else
	gethostname (nodename, sizeof(nodename));
	nodenames[0] = nodename;
#endif

	PrintNodeNames (numtasks, taskid, nodenames);
	DistributeWork (numtasks, taskid);
	NodeCPUinfo = AssignCPUNode (nTraces, InputTraces);

	if (AutoSincronitzaTasks)
	{
		unsigned i;
		unsigned all_nodes_are_equal = TRUE;
		unsigned first_node = InputTraces[0].nodeid;
		for (i = 1; i < nTraces && all_nodes_are_equal; i++)
			all_nodes_are_equal = (first_node == InputTraces[i].nodeid);
		set_option_merge_SincronitzaTasks (!all_nodes_are_equal);

		if (0 == taskid)
		{
			fprintf (stdout, "mpi2prv: Time synchronization has been turned %s\n", get_option_merge_SincronitzaTasks()?"on":"off");
			fflush (stdout);
		}
	}

	if (get_option_merge_TranslateAddresses())
	{
		if (taskid == 0 && strlen(get_merge_ExecutableFileName()) > 0)
			Address2Info_Initialize (get_merge_ExecutableFileName());
		else
			Address2Info_Initialize (NULL);
	}

#if defined(PARALLEL_MERGE)
	if (taskid == 0)
	{
		int res, tmp = get_option_merge_SortAddresses();
		res = MPI_Bcast (&tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Bcast, "Cannot share whether option SortAddresses is turned on");
	}
	else
	{
		int res, tmp;
		res = MPI_Bcast (&tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Bcast, "Cannot share whether option SortAddresses is turned on");
		set_option_merge_SortAddresses (tmp);
	}
#endif

	if (taskid == 0 &&
		strlen(get_merge_SymbolFileName()) == 0 &&
		last_mpits_file != NULL)
	{
		char tmp[1024];
		strncpy (tmp, last_mpits_file, sizeof(tmp));

		if (strcmp (&tmp[strlen(tmp)-strlen(".mpits")], ".mpits") == 0)
		{
			strncpy (&tmp[strlen(tmp)-strlen(".mpits")], ".sym", strlen(".sym")+1);
			Labels_loadSYMfile (taskid, TRUE, 0, 0, tmp, TRUE);
		}
	}
	else
	{
		if (taskid == 0)
			Labels_loadSYMfile (taskid, FALSE, 0, 0, get_merge_SymbolFileName(), TRUE);
	}

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Checking for target directory existance...");
		char *dirn = dirname(strdup(trim(get_merge_OutputTraceName())));
		if (!directory_exists(dirn))
		{
			fprintf (stdout, " does not exist. Creating ...");
			if (!mkdir_recursive(dirn))
			{
				fprintf (stdout, " failed to create (%s)!\n", dirn);
				exit (-1);
			}
			else
				fprintf (stdout, " done\n");
		}
		else
			fprintf (stdout, " exists, ok!\n");
	}

	if (get_option_merge_ParaverFormat())
		error = Paraver_ProcessTraceFiles (nTraces, InputTraces,
		    get_option_merge_NumApplications(),
			NodeCPUinfo, numtasks, taskid);
	else
		error = Dimemas_ProcessTraceFiles (trim(get_merge_OutputTraceName()),
			nTraces, InputTraces, get_option_merge_NumApplications(),
			NodeCPUinfo, numtasks, taskid);

	if (!error)
	{
		if (get_option_merge_RemoveFiles())
		{
			unsigned u;

			/* Remove MPITS and their SYM related files */
			for (u = 0; u < Num_MPITS_Files; u++)
			{
				char tmp[1024];
				strncpy (tmp, MPITS_Files[u], sizeof(tmp));

				if (strcmp (&tmp[strlen(tmp)-strlen(".mpits")], ".mpits") == 0)
				{
					strncpy (&tmp[strlen(tmp)-strlen(".mpits")], ".sym", strlen(".sym")+1);
					unlink (tmp);
				}
				unlink (MPITS_Files[u]);
			}

			for (u = 0; u < nTraces; u++)
			{
				/* Remove the .mpit file */
				unlink (InputTraces[u].name);

				/* Remove the local .sym file for that .mpit file */
				{
					char tmp[1024];
					strncpy (tmp, InputTraces[u].name, sizeof(tmp));
					strncpy (&tmp[strlen(tmp)-strlen(".mpit")], ".sym", strlen(".sym")+1);
					unlink (tmp);
				}

				/* Try to remove the container set-X directory */
				rmdir (dirname (InputTraces[u].name));
			}
		}
	}
	else
		fprintf (stderr, "mpi2prv: An error has been encountered when generating the tracefile. Dying...\n");

#if defined(HAVE_BFD)
	if (get_option_merge_VerboseLevel() > 0)
		Addr2Info_HashCache_ShowStatistics();
#endif

	return 0;
}
