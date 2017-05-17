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
#include <errno.h>

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef PARALLEL_MERGE
# include <mpi.h>
# include "mpi-tags.h"
# include "mpi-aux.h"
# include "tree-logistics.h"
#endif
#if defined(HAVE_SIONLIB)
# include "sion.h"
#endif
#include "labels.h"
#include "utils.h"
#include "semantics.h"
#include "cpunode.h"
#include "timesync.h"
#include "bswap.h"
#include "file_set.h"
#include "mpi2out.h"
#include "events.h"
#include "object_tree.h"
#include "HardwareCounters.h"
#include "trace_to_prv.h"
#include "communication_queues.h"
#include "intercommunicators.h"

#define EVENTS_FOR_NUM_GLOBAL_OPS(x) \
     ((x) == MPI_BARRIER_EV  || (x) == MPI_BCAST_EV       || (x) == MPI_ALLREDUCE_EV       || \
      (x) == MPI_ALLTOALL_EV || (x) == MPI_ALLTOALLV_EV   || (x) == MPI_SCAN_EV            || \
      (x) == MPI_REDUCE_EV   || (x) == MPI_ALLGATHER_EV   || (x) == MPI_ALLGATHERV_EV      || \
      (x) == MPI_GATHER_EV   || (x) == MPI_GATHERV_EV     || (x) == MPI_SCATTER_EV         || \
      (x) == MPI_SCATTERV_EV || (x) == MPI_REDUCESCAT_EV  || (x) == MPI_REDUCE_SCATTER_BLOCK_EV || \
      (x) == MPI_IREDUCE_SCATTER_BLOCK_EV || (x) == MPI_ALLTOALLW_EV || (x) == MPI_IALLTOALLW_EV)

static int Is_FS_Rewound = TRUE;
static int LimitOfEvents = 0;

void setLimitOfEvents (int limit)
{
	LimitOfEvents = limit;
}

/******************************************************************************
 ***  num_Files_FS
 ******************************************************************************/

int num_Files_FS (FileSet_t * fset)
{
  return (fset->nfiles);
}

/******************************************************************************
 ***  GetNextObj_FS
 ******************************************************************************/

void GetNextObj_FS (FileSet_t * fset, int file, unsigned int *cpu, unsigned int *ptask, unsigned int *task,
                    unsigned int *thread)
{
  FileItem_t *sfile;

  ASSERT (file >= 0 && file < fset->nfiles, "Invalid file identifier");

  sfile = &(fset->files[file]);
  CurrentObj_FS (sfile, *cpu, *ptask, *task, *thread);
}

#if defined(SAMPLING_SUPPORT) || defined(HAVE_ONLINE)
static int event_timing_sort (const void *e1, const void *e2)
{
	event_t *ev1 = (event_t*) e1;
	event_t *ev2 = (event_t*) e2;

	if (Get_EvTime(ev1) < Get_EvTime(ev2))
		return -1;
	else if (Get_EvTime(ev1) > Get_EvTime(ev2))
		return 1;
	else
		return 0;
}
#endif

int isTaskInMyGroup (FileSet_t *fset, int ptask, int task)
{
	unsigned i;
	
	for (i = 0; i < fset->nfiles; i++)
		if ((fset->files[i].ptask-1 == ptask) && (fset->files[i].task-1 == task))
			return TRUE;
	return FALSE;
}

int inWhichGroup (int ptask, int task, FileSet_t *fset)
{
	unsigned i;

	for (i = 0; i < fset->num_input_files; i++)
		if ((fset->input_files[i].ptask-1 == ptask) && (fset->input_files[i].task-1 == task))
			return fset->input_files[i].InputForWorker;
	return -1;
}

static int newTemporalFile (int taskid, int initial, int depth, char *filename)
{
	int ID;

	if (initial)
	{
		if (getenv ("MPI2PRV_TMP_DIR") == NULL)
		{
			if (getenv ("TMPDIR") == NULL)
				sprintf (filename, "TmpFile-taskid%d-initial-XXXXXX", taskid);
			else
				sprintf (filename, "%s/TmpFile-taskid%d-initial-XXXXXX", getenv ("TMPDIR"), taskid);
		}
		else
			sprintf (filename, "%s/TmpFile-taskid%d-initial-XXXXXX", getenv ("MPI2PRV_TMP_DIR"), taskid);
	}
	else
	{
		if (getenv ("MPI2PRV_TMP_DIR") == NULL)
		{
			if (getenv ("TMPDIR") == NULL)
				sprintf (filename, "TmpFile-taskid%d-depth%d-XXXXXX", taskid, depth);
			else
				sprintf (filename, "%s/TmpFile-taskid%d-depth%d-XXXXXX", getenv ("TMPDIR"), taskid, depth);
		}
		else
			sprintf (filename, "%s/TmpFile-taskid%d-depth%d-XXXXXX", getenv ("MPI2PRV_TMP_DIR"), taskid, depth);
	}

	/* Make a temporal name for a file */	
	if ((ID = mkstemp (filename)) == -1)
	{
		perror ("mkstemp");
		fprintf (stderr, "mpi2prv: Error! Unable to create temporal file using mkstemp");
		fflush (stderr);
		exit (-1);
	}

	return ID;
}


/******************************************************************************
 ***  AddFile_FS
 ******************************************************************************/

static int AddFile_FS (FileItem_t * fitem, struct input_t *IFile, int taskid)
{
	int tmp_fd;
	int ret;
	FILE *fd_trace;
	ssize_t res;
	char *tmp;
	char paraver_tmp[PATH_MAX];
	char trace_file_name[PATH_MAX];
	long long trace_file_size;
#if defined(SAMPLING_SUPPORT)
	char sample_file_name[PATH_MAX];
	long long sample_file_size;
	FILE *fd_sample;
#endif
#if defined(HAVE_ONLINE)
	char online_file_name[PATH_MAX];
	long long online_file_size;
	int fd_online;
#endif
#if defined(SAMPLING_SUPPORT) || defined(HAVE_ONLINE)
	int sort_needed = FALSE;
#endif
	event_t *ptr_last = NULL;

	strcpy (trace_file_name, IFile->name);
#if defined(HAVE_SIONLIB)
	int rc, sid;
	int ntasks, nfiles;
	FILE *fp;
	sion_int64 *chunksizes = NULL; 
	sion_int32 fsblksize;
	int *globalranks = NULL; // allocated by sion_open

	// default is ANSI-C
	sid = sion_open("data.mpit", "rb", &ntasks, &nfiles, &chunksizes,
		&fsblksize, &globalranks, &fp);
	if (sid == -1)
	{
		perror ("sion_open");
		fprintf (stderr, "mpi2prv Error: Failed to open trace-file\n");
	}

	int size, block;
	sion_int64 globalskip;
	sion_int64 start_of_varheader;
	sion_int64 *sion_chunksizes;
	sion_int64 *sion_globalranks;
	sion_int64 *sion_blockcount;
	sion_int64 *sion_blocksizes;

	sion_get_locations(sid, &size, &block, &globalskip, &start_of_varheader,
	  &sion_chunksizes, &sion_globalranks, &sion_blockcount, &sion_blocksizes);

	int rank; 
	int blksize; 
	int blknum;

	for (blknum = 0; blknum < sion_blockcount[taskid]; blknum++)
		blksize = sion_blocksizes[size * blknum + IFile->task -1];
	//fprintf(stdout, "rank %d byes %d\n", IFile->task - 1, blksize);

	fd_trace = fp;
#else
	fd_trace = fopen (trace_file_name, "r");
	if (NULL == fd_trace)
	{
		perror ("fopen");
		fprintf (stderr, "mpi2prv Error: Opening trace file %s\n", trace_file_name);
		return (-1);
	}
#endif

#if defined(SAMPLING_SUPPORT)
	strcpy (sample_file_name, IFile->name);
	sample_file_name[strlen(sample_file_name)-strlen(EXT_MPIT)] = (char) 0; /* remove ".mpit" extension */
	strcat (sample_file_name, EXT_SAMPLE);

	fd_sample = fopen (sample_file_name, "r");
#endif
#if defined(HAVE_ONLINE)
	strcpy (online_file_name, IFile->name);
	online_file_name[strlen(online_file_name)-strlen(EXT_MPIT)] = (char) 0; /* remove ".mpit" extension */
	strcat (online_file_name, EXT_ONLINE);

	fd_online = open (online_file_name, O_RDONLY);
#endif
#if defined(HAVE_SIONLIB)
	trace_file_size = blksize;
#else
	ret = fseeko (fd_trace, 0, SEEK_END);
	if (0 != ret)
	{
		fprintf (stderr, "mpi2prv: `fseeko` failed to set file pointer of file %s\n", trace_file_name);
		exit (1);
	}
	trace_file_size = ftello (fd_trace);
#endif
#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
	{
		ret = fseeko (fd_sample, 0, SEEK_END);
		if (0 != ret)
		{
			fprintf (stderr, "mpi2prv: `fseeko` failed to set file pointer of file %s\n", sample_file_name);
			exit (1);
		}
		sample_file_size = ftello (fd_sample);
	}
	else
		sample_file_size = 0;
#endif

#if defined(HAVE_ONLINE)
	online_file_size = (fd_online != -1)?lseek (fd_online, 0, SEEK_END):0;
#endif

	fitem->size = trace_file_size;
#if defined(SAMPLING_SUPPORT)
	fitem->size += sample_file_size;
#endif
#if defined(HAVE_ONLINE)
	fitem->size += online_file_size;
#endif

#if 0
	if (LimitOfEvents != 0)
		fitem->size = MIN(fitem->size, LimitOfEvents*sizeof(event_t));
#endif
	fitem->num_of_events = (fitem->size>0)?fitem->size/sizeof(event_t):0;

#if !defined(HAVE_SIONLIB)
	rewind (fd_trace);
#endif

#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
		rewind (fd_sample);
#endif
#if defined(HAVE_ONLINE)
	if (fd_online != -1)
		lseek (fd_online, 0, SEEK_SET);
#endif

	{
		int extra = (trace_file_size % sizeof (event_t));
		if (extra != 0)
			printf ("PANIC! Trace file %s is %d bytes too big!\n", trace_file_name, extra);

#if defined(SAMPLING_SUPPORT)
		extra = (sample_file_size % sizeof (event_t));
		if (extra != 0)
			printf ("PANIC! Sample file %s is %d bytes too big!\n", sample_file_name, extra);
#endif
#if defined(HAVE_ONLINE)
		extra = (online_file_size % sizeof (event_t));
		if (extra != 0)
			printf ("PANIC! Online file %s is %d bytes too big!\n", online_file_name, extra);
#endif
	}

	fitem->first = (event_t*) malloc (fitem->size);
	if (fitem->first == NULL)
	{
		fprintf (stderr, "mpi2prv: `malloc` failed to allocate memory for file %s\n",
			IFile->name);
		exit (1);
	}

	/* Read files */
	res = fread (fitem->first, 1, trace_file_size, fd_trace);
	if (res != trace_file_size)
	{
		fprintf (stderr, "mpi2prv: `fread` failed to read from file %s\n", trace_file_name);
		fprintf (stderr, "mpi2prv:        returned %Zu (instead of %lld)\n", res, trace_file_size);
		exit (1);
	}
	ptr_last = fitem->first + (trace_file_size/sizeof(event_t));

#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
	{
		res = fread (ptr_last, 1, sample_file_size, fd_sample);
		if (res != sample_file_size)
		{
			fprintf (stderr, "mpi2prv: `fread` failed to read from file %s\n", sample_file_name);
			fprintf (stderr, "mpi2prv:        returned %Zu (instead of %lld)\n", res, sample_file_size);
			exit (1);
		}
	}
	sort_needed = sample_file_size > 0;
	ptr_last += (sample_file_size/sizeof(event_t));
#endif
#if defined(HAVE_ONLINE)
	if (fd_online != -1)
	{
		res = read (fd_online, ptr_last, online_file_size);
		if (res != online_file_size)
		{
			fprintf (stderr, "mpi2prv: `read` failed to read from file %s\n", online_file_name);
			fprintf (stderr, "mpi2prv:        returned %Zu (instead of %lld)\n", res, online_file_size);
			exit (1);
		}
	}
	if (online_file_size > 0) sort_needed = TRUE;
	ptr_last += (online_file_size/sizeof(event_t));	
#endif

#if defined(SAMPLING_SUPPORT) || defined(HAVE_ONLINE)
	if (sort_needed)
	{
		qsort (fitem->first, fitem->num_of_events, sizeof(event_t), event_timing_sort);
	}
#endif

	/* We no longer need this/these descriptor/s */
	fclose (fd_trace);
#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
		fclose (fd_sample);
#endif
#if defined(HAVE_ONLINE)
	if (fd_online != -1)
		close (fd_online);
#endif

	tmp = (char *) fitem->first;
	tmp = tmp + fitem->size;
	fitem->last = (event_t *) tmp;

	fitem->first_glop = NULL;
	fitem->current = fitem->next_cpu_burst = fitem->last_recv = fitem->first;

	fitem->ptask = IFile->ptask;
	fitem->task = IFile->task;
	fitem->thread = IFile->thread;
	fitem->cpu = IFile->cpu;

	(GET_THREAD_INFO(fitem->ptask,IFile->task,IFile->thread))->file = fitem;

	/* Create a buffered file with 512 entries of paraver_rec_t */
	tmp_fd = newTemporalFile (taskid, TRUE, 0, paraver_tmp);
	fitem->wfb = WriteFileBuffer_new (tmp_fd, paraver_tmp, 512, sizeof(paraver_rec_t));

	/* Remove the created file... while we don't die, it won't be removed */
	unlink (paraver_tmp);

	return 0;
}

/******************************************************************************
 ***  Create_FS
 ******************************************************************************/

FileSet_t *Create_FS (unsigned long nfiles, struct input_t * IFiles, int idtask, int trace_format)
{
	unsigned long file;
	FileSet_t *fset;
	FileItem_t *fitem;

	if ((fset = malloc (sizeof (FileSet_t))) == NULL)
	{
		perror ("malloc");
		fprintf (stderr, "mpi2prv: Error creating file set\n");
		return NULL;
	}

	fset->input_files = IFiles;
	fset->num_input_files = nfiles;
	fset->traceformat = trace_format;
	xmalloc(fset->files, nTraces * sizeof(FileItem_t));
	fset->nfiles = 0;
	for (file = 0; file < nfiles; file++)
		if (IFiles[file].InputForWorker == idtask)
		{
			fitem = &(fset->files[fset->nfiles]);
			fitem->mpit_id = file;
			if (AddFile_FS (fitem, &(IFiles[file]), idtask) != 0)
			{
				perror ("AddFile_FS");
				fprintf (stderr, "mpi2prv: Error creating file set\n");
				free (fset);
				return NULL;
			}
			fset->nfiles++;
		}

	return fset;
}

void Flush_FS (FileSet_t *fset, int remove_last)
{
	unsigned i;

	if (fset != NULL)
		for (i = 0; i < fset->nfiles; i++)
		{
			if (remove_last)
				WriteFileBuffer_removeLast (fset->files[i].wfb);
			WriteFileBuffer_flush (fset->files[i].wfb);
		}
}

void Free_FS (FileSet_t *fset)
{
	unsigned i;
	FileItem_t *fitem;

	if (fset != NULL)
	{
		for (i = 0; i < fset->nfiles; i++)
		{
			fitem = &(fset->files[i]);
			if (fitem->first != NULL)
				free (fitem->first);
			fitem->first = fitem->last = fitem->current = NULL;
		}
		free (fset);
	}
}

#if defined(PARALLEL_MERGE)
PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block, int tree_fan_out)
{
	int i;
	int res;
	unsigned long long total = 0;
	PRVFileSet_t *prvfset = NULL;

	*num_of_events = total;


	if ((prvfset = malloc (sizeof (PRVFileSet_t))) == NULL)
	{
		perror ("malloc");
		fprintf (stderr, "mpi2prv: Error creating PRV file set\n");
		return 0;
	}

	prvfset->fset = fset;

        xmalloc(prvfset->files, nTraces * sizeof(PRVFileItem_t));
	prvfset->nfiles = fset->nfiles;
	prvfset->records_per_block = records_per_block / (fset->nfiles + tree_fan_out);

	/* Set local files first */
	for (i = 0; i < fset->nfiles; i++)
	{
		if (i == 0 && tree_MasterOfSubtree (taskid, tree_fan_out, 0))
		{
			char paraver_tmp[PATH_MAX];

			/* Create a temporal file */
			int fd = newTemporalFile (taskid, FALSE, 0, paraver_tmp);
			prvfset->files[i].destination = WriteFileBuffer_new (fd, paraver_tmp, 512, sizeof(paraver_rec_t));
			unlink (paraver_tmp);
		}
		else
			prvfset->files[i].destination = (WriteFileBuffer_t*) 0xbeefdead;

		prvfset->files[i].source = WriteFileBuffer_getFD(fset->files[i].wfb);
		prvfset->files[i].type = LOCAL;
		prvfset->files[i].mapped_records = 0;
		prvfset->files[i].current_p =
			prvfset->files[i].last_mapped_p =
			prvfset->files[i].first_mapped_p = NULL;
		prvfset->files[i].remaining_records = lseek (prvfset->files[i].source, 0, SEEK_END);
		lseek (prvfset->files[i].source, 0, SEEK_SET);
		if (-1 == prvfset->files[i].remaining_records)
		{
			fprintf (stderr, "mpi2prv: Failed to seek the end of a temporal file\n");
			fflush (stderr);
			exit (0);
		}
		else
			prvfset->files[i].remaining_records /= sizeof(paraver_rec_t);

		total += prvfset->files[i].remaining_records;
	}

	/* Set remote files now (if exist), receive how many events they have */
 	if (tree_MasterOfSubtree (taskid, tree_fan_out, 0))
	{
		int i = 1;
		while (taskid+i*tree_pow(tree_fan_out,0) < numtasks && i < tree_fan_out)
		{
			MPI_Status s;

			prvfset->files[fset->nfiles+i-1].source = taskid + i*tree_pow(tree_fan_out, 0);
			prvfset->files[fset->nfiles+i-1].type = REMOTE;
			prvfset->files[fset->nfiles+i-1].mapped_records = 0;
			prvfset->files[fset->nfiles+i-1].current_p =
			prvfset->files[fset->nfiles+i-1].last_mapped_p =
				prvfset->files[fset->nfiles+i-1].first_mapped_p = NULL;

			res = MPI_Recv (&(prvfset->files[fset->nfiles+i-1].remaining_records), 1, MPI_LONG_LONG, prvfset->files[fset->nfiles+i-1].source, REMAINING_TAG, MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Cannot receive information of remaining records");

			total += prvfset->files[fset->nfiles+i-1].remaining_records;

			prvfset->nfiles++;
			i++;
		}
	}
	else
	{
		/* NON ROOT WORK,
		   send how many events are on this slave */
		int my_master = tree_myMaster (taskid, tree_fan_out, 0);

		res = MPI_Send (&total, 1, MPI_LONG_LONG, my_master, REMAINING_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Cannot send information of remaining records");
	}

	*num_of_events = total;

	prvfset->SkipAsMasterOfSubtree = FALSE;

	return prvfset;
}

PRVFileSet_t * ReMap_Paraver_files_binary (PRVFileSet_t * infset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block, int depth, int tree_fan_out)
{
	int i;
	int res;
	unsigned long long total = 0;

	*num_of_events = total;

	infset->records_per_block = records_per_block / tree_fan_out;

	/* Master process will have its own files plus references to a single file of every other
	   task (which represents a its set of assigned files) */

	if (tree_MasterOfSubtree (taskid, tree_fan_out, depth))
	{
		char paraver_tmp[PATH_MAX];

		if (infset->nfiles > 1)
		{
			int fd;

			infset->files[0].source = WriteFileBuffer_getFD(infset->files[0].destination);

			/* Create a temporal file */
			fd = newTemporalFile (taskid, FALSE, 0, paraver_tmp);
			infset->files[0].destination = WriteFileBuffer_new (fd, paraver_tmp, 512, sizeof(paraver_rec_t));
			unlink (paraver_tmp);

			/* Set local file first */
			infset->nfiles = 1;
			infset->files[0].type = LOCAL;
			infset->files[0].mapped_records = 0;
			infset->files[0].current_p =
				infset->files[0].last_mapped_p =
				infset->files[0].first_mapped_p = NULL;
			infset->files[0].remaining_records = lseek (infset->files[0].source, 0, SEEK_END);
			lseek (infset->files[0].source, 0, SEEK_SET);
			if (-1 == infset->files[0].remaining_records)
			{
				fprintf (stderr, "mpi2prv: Failed to seek the end of a temporal file\n");
				fflush (stderr);
				exit (0);
			}
			else
				infset->files[0].remaining_records /= sizeof(paraver_rec_t);
		
			total += infset->files[0].remaining_records;

			i = 1;
			while (taskid+i*tree_pow(tree_fan_out,depth) < numtasks && i < tree_fan_out)
			{
				MPI_Status s;

				infset->files[i].source = taskid + i*tree_pow(tree_fan_out, depth);
				infset->files[i].type = REMOTE;
				infset->files[i].mapped_records = 0;
				infset->files[i].current_p =
					infset->files[i].last_mapped_p =
					infset->files[i].first_mapped_p = NULL;

				res = MPI_Recv (&(infset->files[i].remaining_records), 1, MPI_LONG_LONG, infset->files[i].source, REMAINING_TAG, MPI_COMM_WORLD, &s);
				MPI_CHECK(res, MPI_Recv, "Cannot receive information of remaining records");

				total += infset->files[i].remaining_records;

				infset->nfiles++;
				i++;
			}

				infset->SkipAsMasterOfSubtree = FALSE;
		} /* if infset->nfiles > 1 */
		else
		{
			infset->SkipAsMasterOfSubtree = TRUE;
		}
	}
	else
	{
		/* NON ROOT WORK,
	   send how many events are on this slave */
		int my_master = tree_myMaster (taskid, tree_fan_out, depth);

		infset->nfiles = 1;
		infset->files[0].source = WriteFileBuffer_getFD(infset->files[0].destination);
		infset->files[0].destination = (WriteFileBuffer_t*) 0xdeadbeef;
		infset->files[0].type = LOCAL;

		/* Set local file first */
		infset->files[0].mapped_records = 0;
		infset->files[0].current_p =
			infset->files[0].last_mapped_p =
			infset->files[0].first_mapped_p = NULL;

		infset->files[0].remaining_records = lseek (infset->files[0].source, 0, SEEK_END);
		lseek (infset->files[0].source, 0, SEEK_SET);

		if (-1 == infset->files[0].remaining_records)
		{
			fprintf (stderr, "mpi2prv: Failed to seek the end of a temporal file\n");
			fflush (stderr);
			exit (0);
		}
		else
			infset->files[0].remaining_records /= sizeof(paraver_rec_t);

		total = infset->files[0].remaining_records;

		res = MPI_Send (&total, 1, MPI_LONG_LONG, my_master, REMAINING_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Cannot send information of remaining records");
	}	

	*num_of_events = total;

	return infset;
}

void Free_Map_Paraver_Files (PRVFileSet_t * infset)
{
	int i;

	/* Frees memory allocated by Read_PRV_LocalFile and Read_PRV_RemoteFile */
	for (i = 0; i < infset->nfiles; i++)
		{
			xfree (infset->files[i].first_mapped_p)
			infset->files[i].first_mapped_p = NULL;
		}
}

void Flush_Paraver_Files_binary (PRVFileSet_t *prvfset, int taskid, int depth,
	int tree_fan_out)
{
	if (tree_MasterOfSubtree (taskid, tree_fan_out, depth))
		WriteFileBuffer_flush ( prvfset->files[0].destination );
}

#else /* PARALLEL_MERGE */

PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block)
{
	unsigned long long total = 0;
	PRVFileSet_t *prvfset = NULL;
	unsigned i;

	*num_of_events = total;

	if ((prvfset = malloc (sizeof (PRVFileSet_t))) == NULL)
	{
		perror ("malloc");
		fprintf (stderr, "mpi2prv: Error creating PRV file set\n");
		return 0;
	}

	prvfset->fset = fset;

	/* Master process will have its own files plus references to a single file of every other
	   task (which represents a its set of assigned files) */
	if (0 == taskid)
	{
		prvfset->nfiles = fset->nfiles + numtasks - 1;
		prvfset->records_per_block = records_per_block / prvfset->nfiles;
	}
	else
		prvfset->nfiles = fset->nfiles;


        xmalloc(prvfset->files, nTraces * sizeof(PRVFileItem_t));
	/* Set local files first */
	for (i = 0; i < fset->nfiles; i++)
	{
		prvfset->files[i].mapped_records = 0;
		prvfset->files[i].source = WriteFileBuffer_getFD(fset->files[i].wfb);
		prvfset->files[i].type = LOCAL;
		prvfset->files[i].current_p =
			prvfset->files[i].last_mapped_p =
			prvfset->files[i].first_mapped_p = NULL;
		prvfset->files[i].remaining_records = lseek (prvfset->files[i].source, 0, SEEK_END);
		lseek (prvfset->files[i].source, 0, SEEK_SET);
		if (-1 == prvfset->files[i].remaining_records)
		{
			fprintf (stderr, "mpi2prv: Failed to seek the end of a temporal file\n");
			fflush (stderr);
			exit (0);
		}
		else
			prvfset->files[i].remaining_records /= sizeof(paraver_rec_t);
			
		total += prvfset->files[i].remaining_records;
	}

	*num_of_events = total;

	return prvfset;
}

#endif /* PARALLEL_MERGE */

static void Read_PRV_LocalFile (PRVFileItem_t *file, unsigned records_per_block)
{
	size_t want_to_read;
	ssize_t res;
	char *tmp;
	unsigned nrecords;

	nrecords = MIN(records_per_block,file->remaining_records);
	want_to_read = nrecords * sizeof(paraver_rec_t);

	/* Reuse buffer if size match */
	if (nrecords != file->mapped_records)
	{
		if (file->first_mapped_p != NULL)
			free (file->first_mapped_p);
		file->first_mapped_p = (paraver_rec_t*) malloc (want_to_read);
		file->mapped_records = nrecords;
	}

	if (file->first_mapped_p == NULL)
	{
		perror ("malloc");
		fprintf (stderr, "mpi2prv: Failed to obtain memory for block of %u events (size %Zu)\n", records_per_block, want_to_read);
		fflush (stderr);
		exit (0);
	}
	
	res = read (file->source, file->first_mapped_p, want_to_read);
	if (-1 == res)
	{
		perror ("read");
		fprintf (stderr, "mpi2prv: Failed to read %Zu bytes on local file (result = %Zu)\n", want_to_read, res);
		fflush (stderr);
		exit (0);
	}

	file->current_p = file->first_mapped_p;
	tmp = (char *) file->first_mapped_p;
	tmp = tmp + want_to_read;
	file->last_mapped_p = (paraver_rec_t *) tmp;

	file->remaining_records -= nrecords;
}

#if defined(PARALLEL_MERGE)
/* This is called by the master/root process */
static void Read_PRV_RemoteFile (PRVFileItem_t *file)
{
	int res;
	unsigned int howmany;
	MPI_Status s;
	char *tmp;

	res = MPI_Send (&res, 1, MPI_INT, file->source, ASK_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD); 
	MPI_CHECK(res, MPI_Send, "Failed to ask to a remote task a block of merged events!");

	res = MPI_Recv (&howmany, 1, MPI_UNSIGNED, file->source, HOWMANY_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD, &s);
	MPI_CHECK(res, MPI_Recv, "Failed to receive how many events are on the incoming buffer!");

	if (0 != howmany)
	{
		if (file->first_mapped_p != NULL)
			free (file->first_mapped_p);

		file->first_mapped_p = (paraver_rec_t*) malloc (howmany*sizeof(paraver_rec_t));
		if (file->first_mapped_p == NULL)
		{
			perror ("malloc");
			fprintf (stderr, "mpi2prv: Failed to obtain memory for block of %u remote events\n", howmany);
			fflush (stderr);
			exit (0);
		}

		file->current_p = file->first_mapped_p;
		tmp = (char *) file->first_mapped_p;
		tmp += howmany*sizeof(paraver_rec_t);
		file->last_mapped_p = (paraver_rec_t *) tmp;
		file->remaining_records -= howmany;

		res = MPI_Recv (file->first_mapped_p, howmany*sizeof(paraver_rec_t), MPI_BYTE, file->source, BUFFER_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD, &s);
		MPI_CHECK(res, MPI_Recv, "ERROR! Failed to receive how many events are on the incoming buffer!");
	}
}
#endif /* PARALLEL_MERGE */


paraver_rec_t *GetNextParaver_Rec (PRVFileSet_t * fset)
{
	paraver_rec_t *minimum = NULL, *current = NULL;
	PRVFileItem_t *sfile;
	unsigned int fminimum = 0, i;
	
	/* Check if we have to reload any files */
	for (i = 0; i < fset->nfiles; i++)
		if (fset->files[i].current_p == fset->files[i].last_mapped_p)
			if (fset->files[i].remaining_records > 0)
			{
#if defined(PARALLEL_MERGE)
				/* This can only happen for task = 0 and parallel merger */
				if (fset->files[i].type == REMOTE)
					Read_PRV_RemoteFile (&(fset->files[i]));
				else
#endif
					Read_PRV_LocalFile (&(fset->files[i]), fset->records_per_block);
			}

	for (i = 0; i < fset->nfiles; i++)
	{
		sfile = &(fset->files[i]);

		current = (sfile->current_p == sfile->last_mapped_p) ? NULL : sfile->current_p;

		if (!(minimum) && current)
		{
			minimum = current;
			fminimum = i;
		}
		else if (minimum && current)
		{
			if (minimum->time > current->time)
			{
				minimum = current;
				fminimum = i;
			}
			else if (minimum->time == current->time)
			{
				if (minimum->type < current->type)
				{
					minimum = current;
					fminimum = i;
				}
			}
		}
	}

	sfile = &(fset->files[fminimum]);
	if (sfile->current_p != sfile->last_mapped_p)
		sfile->current_p++;

	return minimum;
}

/******************************************************************************
 ***  Search_CPU_Burst
 ******************************************************************************/

/* Search for the following CPU burst from any file */
static event_t * Search_CPU_Burst (
	FileSet_t * fset, 
	unsigned int *cpu,
	unsigned int *ptask, 
	unsigned int *task, 
	unsigned int *thread )
{
	event_t * minimum = NULL, * current = NULL;
	unsigned int fminimum = 0;
	unsigned int file;
	FileItem_t * sfile;

	for (file = 0; file < fset->nfiles; file++)
	{  
		current = fset->files[file].next_cpu_burst;
		/* Look forward until you find the next CPU burst (or end of file) */
		while ((current < (fset->files[file].last)) && (current->event != CPU_BURST_EV) && (current->event != MPI_STATS_EV))
		{
			current = ++(fset->files[file].next_cpu_burst);
		}
		if (current < (fset->files[file].last)) 
		{
			if (minimum == NULL)
			{
				minimum = current;
				fminimum = file;
			}
			else 
			{ 
				if (TIMESYNC(fset->files[fminimum].ptask-1, fset->files[fminimum].task-1, minimum->time) > TIMESYNC(fset->files[file].ptask-1, fset->files[file].task-1, current->time))
				{
					minimum = current;
					fminimum = file;
				}
			}
		}
	}

	sfile = &(fset->files[fminimum]);
	CurrentObj_FS (sfile, *cpu, *ptask, *task, *thread);
	sfile->next_cpu_burst ++; /* StepOne */

	return minimum;
}

/******************************************************************************
 ***  GetNextEvent_FS_prv
 ******************************************************************************/

static event_t *GetNextEvent_FS_prv (FileSet_t * fset, unsigned int *cpu,
  unsigned int *ptask, unsigned int *task, unsigned int *thread)
{
	event_t *minimum = NULL, *current = NULL;
	unsigned int fminimum = 0, file;
	FileItem_t *sfile;

	for (file = 0; file < fset->nfiles; file++)
	{
		current = Current_FS (&(fset->files[file]));
		while ((current != NULL) && ((current->event == CPU_BURST_EV) || (current->event == MPI_STATS_EV)))
		{
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));
		}
		if (minimum == NULL && current != NULL)
		{
			minimum = current;
			fminimum = file;
		}
		else if (minimum != NULL && current != NULL)
		{
			if (TIMESYNC(fset->files[fminimum].ptask-1, fset->files[fminimum].task-1, minimum->time) > TIMESYNC(fset->files[file].ptask-1, fset->files[file].task-1, current->time))
			{
				minimum = current;
				fminimum = file;
			}
		}
	}
	sfile = &(fset->files[fminimum]);
	CurrentObj_FS (sfile, *cpu, *ptask, *task, *thread);
	StepOne_FS (sfile);

	return minimum;
}

static event_t *GetNextEvent_FS_trf (FileSet_t * fset, unsigned int *cpu,
  unsigned int *ptask, unsigned int *task, unsigned int *thread)
{
	event_t *current = NULL;
	FileItem_t *sfile; 

	current = Current_FS(&(fset->files[fset->active_file]));
	if (NULL == current && fset->active_file < fset->nfiles-1)
	{
		fset->active_file++;
		current = Current_FS (&(fset->files[fset->active_file]));
	}
	sfile = &(fset->files[fset->active_file]);
	CurrentObj_FS (sfile, *cpu, *ptask, *task, *thread);
	StepOne_FS (sfile);

	return current;
}

event_t * GetNextEvent_FS (
	FileSet_t * fset,
	unsigned int *cpu, 
	unsigned int *ptask,
	unsigned int *task, 
	unsigned int *thread)
{
	event_t * ret = NULL;
	static event_t * min_event = NULL, * min_burst = NULL;
	static unsigned int min_event_ptask = 0, min_event_task = 0, min_event_thread = 0, min_event_cpu = 0;
	static unsigned int min_burst_ptask = 0, min_burst_task = 0, min_burst_thread = 0, min_burst_cpu = 0;

	if (PRV_SEMANTICS == fset->traceformat)
	{
		if (Is_FS_Rewound)
		{
			/* Select the first CPU burst event and the first normal event */
			min_event = GetNextEvent_FS_prv (fset, &min_event_cpu, &min_event_ptask, &min_event_task, &min_event_thread);
			min_burst = Search_CPU_Burst (fset, &min_burst_cpu, &min_burst_ptask, &min_burst_task, &min_burst_thread);
			Is_FS_Rewound = FALSE;
		}
		if ((min_event == NULL) && (min_burst == NULL))
		{
			/* No more events to parse */
			ret = NULL;
		}
		else if ((min_event == NULL) || (min_burst != NULL && TIMESYNC(min_burst_ptask-1, min_burst_task-1, min_burst->time) < TIMESYNC(min_event_ptask-1, min_event_task-1, min_event->time)))
		{
			/* Return the minimum CPU burst event */
			ret = min_burst;
			*cpu    = min_burst_cpu;
			*ptask  = min_burst_ptask;
			*task   = min_burst_task;
			*thread = min_burst_thread;
			/* Select the following CPU burst event */
			min_burst = Search_CPU_Burst (fset, &min_burst_cpu, &min_burst_ptask, &min_burst_task, &min_burst_thread);
		}
		else if ((min_burst == NULL) || (min_event != NULL && TIMESYNC(min_event_ptask-1, min_event_task-1, min_event->time) <= TIMESYNC(min_burst_ptask-1, min_burst_task-1, min_burst->time)))
		{
			/* Return the minimum normal event */
			ret = min_event;
			*cpu    = min_event_cpu;
			*ptask  = min_event_ptask;
			*task   = min_event_task;
			*thread = min_event_thread;
			/* Select the following normal event */
			min_event = GetNextEvent_FS_prv (fset, &min_event_cpu, &min_event_ptask, &min_event_task, &min_event_thread);
		}
		else
		{
			/* Should not happen */
			ret = NULL;
		}
	}
	else if (TRF_SEMANTICS == fset->traceformat)
	{
		ret = GetNextEvent_FS_trf (fset, cpu, ptask, task, thread);
	}
	return ret;
}

/******************************************************************************
 ***  Search_MPI_IRECVED
 ******************************************************************************/
event_t *Search_MPI_IRECVED (event_t * current, long long request, FileItem_t * freceive)
{
	event_t *irecved = current;

	freceive->tmp = irecved;

	do
	{
		if (Get_EvEvent (irecved) == MPI_IRECVED_EV)
		{
			if (Get_EvAux (irecved) == request)
			{
				int cancelled = Get_EvValue(irecved);
				if (cancelled)
				{
					return NULL;
				}
				else
				{
					return irecved;
				}
			}
	        }
	}
        while ((irecved = NextRecvG_FS (freceive)) != NULL);

	return NULL;
}

void Rewind_FS (FileSet_t * fs)
{
	unsigned int i;
  
	Is_FS_Rewound = TRUE;

	/* If we are using a circular buffer, rewind to the first useful event */
	for (i = 0; i < fs->nfiles; i++)
	{
		/* Rewind to the first useful event */
		if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_EVENTS))
		{
			/* All pointers are set to the 1st event after the 1st common global op.
			 * This is because the search for the first global op leaves first_glop 
			 * pointing to the EVT_END of the collective. 
			 */ 
			fs->files[i].current = fs->files[i].first_glop ++;
			fs->files[i].next_cpu_burst = fs->files[i].first_glop ++;
			fs->files[i].last_recv = fs->files[i].first_glop ++;
		}
		else if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES))
		{
			/* Pointers are set to the 1st event, but we search for comm matches after the 1st glop */
            fs->files[i].current = fs->files[i].first;
            fs->files[i].next_cpu_burst = fs->files[i].first;
            fs->files[i].last_recv = fs->files[i].first_glop ++;
		}
		else if (!tracingCircularBuffer())
		{
			/* All pointers are set to the 1st event */
			fs->files[i].current = fs->files[i].first;
			fs->files[i].next_cpu_burst = fs->files[i].first;
			fs->files[i].last_recv = fs->files[i].first;
		}
	}
	fs->active_file = 0;
}

unsigned long long EventsInFS (FileSet_t * fs)
{
		unsigned int i;
		unsigned long long tmp = 0;

		for (i = 0; i < fs->nfiles; i++)
			tmp += fs->files[i].num_of_events;

		return tmp;
}

/******************************************************************************
 *** Gestio trivial del buffer circular
 ******************************************************************************/
 
static int circular_behaviour = CIRCULAR_SKIP_MATCHES;
static int max_tag_circular_buffer = -1;
static int circular_buffer_enabled = FALSE;

int getBehaviourForCircularBuffer (void)
{
	return circular_behaviour;
}

int tracingCircularBuffer (void)
{
	/* return max_tag_circular_buffer != 0; */
	return circular_buffer_enabled;
}

int getTagForCircularBuffer (void)
{
	return max_tag_circular_buffer;
}

/*****************************************************************************
 *** Communications matching
 *****************************************************************************/

void MatchComms_ChangeZone(unsigned int ptask, unsigned int task)
{
  task_t *task_info = GET_TASK_INFO(ptask, task);

  task_info->match_zone ++;
}

int MatchComms_GetZone(unsigned int ptask, unsigned int task)
{
  task_t *task_info = GET_TASK_INFO(ptask, task);

  return task_info->match_zone;
}

void MatchComms_On(unsigned int ptask, unsigned int task)
{   
  task_t *task_info = GET_TASK_INFO(ptask, task);

  MatchComms_ChangeZone(ptask, task);

  task_info->MatchingComms = TRUE;
}

void MatchComms_Off(unsigned int ptask, unsigned int task)
{   
  task_t *task_info = GET_TASK_INFO(ptask, task);

  MatchComms_ChangeZone(ptask, task);

  task_info->MatchingComms = FALSE;

  CommunicationQueues_Clear (task_info->send_queue);
  CommunicationQueues_Clear (task_info->recv_queue);
}

int MatchComms_Enabled(unsigned int ptask, unsigned int task)
{   
  task_t *task_info = GET_TASK_INFO(ptask, task);

  return task_info->MatchingComms;
}

/******************************************************************************
 ***  Search_Synchronization_Times
 ******************************************************************************/

int Search_Synchronization_Times (int taskid, int ntasks, FileSet_t * fset,
	UINT64 **io_StartingTimes, UINT64 **io_SynchronizationTimes)
{
#if defined(PARALLEL_MERGE)
	int rc = 0;
	UINT64 *tmp_StartingTimes = NULL;
	UINT64 *tmp_SynchronizationTimes = NULL;
#endif
	unsigned i = 0;
	int total_mpits = 0;
	UINT64 *StartingTimes = NULL;
	UINT64 *SynchronizationTimes = NULL;
	event_t *current = NULL;

	UNREFERENCED_PARAMETER(taskid); /* May be used for debugging purposes */
	UNREFERENCED_PARAMETER(ntasks); /* May be used for debugging purposes */

	Rewind_FS (fset);

	/* Calculate the total number of mpits (note this doesn't check the application they belong!) */
#if defined(PARALLEL_MERGE)
	rc = MPI_Allreduce(&fset->nfiles, &total_mpits, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_CHECK(rc, MPI_Allreduce, "Failed to share total number of mpits!");
#else
	total_mpits = fset->nfiles;
#endif

	/* Allocate space for the synchronization times of each task */
	xmalloc(StartingTimes, total_mpits * sizeof(UINT64));
	memset (StartingTimes, 0, total_mpits * sizeof(UINT64));
	xmalloc(SynchronizationTimes, total_mpits * sizeof(UINT64));
	memset (SynchronizationTimes, 0, total_mpits * sizeof(UINT64));

#if defined(PARALLEL_MERGE)
	xmalloc(tmp_StartingTimes, total_mpits * sizeof(UINT64));
	memset (tmp_StartingTimes, 0, total_mpits * sizeof(UINT64));
	xmalloc(tmp_SynchronizationTimes, total_mpits * sizeof(UINT64));
	memset (tmp_SynchronizationTimes, 0, total_mpits * sizeof(UINT64));
#endif

	for (i=0; i<fset->nfiles; i++)
	{
		/* All threads within a task share the synchronization times */
		if (fset->files[i].thread - 1 == 0)
		{
			current = Current_FS (&(fset->files[i]));
			if (current != NULL)
			{
				FileItem_t *fi = &(fset->files[i]);

				UINT64 mpi_init_end_time, trace_init_end_time, shmem_init_end_time;
				int found_mpi_init_end_time, found_trace_init_end_time, found_shmem_init_end_time;

				found_mpi_init_end_time = found_trace_init_end_time = found_shmem_init_end_time = FALSE;
				mpi_init_end_time = trace_init_end_time = shmem_init_end_time = 0;

				/* Save the starting tracing time of this task */
				StartingTimes[fi->mpit_id] = current->time;

				/* Locate MPI_INIT_EV or TRACE_INIT_EV
				   Be careful not to stop on TRACE_INIT_EV because a MPI_INIT_EV may
				   appear in future and they're always synchronized, not as TRACE_INIT_EV */
				while (current != NULL && !found_mpi_init_end_time)
				{
					if (Get_EvEvent(current) == MPI_INIT_EV && Get_EvValue(current) == EVT_END)
					{
						mpi_init_end_time = Get_EvTime(current);
						found_mpi_init_end_time = TRUE;
					}
					else if (Get_EvEvent(current) == TRACE_INIT_EV && Get_EvValue(current) == EVT_END)
					{
						trace_init_end_time = Get_EvTime(current);
						found_trace_init_end_time = TRUE;
					}
					else if (Get_EvEvent(current) == START_PES_EV && Get_EvValue(current) == EVT_END)
					{
						shmem_init_end_time = Get_EvTime(current);
						found_shmem_init_end_time = TRUE;
					}
					StepOne_FS (&(fset->files[i]));
					current = Current_FS (&(fset->files[i]));
				}

				if (found_mpi_init_end_time)
					SynchronizationTimes[fi->mpit_id] = mpi_init_end_time;
				else if (found_trace_init_end_time)
					SynchronizationTimes[fi->mpit_id] = trace_init_end_time;
				else if (found_shmem_init_end_time)
					SynchronizationTimes[fi->mpit_id] = shmem_init_end_time;
			}
		}
	}


#if defined(PARALLEL_MERGE)
	/* Share information among all tasks */
	MPI_Allreduce( StartingTimes, tmp_StartingTimes, total_mpits, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce( SynchronizationTimes, tmp_SynchronizationTimes, total_mpits, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);

	*io_StartingTimes = tmp_StartingTimes;
	*io_SynchronizationTimes = tmp_SynchronizationTimes;

	xfree(StartingTimes);
	xfree(SynchronizationTimes);

#else
	*io_StartingTimes = StartingTimes;
	*io_SynchronizationTimes = SynchronizationTimes;
#endif

	Rewind_FS (fset);

	return 0;
}

#if defined(HETEROGENEOUS_SUPPORT)
/******************************************************************************
 *** EndianCorrection (fset)
 *** Checks whether the files must be converted into the host endianness
 *****************************************************************************/
static void PerformEndianCorrection (FileItem_t *file)
{
	int cnt;
	unsigned long long i;

	/* Change the endianness in all the fields of event_t in all the records of
	   the file! */	
	for (i = 0; i < file->num_of_events; i++)
	{
		file->first[i].event = bswap32 (file->first[i].event);
		file->first[i].time = bswap64 (file->first[i].time);
		file->first[i].value = bswap64 (file->first[i].value);

		/* If the event is MPI related, change all the fields of the MPI event */
		if (file->first[i].event>=MPI_MIN_EV  && file->first[i].event<=MPI_MAX_EV)
		{
			file->first[i].param.mpi_param.target = 
				bswap32 (file->first[i].param.mpi_param.target);
			file->first[i].param.mpi_param.size = 
				bswap32 (file->first[i].param.mpi_param.size);
			file->first[i].param.mpi_param.tag = 
				bswap32 (file->first[i].param.mpi_param.tag);
			file->first[i].param.mpi_param.comm = 
				bswap32 (file->first[i].param.mpi_param.comm);
			file->first[i].param.mpi_param.aux = 
				bswap32 (file->first[i].param.mpi_param.aux);
		}
		else
		{
			/* In any other case, just change address_t of the structure because
				 params of misc_param_t / omp_param_t are "unioned" and will be changed
			   together */
			file->first[i].param.misc_param.param =
				bswap64 (file->first[i].param.misc_param.param);
		}

		file->first[i].HWCReadSet = bswap32 (file->first[i].HWCReadSet);
		for (cnt = 0; cnt < MAX_HWC; cnt++)
			file->first[i].HWCValues[cnt] = bswap64 (file->first[i].HWCValues[cnt]);
	}
}

/******************************************************************************
 *** SearchEndianCorrection (fset)
 *** Checks whether the files must be converted into the host endianness
 *****************************************************************************/
static int SearchEndianCorrection (FileItem_t *file)
{
	event_t *ptr = file->first;

	while (ptr != file->last)
	{
		if ((ptr->event == APPL_EV) || (ptr->event == bswap32 (APPL_EV)))
			break;
		ptr++;
	}
	if (ptr->event == APPL_EV)
		return 1;
	else if (ptr->event == bswap32(APPL_EV))
		return -1;
	else
		return 0;
}

/******************************************************************************
 *** EndianCorrection (fset)
 *** Checks whether the files must be converted into the host endianness
 *****************************************************************************/
void EndianCorrection (FileSet_t *fset, int numtasks, int taskid)
{
	int somechange = FALSE;
	int *changes;
	int file;
	double last_pct, pct;
#if defined(PARALLEL_MERGE)
	int big_endian, tmp1, tmp2, res;
#endif

#if defined(IS_BIG_ENDIAN)
	fprintf (stdout, "mpi2prv: Host %d endianness: big endian\n", taskid);
#elif defined(IS_LITTLE_ENDIAN)
	fprintf (stdout, "mpi2prv: Host %d endianness: little endian\n", taskid);
#else
# error "Unhandled byte ordering!"
#endif

#if defined(PARALLEL_MERGE)
	/* All tasks must run on the same endianess! */
# if defined(IS_BIG_ENDIAN)
	big_endian = TRUE;
# else
	big_endian = FALSE;
# endif
	res = MPI_Allreduce (&big_endian, &tmp1, 1, MPI_INTEGER, MPI_BOR, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Allreduce, "Matching (step 1) endianess");

	res = MPI_Allreduce (&big_endian, &tmp2, 1, MPI_INTEGER, MPI_BAND, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Allreduce, "Matching (step 2) endianess");

	if (tmp1 != tmp2)
	{
		fprintf (stderr, "mpi2prv: Error! mpi2prv should not be run in heterogeneous environments\n");
		fflush (stderr);
		MPI_Finalize ();
		exit (0);
	}
#endif

	if (taskid == 0)
	{	
		fprintf (stdout, "mpi2prv: Endian correction ... ");
		fflush (stdout);
	}
	
	changes = malloc (fset->nfiles*sizeof(int));
	if (changes == NULL)
	{
		fprintf (stderr, "mpi2prv: ERROR couldn't allocate memory for endian correction!\n");
		exit (1);
	}
		
	/* Just check all the TASKS for their endianness */
	/* Only MASTER thread have APPL_EV info, so search this info at the MASTER
	   and then touch all threads to the required ptask/task */

	last_pct = 0;
	for (file = 0; file < fset->nfiles; file++)
	{
		if (fset->files[file].thread == 1)
		{
			int i;
			int res;
			res = SearchEndianCorrection (&(fset->files[file]));
			if (res == -1)
			{
				/* Modify other threads of this ptask/task */
				for (i = 0; i < fset->nfiles; i++)
					if (fset->files[file].task == fset->files[i].task &&
					    fset->files[file].ptask == fset->files[i].ptask)
					{
						PerformEndianCorrection (&(fset->files[i]));
						changes[i] = TRUE;
					}
			}
			else if (res == 1)
			{
				/* Do not touch other threads of this ptask/task */
				for (i = 0; i < fset->nfiles; i++)
					if (fset->files[file].task == fset->files[i].task &&
					    fset->files[file].ptask == fset->files[i].ptask)
						changes[i] = FALSE;
			}
			else
			{
				/* There isn't an APPL_EV! FAIL! */
				fprintf (stderr, "mpi2prv: Error! Cannot obtain endian information for %d.%d.%d\n",
					fset->files[file].ptask, fset->files[file].task, fset->files[file].thread);
				exit (0);
			}
		}

		if (1 == numtasks)
		{
			pct = ((double) file)/((double) fset->nfiles)*100.0f;
			if (pct > last_pct + 5.0 && pct <= 100.0)
			{
				fprintf (stdout, "%.1lf%% ", pct);
				fflush (stdout);
				last_pct += 5.0;
			}
		}
	}

	if (0 == taskid)
	{
		fprintf (stdout, "done\n");
		fflush (stdout);
	}

	if (1 == numtasks)
	{
		/* Now print which files has been corrected previously */
		fprintf (stdout, "mpi2prv: Endian correction applied to: ");
		fflush (stdout);

		for (file = 0; file < fset->nfiles; file++)
		{
			if (changes[file])
			{
				if (!somechange)
					fprintf (stdout, "%d.%d.%d", fset->files[file].ptask, 
						fset->files[file].task, fset->files[file].thread);
				else
					fprintf (stdout, ",%d.%d.%d", fset->files[file].ptask,
						fset->files[file].task, fset->files[file].thread);
				somechange = TRUE;
			}
		}
		fprintf (stdout, "%s\n", somechange?"":"none file");
		fflush (stdout);
	}
	
#if defined(PARALLEL_MERGE)
	res = MPI_Barrier (MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Barrier, "Synchronizing endianess");
#endif

	free (changes);
}

#endif /* HETEROGENEOUS_SUPPORT */

/******************************************************************************
 ***  GetTraceOptions
 ******************************************************************************/
long long GetTraceOptions (FileSet_t * fset, int numtasks, int taskid)
{
	long long options = TRACEOPTION_NONE;
	event_t *current;

	UNREFERENCED_PARAMETER(numtasks);
	UNREFERENCED_PARAMETER(taskid);

	/* All tasks share the same initialization, so check once only! */
	current = Current_FS (&(fset->files[0]));

	while (current != NULL)
	{
		if ((Get_EvEvent (current) == MPI_INIT_EV && Get_EvValue (current) == EVT_END) ||
		    (Get_EvEvent (current) == TRACE_INIT_EV && Get_EvValue (current) == EVT_END))
		{
			options = Get_EvAux(current);
			break;
		}
		StepOne_FS (&(fset->files[0]));
		current = Current_FS (&(fset->files[0]));
	}
	Rewind_FS (fset);

	return options;
}

void FSet_Forward_To_First_GlobalOp (FileSet_t *fset, int numtasks, int taskid)
{
	event_t *current = NULL;
	unsigned int file = 0;

#if !defined(PARALLEL_MERGE)
	UNREFERENCED_PARAMETER(numtasks);
#endif

	/* Es calcula el temps minim i es guarda el minim de cada fitxer */
	for (file = 0; file < fset->nfiles; file++)
	{
		/* Si el buffer es circular, cal cercar el primer MPI_Barrier de tots
		 * per veure com ho sincronitzem! */
		current = Current_FS (&(fset->files[file]));
		while (current != NULL)
		{
			if (EVENTS_FOR_NUM_GLOBAL_OPS(Get_EvEvent (current)) &&
			   (Get_EvValue (current) == EVT_END) && (Get_EvAux (current) != 0))
				break;
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));
		}
		if (current == NULL)
		{
			if (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_EVENTS)
			{
				/* FIXME this needs better handling for the -D PARALLEL_MERGE */
				fprintf (stderr, "mpi2prv: Error! current == NULL when searching NumGlobalOps on file %d\n", file);
				exit (0);
			}
			else if (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES)
			{
				fprintf (stderr, "mpi2prv: No global operations found on file %d... Communication matching disabled.\n", file);
			}
		}
		else
		{
			max_tag_circular_buffer = MAX(max_tag_circular_buffer, Get_EvAux (current));
		}
	}

#if defined(PARALLEL_MERGE)
	if (numtasks > 1)
	{
		int res;
		unsigned int temp;

		fprintf (stdout, "mpi2prv: Processor %d suggests tag %u to for the circular buffering.\n", taskid, max_tag_circular_buffer);
		fflush (stdout);

		res = MPI_Allreduce (&max_tag_circular_buffer, &temp, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Allreduce, "Failed to share maximum tag id in circular buffer!");

		max_tag_circular_buffer = temp;
	}
#endif

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Tag used for circular buffering: %d\n", max_tag_circular_buffer);
		fflush (stdout);
	}

	/* Es calcula el temps minim i es guarda el minim de cada fitxer */
	for (file = 0; file < fset->nfiles; file++)
	{
		/* Rebobinem aquest fitxer! */
		int local_max = 0;
		int skip_events = 0;
		fset->files[file].current = fset->files[file].first;
		current = Current_FS (&(fset->files[file]));

		/* Si el buffer es circular, cal cercar el primer MPI_Barrier de tots
		 * per veure com ho sincronitzem! */
		while (current != NULL)
		{
			if (EVENTS_FOR_NUM_GLOBAL_OPS(Get_EvEvent(current)) &&
			   (Get_EvValue (current) == EVT_END) && 
			   (Get_EvAux (current) == max_tag_circular_buffer ))
				break;

			if (EVENTS_FOR_NUM_GLOBAL_OPS(Get_EvEvent(current)) &&
			   (Get_EvValue (current) == EVT_END) && 
				 (Get_EvAux (current) != max_tag_circular_buffer ))
			{
				local_max = MAX(Get_EvAux (current), local_max);
			}
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));

			skip_events ++;
		}
		if ((current == NULL) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_EVENTS))
		{
			fprintf (stderr, "Error! current == NULL when searching NumGlobalOps on file %d (local_max = %d)\n",
			file, local_max);
			exit (0);
		}
		else
		{
			fset->files[file].first_glop = current;

			if (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_EVENTS)
			{
				/* Previous events will be skipped after Rewind_FS */
				fset->files[file].num_of_events -= skip_events;
			}
			else if (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES)
			{
				FileItem_t *sfile = &(fset->files[file]);
				unsigned int cpu, ptask, task, thread;
				CurrentObj_FS (sfile, cpu, ptask, task, thread);
				UNREFERENCED_PARAMETER(cpu);
				UNREFERENCED_PARAMETER(thread);

				/* Disable communications matching */
				MatchComms_Off (ptask, task);
			}
		}
	}
	/*
 	 * Es rebobina el fileset perque sino es perdrien tots els
	 * events que ens hem saltat 
	 */
	Rewind_FS (fset);
}

/******************************************************************************
 ***  CheckCircularBufferWhenTracing
 ******************************************************************************/
void CheckCircularBufferWhenTracing (FileSet_t * fset, int numtasks, int taskid)
{
#if defined(PARALLEL_MERGE)
	int res;
#endif
  unsigned int file = 0, circular_buffer = FALSE;
  event_t *current;

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Circular buffer enabled at tracing time? ");
		fflush (stdout);

		current = Current_FS (&(fset->files[file]));
		while ((current != NULL) &&
		((Get_EvEvent (current) != MPI_INIT_EV) || (Get_EvValue (current) != EVT_END)))
		{
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));
		}

		if (current != NULL)
			circular_buffer = Get_EvAux(current) & TRACEOPTION_CIRCULAR_BUFFER;
		else
			circular_buffer = 0;

 		Rewind_FS (fset);
	}

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&circular_buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to share CircularBuffer result!");
#endif

	if (!circular_buffer)
	{
		if (0 == taskid)
		{
			fprintf (stdout, "NO\n");
			fflush (stdout);
		}
	}
	else
	{
		circular_buffer_enabled = TRUE;
		if (0 == taskid)
		{
			fprintf (stdout, "YES\nmpi2prv: Searching required information...\n");
			fflush (stdout);
		}
		FSet_Forward_To_First_GlobalOp (fset, numtasks, taskid);
	}
}

unsigned int GetActiveFile (FileSet_t *fset)
{
	return fset->active_file;
}

