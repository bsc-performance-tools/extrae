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
#ifdef HAVE_ASSERT_H
# include <assert.h>
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
#endif
#include "utils.h"
#include "semantics.h"
#include "cpunode.h"
#include "timesync.h"

#if defined(OS_LINUX)
# ifdef HAVE_BYTESWAP_H
#  include <byteswap.h>
# endif
# define bswap16(x) bswap_16(x)
# define bswap32(x) bswap_32(x)
# define bswap64(x) bswap_64(x)
#elif defined(OS_FREEBSD)
# ifdef HAVE_SYS_ENDIAN_H
#  include <sys/endian.h>
# endif
#elif defined(OS_AIX) || defined (OS_SOLARIS)

unsigned bswap32(unsigned value)
{
  unsigned newValue;
  char* pnewValue = (char*) &newValue;
  char* poldValue = (char*) &value;
 
  pnewValue[0] = poldValue[3];
  pnewValue[1] = poldValue[2];
  pnewValue[2] = poldValue[1];
  pnewValue[3] = poldValue[0];
 
  return newValue;
}

unsigned long long bswap64 (unsigned long long value)
{
  unsigned long long newValue;
  char* pnewValue = (char*) &newValue;
  char* poldValue = (char*) &value;
 
  pnewValue[0] = poldValue[7];
  pnewValue[1] = poldValue[6];
  pnewValue[2] = poldValue[5];
  pnewValue[3] = poldValue[4];
  pnewValue[4] = poldValue[3];
  pnewValue[5] = poldValue[2];
  pnewValue[6] = poldValue[1];
  pnewValue[7] = poldValue[0];
 
  return newValue;
}

#endif

#include "file_set.h"
#include "mpi2out.h"
#include "events.h"
#include "object_tree.h"
#include "HardwareCounters.h"
#include "trace_to_prv.h"

#define EVENTS_FOR_NUM_GLOBAL_OPS(x) \
	((x) == MPI_BARRIER_EV || (x) == MPI_BCAST_EV || (x) == MPI_ALLREDUCE_EV ||  \
	 (x) == MPI_ALLTOALL_EV || (x) == MPI_ALLTOALLV_EV || (x) == MPI_SCAN_EV)

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

  assert (file < fset->nfiles);

  sfile = &(fset->files[file]);
  CurrentObj_FS (sfile, *cpu, *ptask, *task, *thread);
}

#if defined(SAMPLING_SUPPORT) || defined(HAVE_MRNET)
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

int isTaskInMyGroup (FileSet_t *fset, int task)
{
	int i;

	for (i = 0; i < fset->nfiles; i++)
		if (fset->files[i].task-1 == task)
		return TRUE;
	return FALSE;
}

int inWhichGroup (int task, FileSet_t *fset)
{
	int i;

	for (i = 0; i < fset->num_input_files; i++)
		if (fset->input_files[i].task-1 == task)
			return fset->input_files[i].InputForWorker;
	return -1;
}


/******************************************************************************
 ***  AddFile_FS
 ******************************************************************************/

static int AddFile_FS (FileItem_t * fitem, struct input_t *IFile/*, int nfile*/)
{
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
#if defined(HAVE_MRNET)
	char mrn_file_name[PATH_MAX];
	long long mrn_file_size;
	int fd_mrn;
#endif
#if defined(SAMPLING_SUPPORT) || defined(HAVE_MRNET)
	int sort_needed = FALSE;
#endif
	event_t *ptr_last = NULL;

	if (getenv ("MPI2PRV_TMP_DIR") == NULL)
	{
		if (getenv ("TMPDIR") == NULL)
			sprintf (paraver_tmp, "TmpFileXXXXXX");
		else
			sprintf (paraver_tmp, "%s/TmpFileXXXXXX", getenv ("TMPDIR"));
	}
	else
		sprintf (paraver_tmp, "%s/TmpFileXXXXXX", getenv ("MPI2PRV_TMP_DIR"));

	strcpy (trace_file_name, IFile->name);
	fd_trace = fopen (trace_file_name, "r");
	if (NULL == fd_trace)
	{
		perror ("fopen");
		fprintf (stderr, "mpi2prv Error: Opening trace file %s\n", trace_file_name);
		return (-1);
	}
#if defined(SAMPLING_SUPPORT)
	strcpy (sample_file_name, IFile->name);
	sample_file_name[strlen(sample_file_name)-strlen(EXT_MPIT)] = (char) 0; /* remove ".mpit" extension */
	strcat (sample_file_name, EXT_SAMPLE);

	fd_sample = fopen (sample_file_name, "r");
#endif
#if defined(HAVE_MRNET)
	strcpy (mrn_file_name, IFile->name);
	mrn_file_name[strlen(mrn_file_name)-strlen(EXT_MPIT)] = (char) 0; /* remove ".mpit" extension */
	strcat (mrn_file_name, EXT_MRN);

	fd_mrn = open (mrn_file_name, O_RDONLY);
#endif

	ret = fseeko (fd_trace, 0, SEEK_END);
	if (0 != ret)
	{
		fprintf (stderr, "mpi2prv: `fseeko` failed to set file pointer of file %s\n", trace_file_name);
		exit (1);
	}
	trace_file_size = ftello (fd_trace);

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

#if defined(HAVE_MRNET)
	mrn_file_size = (fd_mrn != -1)?lseek (fd_mrn, 0, SEEK_END):0;
#endif

	fitem->size = trace_file_size;
#if defined(SAMPLING_SUPPORT)
	fitem->size += sample_file_size;
#endif
#if defined(HAVE_MRNET)
	fitem->size += mrn_file_size;
#endif

#if 0
	if (LimitOfEvents != 0)
		fitem->size = MIN(fitem->size, LimitOfEvents*sizeof(event_t));
#endif
	fitem->num_of_events = (fitem->size>0)?fitem->size/sizeof(event_t):0;

	rewind (fd_trace);
#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
		rewind (fd_sample);
#endif
#if defined(HAVE_MRNET)
	if (fd_mrn != -1) lseek (fd_mrn, 0, SEEK_SET);
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
#if defined(HAVE_MRNET)
		extra = (mrn_file_size % sizeof (event_t));
		if (extra != 0)
			printf ("PANIC! MRNet file %s is %d bytes too big!\n", mrn_file_name, extra);
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
		fprintf (stderr, "mpi2prv:        returned %d (instead of %d)\n", res, trace_file_size);
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
			fprintf (stderr, "mpi2prv:        returned %d (instead of %d)\n", res, sample_file_size);
			exit (1);
		}
	}
	sort_needed = sample_file_size > 0;
	ptr_last += (sample_file_size/sizeof(event_t));
#endif
#if defined(HAVE_MRNET)
	if (fd_mrn != -1)
	{
		res = read (fd_mrn, ptr_last, mrn_file_size);
		if (res != mrn_file_size)
		{
			fprintf (stderr, "mpi2prv: `read` failed to read from file %s\n", mrn_file_name);
			fprintf (stderr, "mpi2prv:        returned %d (instead of %d)\n", res, mrn_file_size);
			exit (1);
		}
	}
	if (mrn_file_size > 0) sort_needed = TRUE;
	ptr_last += (mrn_file_size/sizeof(event_t));	
#endif

#if defined(SAMPLING_SUPPORT) || defined(HAVE_MRNET)
	if (sort_needed)
		qsort (fitem->first, fitem->num_of_events, sizeof(event_t), event_timing_sort);
#endif

	/* We no longer need this/these descriptor/s */
	fclose (fd_trace);
#if defined(SAMPLING_SUPPORT)
	if (NULL != fd_sample)
		fclose (fd_sample);
#endif
#if defined(HAVE_MRNET)
	close (fd_mrn);
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

	obj_table[fitem->ptask-1].tasks[IFile->task-1].threads[IFile->thread-1].file = fitem;

	CommunicationQueues_Init (&(fitem->send_queue), &(fitem->recv_queue));

	/* Make a temporal name for a file */	
	if (mkstemp (paraver_tmp) == -1)
	{
		perror ("mkstemp");
		fprintf (stderr, "mpi2prv: Unable to create temporal file using mkstemp");
		fflush (stderr);
		exit (-1);
	}

	/* Create a buffered file with 512 entries of paraver_rec_t */
	fitem->wfb = WriteFileBuffer_new (paraver_tmp, 512, sizeof(paraver_rec_t));

	/* Remove the created file... while we don't die, it won't be removed */
	unlink (paraver_tmp);

	return 0;
}

/******************************************************************************
 ***  Create_FS
 ******************************************************************************/

FileSet_t *Create_FS (unsigned long nfiles, struct input_t * IFiles, int idtask, int trace_format)
{
	int file;
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
	fset->nfiles = 0;
	for (file = 0; file < nfiles; file++)
		if (IFiles[file].InputForWorker == idtask)
		{
			fitem = &(fset->files[fset->nfiles]);
			if (AddFile_FS (fitem, &(IFiles[file])/*, fset->nfiles + 1*/) != 0)
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

PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block)
{
#if defined(PARALLEL_MERGE)
	int res;
#endif
	unsigned long long total = 0;
	PRVFileSet_t *prvfset = NULL;
	int i;

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

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&(prvfset->records_per_block), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to share number of records per block and task!");
#endif

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
		if (-1 == prvfset->files[i].remaining_records)
		{
			fprintf (stderr, "mpi2prv: Failed to seek the end of a temporal file\n");
			fflush (stderr);
			exit (0);
		}
		else
			prvfset->files[i].remaining_records /= sizeof(paraver_rec_t);
			
		total += prvfset->files[i].remaining_records;
		lseek (prvfset->files[i].source, 0, SEEK_SET);
	}

#if defined(PARALLEL_MERGE)
	if (0 == taskid)
	{
		/* Set remote files now (if exist), receive how many events they have */
		for (i = 0; i < numtasks-1; i++)
		{
			MPI_Status s;

			prvfset->files[fset->nfiles+i].mapped_records = 0;
			prvfset->files[fset->nfiles+i].source = i+1;
			prvfset->files[fset->nfiles+i].type = REMOTE;
			prvfset->files[fset->nfiles+i].current_p =
				prvfset->files[fset->nfiles+i].last_mapped_p =
				prvfset->files[fset->nfiles+i].first_mapped_p = NULL;

			res = MPI_Recv (&(prvfset->files[fset->nfiles+i].remaining_records), 1, MPI_LONG_LONG, i+1, REMAINING_TAG, MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Cannot receive information of remaining records");

			total += prvfset->files[fset->nfiles+i].remaining_records;
		}
	}
	else
	{
		/* NON ROOT WORK,
		   send how many events are on this slave */
		res = MPI_Send (&total, 1, MPI_LONG_LONG, 0, REMAINING_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Cannot send information of remaining records");
	}
#endif /* PARALLEL_MERGE */

	*num_of_events = total;

	return prvfset;
}

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
		fprintf (stderr, "mpi2prv: Failed to obtain memory for block of %u events (size %u)\n", records_per_block, want_to_read);
		fflush (stderr);
		exit (0);
	}
	
	res = read (file->source, file->first_mapped_p, want_to_read);
	if (-1 == res)
	{
		perror ("read");
		fprintf (stderr, "mpi2prv: Failed to read %u bytes on local file (result = %u)\n", want_to_read, res);
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
static void Read_PRV_RemoteFile (PRVFileItem_t *file, unsigned records_per_block, int taskid)
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


paraver_rec_t *GetNextParaver_Rec (PRVFileSet_t * fset, int taskid)
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
					Read_PRV_RemoteFile (&(fset->files[i]), fset->records_per_block, fset->files[i].source);
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
				if (TIMESYNC(fset->files[fminimum].task - 1, minimum->time) > TIMESYNC(fset->files[file].task - 1, current->time))
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
			if (TIMESYNC(fset->files[fminimum].task - 1, minimum->time) > TIMESYNC(fset->files[file].task - 1, current->time))
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
#warning "We could close unused files so as to free mem!"
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
		else if ((min_event == NULL) || (min_burst != NULL && TIMESYNC(min_burst_task-1, min_burst->time) < TIMESYNC(min_event_task-1, min_event->time)))
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
		else if ((min_burst == NULL) || (min_event != NULL && TIMESYNC(min_event_task-1, min_event->time) <= TIMESYNC(min_burst_task-1, min_burst->time)))
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
	/* freceive->tmp = freceive->first; */

	if (Get_EvEvent (irecved) == MPI_IRECVED_EV)
		if (Get_EvAux (irecved) == request)
			return irecved;

	while ((irecved = NextRecvG_FS (freceive)) != NULL)
		if (Get_EvEvent (irecved) == MPI_IRECVED_EV)
			if (Get_EvAux (irecved) == request)
				return irecved;
	return NULL;
}

/******************************************************************************
 ***  Search_PACX_IRECVED
 ******************************************************************************/
event_t *Search_PACX_IRECVED (event_t * current, long long request, FileItem_t * freceive)
{
	event_t *irecved = current;

	freceive->tmp = irecved;
	/* freceive->tmp = freceive->first; */

	if (Get_EvEvent (irecved) == PACX_IRECVED_EV)
		if (Get_EvAux (irecved) == request)
			return irecved;

	while ((irecved = NextRecvG_FS (freceive)) != NULL)
		if (Get_EvEvent (irecved) == PACX_IRECVED_EV)
			if (Get_EvAux (irecved) == request)
				return irecved;
	return NULL;
}

#if defined(DEAD_CODE)
/******************************************************************************
 ***  SearchRecvEvent_FS
 ******************************************************************************/

int SearchRecvEvent_FS (FileSet_t *fset, unsigned int ptask, unsigned int receiver,
	unsigned int sender, unsigned int tag, event_t ** recv_begin, event_t ** recv_end)
{
  RecvQ_t *recv_queue;
  RecvQ_t *recv_rec;
  FileItem_t *freceive;
  event_t *current, *tmp_recv_begin = NULL, *tmp_sendrecv_begin = NULL, *recved;

  *recv_begin = NULL;
  *recv_end = NULL;

	if (!isTaskInMyGroup (fset, receiver))
		return 1;

  /*
   * Compte amb aquest acces, en aquest cas va perque receiver es mou
   * entre [0..n-1] i perque el fitxer "i" es del receiver "i". En un
   * cas general potser no coincideix.
   */
  freceive = obj_table[ptask - 1].tasks[receiver].threads[0].file;

  recv_queue = Queue_FS (freceive);

  recv_rec = QueueSearch ((void *) freceive, recv_queue, tag, sender);

  if (recv_rec != NULL)
  {
    *recv_begin = GetRecv_RecordQ (recv_rec, RECV_BEGIN_RECORD);
    *recv_end = GetRecv_RecordQ (recv_rec, RECV_END_RECORD);
    Remove_RecvQ (recv_rec);
  }
  else
  {
    current = NextRecv_FS (freceive);
    while (current != NULL)
    {
      if ((Get_EvEvent (current) == RECV_EV) &&
          (Get_EvValue (current) == EVT_BEGIN))
      {
        tmp_recv_begin = current;
      }
			else if ((Get_EvEvent (current) == SENDRECV_EV) &&
               (Get_EvValue (current) == EVT_BEGIN))
			{
				tmp_sendrecv_begin = current;
			}
			else if ((Get_EvEvent (current) == SENDRECV_REPLACE_EV) &&
               (Get_EvValue (current) == EVT_BEGIN))
			{
				tmp_sendrecv_begin = current;
			}
      else if (((Get_EvEvent (current) == IRECV_EV) &&
                (Get_EvValue (current) == EVT_END)) ||
               ((Get_EvEvent (current) == PERSIST_REQ_EV)
                && (Get_EvValue (current) == IRECV_EV)))
      {
        recved = SearchIRECVED (current, Get_EvAux(current), freceive);
        if (recved == NULL)
				{
          printf ("No IRECVED!!!: ");
		  		break;
				}
        if ((Get_EvTarget (recved) == sender) && (Get_EvTag (recved) == tag))
        {
          /*
           * We have already encountered the receive records
           */
          *recv_begin = current;
          *recv_end = recved;
          break;
        }
        else
        {
          /*
           * Those receive records don't match with the sent.
           * We will store it in the queue.
           */
          recv_rec = Alloc_RecvQ_Item ();
/*		     recv_rec = (RecvQ_t *)malloc(sizeof(RecvQ_t));*/
          SetRecv_RecordQ (recv_rec, current, RECV_BEGIN_RECORD);
          SetRecv_RecordQ (recv_rec, recved, RECV_END_RECORD);

          Queue_RecvQ (recv_queue, recv_rec);
        }
      }
      else if ((Get_EvEvent (current) == RECV_EV) &&
               (Get_EvValue (current) == EVT_END))
      {
        if ((Get_EvTarget (current) == sender) &&
            (Get_EvTag (current) == tag))
        {
          /*
           * We have already encountered the receive records
           */
          *recv_begin = tmp_recv_begin;
          *recv_end = current;
          break;
        }
        else
        {
          /*
           * Those receive records don't match with the sent.
           * We will store it in the queue.
           */
          recv_rec = Alloc_RecvQ_Item ();
          SetRecv_RecordQ (recv_rec, tmp_recv_begin, RECV_BEGIN_RECORD);
          SetRecv_RecordQ (recv_rec, current, RECV_END_RECORD);
          Queue_RecvQ (recv_queue, recv_rec);
        }
      }
#if !defined(AVOID_SENDRECV)
      else if ((Get_EvEvent (current) == SENDRECV_EV) &&
               (Get_EvValue (current) == EVT_END))
      {
        if ((Get_EvTarget (current) == sender) &&
            (Get_EvTag (current) == tag))
        {
          /*
           * We have already encountered the receive records
           */
          *recv_begin = tmp_sendrecv_begin;
          *recv_end = current;
          break;
        }
        else
        {
          /*
           * Those receive records don't match with the sent.
           * We will store it in the queue.
           */
          recv_rec = Alloc_RecvQ_Item ();
          SetRecv_RecordQ (recv_rec, tmp_sendrecv_begin, RECV_BEGIN_RECORD);
          SetRecv_RecordQ (recv_rec, current, RECV_END_RECORD);
          Queue_RecvQ (recv_queue, recv_rec);
        }
			}
      else if ((Get_EvEvent (current) == SENDRECV_REPLACE_EV) &&
               (Get_EvValue (current) == EVT_END))
      {
        if ((Get_EvTarget (current) == sender) &&
            (Get_EvTag (current) == tag))
        {
          /*
           * We have already encountered the receive records
           */
          *recv_begin = tmp_sendrecv_begin;
          *recv_end = current;
          break;
        }
        else
        {
          /*
           * Those receive records don't match with the sent.
           * We will store it in the queue.
           */
          recv_rec = Alloc_RecvQ_Item ();
          SetRecv_RecordQ (recv_rec, tmp_sendrecv_begin, RECV_BEGIN_RECORD);
          SetRecv_RecordQ (recv_rec, current, RECV_END_RECORD);
          Queue_RecvQ (recv_queue, recv_rec);
        }
     }
#endif
      current = NextRecv_FS (freceive);
    }
  }
	return 0;
}
#endif

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
			/* All pointers are set to the 1st glop */
			fs->files[i].current = fs->files[i].first_glop;
			fs->files[i].next_cpu_burst = fs->files[i].first_glop;
			fs->files[i].last_recv = fs->files[i].first_glop;
		}
		else if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES))
		{
			/* Pointers are set to the 1st event, but we search for comm matches after the 1st glop */
            fs->files[i].current = fs->files[i].first;
            fs->files[i].next_cpu_burst = fs->files[i].first;
            fs->files[i].last_recv = fs->files[i].first_glop;
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

void MatchComms_On(unsigned int ptask, unsigned int task, unsigned int thread)
{   
    struct thread_t * thread_info;
    thread_info = GET_THREAD_INFO(ptask, task, thread);
    thread_info->MatchingComms = TRUE;
}

void MatchComms_Off(unsigned int ptask, unsigned int task, unsigned int thread)
{   
    struct thread_t * thread_info;
    thread_info = GET_THREAD_INFO(ptask, task, thread);
    thread_info->MatchingComms = FALSE;
}

int MatchComms_Enabled(unsigned int ptask, unsigned int task, unsigned int thread)
{   
    struct thread_t * thread_info;
    thread_info = GET_THREAD_INFO(ptask, task, thread);
    return thread_info->MatchingComms;
}

/******************************************************************************
 ***  Search_Synchronization_Times
 ******************************************************************************/

int Search_Synchronization_Times (FileSet_t * fset, UINT64 **io_StartingTimes, UINT64 **io_SynchronizationTimes)
{
#if defined(PARALLEL_MERGE)
	int rc = 0;
#endif
	int i = 0;
	int total_mpits = 0;
	int mpit_taskid = 0;
	int num_tasks = 0, sum_num_tasks = 0;
	UINT64 *StartingTimes = NULL;
	UINT64 *SynchronizationTimes = NULL;
	event_t *current = NULL;

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
	bzero(StartingTimes, total_mpits * sizeof(UINT64));

	xmalloc(SynchronizationTimes, total_mpits * sizeof(UINT64));
	bzero(SynchronizationTimes, total_mpits * sizeof(UINT64));

	for (i=0; i<fset->nfiles; i++)
	{
		/* All threads within a task share the synchronization times */
		if (fset->files[i].thread - 1 == 0)
		{
			num_tasks ++;

			/* Which task is this mpit? */
			mpit_taskid = fset->files[i].task - 1;

			current = Current_FS (&(fset->files[i]));
			if (current != NULL)
			{
				/* Save the starting tracing time of this task */
				StartingTimes[mpit_taskid] = current->time;

				/* Locate the MPI_Init end event */
				while ((current != NULL) && ((Get_EvEvent(current) != MPI_INIT_EV) || (Get_EvValue(current) != EVT_END)))
				{
					StepOne_FS (&(fset->files[i]));
					current = Current_FS (&(fset->files[i]));
				}
				if (current != NULL)
				{
					/* Save the synchronization time (MPI_Init end) of this task */
					SynchronizationTimes[mpit_taskid] = current->time;
				}
			}
		}
	}

#if defined(PARALLEL_MERGE)
	/* Share information among all tasks */
	xmalloc(*io_StartingTimes, total_mpits * sizeof(UINT64));
	bzero(*io_StartingTimes, total_mpits * sizeof(UINT64));

	xmalloc(*io_SynchronizationTimes, total_mpits * sizeof(UINT64));
	bzero(*io_SynchronizationTimes, total_mpits * sizeof(UINT64));

	rc = MPI_Allreduce(StartingTimes, *io_StartingTimes, total_mpits, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_CHECK(rc, MPI_Allreduce, "Failed to share starting times information!");

	rc = MPI_Allreduce(SynchronizationTimes, *io_SynchronizationTimes, total_mpits, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_CHECK(rc, MPI_Allreduce, "Failed to share synchronization times information!");
	
	xfree(StartingTimes);
	xfree(SynchronizationTimes);

	/* Every process has a subset of mpits => num_tasks has to be aggregated */
	rc = MPI_Allreduce(&num_tasks, &sum_num_tasks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	*io_StartingTimes = StartingTimes;
	*io_SynchronizationTimes = SynchronizationTimes;

	sum_num_tasks = num_tasks;
#endif

	Rewind_FS (fset);

	/*return total_mpits;*/
	return sum_num_tasks;
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

	/* All tasks share the same initialization, so check once only! */
	current = Current_FS (&(fset->files[0]));

	while (current != NULL)
	{
		if (Get_EvEvent (current) == MPI_INIT_EV &&
		    Get_EvValue (current) == EVT_END)
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


#if defined(DEAD_CODE)
/******************************************************************************
 ***  CheckBursts
 ******************************************************************************/

int CheckBursts (FileSet_t * fset, int numtasks, int taskid)
{
#if defined(PARALLEL_MERGE)
	int res;
#endif
	unsigned int file = 0, lookahead = 100;
	int result = FALSE;
	event_t *current;

	/* This is just informational! */

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Bursts library ... ");
		fflush (stdout);

		/* All tasks share the same initialization, so check once only! */
		current = Current_FS (&(fset->files[file]));
		while ((current != NULL) &&
		((Get_EvEvent (current) != MPI_INIT_EV) || (Get_EvValue (current) != EVT_END)))
		{
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));
		}

		if (current != NULL)
			if (Get_EvAux(current) & TRACEOPTION_BURSTS)
			{
				fprintf (stdout, " yes! (by header-info)\n");
				Rewind_FS (fset);
				return TRUE;
			}

		/* If we're here, then we didn't found the header, or the header didn't
		   know about the BURSTS! ... Just to have backward compatibility, check
		   for some records inside the MPITs */
		while (current != NULL && lookahead > 0)
		{
			if (Get_EvEvent(current) == CPU_BURST_EV)
			{
				fprintf (stdout, " yes! (by content)\n");
				Rewind_FS (fset);
				result = TRUE;
			}
			StepOne_FS (&(fset->files[file]));
			current = Current_FS (&(fset->files[file]));
			lookahead--;
		}

		fprintf (stdout, " NO\n");
		fflush (stdout);

		Rewind_FS (fset);
	}

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to share CheckBursts result!");
#endif

	return result;
}
#endif


void FSet_Forward_To_First_GlobalOp (FileSet_t *fset, int numtasks, int taskid)
{
	event_t *current = NULL;
	unsigned int file = 0;

	/* Es calcula el temps minim i es guarda el minim de cada fitxer */
	for (file = 0; file < fset->nfiles; file++)
	{
		/* Si el buffer es circular, cal cercar el primer MPI_Barrier de tots
		 * per veure com ho sincronitzem! */
		current = Current_FS (&(fset->files[file]));
		while (current != NULL)
		{
			if (EVENTS_FOR_NUM_GLOBAL_OPS(Get_EvEvent (current)) &&
			   (Get_EvValue (current) == EVT_BEGIN) && (Get_EvAux (current) != 0))
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
			   (Get_EvValue (current) == EVT_BEGIN) && 
			   (Get_EvAux (current) == max_tag_circular_buffer ))
				break;

			if (EVENTS_FOR_NUM_GLOBAL_OPS(Get_EvEvent(current)) &&
			   (Get_EvValue (current) == EVT_BEGIN) && 
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

				/* Disable communications matching */
				MatchComms_Off (ptask, task, thread);
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
		if (0 == taskid)
		{
			circular_buffer_enabled = TRUE;
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

#ifdef MPI_PHYSICAL_COMM
#include "MPI_Physical.h"
/*****************************************************************************
 **  Function name: BuscaComunicacionsFisiques ( RecTraceSet ) 
 **  Description:   Busca informacio de les comunicacions fisiques presents
 **                 a la trasa. Generara dues taules (una amb els sends
 **                 fisics i l'altre amb els receives fisics.
 *****************************************************************************/

int *num_phys_Sends, *num_phys_Receives, *num_log2phys_tag;
InformacioFisica **phys_Sends, **phys_Receives;

void BuscaComunicacionsFisiques (FileSet_t * TraceSet)
{
  int ptask, task, thread, cpu, ii;
  int SkipEvent, filter;
  int sum_phys_sends, sum_phys_recvs, max_send_value;
  int numTraces = num_Files_FS (TraceSet);
  event_t *current = NULL;
  event_t **prevEvent;

  /************************************************************************
   ** Primer mirem quants events hi ha!
   ***********************************************************************/

  num_phys_Sends = (int *) malloc (sizeof (int) * numTraces);
  num_phys_Receives = (int *) malloc (sizeof (int) * numTraces);
  num_log2phys_tag = (int *) malloc (sizeof (int) * numTraces);
  if (num_phys_Sends == NULL ||
      num_phys_Receives == NULL || num_log2phys_tag == NULL)
  {
    fprintf (stderr,
             "Error while allocating num_phys_Sends/num_phys_Receives/num_log2phys_tag\n");
    exit (0);
  }
  bzero (num_phys_Sends, sizeof (int) * numTraces);
  bzero (num_phys_Receives, sizeof (int) * numTraces);
  bzero (num_log2phys_tag, sizeof (int) * numTraces);

  fprintf (stderr, "Analyzing physical communications...\n");
  fflush (stderr);

  Rewind_FS (TraceSet);

  for (ii = 0; ii < numTraces; ii++)
  {
    current = Current_FS (&(TraceSet->files[ii]));
    do
    {
      int Type = Get_EvEvent (current);
      int Value = Get_EvValue (current);
      int subValue = Get_EvMiscParam (current);
      int task = TraceSet->files[ii].task;

      if (IsMISC (Type, &filter))
      {
        if (Value == TAG_SND_FISIC)
          if (subValue > num_phys_Sends[task - 1])
            num_phys_Sends[task - 1] = subValue;

        if (Value == TAG_RCV_FISIC)
          if (subValue > num_phys_Receives[task - 1])
            num_phys_Receives[task - 1] = subValue;

        if (Value == TAG_RCV_F_L)
          if (subValue > num_log2phys_tag[task - 1])
            num_log2phys_tag[task - 1] = subValue;
      }

      StepOne_FS (&(TraceSet->files[ii]));
      current = Current_FS (&(TraceSet->files[ii]));
    }
    while (current);
  }

  fprintf (stderr, "Physical&Logical communications briefing:\n");
  for (ii = 0; ii < numTraces; ii++)
    fprintf (stderr,
             "task %02d => { #send = %d / #precv = %d / #lrecv = %d }\n",
             ii + 1, num_phys_Sends[ii], num_phys_Receives[ii],
             num_log2phys_tag[ii]);
  fprintf (stderr, "\n");
  fflush (stderr);

  sum_phys_sends = sum_phys_recvs = 0;
  for (ii = 0; ii < numTraces; ii++)
  {
    sum_phys_sends += num_phys_Sends[ii];
    sum_phys_recvs += num_log2phys_tag[ii];
  }

  if (sum_phys_sends != sum_phys_recvs)
  {
    fprintf (stderr,
             "Error: Sum(logical Sends) differs from Sum(logical Recvs)\n");
    fflush (stderr);
  }

  Rewind_FS (TraceSet);

  fprintf (stderr, "Reordering physical communications\n");
  fflush (stderr);

  /************************************************************************
   ** Ara els agafem i els assignem a una taula
   ***********************************************************************/
  phys_Sends =
    (InformacioFisica **) malloc (sizeof (InformacioFisica) * numTraces);
  phys_Receives =
    (InformacioFisica **) malloc (sizeof (InformacioFisica *) * numTraces);
  if (phys_Sends == NULL || phys_Receives == NULL)
  {
    fprintf (stderr, "Error while allocating phys_Sends/phys_Receives\n");
    exit (0);
  }
  for (ii = 0; ii < numTraces; ii++)
  {
    phys_Sends[ii] =
      (InformacioFisica *) malloc (sizeof (InformacioFisica) *
                                   (num_phys_Sends[ii] + 1));
    phys_Receives[ii] =
      (InformacioFisica *) malloc (sizeof (InformacioFisica) *
                                   (num_phys_Receives[ii] + 1));

    if (phys_Sends[ii] == NULL || phys_Receives[ii] == NULL)
    {
      fprintf (stderr,
               "Error while allocating phys_Sends[%d]/phys_Receives[%d]n", ii,
               ii);
      exit (0);
    }
    else
    {
      bzero (phys_Sends[ii],
             sizeof (InformacioFisica) * (num_phys_Sends[ii] + 1));
      bzero (phys_Receives[ii],
             sizeof (InformacioFisica) * (num_phys_Receives[ii] + 1));
    }
  }

  prevEvent = (event_t **) malloc (sizeof (event_t *) * numTraces);
  if (prevEvent == NULL)
  {
    fprintf (stderr,
             "Error while allocating prevEvent/matchFound_s/matchFound_r\n");
    exit (0);
  }
  bzero (prevEvent, sizeof (event_t *) * numTraces);

  /*
   * Que fa aquest bucle.
   * 
   * Necessitem omplir les taules que relacionen
   * #send/receive -> temps fisic / temps entrada rutina.
   * 
   * En els sends, el send fisic sempre esta entre l'entrada i la sortida.
   * Per tant, cal guardar l'event anterior de cada tasca i marcar d'alguna
   * forma que el seguent event tambe servira per omplir la taula.
   * 
   * Pels receives, es diferent. Primer tenim un event que indica quan es
   * reb el receive fisic -- pot ser a qualsevol lloc de la trasa. Tambe
   * hi ha un event que indica si aquell receive correspon a una P2P. En
   * cas de trobar-lo, cal desar els punts d'entrada d'aquesta  funcio!
   * 
   */
  current = GetNextEvent_FS (TraceSet, &cpu, &ptask, &task, &thread);

  do
  {
    int Type = Get_EvEvent (current);
    int Value = Get_EvValue (current);
    int subValue = Get_EvMiscParam (current);
    UINT64 temps = Get_EvTime (current);

    SkipEvent = FALSE;

    if (IsMISC (Type, &filter))
    {
      if (Value == TAG_SND_FISIC && Type == USER_EV
          && prevEvent[task - 1] != NULL)
      {
        phys_Sends[task - 1][subValue].Temps = temps;
        phys_Sends[task - 1][subValue].Temps_entrada =
          Get_EvTime (prevEvent[task - 1]);
        phys_Sends[task - 1][subValue].Valid = TRUE;

#if defined (DEBUG_MPI_PHYSICAL)
        printf ("task %d: TAG_SND_FISIC #%d at time %lld\n", task, subValue,
                temps);
#endif
      }

      if (Value == TAG_RCV_FISIC && Type == USER_EV)
      {
        phys_Receives[task - 1][subValue].Temps = temps;
#if defined (DEBUG_MPI_PHYSICAL)
        printf ("task %d: TAG_RCV_FISIC #%d at time %lld\n", task, subValue,
                temps);
#endif
      }

      if (Value == TAG_RCV_F_L && Type == USER_EV
          && prevEvent[task - 1] != NULL)
      {
        phys_Receives[task - 1][subValue].Valid = TRUE;
        phys_Receives[task - 1][subValue].Temps_entrada =
          Get_EvTime (prevEvent[task - 1]);

#if defined (DEBUG_MPI_PHYSICAL)
        printf ("task %d: TAG_RCV_F_L #%d at time %lld\n", task, subValue,
                temps);
#endif
      }
    }

    /*
     * Hem d'ignorar els events que identifiquen la comunicacio fisica
     * a l'hora de coneixer l'event precedent de la task. 
     */

    SkipEvent = !(IsMPI (Type, &filter));

    if (!SkipEvent)
      prevEvent[task - 1] = current;

    current = GetNextEvent_FS (TraceSet, &cpu, &ptask, &task, &thread);
  }
  while (current);

  Rewind_FS (TraceSet);
}

#endif
