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
#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif

#include <mpi.h>
#include "mpi-tags.h"
#include "mpi-aux.h"
#include "parallel_merge_aux.h"
#include "mpi_comunicadors.h"

#include "mpi_prv_events.h"
#include "pthread_prv_events.h"
#include "omp_prv_events.h"
#include "misc_prv_events.h"
#include "cuda_prv_events.h"
#include "addr2info.h"
#include "options.h"

//#define DEBUG

struct PendingCommunication_t
{
	int sender, recver, tag, descriptor, match, match_zone;
	off_t offset;
};
struct PendingComms_t 
{
	struct PendingCommunication_t *data;
	int count, size;
};


static struct ForeignRecv_t **myForeignRecvs;
static int *myForeignRecvs_count;
static char **myForeignRecvs_used;

struct IntraCommunicator_t
{
	int *tasks;
	int type;
	int task;
	int ptask;
	int id;
	int ntasks;
};
struct IntraCommunicators_t
{
	struct IntraCommunicator_t *comms;
	int count, size;
};
struct InterCommunicator_t
{
	int task;
	int ptask;
	int id;
	int commids[2];
	int leaders[2];
};
struct InterCommunicators_t
{
	struct InterCommunicator_t *comms;
	int count, size;
};
static struct IntraCommunicators_t IntraCommunicators;
static struct InterCommunicators_t InterCommunicators;

static struct PendingComms_t PendingComms;
static struct ForeignRecvs_t *ForeignRecvs;

#define FOREIGN_RECV_RESIZE_STEP ((1024*1024)/sizeof(struct ForeignRecv_t))
#define INTRA_COMMUNICATORS_RESIZE_STEP ((1024*1024)/sizeof(struct IntraCommunicator_t))
#define INTER_COMMUNICATORS_RESIZE_STEP ((1024*1024)/sizeof(struct InterCommunicator_t))
#define PENDING_COMM_RESIZE_STEP ((1024*1024)/sizeof(struct PendingCommunication_t))

void ParallelMerge_AddIntraCommunicator (int ptask, int task, int type, int id, int ntasks, int *tasks)
{
	int count = IntraCommunicators.count;

	if (IntraCommunicators.count == IntraCommunicators.size)
	{
		IntraCommunicators.size += INTRA_COMMUNICATORS_RESIZE_STEP;
		IntraCommunicators.comms = (struct IntraCommunicator_t*)
			realloc (IntraCommunicators.comms,
			IntraCommunicators.size*sizeof(struct IntraCommunicator_t));
	}
	IntraCommunicators.comms[count].ptask = ptask;
	IntraCommunicators.comms[count].task = task;
	IntraCommunicators.comms[count].type = type;
	IntraCommunicators.comms[count].id = id;
	IntraCommunicators.comms[count].ntasks = ntasks;
	if (MPI_COMM_WORLD_ALIAS != type && MPI_COMM_SELF_ALIAS != type)
	{
		int i;

		IntraCommunicators.comms[count].tasks = (int*) malloc (sizeof(int)*ntasks);
		if (NULL == IntraCommunicators.comms[count].tasks)
		{
			fprintf (stderr, "mpi2prv: ERROR! Unable to store communicator information\n");
			fflush (stderr);
			exit (-1);
		}
		for (i = 0; i < ntasks; i++)
			IntraCommunicators.comms[count].tasks[i] = tasks[i];
	}
	else
		IntraCommunicators.comms[count].tasks = NULL;

	IntraCommunicators.count++;
}

void ParallelMerge_AddInterCommunicator (int ptask, int task, int id, int comm1,
	int leader1, int comm2, int leader2)
{
	int count = InterCommunicators.count;

	if (InterCommunicators.count == InterCommunicators.size)
	{
		InterCommunicators.size += INTER_COMMUNICATORS_RESIZE_STEP;
		InterCommunicators.comms = (struct InterCommunicator_t*)
			realloc (InterCommunicators.comms,
			InterCommunicators.size*sizeof(struct InterCommunicator_t));
	}

	InterCommunicators.comms[count].id = id;
	InterCommunicators.comms[count].task  = task;
	InterCommunicators.comms[count].ptask = ptask;
	InterCommunicators.comms[count].commids[0] = comm1;
	InterCommunicators.comms[count].commids[1] = comm2;
	InterCommunicators.comms[count].leaders[0] = leader1;
	InterCommunicators.comms[count].leaders[1] = leader2;

	InterCommunicators.count++;
}

void ParallelMerge_InitCommunicators(void)
{
	IntraCommunicators.size = IntraCommunicators.count = 0;
	IntraCommunicators.comms = NULL;
	InterCommunicators.size = InterCommunicators.count = 0;
	InterCommunicators.comms = NULL;
}

void AddPendingCommunication (int descriptor, off_t offset, int tag, int task_r,
	int task_s, int mz)
{
	int count = PendingComms.count;

	if (PendingComms.count == PendingComms.size)
	{
		PendingComms.size += PENDING_COMM_RESIZE_STEP;
		PendingComms.data = (struct PendingCommunication_t*) 
			realloc (PendingComms.data, 
			PendingComms.size*sizeof(struct PendingCommunication_t));
	}
#if defined(DEBUG)
	fprintf (stdout, "DEBUG: AddPendingCommunication (descriptor = %d, offset = %ld, tag = %d, task_r = %d, task_s = %d)\n",
		descriptor, offset, tag, task_r, task_s);
#endif
	PendingComms.data[count].offset = offset;
	PendingComms.data[count].descriptor = descriptor;
	PendingComms.data[count].recver = task_r;
	PendingComms.data[count].sender = task_s;
	PendingComms.data[count].tag = tag;
	PendingComms.data[count].match = 0;
	PendingComms.data[count].match_zone = mz;
	PendingComms.count++;
}

void InitPendingCommunication (void)
{
	PendingComms.size = PendingComms.count = 0;
	PendingComms.data = NULL;
}

void AddForeignRecv (UINT64 physic, UINT64 logic, int tag, int ptask_r, int task_r, 
	unsigned thread_r, unsigned vthread_r, int ptask_s, int task_s, FileSet_t *fset, int mz)
{
	int count;
	int group = inWhichGroup (ptask_s, task_s, fset);

	if (-1 == group)
	{
		fprintf (stderr, "mpi2prv: Error! Invalid group for foreign receive. Dying...\n");
		fflush (stderr);
		exit (0);
	}
	count = ForeignRecvs[group].count;

	if (count == ForeignRecvs[group].size)
	{
		ForeignRecvs[group].size += FOREIGN_RECV_RESIZE_STEP;
		ForeignRecvs[group].data = (struct ForeignRecv_t*) 
			realloc (ForeignRecvs[group].data, 
					ForeignRecvs[group].size*sizeof(struct ForeignRecv_t));
	}

#if defined(DEBUG)
	fprintf (stdout, "DEBUG: AddForeignRecv (phy_time = %llu, log_time = %llu, tag = %d, task_r = %d, thread_r = %d task_s = %d)\n",
		physic, logic, tag, task_r, thread_r, task_s);
#endif

	ForeignRecvs[group].data[count].sender = task_s;
	ForeignRecvs[group].data[count].sender_app = ptask_s;
	ForeignRecvs[group].data[count].recver = task_r;
	ForeignRecvs[group].data[count].recver_app = ptask_r;
	ForeignRecvs[group].data[count].tag = tag;
	ForeignRecvs[group].data[count].physic = physic;
	ForeignRecvs[group].data[count].logic = logic;
	ForeignRecvs[group].data[count].thread = thread_r;
	ForeignRecvs[group].data[count].vthread = vthread_r;
	ForeignRecvs[group].data[count].match_zone = mz; 
	ForeignRecvs[group].count++;
}

void InitForeignRecvs (int numtasks)
{
	int i;

	ForeignRecvs = (struct ForeignRecvs_t *) malloc (sizeof(struct ForeignRecvs_t)*numtasks);

	for (i = 0; i < numtasks; i++)
	{
		ForeignRecvs[i].count = ForeignRecvs[i].size = 0;
		ForeignRecvs[i].data  = NULL;
	}
}

static void MatchRecv (int fd, off_t offset, UINT64 physic_time, UINT64 logic_time)
{
	paraver_rec_t r;
	ssize_t size;
	off_t ret;
	unsigned long long receives[NUM_COMMUNICATION_TYPE];
	long offset_in_struct;

	/* Search offset of receives within the paraver_rec_t struct */
	offset_in_struct = ((long) &r.receive) - ((long) &r);

	receives[LOGICAL_COMMUNICATION] = logic_time;
	receives[PHYSICAL_COMMUNICATION] = physic_time;

	ret = lseek (fd, offset+offset_in_struct, SEEK_SET);
	if (ret != offset)
	{
		perror ("lseek");
#if SIZEOF_OFF_T == SIZEOF_LONG
		fprintf (stderr, "mpi2prv: Error on MatchRecv! Unable to lseek (fd = %d, offset = %ld)\n", fd, offset);
#elif SIZEOF_OFF_T == SIZEOF_LONG_LONG
		fprintf (stderr, "mpi2prv: Error on MatchRecv! Unable to lseek (fd = %d, offset = %lld)\n", fd, offset);
#elif SIZEOF_OFF_T == 4
		fprintf (stderr, "mpi2prv: Error on MatchRecv! Unable to lseek (fd = %d, offset = %d)\n", fd, offset);
#endif
		exit (-2);
	}

	size = write (fd, &receives, sizeof(receives));
	if (sizeof(receives) != size)
	{
		perror ("write");
		fprintf (stderr, "mpi2prv: Error on MatchRecv! Unable to write (fd = %d, size = %ld, written = %Zu)\n", fd, sizeof(r), size);
		exit (-2);
	}
}


static int MatchRecvs (struct ForeignRecv_t *data, int count)
{
	/* Match a set of "incomplete receives" */
	int i, j, result;

	result = 0;
	for (i = 0; i < count; i++)
		for (j = 0; j < PendingComms.count; j++)
		{
			if (data[i].match_zone == PendingComms.data[j].match_zone &&
			    data[i].sender == PendingComms.data[j].sender &&
			    data[i].recver == PendingComms.data[j].recver &&
			    (data[i].tag == PendingComms.data[j].tag  || data[i].tag == MPI_ANY_TAG) &&
			    !PendingComms.data[j].match)
			{
				PendingComms.data[j].match = 1;
				MatchRecv (PendingComms.data[j].descriptor, PendingComms.data[j].offset, data[i].physic, data[i].logic);
				result++;
				break;
			}
		}
	return result;
}


struct ForeignRecv_t* SearchForeignRecv (int group, int sender_app, int sender, int recver_app, int recver, int tag, int mz)
{
	int i;

#if defined(DEBUG)
	fprintf (stdout, "DEBUG: SearchForeignRecv (group = %d, sender/app = %d/%d, recver/app = %d, tag = %d)\n",
		group, sender, sender_app, recver, recver_app, tag);
#endif

	if (myForeignRecvs_count != NULL && myForeignRecvs != NULL)
	{
		if (myForeignRecvs[group] != NULL)
			for (i = 0; i < myForeignRecvs_count[group]; i++)
			{
#if defined(DEBUG)
				fprintf (stdout, "DEBUG: (sender/app = %d/%d, recver/app = %d/%d, tag = %d) vs (sender/app = %d/%d, recver = %d/%d, tag = %d)\n",
				  sender, sender_app, recver, recver_app, tag,
				  myForeignRecvs[group][i].sender, myForeignRecvs[group][i].sender_app,
				  myForeignRecvs[group][i].recver, myForeignRecvs[group][i].recver_app,
				  myForeignRecvs[group][i].tag);
#endif
				if (myForeignRecvs[group][i].match_zone == mz &&
				    myForeignRecvs[group][i].sender == sender &&
				    myForeignRecvs[group][i].sender_app == sender_app &&
				    myForeignRecvs[group][i].recver == recver &&
				    myForeignRecvs[group][i].recver_app == recver_app &&
			  	  (myForeignRecvs[group][i].tag == tag || myForeignRecvs[group][i].tag == MPI_ANY_TAG) &&
			    	!myForeignRecvs_used[group][i])
				{
					myForeignRecvs_used[group][i] = TRUE;
					return &myForeignRecvs[group][i];
				}
			}
	}
	return NULL;
}

static int RecvMine (int taskid, int from, int match, int *out_count, struct ForeignRecv_t **out, char **used)
{
	MPI_Status s;
	int res, count;
	struct ForeignRecv_t *data;
	int num_match = 0;

	res = MPI_Recv (&count, 1, MPI_INT, from, HOWMANY_FOREIGN_RECVS_TAG, MPI_COMM_WORLD, &s);
	MPI_CHECK(res, MPI_Recv, "Failed to receive count of foreign receives");

	if (count > 0)
	{
		data = (struct ForeignRecv_t*) malloc (count*sizeof(struct ForeignRecv_t));
		if (NULL == data)
		{
			fprintf (stderr, "mpi2prv: Error! Failed to allocate memory to receive foreign receives\n");
			fflush (stderr);
			exit (0);
		}

		res = MPI_Recv (data, count*sizeof(struct ForeignRecv_t), MPI_BYTE, from, BUFFER_FOREIGN_RECVS_TAG, MPI_COMM_WORLD, &s);
		MPI_CHECK(res, MPI_Recv, "Failed to receive foreign receives");
		if (match)
		{
			num_match = MatchRecvs (data, count);
			free (data);
		}
		else
		{
			int i;
			char *data_used;

			data_used = (char*) malloc (sizeof(char)*count);
			if (NULL == data_used)
			{
				fprintf (stderr, "mpi2prv: Error! Cannot create 'used' structure for foreign receives.\n");
				exit (-1);
			}
			for (i = 0; i < count; i++)
				data_used[i] = FALSE;

			*used = data_used;
			*out_count = count;
			*out = data;
		}
	}

	if (match)
	{
		if (get_option_merge_VerboseLevel() >= 1)
		{
			if (count > 0)
				fprintf (stdout, "mpi2prv: Processor %d matched %d of %d communications from processor %d\n", taskid, num_match, count, from);
			else
				fprintf (stdout, "mpi2prv: Processor %d did not receive communications from processor %d\n", taskid, from);
		}
	}
	
	fflush (stdout);

	return num_match;
}

static void SendMine (int taskid, int to, MPI_Request *req1, MPI_Request *req2)
{	
	int res;

	/* Send info on how many recvs will be transmitted */
	res = MPI_Isend (&(ForeignRecvs[to].count), 1, MPI_INT, to, HOWMANY_FOREIGN_RECVS_TAG, MPI_COMM_WORLD, req1);
	MPI_CHECK(res, MPI_Isend, "Failed to send quantity of foreign receives");

	if (ForeignRecvs[to].count > 0)
	{
		if (get_option_merge_VerboseLevel() >= 1)
		{
			fprintf (stdout, "mpi2prv: Processor %d distributes %d foreign receives to processor %d\n", taskid, ForeignRecvs[to].count, to);
			fflush (stdout);
		}

		/* Send data */
		res = MPI_Isend (ForeignRecvs[to].data, ForeignRecvs[to].count*sizeof(struct ForeignRecv_t), MPI_BYTE, to, BUFFER_FOREIGN_RECVS_TAG, MPI_COMM_WORLD, req2);
		MPI_CHECK(res, MPI_Isend, "Failed to send foreign receives");
	}
	else
	{
		if (get_option_merge_VerboseLevel() >= 1)
			fprintf(stdout, "mpi2prv: Processor %d does not have foreign receives for processor %d\n", taskid, to);
	}
}

void NewDistributePendingComms (int numtasks, int taskid, int match)
{
	int i, skew, res;

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Starting the distribution of foreign receives.\n");
		fflush (stdout);
	}

	if (!match)
	{
		myForeignRecvs = (struct ForeignRecv_t**) malloc (sizeof(struct ForeignRecv_t*)*numtasks);
		if (NULL == myForeignRecvs)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot allocate memory to control foreign receives!\n");
			exit (-1);
		}
		myForeignRecvs_used = (char**) malloc (sizeof(char*)*numtasks);
		if (NULL == myForeignRecvs_used)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot allocate memory to control foreign receives!\n");
			exit (-1);
		}
		myForeignRecvs_count = (int*) malloc (sizeof(int)*numtasks);
		if (NULL == myForeignRecvs_count)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot allocate memory to control the number of foreign receives!\n");
			exit (-1);
		}
		for (i = 0; i < numtasks; i++)
		{
			myForeignRecvs_count[i] = 0;
			myForeignRecvs[i] = NULL;
			myForeignRecvs_used[i] = NULL;
		}
	}

	for (skew = 1; skew < numtasks; skew++)
	{
		MPI_Request send_req1, send_req2;
		MPI_Status sts;
		int to, from;

		to = (taskid + skew) % numtasks;
		from = (taskid - skew + numtasks) % numtasks;

		SendMine (taskid, to, &send_req1, &send_req2);
		RecvMine (taskid, from, match, &myForeignRecvs_count[from], &myForeignRecvs[from], &myForeignRecvs_used[from]);

		MPI_Wait (&send_req1, &sts);
		if (ForeignRecvs[to].count > 0)
			MPI_Wait (&send_req2, &sts);

		/* Free data */
		free (ForeignRecvs[to].data);

/*
		res = MPI_Barrier (MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Barrier, "Failed to synchronize distribution of pending communications");
*/
	}

	res = MPI_Barrier (MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Barrier, "Failed to synchronize distribution of pending communications");

	if (!match)
	{
		int total;

		for (i = 0, total = 0; i < numtasks; i++)
			total += myForeignRecvs_count[i];

		fprintf (stdout, "mpi2prv: Processor %d is storing %d foreign receives (%lld Kbytes) for the next phase.\n",
			taskid, total, (((long long) total)*(sizeof(struct ForeignRecv_t)+sizeof(char)))/1024);
	}

	if (0 == taskid)
	{
		fprintf (stdout, "mpi2prv: Ended the distribution of foreign receives.\n");
		fflush (stdout);
	}

	/* Free pending communication info */
	if (PendingComms.count > 0)
		free (PendingComms.data);
}


static void BuildIntraCommunicator (struct IntraCommunicator_t *new_comm)
{
	unsigned j;
	TipusComunicador com;

#if defined(DEBUG_COMMUNICATORS)
	fprintf (stdout, "mpi2prv: DEBUG Adding intra-communicator type = %d ptask = %d task = %d\n",
	  new_comm->type, new_comm->ptask, new_comm->task);
	if (new_comm->type != MPI_COMM_WORLD_ALIAS && new_comm->type != MPI_COMM_SELF_ALIAS)
	{
		fprintf (stdout, "mpi2prv: tasks:");
		for (j = 0; j < new_comm->ntasks; j++)
			fprintf (stdout, "%d \n", new_comm->tasks[j]);
		fprintf (stdout, "\n");
	}
#endif

	com.id = new_comm->id;
	com.num_tasks = new_comm->ntasks;
	com.tasks = (int*) malloc(sizeof(int)*com.num_tasks);
	if (NULL == com.tasks)
	{
		fprintf (stderr, "mpi2prv: Error! Unable to allocate memory for transferred communicator!\n");
		fflush (stderr);
		exit (-1);
	}

	if (MPI_COMM_WORLD_ALIAS == new_comm->type)
		for (j = 0; j < com.num_tasks; j++)
			com.tasks[j] = j;
	else if (MPI_COMM_SELF_ALIAS == new_comm->type)
		com.tasks[0] = new_comm->task-1;
	else
		for (j = 0; j < com.num_tasks; j++)
			com.tasks[j] = new_comm->tasks[j];

	afegir_comunicador (&com, new_comm->ptask, new_comm->task);

	free (com.tasks);
}

static void BroadCastIntraCommunicator (int id, struct IntraCommunicator_t *new_comm)
{
	int res;

	res = MPI_Bcast (new_comm, sizeof(struct IntraCommunicator_t), MPI_BYTE, id, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated intra-communicators");

	/* If comm isn't a predefined, send the involved tasks */
	if (MPI_COMM_SELF_ALIAS != new_comm->type && MPI_COMM_WORLD_ALIAS != new_comm->type)
	{
		res = MPI_Bcast (new_comm->tasks, new_comm->ntasks, MPI_INT, id, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated intra-communicators");
	}
}

static void ReceiveIntraCommunicator (int id)
{
	int res;
	struct IntraCommunicator_t tmp;

	res = MPI_Bcast (&tmp, sizeof(struct IntraCommunicator_t), MPI_BYTE, id, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated intra-communicators");

	/* If comm isn't a predefined, receive the involved tasks */
	if (MPI_COMM_SELF_ALIAS != tmp.type && MPI_COMM_WORLD_ALIAS != tmp.type)
	{
		tmp.tasks = (int*) malloc (sizeof(int)*tmp.ntasks);
		if (NULL == tmp.tasks)
		{
			fprintf (stderr, "mpi2prv: ERROR! Failed to allocate memory for a new intra-communicator body\n");
			fflush (stderr);
			exit (0);
		}
		res = MPI_Bcast (tmp.tasks, tmp.ntasks, MPI_INT, id, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated communicators");
	}
	BuildIntraCommunicator (&tmp);

	/* Free data structures */
	if (tmp.tasks != NULL)
		free (tmp.tasks);
}

static void ParallelMerge_BuildIntraCommunicators (int num_tasks, int taskid)
{
	int i, j;
	int res, count;

	for (i = 0; i < num_tasks; i++)
	{
		if (i == taskid)
		{
			for (j = 0; j < IntraCommunicators.count; j++)
				BuildIntraCommunicator (&(IntraCommunicators.comms[j]));

			res = MPI_Bcast (&IntraCommunicators.count, 1, MPI_INT, i, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Bcast, "Failed to broadcast number of generated intra-communicators");

			for (j = 0; j < IntraCommunicators.count; j++)
				BroadCastIntraCommunicator (i, &(IntraCommunicators.comms[j]));

			/* Free data structures */
			for (j = 0; j < IntraCommunicators.count; j++)
				if (IntraCommunicators.comms[j].tasks != NULL)
					free (IntraCommunicators.comms[j].tasks);
			free (IntraCommunicators.comms);
		}
		else
		{
			res = MPI_Bcast (&count, 1, MPI_INT, i, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Bcast, "Failed to broadcast number of generated intra-communicators");
			for (j = 0; j < count; j++)
				ReceiveIntraCommunicator (i);
		}
	}
}

static void BroadCastInterCommunicator (int id, struct InterCommunicator_t *new_comm)
{
	int res;
	res = MPI_Bcast (new_comm, sizeof(struct InterCommunicator_t), MPI_BYTE, id, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated inter-communicators");
}

static void BuildInterCommunicator (struct InterCommunicator_t *new_comm)
{
	addInterCommunicator (new_comm->id,
	  new_comm->commids[0], new_comm->leaders[0],
	  new_comm->commids[1], new_comm->leaders[1],
	  new_comm->ptask, new_comm->task);
}

static void ReceiveInterCommunicator (int id)
{
	int res;
	struct InterCommunicator_t tmp;

	res = MPI_Bcast (&tmp, sizeof(struct InterCommunicator_t), MPI_BYTE, id, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to broadcast generated inter-communicators");

	BuildInterCommunicator (&tmp);
}

static void ParallelMerge_BuildInterCommunicators (int num_tasks, int taskid)
{
	int i, j;
	int res, count;

	for (i = 0; i < num_tasks; i++)
	{
		if (i == taskid)
		{
			for (j = 0; j < InterCommunicators.count; j++)
				BuildInterCommunicator (&(InterCommunicators.comms[j]));

			res = MPI_Bcast (&InterCommunicators.count, 1, MPI_INT, i, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Bcast, "Failed to broadcast number of generated inter-communicators");

			for (j = 0; j < InterCommunicators.count; j++)
				BroadCastInterCommunicator (i, &(InterCommunicators.comms[j]));

			/* Free data structures */
			free (InterCommunicators.comms);
		}
		else
		{
			res = MPI_Bcast (&count, 1, MPI_INT, i, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Bcast, "Failed to broadcast number of generated inter-communicators");
			for (j = 0; j < count; j++)
				ReceiveInterCommunicator (i);
		}
	}
}

void ParallelMerge_BuildCommunicators (int num_tasks, int taskid)
{
	ParallelMerge_BuildIntraCommunicators (num_tasks, taskid);
	ParallelMerge_BuildInterCommunicators (num_tasks, taskid);
}

void ShareTraceInformation (int numtasks, int taskid)
{
	int res;
	UNREFERENCED_PARAMETER(numtasks);

	res = MPI_Barrier (MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to synchronize when sharing trace information");

	if (0 == taskid)
		fprintf (stdout, "mpi2prv: Sharing information <");
	fflush (stdout);

	if (0 == taskid)
		fprintf (stdout, " MPI");
	fflush (stdout);
	Share_MPI_Softcounter_Operations ();
	Share_MPI_Operations ();

	if (0 == taskid)
		fprintf (stdout, " OpenMP");
	fflush (stdout);
	Share_OMP_Operations ();

	if (0 == taskid)
		fprintf (stdout, " pthread");
	fflush (stdout);
	Share_pthread_Operations ();

	if (0 == taskid)
		fprintf (stdout, " CUDA");
	fflush (stdout);
	Share_CUDA_Operations ();

#if USE_HARDWARE_COUNTERS
	if (0 == taskid)
		fprintf (stdout, " HWC");
	fflush (stdout);
	Share_Counters_Usage (numtasks, taskid);
#endif

	if (0 == taskid)
		fprintf (stdout, " MISC");
	fflush (stdout);
	Share_MISC_Operations ();

#if defined(HAVE_BFD)
	if (0 == taskid)
		fprintf (stdout, " callers");
	fflush (stdout);
	Share_Callers_Usage ();
#endif

	if (0 == taskid)
		fprintf (stdout, " >\n");
	fflush (stdout);
}

static void Receive_Dimemas_Data (void *buffer, int maxmem, int source, FILE *fd)
{
	ssize_t written;
	long long size;
	MPI_Status s;
	int res;
	int min;

	res = MPI_Recv (&size, 1, MPI_LONG_LONG, source, DIMEMAS_CHUNK_FILE_SIZE_TAG, MPI_COMM_WORLD, &s);
	MPI_CHECK(res, MPI_Recv, "Failed to receive file size of Dimemas chunk");

	do
	{
		min = MIN(maxmem, size);

		res = MPI_Recv (buffer, min, MPI_BYTE, source, DIMEMAS_CHUNK_DATA_TAG, MPI_COMM_WORLD, &s);
		MPI_CHECK(res, MPI_Recv, "Failed to receive file size of Dimemas chunk");

		written = write (fileno(fd), buffer, min);
		if (written != min)
		{
			perror ("write");
			fprintf (stderr, "mpi2trf: Error while writing the Dimemas trace file during parallel gather\n");
			fflush (stderr);
			exit (-1);
		}

		size -= min;
	} while (size > 0);
}

static void Send_Dimemas_Data (void *buffer, int maxmem, FILE *fd)
{
	ssize_t read_;
	long long size;
	int res;
	int min;

#if !defined(HAVE_FTELL64) && !defined(HAVE_FTELLO64)
	size = ftell (fd);
#elif defined(HAVE_FTELL64)
	size = ftell64 (fd);
#elif defined(HAVE_FTELLO64)
	size = ftello64 (fd);
#endif

	res = MPI_Send (&size, 1, MPI_LONG_LONG, 0, DIMEMAS_CHUNK_FILE_SIZE_TAG, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Send, "Failed to send file size of Dimemas chunk");

	rewind (fd);
	fflush (fd);

	do
	{
		min = MIN(maxmem, size);
		read_ = read (fileno(fd), buffer, min);
		if (read_ != min)
		{
			perror ("read");
			fprintf (stderr, "mpi2trf: Error while reading the Dimemas trace file during parallel gather\n");
			fflush (stderr);
			exit (-1);
		}

		res = MPI_Send (buffer, min, MPI_BYTE, 0, DIMEMAS_CHUNK_DATA_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Failed to receive file size of Dimemas chunk");

		size -= min;
	} while (size > 0);
}

void Gather_Dimemas_Traces (int numtasks, int taskid, FILE *fd, unsigned int maxmem)
{
	void *buffer;
	int res;
	int i;

	buffer = malloc (maxmem);
	if (NULL == buffer)
	{
		fprintf (stderr, "Error: mpi2trf was unable to allocate gathering buffers for Dimemas trace\n");
		fflush (stderr);
		exit (-1);
	}

	for (i = 1; i < numtasks; i++)
	{
		if (0 == taskid)
			Receive_Dimemas_Data (buffer, maxmem, i, fd);
		else if (i == taskid)
			Send_Dimemas_Data (buffer, maxmem, fd);

		res = MPI_Barrier (MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Barrier, "Failed to synchronize while gathering Dimemas trace");
	}

	free (buffer);
}


void Gather_Dimemas_Offsets (int numtasks, int taskid, int count,
	unsigned long long *in_offsets, unsigned long long **out_offsets,
	unsigned long long local_trace_size, FileSet_t *fset)
{
	unsigned long long other_trace_size;
	unsigned long long *temp = NULL;
	int res;
	int i;
	int j;

	if (0 == taskid)	
	{
		temp = (unsigned long long*) malloc (count*sizeof(unsigned long long));
		if (NULL == temp)
		{
			fprintf (stderr, "mpi2trf: Error! Unable to allocate memory for computing the offsets!\n");
			fflush (stderr);
			exit (-1);
		}
	}

	for (i = 0; i < numtasks-1; i++)
	{
		other_trace_size = (taskid == i)?local_trace_size:0;
		res = MPI_Bcast (&other_trace_size, 1, MPI_LONG_LONG, i, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Bcast, "Failed to broadcast Dimemas local trace size");

		if (taskid > i)
			for (j = 0; j < count; j++)
				if (inWhichGroup (0, j, fset) == taskid)
					in_offsets[j] += other_trace_size;
	}

	res = MPI_Reduce (in_offsets, temp, count, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Failed to gather offsets for Dimemas trace");

	if (0 == taskid)
		*out_offsets = temp;
}

void ShareNodeNames (int numtasks, char ***nodenames)
{
	int i, rc;
	size_t s;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	char *buffer_names = NULL;
	char **TasksNodes;

	/* Get processor name */
	if (gethostname (hostname, sizeof(hostname)) == -1)
	{
		fprintf (stderr, "Error! Cannot get hostname!\n");
		exit (-1);
	}

	/* Change spaces " " into underscores "_" (BGL nodes use to have spaces in their names) */
	for (s = 0; s < strlen(hostname); s++)
		if (' ' == hostname[s])
			hostname[s] = '_';

	/* Share information among all tasks */
	buffer_names = (char*) malloc (sizeof(char) * numtasks * MPI_MAX_PROCESSOR_NAME);
	rc = MPI_Allgather (hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, buffer_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
	MPI_CHECK(rc, MPI_Allgather, "Cannot gather processor names");

	/* Store the information in a global array */
	TasksNodes = (char **) malloc (numtasks * sizeof(char *));
	for (i = 0; i < numtasks; i++)
	{
		char *tmp = &buffer_names[i*MPI_MAX_PROCESSOR_NAME];
		TasksNodes[i] = (char *) malloc((strlen(tmp)+1) * sizeof(char));
		strcpy (TasksNodes[i], tmp);
	}

	/* Free the local array, not the global one */
	free (buffer_names);

	*nodenames = TasksNodes;
}

unsigned * Gather_Paraver_VirtualThreads (unsigned taskid, unsigned ptask,
	FileSet_t *fset)
{
	int res;
	unsigned *temp, *temp_out = NULL;
	ptask_t *ptask_info = GET_PTASK_INFO(ptask+1);
	unsigned ntasks = ptask_info->ntasks;
	unsigned u;

	if (0 == taskid)
		fprintf (stdout, "mpi2prv: Sharing thread accounting information for ptask %d", ptask);
	fflush (stdout);

	temp = (unsigned*) malloc (ntasks*sizeof(unsigned));
	if (NULL == temp)
	{
		fprintf (stderr, "mpi2prv: Error! Task %d unable to allocate memory to gather virtual threads!\n", taskid);
		fflush (stderr);
		exit (-1);
	}

	if (taskid == 0)
	{
		temp_out = (unsigned*) malloc (ntasks*sizeof(unsigned));
		if (NULL == temp_out)
		{
			fprintf (stderr, "mpi2prv: Error! Task %d unable to allocate memory to gather virtual threads!\n", taskid);
			fflush (stderr);
			exit (-1);
		}
	}

	for (u = 0; u < ntasks; u++)
		if (isTaskInMyGroup(fset, ptask, u))
		{
			task_t *task_info = GET_TASK_INFO(ptask+1, u+1);
			temp[u] = task_info->num_virtual_threads;
		}
		else
			temp[u] = 0;

	/* Reduce information into root task */
	res = MPI_Reduce (temp, temp_out, ntasks, MPI_UNSIGNED, MPI_SUM, 0,
	  MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Failed to gather number of virtual threads");

	if (0 == taskid)
		fprintf (stdout, " done\n");
	fflush (stdout);

	free (temp);

	return temp_out;
}

