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

#define _GNU_SOURCE
#include "common.h"

//#define DEBUG_SPAWN

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_FILE_H
# include <sys/file.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef WITH_PMPI_HOOK
# include <dlfcn.h>
#endif

#include "utils.h"
#include "utils_mpi.h"
#include "mpi_wrapper.h"
#include "wrapper.h"
#include "clock.h"
#include "signals.h"
#include "misc_wrapper.h"
#include "mpi_interface.h"
#include "mode.h"
#include "threadinfo.h"

#include <mpi.h>
#include "extrae_mpif.h"

#if defined(C_SYMBOLS) && defined(FORTRAN_SYMBOLS)
# define COMBINED_SYMBOLS
#endif

#if defined(HAVE_MRNET)
# include "mrnet_be.h"
#endif

#define MPI_COMM_WORLD_ALIAS 1
#define MPI_COMM_SELF_ALIAS  2
#define MPI_NEW_INTERCOMM_ALIAS  3

#if !defined(MPI_HAS_MPI_F_STATUS_IGNORE)
# define MPI_F_STATUS_IGNORE   ((MPI_Fint *) 0)
# define MPI_F_STATUSES_IGNORE ((MPI_Fint *) 0)
#endif

/*
	He d'incloure la capc,alera del misc_wrapper per poder comenc,ar 
	a tracejar quan es cridi al MPI_init i acabar al MPI_Finalize.
*/
#include "misc_wrapper.h"

/* Cal tenir requests persistents per algunes operacions */
#include "persistent_requests.h"

#ifdef HAVE_NETINET_IN_H
# include <netinet/in.h>
#endif

#define MPI_CHECK(mpi_error, routine) \
	if (mpi_error != MPI_SUCCESS) \
	{ \
		fprintf (stderr, "Error in MPI call %s (file %s, line %d, routine %s) returned %d\n", \
			#routine, __FILE__, __LINE__, __func__, mpi_error); \
		fflush (stderr); \
		exit (1); \
	}

#define MAX_WAIT_REQUESTS 16384

static unsigned Extrae_MPI_NumTasks (void)
{
	static int run = FALSE;
	static int mysize;

	if (!run)
	{
		PMPI_Comm_size (MPI_COMM_WORLD, &mysize);
		run = TRUE;
	}

	return (unsigned) mysize;
}

static unsigned Extrae_MPI_TaskID (void)
{
	static int run = FALSE;
	static int myrank;

	if (!run)
	{
		PMPI_Comm_rank (MPI_COMM_WORLD, &myrank);
		run = TRUE;
	}

	return (unsigned) myrank;
}

static void Extrae_MPI_Barrier (void)
{
	PMPI_Barrier (MPI_COMM_WORLD);
}

static void Extrae_MPI_Finalize (void)
{
	PMPI_Finalize ();
}

static void Trace_MPI_Communicator (MPI_Comm newcomm, UINT64 time, int trace);
static void Trace_MPI_InterCommunicator (MPI_Comm newcomm, MPI_Comm local_comm,
	int local_leader, MPI_Comm remote_comm, int remote_leader, UINT64 time,
	int trace);

/******************************************************************************
 ********************      L O C A L    V A R I A B L E S        **************
 ******************************************************************************/

char *MpitsFileName    = NULL;    /* Name of the .mpits file (only significant at rank 0) */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
char *SpawnsFileName   = NULL;    /* Name of the .spawn file (all tasks have it defined)  */
int   SpawnGroup       = 0;
int  *ParentWorldRanks = NULL;    /* World ranks of the parent processes 
  (index is local rank for the parent process, value is the parent world rank) */
unsigned long long SpawnOffset = 0;
#endif

char *Extrae_core_get_mpits_file_name (void)
{
	return MpitsFileName;
}

hash_t requests;         /* Receive requests stored in a hash in order to search them fast */
PR_Queue_t PR_queue;     /* Persistent requests queue */
static int *ranks_global;       /* Global ranks vector (from 1 to NProcs) */
static MPI_Group grup_global;   /* Group attached to the MPI_COMM_WORLD */
static MPI_Fint grup_global_F;  /* Group attached to the MPI_COMM_WORLD (Fortran) */

#if defined(IS_BGL_MACHINE)     /* BGL, s'intercepten algunes crides barrier dins d'altres cols */
static int BGL_disable_barrier_inside = 0;
#endif

/******************************************************************************
 *** CheckGlobalOpsTracingIntervals()
 ******************************************************************************/
void CheckGlobalOpsTracingIntervals (void)
{
	int result;

	result = GlobalOp_Changes_Trace_Status (Extrae_MPI_getCurrentOpGlobal());
	if (result == SHUTDOWN)
		Extrae_shutdown_Wrapper();
	else if (result == RESTART)
		Extrae_restart_Wrapper();
}

/******************************************************************************
 ***  get_rank_obj_C
 ******************************************************************************/

int get_rank_obj_C (MPI_Comm comm, int dest, int *receiver, int send_or_recv)
{
	int ret, inter;
	MPI_Group group;

	/* If rank in MPI_COMM_WORLD or if dest is PROC_NULL or any source,
	   return value directly */
	if (comm == MPI_COMM_WORLD || dest == MPI_PROC_NULL || dest == MPI_ANY_SOURCE)
	{
		*receiver = dest;
	}
	else
	{
		ret = PMPI_Comm_test_inter (comm, &inter);	
		MPI_CHECK (ret, PMPI_Comm_test_inter);

		if (inter)
		{
                        MPI_Comm parent;
                        PMPI_Comm_get_parent(&parent);

                        /* The communicator is an intercommunicator */
                        if (send_or_recv == RANK_OBJ_SEND)
                        {
                                if (comm == parent)
                                {
                                        /* Send to parent -- Translate the local rank for the parent into its MPI_COMM_WORLD rank */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
                                        if (ParentWorldRanks != NULL)
                                                *receiver = ParentWorldRanks[dest];
                                        else
#endif
                                                *receiver = dest; /* Should never happen */
                                }
                                else
                                {
                                        /* Send to children -- When sending to specific childen X, there's no need to translate ranks */
                                        *receiver = dest;
                                }
                        }
                        else
                        {
                                if (comm == parent)
                                {
                                        /* Recv from parent -- Translate the local rank for the parent into its MPI_COMM_WORLD rank */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
                                        if (ParentWorldRanks != NULL)
                                                *receiver = ParentWorldRanks[dest];
                                        else
#endif
                                                *receiver = dest; /* Should never happen */
                                }
                                else
                                {
                                        /* Recv from children -- When receiving from specific childen X, there's no need to translate ranks */
                                        *receiver = dest;
                                }
                        }
		}
		else
		{
			/* The communicator is an intracommunicator */
			ret = PMPI_Comm_group (comm, &group);
			MPI_CHECK (ret, PMPI_Comm_group);

			/* Translate the rank */
			ret = PMPI_Group_translate_ranks (group, 1, &dest, grup_global, receiver); 
			MPI_CHECK (ret, PMPI_Group_translate_ranks);
			
			ret = PMPI_Group_free (&group);
			MPI_CHECK (ret, PMPI_Group_free);
		}
	}
	return MPI_SUCCESS;
}

/******************************************************************************
 ***  Traceja_Persistent_Request
 ******************************************************************************/

static void Traceja_Persistent_Request (MPI_Request* reqid, iotimer_t temps)
{
	persistent_req_t *p_request;
	hash_data_t hash_req;
	int inter;
	int size, src_world, ret;
	int send_or_recv;

	/*
	* S'intenta recuperar la informacio d'aquesta request per tracejar-la 
	*/
	p_request = PR_Busca_request (&PR_queue, reqid);
	if (p_request == NULL)
		return;

	/* 
	  HSG, aixo es pot emmagatzemar a la taula de hash! A mes,
	  pot ser que hi hagi un problema a l'hora de calcular els  bytes p2p
	  pq ignora la quantitat de dades enviada
	*/
	ret = PMPI_Type_size (p_request->datatype, &size);
	MPI_CHECK(ret, PMPI_Type_size);

	send_or_recv = (p_request->tipus == MPI_IRECV_EV ? RANK_OBJ_RECV : RANK_OBJ_SEND );
	if (get_rank_obj_C (p_request->comm, p_request->task, &src_world, send_or_recv) != MPI_SUCCESS)
		return;

	if (p_request->tipus == MPI_IRECV_EV)
	{
		/*
		 * Als recv guardem informacio pels WAITs 
		*/
		hash_req.key = *reqid;
		hash_req.commid = p_request->comm;
		hash_req.partner = p_request->task;
		hash_req.tag = p_request->tag;
		hash_req.size = p_request->count * size;

		if (p_request->comm == MPI_COMM_WORLD)
		{
			hash_req.group = MPI_GROUP_NULL;
		}
		else
		{
			ret = PMPI_Comm_test_inter (p_request->comm, &inter);
			MPI_CHECK (ret, PMPI_Comm_test_inter);
			
			if (inter)
			{
				ret = PMPI_Comm_remote_group (p_request->comm, &hash_req.group);
				MPI_CHECK (ret, PMPI_Comm_remote_group);
			}
			else
			{
				ret = PMPI_Comm_group (p_request->comm, &hash_req.group);	
				MPI_CHECK (ret, PMPI_Comm_group);
			}
		}

		hash_add (&requests, &hash_req);
	}

	/*
	*   event : PERSIST_REQ_EV                        value : Request type
	*   target : MPI_ANY_SOURCE or sender/receiver    size  : buffer size
	*   tag : message tag or MPI_ANY_TAG              commid: Communicator id
	*   aux: request id
	*/
	TRACE_MPIEVENT_NOHWC (temps, MPI_PERSIST_REQ_EV, p_request->tipus,
	  src_world, size, p_request->tag, p_request->comm, p_request->req);
}


/******************************************************************************
 *** CheckControlFile()
 ******************************************************************************/

/* This counter indicates when will be the next check for the control file */
unsigned int NumOpsGlobalsCheckControlFile        = 10;
unsigned int NumOpsGlobalsCheckControlFile_backup = 10;

unsigned long long CalculateNumOpsForPeriod (unsigned long long wannaPeriod,
	unsigned long long NumOfGlobals, unsigned long long runnedPeriod)
{
	if (runnedPeriod <= wannaPeriod * NumOfGlobals)
		return (wannaPeriod * NumOfGlobals) / runnedPeriod;

	return 1;
}

void CheckControlFile(void)
{
	unsigned int prevtracejant = tracejant;
	unsigned int wannatrace = 0;

	NumOpsGlobalsCheckControlFile--;
	
	if (!NumOpsGlobalsCheckControlFile)
	{
		if (TASKID == 0)
		{
			wannatrace = file_exists (Extrae_getCheckControlFileName());
			if (wannatrace != prevtracejant)
			{
				fprintf (stdout, PACKAGE_NAME": Tracing is %s via control file\n", (wannatrace)?"activated":"deactivated");
				if (wannatrace)
					mpitrace_on = TRUE;
			}

			if (WantedCheckControlPeriod != 0)
			{
				NumOpsGlobalsCheckControlFile_backup = CalculateNumOpsForPeriod (WantedCheckControlPeriod, NumOpsGlobalsCheckControlFile_backup, TIME - initTracingTime);
				fprintf (stderr, PACKAGE_NAME": Control file check change, now every %u global ops (%llu s)\n", NumOpsGlobalsCheckControlFile_backup, WantedCheckControlPeriod / 1000000000);
			}
		}

		/* Broadcast the following num of global-num-ops before being checked*/
		PMPI_Bcast (&NumOpsGlobalsCheckControlFile_backup, 1, MPI_LONG_LONG_INT, 0, 
			MPI_COMM_WORLD);

		/* Broadcast both mpitrace_on & tracing */
		{
			int valors[2] = { wannatrace, mpitrace_on };
			PMPI_Bcast (valors, 2, MPI_INT, 0, MPI_COMM_WORLD);
			wannatrace = valors[0];
			mpitrace_on = valors[1];

			if (mpitrace_on)
			{
				/* Turn on if it was off, and turn off it it was on */
				if (wannatrace && !prevtracejant)
					Extrae_restart_Wrapper();
				else if (!wannatrace && prevtracejant)
					Extrae_shutdown_Wrapper();
			}
		}

		/* If the tracing has been enabled, just change the init tracing time. */
		/* If not, just reset init tracing time so as the next period will be 
		   calculated from this point */
		if (mpitrace_on && initTracingTime == 0)
			initTracingTime = TIME;

		NumOpsGlobalsCheckControlFile = NumOpsGlobalsCheckControlFile_backup;
	}
}

/******************************************************************************
 ***  InitMPICommunicators
 ******************************************************************************/

static void InitMPICommunicators (void)
{
	unsigned i;

	/** Inicialitzacio de les variables per la creacio de comunicadors **/
	ranks_global = malloc (sizeof(int)*Extrae_get_num_tasks());
	if (ranks_global == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for 'ranks_global'");
		exit (0);
	}

	for (i = 0; i < Extrae_get_num_tasks(); i++)
		ranks_global[i] = i;

	PMPI_Comm_group (MPI_COMM_WORLD, &grup_global);
	grup_global_F = MPI_Group_c2f(grup_global);

	int s = 0;
	PMPI_Group_size( grup_global, &s );
}


/******************************************************************************
 ***  MPI_remove_file_list
 ******************************************************************************/
void MPI_remove_file_list (int all)
{
	char tmpname[1024];

	if (all || (!all && TASKID == 0))
	{
		sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
		unlink (tmpname);
	}
}

/******************************************************************************
 ***  Get_Nodes_Info
 ******************************************************************************/

char **TasksNodes = NULL;

static void Gather_Nodes_Info (void)
{
	unsigned u;
	int rc;
	size_t s;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	char *buffer_names = NULL;

	/* Get processor name */
	if (gethostname (hostname, sizeof(hostname)) == -1)
	{
		fprintf (stderr, "Error! Cannot get hostname!\n");
		exit (-1);
	}

	/* Change spaces " " into underscores "_" (BLG nodes use to have spaces in their names) */
	for (s = 0; s < strlen(hostname); s++)
		if (' ' == hostname[s])
			hostname[s] = '_';

	/* Share information among all tasks */
	buffer_names = (char*) malloc (sizeof(char) * Extrae_get_num_tasks() * MPI_MAX_PROCESSOR_NAME);
	if (buffer_names == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Fatal error! Cannot allocate memory for nodes name\n");
		exit (-1);
	}
	rc = PMPI_Allgather (hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, buffer_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
	MPI_CHECK(rc, PMPI_Allgather);

	/* Store the information in a global array */
	TasksNodes = (char **)malloc (Extrae_get_num_tasks() * sizeof(char *));
	if (TasksNodes == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Fatal error! Cannot allocate memory for nodes info\n");
		exit (-1);
	}
	for (u=0; u<Extrae_get_num_tasks(); u++)
	{
		char *tmp = &buffer_names[u*MPI_MAX_PROCESSOR_NAME];
		TasksNodes[u] = (char *)malloc((strlen(tmp)+1) * sizeof(char));
		if (TasksNodes[u] == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Fatal error! Cannot allocate memory for node info %u\n", u);
			exit (-1);
		}
		strcpy (TasksNodes[u], tmp);
	}

	/* Free the local array, not the global one */
	free (buffer_names);
}


/******************************************************************************
 ***  MPI_Generate_Task_File_List
 ******************************************************************************/
static int MPI_Generate_Task_File_List (char **node_list, int isSpawned)
{
	int filedes, ierror;
	unsigned u, ret, thid;
	char tmpname[1024];
	unsigned *buffer = NULL;
	unsigned tmp[3]; /* we store pid, nthreads and taskid on each position */

	if (TASKID == 0)
	{
		buffer = (unsigned *) malloc (sizeof(unsigned) * Extrae_get_num_tasks() * 3);
		/* we store pid, nthreads and taskid on each position */

		if (buffer == NULL)
		{
			fprintf (stderr, "Fatal error! Cannot allocate memory to transfer MPITS info\n");
			exit (-1);
		}
	}

	tmp[0] = TASKID; 
	tmp[1] = getpid();
	tmp[2] = Backend_getMaximumOfThreads();

	/* Share PID and number of threads of each MPI task */
	ierror = PMPI_Gather (&tmp, 3, MPI_UNSIGNED, buffer, 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_CHECK(ierror, PMPI_Gather);

	/* If I haven't been MPI_Comm_Spawned, let's clean all the *-%d.mpits we
	   have created in earlier execes */
	if (TASKID == 0 && !isSpawned)
	{
		if (Extrae_core_get_mpits_file_name() == NULL)
		{
			int next = TRUE;
			unsigned count = 1;
			do
			{
				if (count > 1)
					sprintf (tmpname, "%s/%s-%d%s", final_dir, appl_name, count, EXT_MPITS);
				else
					sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);

				/* If the file exists, remove it and its associated .spawn file */
				if (file_exists(tmpname))
				{
					if (unlink (tmpname) != 0)
						fprintf (stderr, PACKAGE_NAME": Warning! Could not clean previous file %s\n", tmpname);

					if (count > 1)
						sprintf (tmpname, "%s/%s-%d%s", final_dir, appl_name, count, EXT_SPAWN);
					else
						sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_SPAWN);

					if (file_exists(tmpname))
						if (unlink (tmpname) != 0)
							fprintf (stderr, PACKAGE_NAME": Warning! Could not clean previous file %s\n", tmpname);

					next = TRUE;
				}
				else
					next = FALSE;

				count++;
			} while (next);
		}
	}

	if (TASKID == 0)
	{
		if (Extrae_core_get_mpits_file_name() == NULL)
		{
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
			do
			{
				SpawnGroup ++;
				if (SpawnGroup > 1)
					sprintf (tmpname, "%s/%s-%d%s", final_dir, appl_name, SpawnGroup, EXT_MPITS);
				else
					sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);

				filedes = open (tmpname, O_RDWR | O_CREAT | O_EXCL | O_TRUNC, 0644);
			} while (filedes == -1);
#else
			sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
			filedes = open (tmpname, O_RDWR | O_CREAT | O_TRUNC, 0644);
			if (filedes == -1)
			{
				return -1;
			}
#endif
			MpitsFileName = strdup( tmpname );
		}
		else
		{
			filedes = open (MpitsFileName, O_RDWR | O_CREAT | O_TRUNC, 0644);
			if (filedes == -1) 
			{
				return -1;
			}
		}

		for (u = 0; u < Extrae_get_num_tasks(); u++)
		{
			char tmpline[2048];
			unsigned TID = buffer[u*3+0];
			unsigned PID = buffer[u*3+1];
			unsigned NTHREADS = buffer[u*3+2];

			if (u == 0)
			{
				/* If Im processing MASTER, I know my threads and their names */
				for (thid = 0; thid < NTHREADS; thid++)
				{
					FileName_PTT(tmpname, Get_FinalDir(TID), appl_name,
					  node_list[u], PID, TID, thid, EXT_MPIT);
					sprintf (tmpline, "%s named %s\n", tmpname,
					  Extrae_get_thread_name(thid));
					ret = write (filedes, tmpline, strlen (tmpline));
					if (ret != strlen (tmpline))
					{
						close (filedes);
						return -1;
					}
				}
			}
			else
			{
				/* If Im not processing MASTER, I have to ask for threads and their names */

				int foo;
				MPI_Status s;
				char *tmp = (char*)malloc (NTHREADS*THREAD_INFO_NAME_LEN*sizeof(char));
				if (tmp == NULL)
				{
					fprintf (stderr, "Fatal error! Cannot allocate memory to transfer thread names\n");
					exit (-1);
				}

				/* Ask to slave */
				PMPI_Send (&foo, 1, MPI_INT, TID, 123456, MPI_COMM_WORLD);

				/* Send master info */
				PMPI_Recv (tmp, NTHREADS*THREAD_INFO_NAME_LEN, MPI_CHAR, TID, 123457,
				  MPI_COMM_WORLD, &s);

				for (thid = 0; thid < NTHREADS; thid++)
				{
					FileName_PTT(tmpname, Get_FinalDir(TID), appl_name,
					  node_list[u], PID, TID, thid, EXT_MPIT);
					sprintf (tmpline, "%s named %s\n", tmpname,
					  &tmp[thid*THREAD_INFO_NAME_LEN]);
					ret = write (filedes, tmpline, strlen (tmpline));
					if (ret != strlen (tmpline))
					{
						close (filedes);
						return -1;
					}
				}
				free (tmp);
			}
		}
		close (filedes);
	}
	else
	{
		MPI_Status s;
		int foo;

		char *tmp = (char*)malloc (Backend_getMaximumOfThreads()*THREAD_INFO_NAME_LEN*sizeof(char));
		if (tmp == NULL)
		{
			fprintf (stderr, "Fatal error! Cannot allocate memory to transfer thread names\n");
			exit (-1);
		}
		for (u = 0; u < Backend_getMaximumOfThreads(); u++)
			memcpy (&tmp[u*THREAD_INFO_NAME_LEN], Extrae_get_thread_name(u), THREAD_INFO_NAME_LEN);

		/* Wait for master to ask */
		PMPI_Recv (&foo, 1, MPI_INT, 0, 123456, MPI_COMM_WORLD, &s);

		/* Send master info */
		PMPI_Send (tmp, Backend_getMaximumOfThreads()*THREAD_INFO_NAME_LEN,
		  MPI_CHAR, 0, 123457, MPI_COMM_WORLD);

		free (tmp);
	}

	if (TASKID == 0)
	{
		free (buffer);
	}

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	/* Pass the name of the .mpits file to all tasks (the embedded merger needs to know!) */
	PMPI_Bcast(&SpawnGroup, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (SpawnGroup > 1)
		sprintf (tmpname, "%s/%s-%d%s", final_dir, appl_name, SpawnGroup, EXT_MPITS);
	else
		sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
#else
	sprintf (tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
#endif

	MpitsFileName = strdup( tmpname );

	return 0;
}


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Generate_Spawns_List (void)
 ***  Prepares the name of the .spawn list, and broadcast the name of the file
 ***  to all tasks. The file will be later open and written exclusively by any 
 ***  task that does a spawn.
 ******************************************************************************/
static void MPI_Generate_Spawns_List (void)
{
  int namelen = 0;

  if (TASKID == 0)
  {
    /* Only task 0 knows the name of the .mpits file */
    char *x = NULL;

    SpawnsFileName = strdup( MpitsFileName );

    x = strrchr(SpawnsFileName, '.');
    strcpy(x, EXT_SPAWN); /* No need to realloc SpawnsFileName because the length of EXT_SPAWN is the same of EXT_MPITS */
    namelen = strlen(SpawnsFileName);
  }

  PMPI_Bcast (&namelen, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (TASKID != 0)
  {
    SpawnsFileName = (char *)malloc((namelen+1) * sizeof(char));
  }
  
  PMPI_Bcast (SpawnsFileName, namelen+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  PMPI_Bcast (&SpawnGroup, 1, MPI_INT, 0, MPI_COMM_WORLD);

#if defined(DEBUG_SPAWN)
  fprintf(stderr, "[DEBUG MPI_Generate_Spawn_List] TASKID=%d SpawnsFileName=%s\n", TASKID, SpawnsFileName);
#endif

  /* The latency to the master tasks is 0 */
  if (TASKID == 0)
  {
    FILE *fd = fopen(SpawnsFileName, "a+");
    if (fd == NULL)
    {
      perror("fopen");
    }
    else
    {
      flock(fileno(fd), LOCK_EX);
      fprintf(fd, "%llu\n", SpawnOffset);
      flock(fileno(fd), LOCK_UN);
      fclose(fd);
    }
  }
}
#endif /* MPI_SUPPORTS_MPI_COMM_SPAWN */


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/**
 * Spawn_Parent_Sync()
 *
 * Gathers the mpit id's and the parent intercommunicator id's from all spawned processes,
 * writes this information in the SpawnsFileName file, which will be later processed by
 * the merger, and synchronizes with the spawned processes' MPI_Init.
 */
static void Spawn_Parent_Sync (unsigned long long SpawnStartTime, MPI_Comm intercomm, MPI_Comm spawn_comm)
{
  int i = 0;

  if ((intercomm != MPI_COMM_NULL) && (spawn_comm != MPI_COMM_NULL))
  {
    int       *all_parents_comms = NULL;
    int       *all_parents_ranks = NULL;
    int        RemoteSpawnGroup  = 0;
    int        my_rank;
    int        num_parents;
    int        world_rank = TASKID;
    unsigned long long ChildSpawnOffset = 0;

	UNREFERENCED_PARAMETER(SpawnStartTime);
    
    PMPI_Comm_rank(spawn_comm, &my_rank);

    /* Register the intercommunicator */
    Trace_MPI_Communicator (intercomm, LAST_READ_TIME, FALSE);

    /* Gather the parent comm id's from the participating tasks */
    PMPI_Comm_size(spawn_comm, &num_parents);
    all_parents_comms = (int *)malloc( num_parents * sizeof(int) );
    all_parents_ranks = (int *)malloc( num_parents * sizeof(int) );

    PMPI_Gather(&intercomm, 1, MPI_INT, all_parents_comms, 1, MPI_INT, 0, spawn_comm);
    PMPI_Gather(&world_rank, 1, MPI_INT, all_parents_ranks, 1, MPI_INT, 0, spawn_comm);

    /* Exchange the spawn group id's */
    PMPI_Bcast( &SpawnGroup, 1, MPI_INT, (my_rank == 0 ? MPI_ROOT : MPI_PROC_NULL), intercomm );
    PMPI_Bcast( &RemoteSpawnGroup, 1, MPI_INT, 0, intercomm );

    /* Send the parent's world ranks to the children */
    PMPI_Bcast( &num_parents, 1, MPI_INT, (my_rank == 0 ? MPI_ROOT : MPI_PROC_NULL), intercomm );
    PMPI_Bcast( all_parents_ranks, num_parents, MPI_INT, (my_rank == 0 ? MPI_ROOT : MPI_PROC_NULL), intercomm );

    /* Register each child parent_comm_id in the spawns list */
    if (my_rank == 0)
    {
      FILE *fd = fopen(SpawnsFileName, "a+");
      if (fd == NULL)
      {
        perror("fopen");
      }
      else
      {
        flock(fileno(fd), LOCK_EX);
        for (i=0; i<num_parents; i++)
        {
          fprintf(fd, "%d %d %d\n", all_parents_ranks[i], (int)all_parents_comms[i], RemoteSpawnGroup);
        }
        flock(fileno(fd), LOCK_UN);
        fclose(fd);
      }
    }

    /* Send the synchronization time */
    ChildSpawnOffset = SpawnOffset + (TIME - getApplBeginTime()); /* Changed SpawnStartTime to TIME to see the synchronization at the end of the spawn call */ 
    PMPI_Bcast ( &ChildSpawnOffset, 1, MPI_UNSIGNED_LONG_LONG, (my_rank == 0 ? MPI_ROOT : MPI_PROC_NULL), intercomm );

    /* Synchronize with the MPI_Init of the spawned tasks (see complementary barrier at MPI_Init) */
#if defined(DEBUG_SPAWN)
    fprintf(stderr, "[EXTRAE-MASTER %d] CALLING -> PMPI_Barrier(intercomm)\n", TASKID);
#endif
    PMPI_Barrier( intercomm );
#if defined(DEBUG_SPAWN)
    fprintf(stderr, "[EXTRAE-MASTER %d] PMPI_Barrier(intercomm) -> RETURNS\n", TASKID);
#endif

    xfree(all_parents_comms);
    xfree(all_parents_ranks);
  }
}
#endif /* MPI_SUPPORTS_MPI_COMM_SPAWN */



#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/**
 * Spawn_Children_Sync()
 *
 * Checks if this application has been spawned through MPI_Comm_spawn. 
 * If so, synchronizes with the parent application and sends the 
 * children intercommunicator id's and the corresponding mpit id's to 
 * link the apps.
 */
static void Spawn_Children_Sync(iotimer_t init_time)
{
  MPI_Comm parent;

  PMPI_Comm_get_parent(&parent);

  if (parent != MPI_COMM_NULL)
  {
    int  i                  = 0;
    int  RemoteSpawnGroup   = 0;
    int  num_children       = 0;
    int *all_children_comms = NULL;
    int  num_parents        = 0;
    int *all_parents_ranks  = NULL;

    /* This task has been spawned through MPI_Comm_spawn! */
#if defined(DEBUG_SPAWN)
    fprintf(stderr, "[EXTRAE-WORKER %d] I HAVE BEEN SPAWNED!\n", TASKID);
    fprintf(stderr, "[EXTRAE-WORKER %d] parent_comm_id=%d\n", TASKID, (int)parent);
#endif
    Trace_MPI_Communicator (parent, init_time, FALSE);
    
    /* Gather the children communicators to the parent */
    PMPI_Comm_size(MPI_COMM_WORLD, &num_children);
    all_children_comms = (int *)malloc(sizeof(int) * num_children);
    PMPI_Gather(&parent, 1, MPI_INT, all_children_comms, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Exchange the spawn group id's */
    PMPI_Bcast( &RemoteSpawnGroup, 1, MPI_INT, 0, parent );
    PMPI_Bcast( &SpawnGroup, 1, MPI_INT, (TASKID == 0 ? MPI_ROOT : MPI_PROC_NULL), parent );

    /* Receive the parent's world ranks */
    PMPI_Bcast (&num_parents, 1, MPI_INT, 0, parent);
    all_parents_ranks = (int *)malloc(sizeof(num_parents) * sizeof(int));
    PMPI_Bcast (all_parents_ranks, num_parents, MPI_INT, 0, parent);
    ParentWorldRanks = all_parents_ranks;
  
    /* Receive the synchronization time */
    PMPI_Bcast ( &SpawnOffset, 1, MPI_LONG_LONG, 0, parent);

    if (TASKID == 0)
    {
      FILE *fd = fopen(SpawnsFileName, "w");
      fprintf(fd, "%llu\n", SpawnOffset);
      for (i=0; i<num_children; i++)
      {
        fprintf(fd, "%d %d %d\n", i, (int)all_children_comms[i], RemoteSpawnGroup);
      }
      fclose(fd);
    }

    /* Synchronize with the parent's MPI_Comm_spawn() */
#if defined(DEBUG_SPAWN)
    fprintf(stderr, "[EXTRAE-WORKER %d] CALLING -> PMPI_Barrier(parent)\n", TASKID);
#endif
    PMPI_Barrier( parent );
#if defined(DEBUG_SPAWN)
    fprintf(stderr, "[EXTRAE-WORKER %d] PMPI_Barrier(parent) -> RETURNS\n", TASKID);
#endif
    xfree(all_children_comms);
  }
}
#endif /* MPI_SUPPORTS_MPI_COMM_SPAWN */


#if defined(FORTRAN_SYMBOLS)

/* Some C libraries do not contain the mpi_init symbol (fortran)
	 When compiling the combined (C+Fortran) dyninst module, the resulting
	 module CANNOT be loaded if mpi_init is not found. The top #if def..
	 is a workaround for this situation

   NOTE: Some C libraries (mpich 1.2.x) use the C initialization and do not
   offer mpi_init (fortran).
*/
/*
 HSG: I think that MPI_C_CONTAINS_FORTRAN_MPI_INIT is not the proper check to do here
#if (defined(COMBINED_SYMBOLS) && !defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
     !defined(COMBINED_SYMBOLS))
*/

/******************************************************************************
 ***  PMPI_Init_Wrapper
 ******************************************************************************/
void PMPI_Init_Wrapper (MPI_Fint *ierror)
/* Aquest codi nomes el volem per traceig sequencial i per mpi_init de fortran */
{
	MPI_Comm cparent = MPI_COMM_NULL;
	iotimer_t MPI_Init_start_time, MPI_Init_end_time;

	hash_init (&requests);
	PR_queue_init (&PR_queue);

#ifdef WITH_PMPI_HOOK
        int (*real_mpi_init)(MPI_Fint *ierror) = NULL;
        real_mpi_init = dlsym(RTLD_NEXT, STRINGIFY(CtoF77 (mpi_init)));

        if (real_mpi_init != NULL) {
                CtoF77 (real_mpi_init) (ierror);
        } else
#endif
        {
                CtoF77 (pmpi_init) (ierror);
        }

	Extrae_set_ApplicationIsMPI (TRUE);
	Extrae_Allocate_Task_Bitmap (Extrae_MPI_NumTasks());

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function (Extrae_MPI_TaskID);
	Extrae_set_numtasks_function (Extrae_MPI_NumTasks);
	Extrae_set_barrier_tasks_function (Extrae_MPI_Barrier);
	Extrae_set_finalize_task_function (Extrae_MPI_Finalize);

	InitMPICommunicators();

#if defined(SAMPLING_SUPPORT)
	/* If sampling is enabled, just stop all the processes at the same point
	   and continue */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
#endif

	/* Proceed with initialization if it's not already init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_NOT_INITIALIZED)
	{
		int res;
		char *config_file = getenv ("EXTRAE_CONFIG_FILE");

		if (config_file == NULL)
			config_file = getenv ("MPTRACE_CONFIG_FILE");

		Extrae_set_initial_TASKID (TASKID);
		Extrae_set_is_initialized (EXTRAE_INITIALIZED_MPI_INIT);

		if (config_file != NULL)
			/* Obtain a localized copy *except for the master process* */
			config_file = MPI_Distribute_XML_File (TASKID, Extrae_get_num_tasks(), config_file);

		/* Initialize the backend */
		res = Backend_preInitialize (TASKID, Extrae_get_num_tasks(), config_file, FALSE);
		if (!res)
			return;

		/* Remove the local copy only if we're not the master */
		if (TASKID != 0)
			unlink (config_file);
		free (config_file);
	}
	else
	{
		Extrae_MPI_prepareDirectoryStructures (TASKID, Extrae_get_num_tasks());
		Backend_updateTaskID ();
	}

	Gather_Nodes_Info ();

	/* Generate a tentative file list, remove first if the list was generated
	   by Extrae_init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
		MPI_remove_file_list (TRUE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	MPI_Generate_Spawns_List ();
#endif

	/* Take the time now, we can't put MPIINIT_EV before APPL_EV */
	MPI_Init_start_time = TIME;
	
	/* Call a barrier in order to synchronize all tasks using MPIINIT_EV / END.
	   Three consecutive barriers for a better synchronization (J suggested) */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = MPI_Init_end_time = TIME;

	if (!Backend_postInitialize (TASKID, Extrae_get_num_tasks(), MPI_INIT_EV, MPI_Init_start_time, MPI_Init_end_time, TasksNodes))
		return;

	/* Annotate already built communicators */
	Trace_MPI_Communicator (MPI_COMM_WORLD, MPI_Init_start_time, FALSE);
	Trace_MPI_Communicator (MPI_COMM_SELF, MPI_Init_start_time, FALSE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	Spawn_Children_Sync (MPI_Init_start_time);
#endif

	/* Stats Init */
	global_mpi_stats = mpi_stats_init(Extrae_get_num_tasks());
	updateStats_OTHER(global_mpi_stats);
}


#if defined(MPI_HAS_INIT_THREAD_F)
/******************************************************************************
 ***  PMPI_Init_thread_Wrapper
 ******************************************************************************/
void PMPI_Init_thread_Wrapper (MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierror)
/* Aquest codi nomes el volem per traceig sequencial i per mpi_init de fortran */
{
	MPI_Comm cparent = MPI_COMM_NULL;
	iotimer_t MPI_Init_start_time, MPI_Init_end_time;

	hash_init (&requests);
	PR_queue_init (&PR_queue);

#ifdef WITH_PMPI_HOOK
        int (*real_mpi_init_thread)(MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierror) = NULL;
        real_mpi_init_thread = dlsym(RTLD_NEXT, STRINGIFY(CtoF77 (mpi_init_thread)));

        if (real_mpi_init_thread != NULL) {
                CtoF77 (real_mpi_init_thread) (required, provided, ierror);
        } else
#endif
        {
		CtoF77 (pmpi_init_thread) (required, provided, ierror);
        }

	Extrae_set_ApplicationIsMPI (TRUE);
	Extrae_Allocate_Task_Bitmap (Extrae_MPI_NumTasks());

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function (Extrae_MPI_TaskID);
	Extrae_set_numtasks_function (Extrae_MPI_NumTasks);
	Extrae_set_barrier_tasks_function (Extrae_MPI_Barrier);
	Extrae_set_finalize_task_function (Extrae_MPI_Finalize);

	InitMPICommunicators();

#if defined(SAMPLING_SUPPORT)
	/* If sampling is enabled, just stop all the processes at the same point
	   and continue */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
#endif

	/* Proceed with initialization if it's not already init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_NOT_INITIALIZED)
	{
		int res;
		char *config_file = getenv ("EXTRAE_CONFIG_FILE");

		if (config_file == NULL)
			config_file = getenv ("MPTRACE_CONFIG_FILE");

		Extrae_set_initial_TASKID (TASKID);
		Extrae_set_is_initialized (EXTRAE_INITIALIZED_MPI_INIT);

		if (config_file != NULL)
			/* Obtain a localized copy *except for the master process* */
			config_file = MPI_Distribute_XML_File (TASKID, Extrae_get_num_tasks(), config_file);

		/* Initialize the backend */
		res = Backend_preInitialize (TASKID, Extrae_get_num_tasks(), config_file, FALSE);
		if (!res)
			return;

		/* Remove the local copy only if we're not the master */
		if (TASKID != 0)
			unlink (config_file);
		free (config_file);
	}
	else
	{
		Extrae_MPI_prepareDirectoryStructures (TASKID, Extrae_get_num_tasks());
		Backend_updateTaskID ();
	}

	Gather_Nodes_Info ();

	/* Generate a tentative file list, remove first if the list was generated
	   by Extrae_init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
		MPI_remove_file_list (TRUE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	MPI_Generate_Spawns_List ();
#endif

	/* Take the time now, we can't put MPIINIT_EV before APPL_EV */
	MPI_Init_start_time = TIME;
	
	/* Call a barrier in order to synchronize all tasks using MPIINIT_EV / END
	   Three consecutive barriers for a better synchronization (J suggested) */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = MPI_Init_end_time = TIME;

	if (!Backend_postInitialize (TASKID, Extrae_get_num_tasks(), MPI_INIT_EV, MPI_Init_start_time, MPI_Init_end_time, TasksNodes))
		return;

	/* Annotate already built communicators */
	Trace_MPI_Communicator (MPI_COMM_WORLD, MPI_Init_start_time, FALSE);
	Trace_MPI_Communicator (MPI_COMM_SELF, MPI_Init_start_time, FALSE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	Spawn_Children_Sync (MPI_Init_start_time);
#endif

	/* Stats Init */
	global_mpi_stats = mpi_stats_init(Extrae_get_num_tasks());
	updateStats_OTHER(global_mpi_stats);
}
#endif /* MPI_HAS_INIT_THREAD_F */

//#endif
/* HSG 
     (defined(COMBINED_SYMBOLS) && !defined(MPI_C_CONTAINS_FORTRAN_MPI_INIT) || \
     !defined(COMBINED_SYMBOLS))
     */

/******************************************************************************
 ***  PMPI_Finalize_Wrapper
 ******************************************************************************/
void PMPI_Finalize_Wrapper (MPI_Fint *ierror)
{
	MPI_Comm cparent = MPI_COMM_NULL;

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
        updateStats_OTHER(global_mpi_stats);
		Extrae_MPI_stats_Wrapper (LAST_READ_TIME);
		Trace_mode_switch();
		Trace_Mode_Change (THREADID, LAST_READ_TIME);
	}

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_FINALIZE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY); 

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* Generate the final file list */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

	TRACE_MPIEVENT (TIME, MPI_FINALIZE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	/* Finalize only if its initialized by MPI_init call */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_MPI_INIT)
	{
		Backend_Finalize ();

#ifdef WITH_PMPI_HOOK
		int (*real_mpi_finalize)(MPI_Fint *ierror) = NULL;
		real_mpi_finalize = dlsym(RTLD_NEXT, STRINGIFY(CtoF77 (mpi_finalize)));
		if (real_mpi_finalize != NULL) {
			CtoF77 (real_mpi_finalize) (ierror);
		} else
#endif
		{
			CtoF77 (pmpi_finalize) (ierror);
		}

		mpitrace_on = FALSE;
	}
	else
		*ierror = MPI_SUCCESS;
}


/******************************************************************************
 ***  get_rank_obj
 ******************************************************************************/

int get_rank_obj (int *comm, int *dest, int *receiver, int send_or_recv)
{
	int ret, inter, one = 1;
	int group;
	MPI_Fint comm_world = MPI_Comm_c2f(MPI_COMM_WORLD);

        /* If rank in MPI_COMM_WORLD or if dest is PROC_NULL or any source,
           return value directly */
	if (*comm == comm_world || *dest == MPI_PROC_NULL || *dest == MPI_ANY_SOURCE)
	{
		*receiver = *dest;
	}
	else
	{
		CtoF77 (pmpi_comm_test_inter) (comm, &inter, &ret);
		MPI_CHECK(ret, pmpi_comm_test_inter);

		if (inter)
		{
                        /* The communicator is an intercommunicator */
                        int parent;
                        CtoF77( pmpi_comm_get_parent ) (&parent, &ret);

                        if (send_or_recv == RANK_OBJ_SEND)
                        {
                                if (*comm == parent)
                                {
                                        /* Send to parent -- Translate the local rank for the parent into its MPI_COMM_WORLD rank */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
                                        if (ParentWorldRanks != NULL)
                                                *receiver = ParentWorldRanks[*dest];
                                        else
#endif
                                                *receiver = *dest; /* Should never happen */
                                }
                                else
                                {
                                        /* Send to children -- When sending to specific childen X, there's no need to translate ranks */
                                        *receiver = *dest;
                                }
                        }
                        else
                        {
                                if (*comm == parent)
                                {
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
                                        /* Recv from parent -- Translate the local rank for the parent into its MPI_COMM_WORLD rank */
                                        if (ParentWorldRanks != NULL)
                                                *receiver = ParentWorldRanks[*dest];
                                        else
#endif
                                                *receiver = *dest; /* Should never happen */
                                }
                                else
                                {
                                        /* Recv from children -- When receiving from specific childen X, there's no need to translate ranks */
                                        *receiver = *dest;
                                }
                        }
		}
		else
		{
			/* The communicator is an intracommunicator */
			CtoF77 (pmpi_comm_group) (comm, &group, &ret);
			MPI_CHECK(ret, pmpi_comm_group);

			/* Translate the rank */
			CtoF77 (pmpi_group_translate_ranks) (&group, &one, dest, &grup_global_F, receiver, &ret);
			MPI_CHECK(ret, pmpi_group_translate_ranks);

			CtoF77 (pmpi_group_free) (&group, &ret);
			MPI_CHECK(ret, pmpi_group_free);
		}
	}
	return MPI_SUCCESS;
}

/******************************************************************************
 ***  PMPI_Request_get_status_Wrapper
 ******************************************************************************/

void Bursts_PMPI_Request_get_status_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
	MPI_Fint *ierror)
{
     /*
      *   event : MPI_REQUEST_GET_STATUS_EV     value : EVT_BEGIN
      *   target : ---                          size  : ---
      *   tag : ---
      */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REQUEST_GET_STATUS_EV, EVT_BEGIN, request, EMPTY, EMPTY, EMPTY, EMPTY);

	CtoF77 (pmpi_request_get_status) (request, flag, status, ierror);

     /*
      *   event : MPI_REQUEST_GET_STATUS_EV    value : EVT_END
      *   target : ---                         size  : ---
      *   tag : ---
      */
	TRACE_MPIEVENT (TIME, MPI_REQUEST_GET_STATUS_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

void Normal_PMPI_Request_get_status_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status,
    MPI_Fint *ierror)
{
  static int PMPI_Request_get_status_counter = 0;
  iotimer_t begin_time, end_time;
  static iotimer_t elapsed_time_outside_PMPI_Request_get_status = 0, last_PMPI_Request_get_status_exit_time = 0;


  begin_time = LAST_READ_TIME;

  if (PMPI_Request_get_status_counter == 0) {
    /* First request */
    elapsed_time_outside_PMPI_Request_get_status = 0;
  }
  else {
    elapsed_time_outside_PMPI_Request_get_status += (begin_time - last_PMPI_Request_get_status_exit_time);
  }

  CtoF77 (pmpi_request_get_status) (request, flag, status, ierror);

  end_time = TIME; 
  last_PMPI_Request_get_status_exit_time = end_time;

	if (tracejant_mpi)
  {
    if (*flag)
    {
      if (PMPI_Request_get_status_counter != 0) {
        TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_MPI_REQUEST_GET_STATUS_EV, elapsed_time_outside_PMPI_Request_get_status);
        TRACE_EVENT (begin_time, MPI_REQUEST_GET_STATUS_COUNTER_EV, PMPI_Request_get_status_counter);
      }
      TRACE_MPIEVENT (begin_time, MPI_REQUEST_GET_STATUS_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

      TRACE_MPIEVENT (end_time, MPI_REQUEST_GET_STATUS_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
      PMPI_Request_get_status_counter = 0;
    }
    else
    {
      if (PMPI_Request_get_status_counter == 0)
      {
        /* First request fail */
        TRACE_EVENTANDCOUNTERS (begin_time, MPI_REQUEST_GET_STATUS_COUNTER_EV, 0, TRUE);
      }
      PMPI_Request_get_status_counter ++;
    }
  }
}

void PMPI_Request_get_status_Wrapper (MPI_Fint *request, MPI_Fint *flag, MPI_Fint *status, MPI_Fint *ierror)
{
	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
		Bursts_PMPI_Request_get_status_Wrapper (request, flag, status, ierror);
	}
	else
	{
		Normal_PMPI_Request_get_status_Wrapper (request, flag, status, ierror);
	}
}


/******************************************************************************
 ***  PMPI_Cancel_Wrapper
 ******************************************************************************/

void PMPI_Cancel_Wrapper (MPI_Fint *request, MPI_Fint *ierror)
{
	MPI_Request req = MPI_Request_f2c(*request);

  /*
   *   event : CANCEL_EV                    value : EVT_BEGIN
   *   target : request to cancel           size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_CANCEL_EV, EVT_BEGIN, req, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  CtoF77 (pmpi_cancel) (request, ierror);

  /*
   *   event : CANCEL_EV                    value : EVT_END
   *   target : request to cancel           size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (TIME, MPI_CANCEL_EV, EVT_END, req, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  get_Irank_obj
 ******************************************************************************/

int get_Irank_obj (hash_data_t * hash_req, int *src_world, int *size,
	int *tag, int *status)
{
	int ret, one = 1;
	MPI_Fint tbyte = MPI_Type_c2f(MPI_BYTE);
	int recved_count, dest;

	CtoF77 (pmpi_get_count) (status, &tbyte, &recved_count, &ret);
	MPI_CHECK(ret, pmpi_get_count);

	if (recved_count != MPI_UNDEFINED)
		*size = recved_count;
	else
		*size = 0;

	*tag = status[MPI_TAG_OFFSET];
	dest = status[MPI_SOURCE_OFFSET];

	if (MPI_GROUP_NULL != hash_req->group)
	{
		MPI_Fint group = MPI_Group_c2f(hash_req->group);
		CtoF77 (pmpi_group_translate_ranks) (&group, &one, &dest, &grup_global_F, src_world, &ret);
		MPI_CHECK(ret, pmpi_group_translate_ranks);
	}
	else
		*src_world = dest;

  return MPI_SUCCESS;
}


/******************************************************************************
 ***  PMPI_Comm_Rank_Wrapper
 ******************************************************************************/

void PMPI_Comm_Rank_Wrapper (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierror)
{
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_RANK_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);
	CtoF77 (pmpi_comm_rank) (comm, rank, ierror);
	TRACE_MPIEVENT (TIME, MPI_COMM_RANK_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


/******************************************************************************
 ***  PMPI_Comm_Size_Wrapper
 ******************************************************************************/

void PMPI_Comm_Size_Wrapper (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierror)
{
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_SIZE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);
	CtoF77 (pmpi_comm_size) (comm, size, ierror);
	TRACE_MPIEVENT (TIME, MPI_COMM_SIZE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  PMPI_Comm_Create_Wrapper
 ******************************************************************************/

void PMPI_Comm_Create_Wrapper (MPI_Fint *comm, MPI_Fint *group,
	MPI_Fint *newcomm, MPI_Fint *ierror)
{
	MPI_Fint cnull;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77 (pmpi_comm_create) (comm, group, newcomm, ierror);

	if (*newcomm != cnull && *ierror == MPI_SUCCESS)
	{	
		MPI_Comm comm_id = MPI_Comm_f2c(*newcomm);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT (TIME, MPI_COMM_CREATE_EV, EVT_END, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  PMPI_Comm_Free_Wrapper
 ******************************************************************************/

void PMPI_Comm_Free_Wrapper (MPI_Fint *comm, MPI_Fint *ierror)
{
	UNREFERENCED_PARAMETER(comm);

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_FREE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	*ierror = MPI_SUCCESS;

	TRACE_MPIEVENT (TIME, MPI_COMM_FREE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  PMPI_Comm_Dup_Wrapper
 ******************************************************************************/

void PMPI_Comm_Dup_Wrapper (MPI_Fint *comm, MPI_Fint *newcomm,
	MPI_Fint *ierror)
{
	MPI_Fint cnull;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_DUP_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77 (pmpi_comm_dup) (comm, newcomm, ierror);

	if (*newcomm != cnull && *ierror == MPI_SUCCESS)
	{
		MPI_Comm comm_id = MPI_Comm_f2c (*newcomm);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT (TIME, MPI_COMM_DUP_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
		EMPTY);

	updateStats_OTHER(global_mpi_stats);
}



/******************************************************************************
 ***  PMPI_Comm_Split_Wrapper
 ******************************************************************************/

void PMPI_Comm_Split_Wrapper (MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key,
	MPI_Fint *newcomm, MPI_Fint *ierror)
{
	MPI_Fint cnull;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_SPLIT_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	cnull = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77 (pmpi_comm_split) (comm, color, key, newcomm, ierror);

	if (*newcomm != cnull && *ierror == MPI_SUCCESS)
	{
		MPI_Comm comm_id = MPI_Comm_f2c (*newcomm);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT (TIME, MPI_COMM_SPLIT_EV, EVT_END, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  PMPI_Comm_Spawn_Wrapper
 ******************************************************************************/
void PMPI_Comm_Spawn_Wrapper (char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
{
  unsigned long long SpawnStartTime = LAST_READ_TIME;

  TRACE_MPIEVENT (SpawnStartTime, MPI_COMM_SPAWN_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  CtoF77 (pmpi_comm_spawn) (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes, ierror);

  if (*ierror == MPI_SUCCESS)
  {
    MPI_Comm intercomm_c;
    intercomm_c = PMPI_Comm_f2c(*intercomm);

    MPI_Comm comm_c;
    comm_c = PMPI_Comm_f2c(*comm);
    Spawn_Parent_Sync (SpawnStartTime, intercomm_c, comm_c);
  }

  TRACE_MPIEVENT (TIME, MPI_COMM_SPAWN_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  updateStats_OTHER(global_mpi_stats);
}

/******************************************************************************
 ***  PMPI_Comm_Spawn_Multiple_Wrapper
 ******************************************************************************/
void PMPI_Comm_Spawn_Multiple_Wrapper (MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierror)
{
  unsigned long long SpawnStartTime = LAST_READ_TIME;

  TRACE_MPIEVENT (SpawnStartTime, MPI_COMM_SPAWN_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  CtoF77 (pmpi_comm_spawn_multiple) (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes, ierror);

  if (*ierror == MPI_SUCCESS)
  { 
    MPI_Comm intercomm_c;
    intercomm_c = PMPI_Comm_f2c(*intercomm);

    MPI_Comm comm_c;
    comm_c = PMPI_Comm_f2c(*comm);

    Spawn_Parent_Sync (SpawnStartTime, intercomm_c, comm_c);
  }

  TRACE_MPIEVENT (TIME, MPI_COMM_SPAWN_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  updateStats_OTHER(global_mpi_stats);
}
#endif


/******************************************************************************
 ***  PMPI_Start_Wrapper
 ******************************************************************************/

void PMPI_Start_Wrapper (MPI_Fint *request, MPI_Fint *ierror)
{
	MPI_Request req;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_START_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	/* Execute the real function */
	CtoF77 (pmpi_start) (request, ierror);

	/* Store the resulting request */
	req = MPI_Request_f2c(*request);
	Traceja_Persistent_Request (&req, LAST_READ_TIME);

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
	TRACE_MPIEVENT (TIME, MPI_START_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}


/******************************************************************************
 ***  PMPI_Startall_Wrapper
 ******************************************************************************/

void PMPI_Startall_Wrapper (MPI_Fint *count, MPI_Fint array_of_requests[],
	MPI_Fint *ierror)
{
  MPI_Fint save_reqs[MAX_WAIT_REQUESTS];
  int ii;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_STARTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Algunes implementacions es poden carregar aquesta informacio.
   * Cal salvar-la per poder tracejar després de fer la crida pmpi. 
   */
  memcpy (save_reqs, array_of_requests, (*count) * sizeof (MPI_Fint));

  /*
   * Primer cal fer la crida real 
   */
  CtoF77 (pmpi_startall) (count, array_of_requests, ierror);

  /*
   * Es tracejen totes les requests 
   */
	for (ii = 0; ii < (*count); ii++)
	{
		MPI_Request req = MPI_Request_f2c(save_reqs[ii]);
		Traceja_Persistent_Request (&req, LAST_READ_TIME);
	}

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_STARTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);
}



/******************************************************************************
 ***  PMPI_Request_free_Wrapper
 ******************************************************************************/

void PMPI_Request_free_Wrapper (MPI_Fint *request, MPI_Fint *ierror)
{
	MPI_Request req;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_REQUEST_FREE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY,
                  EMPTY, EMPTY);

  /*
   * Cal guardar la request perque algunes implementacions se la carreguen. 
   */
  req = MPI_Request_f2c (*request);

  /*
   * S'intenta alliberar aquesta persistent request 
   */
  PR_Elimina_request (&PR_queue, &req);

  /*
   * Primer cal fer la crida real 
   */
  CtoF77 (pmpi_request_free) (request, ierror);

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_REQUEST_FREE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);
}


void PMPI_Cart_sub_Wrapper (MPI_Fint *comm, MPI_Fint *remain_dims,
	MPI_Fint *comm_new, MPI_Fint *ierror)
{
	MPI_Fint comm_null;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_CART_SUB_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	comm_null = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77 (pmpi_cart_sub) (comm, remain_dims, comm_new, ierror);

	if (*ierror == MPI_SUCCESS && *comm_new != comm_null)
	{
		MPI_Comm comm_id = MPI_Comm_f2c (*comm_new);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT (TIME, MPI_CART_SUB_EV, EVT_END, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_Cart_create_Wrapper (MPI_Fint *comm_old, MPI_Fint *ndims,
	MPI_Fint *dims, MPI_Fint *periods, MPI_Fint *reorder, MPI_Fint *comm_cart,
	MPI_Fint *ierror)
{
	MPI_Fint comm_null;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_CART_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	comm_null = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77 (pmpi_cart_create) (comm_old, ndims, dims, periods, reorder,
	  comm_cart, ierror);

	if (*ierror == MPI_SUCCESS && *comm_cart != comm_null)
	{
		MPI_Comm comm_id = MPI_Comm_f2c (*comm_cart);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_CART_CREATE_EV, EVT_END, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);
}

void PMPI_Intercomm_create_F_Wrapper (MPI_Fint *local_comm, MPI_Fint *local_leader,
	MPI_Fint *peer_comm, MPI_Fint *remote_leader, MPI_Fint *tag,
	MPI_Fint *newintercomm, MPI_Fint *ierror)
{
	MPI_Fint comm_null;

	TRACE_MPIEVENT(LAST_READ_TIME, MPI_INTERCOMM_CREATE_EV, EVT_BEGIN,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	comm_null = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77(pmpi_intercomm_create) (local_comm, local_leader, peer_comm,
	  remote_leader, tag, newintercomm, ierror);

	if (*ierror == MPI_SUCCESS && *newintercomm != comm_null)
		Trace_MPI_InterCommunicator (MPI_Comm_f2c (*newintercomm),
		  MPI_Comm_f2c(*local_comm), *local_leader,
		  MPI_Comm_f2c(*peer_comm), *remote_leader,
		  LAST_READ_TIME, TRUE);

	TRACE_MPIEVENT(TIME, MPI_INTERCOMM_CREATE_EV, EVT_END,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

void PMPI_Intercomm_merge_F_Wrapper (MPI_Fint *intercomm, MPI_Fint *high,
	MPI_Fint *newintracomm, MPI_Fint *ierror)
{
	MPI_Fint comm_null;

	TRACE_MPIEVENT(LAST_READ_TIME, MPI_INTERCOMM_MERGE_EV, EVT_BEGIN,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	comm_null = MPI_Comm_c2f(MPI_COMM_NULL);

	CtoF77(mpi_intercomm_merge) (intercomm, high, newintracomm, ierror);

	if (*ierror == MPI_SUCCESS && *newintracomm != comm_null)
	{
		MPI_Comm comm_id = MPI_Comm_f2c (*newintracomm);
		Trace_MPI_Communicator (comm_id, LAST_READ_TIME, TRUE);
	}

	TRACE_MPIEVENT(TIME, MPI_INTERCOMM_MERGE_EV, EVT_END,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
}

#endif /* defined(FORTRAN_SYMBOLS) */

#if defined(C_SYMBOLS)

/******************************************************************************
 ***  get_Irank_obj_C
 ******************************************************************************/

int get_Irank_obj_C (hash_data_t * hash_req, int *src_world, int *size,
	int *tag, MPI_Status *status)
{
	int ret, dest, recved_count;

	ret = PMPI_Get_count (status, MPI_BYTE, &recved_count);
	MPI_CHECK(ret, PMPI_Get_count);

	if (recved_count != MPI_UNDEFINED)
		*size = recved_count;
	else
		*size = 0;

	*tag = status->MPI_TAG;
	dest = status->MPI_SOURCE;

	if (MPI_GROUP_NULL != hash_req->group)
	{
		ret = PMPI_Group_translate_ranks (hash_req->group, 1, &dest, grup_global,
			src_world);
		MPI_CHECK(ret, PMPI_Group_translate_ranks);
	}
	else
		*src_world = dest;

	return MPI_SUCCESS;
}


/******************************************************************************
*******************************************************************************
*******************************************************************************
*****************************  Wrappers versio C ******************************
*******************************************************************************
*******************************************************************************
*******************************************************************************/

/******************************************************************************
 ***  MPI_Init_C_Wrapper
 ******************************************************************************/

int MPI_Init_C_Wrapper (int *argc, char ***argv)
{
	MPI_Comm cparent = MPI_COMM_NULL;
	int val = 0;
	iotimer_t MPI_Init_start_time, MPI_Init_end_time;

	hash_init (&requests);
	PR_queue_init (&PR_queue);

#ifdef WITH_PMPI_HOOK
	int (*real_mpi_init)(int *argc, char ***argv) = NULL;
	real_mpi_init = dlsym(RTLD_NEXT, "MPI_Init");
	if (real_mpi_init != NULL) {
		val = real_mpi_init (argc, argv);
	} else
#endif
	{
		val = PMPI_Init (argc, argv);
	}

	Extrae_set_ApplicationIsMPI (TRUE);
	Extrae_Allocate_Task_Bitmap (Extrae_MPI_NumTasks());

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function (Extrae_MPI_TaskID);
	Extrae_set_numtasks_function (Extrae_MPI_NumTasks);
	Extrae_set_barrier_tasks_function (Extrae_MPI_Barrier);
	Extrae_set_finalize_task_function (Extrae_MPI_Finalize);

	InitMPICommunicators();

#if defined(SAMPLING_SUPPORT)
	/* If sampling is enabled, just stop all the processes at the same point
	   and continue */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
#endif

	/* Proceed with initialization if it's not already init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_NOT_INITIALIZED)
	{
		int res;
		char *config_file = getenv ("EXTRAE_CONFIG_FILE");

		if (config_file == NULL)
			config_file = getenv ("MPTRACE_CONFIG_FILE");

		Extrae_set_initial_TASKID (TASKID);
		Extrae_set_is_initialized (EXTRAE_INITIALIZED_MPI_INIT);

		if (config_file != NULL)
			/* Obtain a localized copy *except for the master process* */
			config_file = MPI_Distribute_XML_File (TASKID, Extrae_get_num_tasks(), config_file);

		/* Initialize the backend */
		res = Backend_preInitialize (TASKID, Extrae_get_num_tasks(), config_file, FALSE);
		if (!res)
			return val;

		/* Remove the local copy only if we're not the master */
		if (TASKID != 0)
			unlink (config_file);
		free (config_file);
	}
	else
	{
		Extrae_MPI_prepareDirectoryStructures (TASKID, Extrae_get_num_tasks());
		Backend_updateTaskID ();
	}

	Gather_Nodes_Info ();

	/* Generate a tentative file list, remove first if the list was generated
	   by Extrae_init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
		MPI_remove_file_list (TRUE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	MPI_Generate_Spawns_List ();
#endif 

	/* Take the time now, we can't put MPIINIT_EV before APPL_EV */
	MPI_Init_start_time = TIME;

	/* Call a barrier in order to synchronize all tasks using MPIINIT_EV / END
	   Three consecutive barriers for a better synchronization (J suggested) */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = MPI_Init_end_time = TIME;

	if (!Backend_postInitialize (TASKID, Extrae_get_num_tasks(), MPI_INIT_EV, MPI_Init_start_time, MPI_Init_end_time, TasksNodes))
		return val;

	/* Annotate already built communicators */
	Trace_MPI_Communicator (MPI_COMM_WORLD, MPI_Init_start_time, FALSE);
	Trace_MPI_Communicator (MPI_COMM_SELF, MPI_Init_start_time, FALSE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
        Spawn_Children_Sync( MPI_Init_start_time );
#endif

	/* Stats Init */
        global_mpi_stats = mpi_stats_init(Extrae_get_num_tasks());
	updateStats_OTHER(global_mpi_stats);

	return val;
}


#if defined(MPI_HAS_INIT_THREAD_C)
int MPI_Init_thread_C_Wrapper (int *argc, char ***argv, int required, int *provided)
{
	MPI_Comm cparent = MPI_COMM_NULL;
	int val = 0;
	iotimer_t MPI_Init_start_time, MPI_Init_end_time;

	hash_init (&requests);
	PR_queue_init (&PR_queue);

#ifdef WITH_PMPI_HOOK
	int (*real_mpi_init_thread)(int *argc, char ***argv, int required, int *provided) = NULL;
	real_mpi_init_thread = dlsym(RTLD_NEXT, "MPI_Init_thread");
	if (real_mpi_init_thread != NULL) {
		val = real_mpi_init_thread (argc, argv, required, provided);
	} else
#endif
	{
		val = PMPI_Init_thread (argc, argv, required, provided);
	}

	Extrae_set_ApplicationIsMPI (TRUE);
	Extrae_Allocate_Task_Bitmap (Extrae_MPI_NumTasks());

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function (Extrae_MPI_TaskID);
	Extrae_set_numtasks_function (Extrae_MPI_NumTasks);
	Extrae_set_barrier_tasks_function (Extrae_MPI_Barrier);
	Extrae_set_finalize_task_function (Extrae_MPI_Finalize);

	InitMPICommunicators();

#if defined(SAMPLING_SUPPORT)
	/* If sampling is enabled, just stop all the processes at the same point
	   and continue */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
#endif

	/* Proceed with initialization if it's not already init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_NOT_INITIALIZED)
	{
		int res;
		char *config_file = getenv ("EXTRAE_CONFIG_FILE");

		if (config_file == NULL)
			config_file = getenv ("MPTRACE_CONFIG_FILE");

		Extrae_set_initial_TASKID (TASKID);
		Extrae_set_is_initialized (EXTRAE_INITIALIZED_MPI_INIT);

		if (config_file != NULL)
			/* Obtain a localized copy *except for the master process* */
			config_file = MPI_Distribute_XML_File (TASKID, Extrae_get_num_tasks(), config_file);

		/* Initialize the backend */
		res = Backend_preInitialize (TASKID, Extrae_get_num_tasks(), config_file, FALSE);
		if (!res)
			return val;

		/* Remove the local copy only if we're not the master */
		if (TASKID != 0)
			unlink (config_file);
		free (config_file);
	}
	else
	{
		Extrae_MPI_prepareDirectoryStructures (TASKID, Extrae_get_num_tasks());
		Backend_updateTaskID ();
	}

	Gather_Nodes_Info ();

	/* Generate a tentative file list, remove first if the list was generated
	   by Extrae_init */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
		MPI_remove_file_list (TRUE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	MPI_Generate_Spawns_List ();
#endif

	/* Take the time now, we can't put MPIINIT_EV before APPL_EV */
	MPI_Init_start_time = TIME;

	/* Call a barrier in order to synchronize all tasks using MPIINIT_EV / END
	   Three consecutive barriers for a better synchronization (J suggested) */
	Extrae_barrier_tasks();  /* will default to MPI_BARRIER */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = MPI_Init_end_time = TIME;

	if (!Backend_postInitialize (TASKID, Extrae_get_num_tasks(), MPI_INIT_EV, MPI_Init_start_time, MPI_Init_end_time, TasksNodes))
		return val;

	/* Annotate already built communicators */
	Trace_MPI_Communicator (MPI_COMM_WORLD, MPI_Init_start_time, FALSE);
	Trace_MPI_Communicator (MPI_COMM_SELF, MPI_Init_start_time, FALSE);

#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
        Spawn_Children_Sync (MPI_Init_start_time);
#endif

	/* Stats Init */
        global_mpi_stats = mpi_stats_init(Extrae_get_num_tasks());
	updateStats_OTHER(global_mpi_stats);

	return val;
}
#endif /* MPI_HAS_INIT_THREAD_C */


/******************************************************************************
 ***  MPI_Finalize_C_Wrapper
 ******************************************************************************/

int MPI_Finalize_C_Wrapper (void)
{
	MPI_Comm cparent = MPI_COMM_NULL;
	int ierror = 0;

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 1;
#endif

	if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
	{
        updateStats_OTHER(global_mpi_stats);
		Extrae_MPI_stats_Wrapper (LAST_READ_TIME);
		Trace_mode_switch();
		Trace_Mode_Change (THREADID, LAST_READ_TIME);
	}

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_FINALIZE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

#if defined(IS_BGL_MACHINE)
	BGL_disable_barrier_inside = 0;
#endif

	/* Generate the final file list */
#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
	PMPI_Comm_get_parent (&cparent);
#endif
	MPI_Generate_Task_File_List (TasksNodes, cparent != MPI_COMM_NULL);

	TRACE_MPIEVENT (TIME, MPI_FINALIZE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	/* Finalize only if its initialized by MPI_init call */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_MPI_INIT)
	{
		Backend_Finalize ();

#ifdef WITH_PMPI_HOOK
		int (*real_mpi_finalize)() = NULL;
		real_mpi_finalize = dlsym(RTLD_NEXT, "MPI_Finalize");
		if (real_mpi_finalize != NULL) {
			ierror = real_mpi_finalize();
		} else
#endif
		{
			ierror = PMPI_Finalize();
		}

		mpitrace_on = FALSE;
	}
	else
		ierror = MPI_SUCCESS;

	return ierror;
}

/******************************************************************************
 ***  MPI_Request_get_status_C_Wrapper
 ******************************************************************************/
int Bursts_MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status)
{
	int ierror;
	/*
	*   event : MPI_REQUEST_GET_STATUS_EV                     value : EVT_BEGIN
	*   target : ---                          size  : ---
	*   tag : ---
	*/
	TRACE_MPIEVENT (LAST_READ_TIME, MPI_REQUEST_GET_STATUS_EV, EVT_BEGIN, request, EMPTY, EMPTY, EMPTY, EMPTY);

    ierror = PMPI_Request_get_status(request, flag, status);
	/*
	*   event : MPI_REQUEST_GET_STATUS_EV                     value : EVT_END
	*   target : ---                          size  : ---
	*   tag : ---
	*/
    TRACE_MPIEVENT (TIME, MPI_REQUEST_GET_STATUS_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
    return ierror;
}

int Normal_MPI_Request_get_status(MPI_Request request, int *flag, MPI_Status *status)
{
    static int MPI_Request_get_status_counter = 0;
    static iotimer_t elapsed_time_outside_MPI_Request_get_status_C = 0, last_MPI_Request_get_status_C_exit_time = 0; 
	iotimer_t begin_time, end_time;
	int ierror;

	begin_time = LAST_READ_TIME;

	if (MPI_Request_get_status_counter == 0)
	{
		/* Primer request */
		elapsed_time_outside_MPI_Request_get_status_C = 0;
	}
	else
	{
		elapsed_time_outside_MPI_Request_get_status_C += (begin_time - last_MPI_Request_get_status_C_exit_time);
	}

    ierror = PMPI_Request_get_status(request, flag, status);
	end_time = TIME;
	last_MPI_Request_get_status_C_exit_time = end_time;

	if (tracejant_mpi)
	{
		if (*flag)
		{
			if (MPI_Request_get_status_counter != 0)
			{
				TRACE_EVENT (begin_time, MPI_TIME_OUTSIDE_MPI_REQUEST_GET_STATUS_EV, elapsed_time_outside_MPI_Request_get_status_C);
				TRACE_EVENT (begin_time, MPI_REQUEST_GET_STATUS_COUNTER_EV, MPI_Request_get_status_counter);
			}

			TRACE_MPIEVENT (begin_time, MPI_REQUEST_GET_STATUS_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
    
			TRACE_MPIEVENT (end_time, MPI_REQUEST_GET_STATUS_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
			MPI_Request_get_status_counter = 0;
		} 
		else
		{
			if (MPI_Request_get_status_counter == 0)
			{
				/* El primer request que falla */
				TRACE_EVENTANDCOUNTERS (begin_time, MPI_REQUEST_GET_STATUS_COUNTER_EV, 0, TRUE);
			}
			MPI_Request_get_status_counter ++;
		}
	}
	return ierror;

}

int MPI_Request_get_status_C_Wrapper(MPI_Request request, int *flag, MPI_Status *status)
{
    int ret;

    if (CURRENT_TRACE_MODE(THREADID) == TRACE_MODE_BURSTS)
    {
        ret = Bursts_MPI_Request_get_status(request, flag, status);
    }
    else
    {
        ret = Normal_MPI_Request_get_status(request, flag, status);
    }
    return ret;
}


/******************************************************************************
 ***  MPI_Cancel_C_Wrapper
 ******************************************************************************/

int MPI_Cancel_C_Wrapper (MPI_Request *request)
{
  int ierror;

  /*
   *   event : CANCEL_EV                    value : EVT_BEGIN
   *   target : request to cancel           size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_CANCEL_EV, EVT_BEGIN, *request, EMPTY, EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Cancel (request);

  /*
   *   event : CANCEL_EV                    value : EVT_END
   *   target : request to cancel           size  : ---
   *   tag : ---
   */
  TRACE_MPIEVENT (TIME, MPI_CANCEL_EV, EVT_END, *request, EMPTY, EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Comm_rank_C_Wrapper
 ******************************************************************************/

int MPI_Comm_rank_C_Wrapper (MPI_Comm comm, int *rank)
{
	int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_RANK_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);
	ierror = PMPI_Comm_rank (comm, rank);
	TRACE_MPIEVENT (TIME, MPI_COMM_RANK_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}



/******************************************************************************
 ***  MPI_Comm_size_C_Wrapper
 ******************************************************************************/

int MPI_Comm_size_C_Wrapper (MPI_Comm comm, int *size)
{
	int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_SIZE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);
	ierror = PMPI_Comm_size (comm, size);
	TRACE_MPIEVENT (TIME, MPI_COMM_SIZE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}


/******************************************************************************
 ***  MPI_Comm_create_C_Wrapper
 ******************************************************************************/

int MPI_Comm_create_C_Wrapper (MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)
{
  int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Comm_create (comm, group, newcomm);
  if (*newcomm != MPI_COMM_NULL && ierror == MPI_SUCCESS)
    Trace_MPI_Communicator (*newcomm, LAST_READ_TIME, FALSE);

	TRACE_MPIEVENT (TIME, MPI_COMM_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}

/******************************************************************************
 ***  MPI_Comm_create_C_Wrapper
 ******************************************************************************/

int MPI_Comm_free_C_Wrapper (MPI_Comm *comm)
{
	UNREFERENCED_PARAMETER(comm);

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_FREE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	TRACE_MPIEVENT (TIME, MPI_COMM_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
	  EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return MPI_SUCCESS;
}


/******************************************************************************
 ***  MPI_Comm_dup_C_Wrapper
 ******************************************************************************/

int MPI_Comm_dup_C_Wrapper (MPI_Comm comm, MPI_Comm *newcomm)
{
  int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_DUP_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Comm_dup (comm, newcomm);
  if (*newcomm != MPI_COMM_NULL && ierror == MPI_SUCCESS)
    Trace_MPI_Communicator (*newcomm, LAST_READ_TIME, FALSE);

	TRACE_MPIEVENT (TIME, MPI_COMM_DUP_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
		EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


/******************************************************************************
 ***  MPI_Comm_split_C_Wrapper
 ******************************************************************************/

int MPI_Comm_split_C_Wrapper (MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
  int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_COMM_SPLIT_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Comm_split (comm, color, key, newcomm);
  if (*newcomm != MPI_COMM_NULL && ierror == MPI_SUCCESS)
    Trace_MPI_Communicator (*newcomm, LAST_READ_TIME, FALSE);

	TRACE_MPIEVENT (TIME, MPI_COMM_SPLIT_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
		EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}


#if defined(MPI_SUPPORTS_MPI_COMM_SPAWN)
/******************************************************************************
 ***  MPI_Comm_spawn_C_Wrapper
 ******************************************************************************/
int MPI_Comm_spawn_C_Wrapper (char *command, char **argv, int maxprocs, MPI_Info info,
  int root, MPI_Comm comm, MPI_Comm *intercomm, int *array_of_errcodes)
{
  int ierror;
  unsigned long long SpawnStartTime = LAST_READ_TIME;

  TRACE_MPIEVENT (SpawnStartTime, MPI_COMM_SPAWN_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Comm_spawn (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes);

  if (ierror == MPI_SUCCESS)
  {
    Spawn_Parent_Sync (SpawnStartTime, *intercomm, comm);
  }

  TRACE_MPIEVENT (TIME, MPI_COMM_SPAWN_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  updateStats_COLLECTIVE(global_mpi_stats, 0, 0);

  return ierror;
}
 
/******************************************************************************
 ***  MPI_Comm_spawn_multiple_C_Wrapper
 ******************************************************************************/

int MPI_Comm_spawn_multiple_C_Wrapper (int count, char *array_of_commands[], char* *array_of_argv[],
  int array_of_maxprocs[], MPI_Info array_of_info[], int root, MPI_Comm comm,
  MPI_Comm *intercomm, int array_of_errcodes[])
{
  int ierror;
  unsigned long long SpawnStartTime = LAST_READ_TIME;

  TRACE_MPIEVENT (SpawnStartTime, MPI_COMM_SPAWN_MULTIPLE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  ierror = PMPI_Comm_spawn_multiple (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info, root, comm, intercomm, array_of_errcodes);

  if (ierror == MPI_SUCCESS)
  {
    Spawn_Parent_Sync (SpawnStartTime, *intercomm, comm);
  }

  TRACE_MPIEVENT (TIME, MPI_COMM_SPAWN_MULTIPLE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

  return ierror;
}
#endif /* MPI_SUPPORTS_MPI_COMM_SPAWN */

/******************************************************************************
 ***  MPI_Cart_create
 ******************************************************************************/
int MPI_Cart_create_C_Wrapper (MPI_Comm comm_old, int ndims, int *dims,
                               int *periods, int reorder, MPI_Comm *comm_cart)
{
	int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_CART_CREATE_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	ierror = PMPI_Cart_create (comm_old, ndims, dims, periods, reorder,
	  comm_cart);

	if (ierror == MPI_SUCCESS && *comm_cart != MPI_COMM_NULL)
		Trace_MPI_Communicator (*comm_cart, LAST_READ_TIME, FALSE);

	TRACE_MPIEVENT (TIME, MPI_CART_CREATE_EV, EVT_END, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}

/* -------------------------------------------------------------------------
   MPI_Cart_sub
   ------------------------------------------------------------------------- */
int MPI_Cart_sub_C_Wrapper (MPI_Comm comm, int *remain_dims, MPI_Comm *comm_new)
{
	int ierror;

	TRACE_MPIEVENT (LAST_READ_TIME, MPI_CART_SUB_EV, EVT_BEGIN, EMPTY, EMPTY,
		EMPTY, EMPTY, EMPTY);

	ierror = PMPI_Cart_sub (comm, remain_dims, comm_new);

	if (ierror == MPI_SUCCESS && *comm_new != MPI_COMM_NULL)
		Trace_MPI_Communicator (*comm_new, LAST_READ_TIME, FALSE);

	TRACE_MPIEVENT (TIME, MPI_CART_SUB_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
		EMPTY); 

	updateStats_OTHER(global_mpi_stats);

	return ierror;
}

/* -------------------------------------------------------------------------
   MPI_Intercomm_create
   ------------------------------------------------------------------------- */
int MPI_Intercomm_create_C_Wrapper (MPI_Comm local_comm, int local_leader,
	MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm)
{
	int ierror;

	TRACE_MPIEVENT(LAST_READ_TIME, MPI_INTERCOMM_MERGE_EV, EVT_BEGIN,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	ierror = PMPI_Intercomm_create (local_comm, local_leader, peer_comm,
	  remote_leader, tag, newintercomm);

	if (ierror == MPI_SUCCESS && *newintercomm != MPI_COMM_NULL)
		Trace_MPI_InterCommunicator (*newintercomm, local_comm, local_leader,
		  peer_comm, remote_leader, LAST_READ_TIME, TRUE);

	TRACE_MPIEVENT(TIME, MPI_INTERCOMM_MERGE_EV, EVT_END,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}

/* -------------------------------------------------------------------------
   MPI_Intercomm_merge
   ------------------------------------------------------------------------- */
int MPI_Intercomm_merge_C_Wrapper (MPI_Comm intercomm, int high,
	MPI_Comm *newintracomm)
{
	int ierror;

	TRACE_MPIEVENT(LAST_READ_TIME, MPI_INTERCOMM_MERGE_EV, EVT_BEGIN,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	ierror = PMPI_Intercomm_merge (intercomm, high, newintracomm);

	if (ierror == MPI_SUCCESS && *newintracomm != MPI_COMM_NULL)
		Trace_MPI_Communicator (*newintracomm, LAST_READ_TIME, TRUE);

	TRACE_MPIEVENT(TIME, MPI_INTERCOMM_MERGE_EV, EVT_END,
	  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);

	return ierror;
}


/******************************************************************************
 ***  MPI_Start_C_Wrapper
 ******************************************************************************/

int MPI_Start_C_Wrapper (MPI_Request *request)
{
  int ierror;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_START_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /* Primer cal fer la crida real */
  ierror = PMPI_Start (request);

  /* S'intenta tracejar aquesta request */
  Traceja_Persistent_Request (request, LAST_READ_TIME);

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_START_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY);
  return ierror;
}


/******************************************************************************
 ***  MPI_Startall_C_Wrapper
 ******************************************************************************/

int MPI_Startall_C_Wrapper (int count, MPI_Request *array_of_requests)
{
  MPI_Request save_reqs[MAX_WAIT_REQUESTS];
  int ii, ierror;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_STARTALL_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

  /*
   * Algunes implementacions es poden carregar aquesta informacio.
   * Cal salvar-la per poder tracejar després de fer la crida pmpi. 
   */
  memcpy (save_reqs, array_of_requests, count * sizeof (MPI_Request));

  /* Primer cal fer la crida real */
  ierror = PMPI_Startall (count, array_of_requests);

  /* Es tracejen totes les requests */
  for (ii = 0; ii < count; ii++)
    Traceja_Persistent_Request (&(save_reqs[ii]), LAST_READ_TIME);

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_STARTALL_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);
  return ierror;
}


/******************************************************************************
 ***  MPI_Request_free_C_Wrapper
 ******************************************************************************/
int MPI_Request_free_C_Wrapper (MPI_Request *request)
{
  int ierror;

  /*
   *   type : START_EV                     value : EVT_BEGIN
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (LAST_READ_TIME, MPI_REQUEST_FREE_EV, EVT_BEGIN, EMPTY, EMPTY, EMPTY,
                  EMPTY, EMPTY);

  /* Free from our structures */
  PR_Elimina_request (&PR_queue, request);

  /* Perform the real call */
  ierror = PMPI_Request_free (request);

  /*
   *   type : START_EV                     value : EVT_END
   *   target : ---                        size  : ----
   *   tag : ---                           comm : ---
   *   aux : ---
   */
  TRACE_MPIEVENT (TIME, MPI_REQUEST_FREE_EV, EVT_END, EMPTY, EMPTY, EMPTY, EMPTY,
                  EMPTY);

	updateStats_OTHER(global_mpi_stats);

  return ierror;
}

#endif /* defined(C_SYMBOLS) */

void Extrae_MPI_stats_Wrapper (iotimer_t timestamp)
{
  int i=0;
  unsigned int vec_types [MPI_STATS_EVENTS_COUNT];
  for (i=0; i<MPI_STATS_EVENTS_COUNT; i++)
    vec_types[i] = MPI_STATS_EV;

  unsigned int vec_values[MPI_STATS_EVENTS_COUNT] = {
    MPI_STATS_P2P_COUNT_EV,
    MPI_STATS_P2P_BYTES_SENT_EV,
    MPI_STATS_P2P_BYTES_RECV_EV,
    MPI_STATS_GLOBAL_COUNT_EV,
    MPI_STATS_GLOBAL_BYTES_SENT_EV,
    MPI_STATS_GLOBAL_BYTES_RECV_EV,
    MPI_STATS_TIME_IN_MPI_EV, 
    MPI_STATS_P2P_INCOMING_COUNT_EV,
    MPI_STATS_P2P_OUTGOING_COUNT_EV,
    MPI_STATS_P2P_INCOMING_PARTNERS_COUNT_EV,
    MPI_STATS_P2P_OUTGOING_PARTNERS_COUNT_EV,
    MPI_STATS_TIME_IN_OTHER_EV,
    MPI_STATS_TIME_IN_P2P_EV,
    MPI_STATS_TIME_IN_GLOBAL_EV,
    MPI_STATS_OTHER_COUNT_EV
  };

  unsigned int vec_params[MPI_STATS_EVENTS_COUNT] = {
    global_mpi_stats->P2P_Communications, 
    global_mpi_stats->P2P_Bytes_Sent,
    global_mpi_stats->P2P_Bytes_Recv, 
    global_mpi_stats->COLLECTIVE_Communications,
    global_mpi_stats->COLLECTIVE_Bytes_Sent, 
    global_mpi_stats->COLLECTIVE_Bytes_Recv,
    global_mpi_stats->Elapsed_Time_In_MPI, 
    global_mpi_stats->P2P_Communications_In,
    global_mpi_stats->P2P_Communications_Out,
    mpi_stats_get_num_partners(global_mpi_stats, global_mpi_stats->P2P_Partner_In),
    mpi_stats_get_num_partners(global_mpi_stats, global_mpi_stats->P2P_Partner_Out),
    (global_mpi_stats->Elapsed_Time_In_MPI - global_mpi_stats->Elapsed_Time_In_P2P_MPI - global_mpi_stats->Elapsed_Time_In_COLLECTIVE_MPI),
    global_mpi_stats->Elapsed_Time_In_P2P_MPI,
    global_mpi_stats->Elapsed_Time_In_COLLECTIVE_MPI,
    global_mpi_stats->MPI_Others_count
  };

  if (TRACING_MPI_STATISTICS)
  {
    TRACE_N_MISCEVENT (timestamp, MPI_STATS_EVENTS_COUNT, vec_types, vec_values, vec_params);
  }

  /* Reset the counters */
  mpi_stats_reset(global_mpi_stats);
}

void Extrae_network_counters_Wrapper (void)
{
}

void Extrae_network_routes_Wrapper (int mpi_rank)
{
	UNREFERENCED_PARAMETER(mpi_rank);
}

/******************************************************************************
 **      Function name : Extrae_tracing_tasks_Wrapper
 **      Author: HSG
 **      Description : Let the user choose which tasks must be traced
 ******************************************************************************/
void Extrae_tracing_tasks_Wrapper (unsigned from, unsigned to)
{
	unsigned i, tmp;

	if (Extrae_get_num_tasks() > 1)
	{
		if (tracejant && TracingBitmap != NULL)
		{
			/*
			 * Interchange them if limits are badly given 
			 */
			if (from > to)
			{
				tmp = from;
				from = to;
				to = tmp;
			}

			if (to >= Extrae_get_num_tasks())
				to = Extrae_get_num_tasks() - 1;

			/*
			 * If I'm not in the bitmask, disallow me tracing! 
			 */
			TRACE_EVENT (TIME, SET_TRACE_EV, (from <= TASKID) && (TASKID <= to));

			for (i = 0; i < Extrae_get_num_tasks(); i++)
				TracingBitmap[i] = FALSE;

			/*
			 * Build the bitmask 
			 */
			for (i = from; i <= to; i++)
				TracingBitmap[i] = TRUE;
		}
	}
}

/******************************************************************************
 ***  Trace_MPI_Communicator
 ******************************************************************************/
static void Trace_MPI_Communicator (MPI_Comm newcomm, UINT64 time, int trace)
{
	/* Store in the tracefile the definition of the communicator.
	   If the communicator is self/world, store an alias, otherwise store the
	   involved tasks
	*/
	int i, num_tasks, ierror;
	int result, is_comm_world, is_comm_self;

	/* First check if the communicators are duplicates of comm_world or
	   comm_self */
	ierror = PMPI_Comm_compare (MPI_COMM_WORLD, newcomm, &result);
	is_comm_world = result == MPI_IDENT || result == MPI_CONGRUENT;

	ierror = PMPI_Comm_compare (MPI_COMM_SELF, newcomm, &result);
	is_comm_self = result == MPI_IDENT || result == MPI_CONGRUENT;

	if (!is_comm_world && !is_comm_self)
	{
		MPI_Group group;

		/* Obtain the group of the communicator */
		ierror = PMPI_Comm_group (newcomm, &group);
		MPI_CHECK(ierror, PMPI_Comm_group);
	
		/* Calculate the number of involved tasks */
		ierror = PMPI_Group_size (group, &num_tasks);
		MPI_CHECK(ierror, PMPI_Group_size);

		{
			int ranks_aux[num_tasks];
	
			/* Obtain task id of each element */
			ierror = PMPI_Group_translate_ranks (group, num_tasks, ranks_global, grup_global, ranks_aux);
			MPI_CHECK(ierror, PMPI_Group_translate_ranks);
	
			FORCE_TRACE_MPIEVENT (time, MPI_ALIAS_COMM_CREATE_EV, EVT_BEGIN, EMPTY, num_tasks, EMPTY, newcomm, trace);
	
			/* Dump each of the task ids */
			for (i = 0; i < num_tasks; i++)
				FORCE_TRACE_MPIEVENT (time, MPI_RANK_CREACIO_COMM_EV, ranks_aux[i], EMPTY,
					EMPTY, EMPTY, EMPTY, EMPTY);
		}

		/* Free the group */
		if (group != MPI_GROUP_NULL)
		{
			ierror = PMPI_Group_free (&group);
			MPI_CHECK(ierror, PMPI_Group_free);
		}
	}
	else if (is_comm_world)
	{
		FORCE_TRACE_MPIEVENT (time, MPI_ALIAS_COMM_CREATE_EV, EVT_BEGIN, MPI_COMM_WORLD_ALIAS,
			Extrae_get_num_tasks(), EMPTY, newcomm, trace);
	}
	else if (is_comm_self)
	{
		FORCE_TRACE_MPIEVENT (time, MPI_ALIAS_COMM_CREATE_EV, EVT_BEGIN, MPI_COMM_SELF_ALIAS,
			1, EMPTY, newcomm, trace);
	}

	FORCE_TRACE_MPIEVENT (time, MPI_ALIAS_COMM_CREATE_EV, EVT_END, EMPTY, EMPTY, EMPTY, newcomm, trace);
}

/******************************************************************************
 ***  Trace_MPI_InterCommunicator
 ******************************************************************************/
static void Trace_MPI_InterCommunicator (MPI_Comm newcomm, MPI_Comm local_comm, 
	int local_leader, MPI_Comm remote_comm, int remote_leader, UINT64 time,
	int trace)
{
	int ierror, t_local_leader, t_remote_leader;
	MPI_Group l_group, r_group;

	ierror = PMPI_Comm_group (local_comm, &l_group);
	MPI_CHECK(ierror, PMPI_Comm_group);

	ierror = PMPI_Comm_group (remote_comm, &r_group);
	MPI_CHECK(ierror, PMPI_Comm_group);

	ierror = PMPI_Group_translate_ranks (l_group, 1, &local_leader,
	 grup_global, &t_local_leader);
	MPI_CHECK(ierror, PMPI_Group_translate_ranks);

	ierror = PMPI_Group_translate_ranks (r_group, 1, &remote_leader,
	  grup_global, &t_remote_leader);
	MPI_CHECK(ierror, PMPI_Group_translate_ranks);

	ierror = PMPI_Group_free (&l_group);
	MPI_CHECK(ierror, PMPI_Group_free);

	ierror = PMPI_Group_free (&r_group);
	MPI_CHECK(ierror, PMPI_Group_free);

	FORCE_TRACE_MPIEVENT(time, MPI_ALIAS_COMM_CREATE_EV, EVT_BEGIN, MPI_NEW_INTERCOMM_ALIAS,
	  1, t_local_leader, local_comm, trace);

	FORCE_TRACE_MPIEVENT(time, MPI_ALIAS_COMM_CREATE_EV, EVT_BEGIN, MPI_NEW_INTERCOMM_ALIAS,
	  2, t_remote_leader, remote_comm, trace);

	FORCE_TRACE_MPIEVENT (time, MPI_ALIAS_COMM_CREATE_EV, EVT_END, MPI_NEW_INTERCOMM_ALIAS,
	  EMPTY, EMPTY, newcomm, trace);
}

void Extrae_MPI_prepareDirectoryStructures (int me, int world_size)
{
	/* Before proceeding, check if it's ok to call MPI. We might support
	   MPI but maybe it's not initialized at this moment (nanos+mpi e.g.) */
	if (world_size > 1)
	{
		/* If the directory is shared, then let task 0 create all temporal
	  	 directories. This proves a significant speedup in GPFS */
		if (ExtraeUtilsMPI_CheckSharedDisk (Extrae_Get_TemporalDirNoTask()))
		{
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Temporal directory (%s) is shared among processes.\n",
				  Extrae_Get_TemporalDirNoTask());
			if (me == 0)
			{
				int i;
				for (i = 0; i < world_size; i+=Extrae_Get_TemporalDir_BlockSize())
					Backend_createExtraeDirectory (i, TRUE);
			}
		}
		else
		{
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Temporal directory (%s) is private among processes.\n",
				  Extrae_Get_TemporalDirNoTask());
				Backend_createExtraeDirectory (me, TRUE);
		}
	
		/* Now, wait for every process to reach this point, so directories are
		   created */
		PMPI_Barrier (MPI_COMM_WORLD);
		PMPI_Barrier (MPI_COMM_WORLD);
		PMPI_Barrier (MPI_COMM_WORLD);
	
		/* If the directory is shared, then let task 0 create all final
		   directories. This proves a significant speedup in GPFS */
		if (ExtraeUtilsMPI_CheckSharedDisk (Extrae_Get_FinalDirNoTask()))
		{
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Final directory (%s) is shared among processes.\n",
				  Extrae_Get_FinalDirNoTask());
			if (me == 0)
			{
				int i;
				for (i = 0; i < world_size; i+=Extrae_Get_FinalDir_BlockSize())
					Backend_createExtraeDirectory (i, FALSE);
			}
		}
		else
		{
			if (me == 0)
				fprintf (stdout, PACKAGE_NAME": Final directory (%s) is private among processes.\n",
				  Extrae_Get_FinalDirNoTask());
			Backend_createExtraeDirectory (me, FALSE);
		}
	
		/* Now, wait for every process to reach this point, so directories are
		   created */
		PMPI_Barrier (MPI_COMM_WORLD);
		PMPI_Barrier (MPI_COMM_WORLD);
		PMPI_Barrier (MPI_COMM_WORLD);
	}
	else
	{
		/* If process is alone, create temporal and final directories */
		Backend_createExtraeDirectory (me, TRUE);
		Backend_createExtraeDirectory (me, FALSE);
	}
}

