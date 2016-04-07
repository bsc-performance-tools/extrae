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
# include <stdio.h>
#endif

#ifndef HAVE_MPI_H
# define MPI_PROC_NULL (-1)
#else
# include <mpi.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "events.h"

#include "mpi_trf_semantics.h"
#include "dimemas_generator.h"
#include "mpi_prv_events.h"
#include "mpi_comunicadors.h"

static int ANY_Send_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset)
{
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	int tipus;
	int isimmediate;
	int comunicador;
	UINT64 valor;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	if (MPI_IBSEND_EV == Get_EvEvent(current) 
	    || MPI_ISSEND_EV == Get_EvEvent(current)
	    || MPI_IRSEND_EV == Get_EvEvent(current)
	    || MPI_ISEND_EV == Get_EvEvent(current))
	   	isimmediate = TRUE;
	else
		isimmediate = FALSE;

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);
	
	switch (Get_EvValue(current)) {
		case EVT_BEGIN:
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, temps);
		break;

		case EVT_END: 
			if (Get_EvTarget(current) != MPI_PROC_NULL)
			{
#ifdef CPUZERO
				Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
				if (!isimmediate)
					Dimemas_NX_BlockingSend (fset->output_file, task-1, thread-1, Get_EvTarget(current),
					  comunicador, Get_EvSize(current), Get_EvTag(current));
				else
					Dimemas_NX_ImmediateSend (fset->output_file, task-1, thread-1, Get_EvTarget(current),
					  comunicador, Get_EvSize(current), Get_EvTag(current));
			}
#ifdef CPUZERO
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
		break;
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);
	return 0;
}

/******************************************************************************
 ***  GlobalOP_Event
 ******************************************************************************/

static int Get_GlobalOP_CommID (event_t *current, int task)
{
	return alies_comunicador (Get_EvComm(current), 1, task);
}

static int Get_GlobalOP_RootRank (event_t *current)
{
	int res;

	switch (Get_EvEvent(current))
	{
		case MPI_REDUCE_EV:
		case MPI_REDUCESCAT_EV:
		case MPI_SCAN_EV:
			res = Get_EvAux(current);
		break;

		case MPI_BARRIER_EV:
		case MPI_BCAST_EV:
		case MPI_ALLGATHER_EV:
		case MPI_ALLGATHERV_EV:
		case MPI_GATHER_EV:
		case MPI_GATHERV_EV:
		case MPI_SCATTER_EV:
		case MPI_SCATTERV_EV:
			res = Get_EvTarget(current);
		break;

		case MPI_ALLREDUCE_EV:
		case MPI_ALLTOALL_EV:
		case MPI_ALLTOALLV_EV:
		default:
			res = 0;
		break;
	}
	return res;
}

static int Get_GlobalOP_RootThd (event_t *current)
{
	UNREFERENCED_PARAMETER(current);
	return 0;
}

static UINT64 Get_GlobalOP_SendSize (event_t *current)
{
	UINT64 res;

	switch (Get_EvEvent(current))
	{
		case MPI_REDUCE_EV:
			if (Get_EvTag(current) != Get_EvAux(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case MPI_BCAST_EV:
			if (Get_EvTag(current) == Get_EvTarget(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case MPI_REDUCESCAT_EV:
		case MPI_SCAN_EV:
		case MPI_ALLGATHER_EV:
		case MPI_ALLGATHERV_EV:
		case MPI_GATHER_EV:
		case MPI_GATHERV_EV:
		case MPI_SCATTER_EV:
		case MPI_SCATTERV_EV:
		case MPI_ALLREDUCE_EV:
		case MPI_ALLTOALL_EV:
		case MPI_ALLTOALLV_EV:
			res = Get_EvSize(current);
		break;

		case MPI_BARRIER_EV:
		default:
			res = 0;
		break;
	}
	return res;
}

static UINT64 Get_GlobalOP_RecvSize (event_t *current)
{
	UINT64 res;

	switch (Get_EvEvent(current))
	{
		case MPI_REDUCE_EV:
			if (Get_EvTag(current) == Get_EvAux(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case MPI_BCAST_EV:
			if (Get_EvTag(current) != Get_EvTarget(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case MPI_REDUCESCAT_EV:
		case MPI_SCAN_EV:
		case MPI_ALLREDUCE_EV:
			res = Get_EvSize(current);
		break;

		case MPI_ALLGATHER_EV:
		case MPI_ALLGATHERV_EV:
		case MPI_GATHER_EV:
		case MPI_GATHERV_EV:
		case MPI_SCATTER_EV:
		case MPI_SCATTERV_EV:
			res = Get_EvAux(current);
		break;

		case MPI_ALLTOALL_EV:
		case MPI_ALLTOALLV_EV:
			res = Get_EvTarget(current);
		break;

		case MPI_BARRIER_EV:
		default:
			res = 0;
		break;
	}
	return res;
}

static int Get_GlobalOP_ID (int type)
{
	int res = 0;

	if (MPI_REDUCE_EV == type)
		res = GLOP_ID_MPI_Reduce;
	else if (MPI_ALLREDUCE_EV == type)
		res = GLOP_ID_MPI_Allreduce;
	else if (MPI_BARRIER_EV == type)
		res = GLOP_ID_MPI_Barrier;
	else if (MPI_BCAST_EV == type)
		res = GLOP_ID_MPI_Bcast;
	else if (MPI_ALLTOALL_EV == type)
		res = GLOP_ID_MPI_Alltoall;
	else if (MPI_ALLTOALLV_EV == type)
		res = GLOP_ID_MPI_Alltoallv;
	else if (MPI_ALLGATHER_EV == type)
		res = GLOP_ID_MPI_Allgather;
	else if (MPI_ALLGATHERV_EV == type)
		res = GLOP_ID_MPI_Allgatherv;
	else if (MPI_GATHER_EV == type)
		res = GLOP_ID_MPI_Gather;
	else if (MPI_GATHERV_EV == type)
		res = GLOP_ID_MPI_Gatherv;
	else if (MPI_SCAN_EV == type)
		res = GLOP_ID_MPI_Scan;
	else if (MPI_REDUCESCAT_EV == type)
		res = GLOP_ID_MPI_Reduce_scatter;
	else if (MPI_SCATTER_EV == type)
		res = GLOP_ID_MPI_Scatter;
	else if (MPI_SCATTERV_EV == type)
		res = GLOP_ID_MPI_Scatterv;

	return res;
}

static int GlobalOP_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;

	if (EVT_BEGIN == Get_EvValue(current))
	{
		UINT64 bsent = Get_GlobalOP_SendSize(current);
		UINT64 brecv = Get_GlobalOP_RecvSize(current);
		int root_rank = Get_GlobalOP_RootRank(current);
		int thd_rank = Get_GlobalOP_RootThd(current);
		int commid = Get_GlobalOP_CommID(current, task);
		int ID = Get_GlobalOP_ID(Get_EvEvent(current));

		Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, temps);
		Dimemas_Global_OP (fset->output_file, task-1, thread-1, ID, commid, root_rank, thd_rank, bsent, brecv );
	}
	else if (EVT_END == Get_EvValue(current))
	{
#ifdef CPUZERO
		Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);
	return 0;
}

/******************************************************************************
 ***  Receive_Event
 ******************************************************************************/
static int Receive_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int isimmediate = (MPI_IRECV_EV == Get_EvEvent(current));
	int tipus;
	int comunicador;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;
	
	switch (Get_EvValue(current))
	{
		case EVT_BEGIN:
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, temps);
		break;

		case EVT_END:
			if (Get_EvTarget(current)!=MPI_PROC_NULL)
			{
#ifdef CPUZERO
				Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
				if (!isimmediate)
					Dimemas_NX_Recv (fset->output_file, task-1, thread-1, Get_EvTarget(current), comunicador, Get_EvSize(current), Get_EvTag(current));
				else
					Dimemas_NX_Irecv (fset->output_file, task-1, thread-1, Get_EvTarget(current), comunicador, Get_EvSize(current), Get_EvTag(current));
			}
#ifdef CPUZERO
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
		break;
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);

	return 0;
}


/******************************************************************************
 ***  MPI_Common_Event
 ******************************************************************************/
static int MPI_Common_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;
	
	switch (Get_EvValue(current)) {
		case EVT_BEGIN:
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, temps);
		break;

		case EVT_END:
#ifdef CPUZERO
			Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
		break;
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);
	return 0;
}

/******************************************************************************
 ***  Irecved_Event
 ******************************************************************************/
static int Irecved_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	int comunicador;

	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(current_time);

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);

	if (MPI_PROC_NULL != Get_EvTarget(current))
	{
#ifdef CPUZERO
		Dimemas_CPU_Burst (fset->output_file, task-1, thread-1, 0);
#endif
		Dimemas_NX_Wait (fset->output_file, task-1, thread-1, Get_EvTarget(current), comunicador, Get_EvSize(current), Get_EvTag(current));
	}

	return 0;
}

/******************************************************************************
 ***  Sendrecv_Event
 ******************************************************************************/
static int SendRecv_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	static unsigned int receiver, send_tag;
	static int send_size;
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	unsigned int sender = 0;
	unsigned int receive_tag = 0;
	unsigned int receive_size = 0;
	int tipus;
	int comunicador;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);
	
	switch( Get_EvValue(current) )
	{
		case EVT_BEGIN:
			Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, temps);
			receiver= Get_EvTarget(current);
			send_tag= Get_EvTag(current);
			send_size= Get_EvSize(current);
		break;

		case EVT_END:
#ifdef CPUZERO
			MPTRACE_ERROR(Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0));
#endif
			if (Get_EvTarget( current ) != MPI_PROC_NULL)
			{
				sender= Get_EvTarget(current);
				receive_tag= Get_EvTag(current);
				receive_size= Get_EvSize(current);                       

#ifdef CPUZERO
				Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
				Dimemas_NX_Irecv (fset->output_file, task-1, thread-1, sender, comunicador, receive_size, receive_tag);
			}

			if (receiver != MPI_PROC_NULL)
			{
#ifdef CPUZERO
				Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
				Dimemas_NX_BlockingSend (fset->output_file, task-1, thread-1, receiver, Get_EvComm(current), send_size, send_tag);
			}

			if (Get_EvTarget( current ) != MPI_PROC_NULL)
			{
#ifdef CPUZERO
				Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
				Dimemas_NX_Wait (fset->output_file, task-1, thread-1, sender, comunicador, receive_size, receive_tag);
			}
#ifdef CPUZERO
			Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
		break;
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);

	return 0;
}

/******************************************************************************
 ***  MPI_Persistent_req_use_Event
 ******************************************************************************/
static int MPI_Persistent_req_use_Event (event_t * current,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

	UNREFERENCED_PARAMETER(cpu);

	temps = current_time-thread_info->Previous_Event_Time;
	temps /= 1000000000.0f;
	
	switch (Get_EvValue(current))
	{
		case EVT_BEGIN:
			Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, temps);
		break;

		case EVT_END:
#ifdef CPUZERO
			Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
		break;
	}

	Translate_MPI_MPIT2PRV (Get_EvEvent(current), Get_EvValue(current), &tipus, &valor);
	Dimemas_User_Event (fset->output_file, task-1, thread-1, tipus, valor);

	return 0;
}


/******************************************************************************
 ***  PersistentRequest_Event
 ******************************************************************************/
static int PersistentRequest_Event (event_t * current,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	int comunicador;

	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(ptask);

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);

	if (Get_EvTarget(current) == MPI_PROC_NULL)
		return 0;

#ifdef CPUZERO
	Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif

	switch (Get_EvValue(current))
	{ 
		case MPI_IRECV_EV:
			Dimemas_NX_Irecv (fset->output_file, task-1, thread-1, Get_EvTarget(current), comunicador, Get_EvSize(current), Get_EvTag(current) );
		break;

		case MPI_ISEND_EV:
		case MPI_IBSEND_EV:
		case MPI_IRSEND_EV:
		case MPI_ISSEND_EV:
			if (MPI_PROC_NULL != Get_EvTarget(current))
			{
#ifdef CPUZERO
				Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif
				Dimemas_NX_ImmediateSend(fset->output_file, task-1, thread-1, Get_EvTarget(current),
				  comunicador, Get_EvSize(current), Get_EvTag(current));
			}
			break;
		default:
			break;
	}
  
	return 0;
}

SingleEv_Handler_t TRF_MPI_Event_Handlers[] = {
	{ MPI_SEND_EV, ANY_Send_Event },
	{ MPI_BSEND_EV, ANY_Send_Event },
	{ MPI_SSEND_EV, ANY_Send_Event },
	{ MPI_RSEND_EV, ANY_Send_Event },
	{ MPI_SENDRECV_EV, SendRecv_Event },
	{ MPI_SENDRECV_REPLACE_EV, SendRecv_Event },
	{ MPI_RECV_EV, Receive_Event },
	{ MPI_IRECV_EV, Receive_Event },
	{ MPI_REDUCE_EV, GlobalOP_Event },
	{ MPI_ALLREDUCE_EV, GlobalOP_Event },
	{ MPI_PROBE_EV, NULL }, /* MUST BE IMPLEMENTED? */
	{ MPI_IPROBE_EV, NULL }, /* MUST BE IMPLEMENTED? */
	{ MPI_IBSEND_EV, ANY_Send_Event },
	{ MPI_ISSEND_EV, ANY_Send_Event },
	{ MPI_IRSEND_EV, ANY_Send_Event },
	{ MPI_ISEND_EV, ANY_Send_Event },
	{ MPI_BARRIER_EV, GlobalOP_Event },
	{ MPI_CANCEL_EV, MPI_Common_Event },
	{ MPI_TEST_EV, MPI_Common_Event },
	{ MPI_WAIT_EV, MPI_Common_Event },
	{ MPI_WAITALL_EV, MPI_Common_Event },
	{ MPI_WAITANY_EV, MPI_Common_Event },
	{ MPI_WAITSOME_EV, MPI_Common_Event },
	{ MPI_IRECVED_EV, Irecved_Event },
	{ MPI_BCAST_EV, GlobalOP_Event },
	{ MPI_ALLTOALL_EV, GlobalOP_Event },
	{ MPI_ALLTOALLV_EV, GlobalOP_Event },
	{ MPI_ALLGATHER_EV, GlobalOP_Event },
	{ MPI_ALLGATHERV_EV, GlobalOP_Event },
	{ MPI_GATHER_EV, GlobalOP_Event },
	{ MPI_GATHERV_EV, GlobalOP_Event },
	{ MPI_SCATTER_EV, GlobalOP_Event },
	{ MPI_SCATTERV_EV, GlobalOP_Event },
	{ MPI_REDUCESCAT_EV, GlobalOP_Event },
	{ MPI_SCAN_EV, GlobalOP_Event },
	{ MPI_INIT_EV, SkipHandler },   /* Skip MPI_INIT */
	{ MPI_FINALIZE_EV, MPI_Common_Event },
	{ MPI_RECV_INIT_EV, MPI_Persistent_req_use_Event },
	{ MPI_SEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ MPI_BSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ MPI_RSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ MPI_SSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ MPI_PERSIST_REQ_EV, PersistentRequest_Event },
	{ MPI_START_EV, MPI_Persistent_req_use_Event },
	{ MPI_STARTALL_EV, MPI_Persistent_req_use_Event },
	{ MPI_REQUEST_FREE_EV, MPI_Persistent_req_use_Event },
	{ MPI_COMM_RANK_EV, MPI_Common_Event },
	{ MPI_COMM_SIZE_EV, MPI_Common_Event },
	{ MPI_IPROBE_COUNTER_EV, NULL }, /* Software counters are unimplemented in TRF */
	{ MPI_TIME_OUTSIDE_IPROBES_EV, NULL },
	{ MPI_TEST_COUNTER_EV, NULL },
	{ MPI_FILE_OPEN_EV, NULL }, /* IO is unimplemented in TRF */
	{ MPI_FILE_CLOSE_EV, NULL },
	{ MPI_FILE_READ_EV, NULL },
	{ MPI_FILE_READ_ALL_EV, NULL },
	{ MPI_FILE_WRITE_EV, NULL },
	{ MPI_FILE_WRITE_ALL_EV, NULL },
	{ MPI_FILE_READ_AT_EV, NULL },
	{ MPI_FILE_READ_AT_ALL_EV, NULL },
	{ MPI_FILE_WRITE_AT_EV, NULL },
	{ MPI_FILE_WRITE_AT_ALL_EV, NULL },
	{ NULL_EV, NULL }
};

