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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/dimemas/mpi_trf_semantics.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:04:49 $
 | @version:     $Revision: 1.10 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: mpi_trf_semantics.c,v 1.10 2009/05/28 13:04:49 harald Exp $";

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
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	int tipus;
	int isimmediate;
	int comunicador;
	UINT64 valor;
	double temps;

	if (IBSEND_EV == Get_EvEvent(current) 
	    || ISSEND_EV == Get_EvEvent(current)
	    || IRSEND_EV == Get_EvEvent(current)
	    || ISEND_EV == Get_EvEvent(current))
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
		case REDUCE_EV:
		case REDUCESCAT_EV:
		case SCAN_EV:
			res = Get_EvAux(current);
		break;

		case BARRIER_EV:
		case BCAST_EV:
		case ALLGATHER_EV:
		case ALLGATHERV_EV:
		case GATHER_EV:
		case GATHERV_EV:
		case SCATTER_EV:
		case SCATTERV_EV:
			res = Get_EvTarget(current);
		break;

		case ALLREDUCE_EV:
		case ALLTOALL_EV:
		case ALLTOALLV_EV:
		default:
			res = 0;
		break;
	}
	return res;
}

static int Get_GlobalOP_RootThd (event_t *current)
{
	return 0;
}

static UINT64 Get_GlobalOP_SendSize (event_t *current)
{
	UINT64 res;

	switch (Get_EvEvent(current))
	{
		case REDUCE_EV:
			if (Get_EvTag(current) != Get_EvAux(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case BCAST_EV:
			if (Get_EvTag(current) == Get_EvTarget(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case REDUCESCAT_EV:
		case SCAN_EV:
		case ALLGATHER_EV:
		case ALLGATHERV_EV:
		case GATHER_EV:
		case GATHERV_EV:
		case SCATTER_EV:
		case SCATTERV_EV:
		case ALLREDUCE_EV:
		case ALLTOALL_EV:
		case ALLTOALLV_EV:
			res = Get_EvSize(current);
		break;

		case BARRIER_EV:
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
		case REDUCE_EV:
			if (Get_EvTag(current) == Get_EvAux(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case BCAST_EV:
			if (Get_EvTag(current) != Get_EvTarget(current))
				res = Get_EvSize(current);
			else
				res = 0;
		break;

		case REDUCESCAT_EV:
		case SCAN_EV:
		case ALLREDUCE_EV:
			res = Get_EvSize(current);
		break;

		case ALLGATHER_EV:
		case ALLGATHERV_EV:
		case GATHER_EV:
		case GATHERV_EV:
		case SCATTER_EV:
		case SCATTERV_EV:
			res = Get_EvAux(current);
		break;

		case ALLTOALL_EV:
		case ALLTOALLV_EV:
			res = Get_EvTarget(current);
		break;

		case BARRIER_EV:
		default:
			res = 0;
		break;
	}
	return res;
}

static int Get_GlobalOP_ID (int type)
{
	int res = 0;

	if (REDUCE_EV == type)
		res = GLOP_ID_MPI_Reduce;
	else if (ALLREDUCE_EV == type)
		res = GLOP_ID_MPI_Allreduce;
	else if (BARRIER_EV == type)
		res = GLOP_ID_MPI_Barrier;
	else if (BCAST_EV == type)
		res = GLOP_ID_MPI_Bcast;
	else if (ALLTOALL_EV == type)
		res = GLOP_ID_MPI_Alltoall;
	else if (ALLTOALLV_EV == type)
		res = GLOP_ID_MPI_Alltoallv;
	else if (ALLGATHER_EV == type)
		res = GLOP_ID_MPI_Allgather;
	else if (ALLGATHERV_EV == type)
		res = GLOP_ID_MPI_Allgatherv;
	else if (GATHER_EV == type)
		res = GLOP_ID_MPI_Gather;
	else if (GATHERV_EV == type)
		res = GLOP_ID_MPI_Gatherv;
	else if (SCAN_EV == type)
		res = GLOP_ID_MPI_Scan;
	else if (REDUCESCAT_EV == type)
		res = GLOP_ID_MPI_Reduce_scatter;
	else if (SCATTER_EV == type)
		res = GLOP_ID_MPI_Scatter;
	else if (SCATTERV_EV == type)
		res = GLOP_ID_MPI_Scatterv;

	return res;
}

static int GlobalOP_Event (event_t * current, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

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
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int isimmediate = (IRECV_EV == Get_EvEvent(current));
	int tipus;
	int comunicador;
	double temps;

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
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

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
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	unsigned int sender = 0;
	unsigned int receive_tag = 0;
	unsigned int receive_size = 0;
	int comm_id;
	int tipus;
	int comunicador;
	double temps;

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
				comm_id= Get_EvComm(current);

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
	struct thread_t * thread_info = GET_THREAD_INFO(ptask, task, thread);
	UINT64 valor;
	int tipus;
	double temps;

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

	comunicador = alies_comunicador (Get_EvComm(current), 1, task);

	if (Get_EvTarget(current) == MPI_PROC_NULL)
		return 0;

#ifdef CPUZERO
	Dimemas_CPU_Burst(fset->output_file, task-1, thread-1, 0);
#endif

	switch (Get_EvValue(current))
	{ 
		case IRECV_EV:
			Dimemas_NX_Irecv (fset->output_file, task-1, thread-1, Get_EvTarget(current), comunicador, Get_EvSize(current), Get_EvTag(current) );
		break;

		case ISEND_EV:
		case IBSEND_EV:
		case IRSEND_EV:
		case ISSEND_EV:
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
	{ SEND_EV, ANY_Send_Event },
	{ BSEND_EV, ANY_Send_Event },
	{ SSEND_EV, ANY_Send_Event },
	{ RSEND_EV, ANY_Send_Event },
	{ SENDRECV_EV, SendRecv_Event },
	{ SENDRECV_REPLACE_EV, SendRecv_Event },
	{ RECV_EV, Receive_Event },
	{ IRECV_EV, Receive_Event },
	{ REDUCE_EV, GlobalOP_Event },
	{ ALLREDUCE_EV, GlobalOP_Event },
	{ PROBE_EV, NULL }, /* MUST BE IMPLEMENTED? */
	{ IPROBE_EV, NULL }, /* MUST BE IMPLEMENTED? */
	{ IBSEND_EV, ANY_Send_Event },
	{ ISSEND_EV, ANY_Send_Event },
	{ IRSEND_EV, ANY_Send_Event },
	{ ISEND_EV, ANY_Send_Event },
	{ BARRIER_EV, GlobalOP_Event },
	{ CANCEL_EV, MPI_Common_Event },
	{ TEST_EV, MPI_Common_Event },
	{ WAIT_EV, MPI_Common_Event },
	{ WAITALL_EV, MPI_Common_Event },
	{ WAITANY_EV, MPI_Common_Event },
	{ WAITSOME_EV, MPI_Common_Event },
	{ IRECVED_EV, Irecved_Event },
	{ BCAST_EV, GlobalOP_Event },
	{ ALLTOALL_EV, GlobalOP_Event },
	{ ALLTOALLV_EV, GlobalOP_Event },
	{ ALLGATHER_EV, GlobalOP_Event },
	{ ALLGATHERV_EV, GlobalOP_Event },
	{ GATHER_EV, GlobalOP_Event },
	{ GATHERV_EV, GlobalOP_Event },
	{ SCATTER_EV, GlobalOP_Event },
	{ SCATTERV_EV, GlobalOP_Event },
	{ REDUCESCAT_EV, GlobalOP_Event },
	{ SCAN_EV, GlobalOP_Event },
	{ MPIINIT_EV, SkipHandler },   /* Skip MPI_INIT */
	{ FINALIZE_EV, MPI_Common_Event },
	{ RECV_INIT_EV, MPI_Persistent_req_use_Event },
	{ SEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ BSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ RSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ SSEND_INIT_EV, MPI_Persistent_req_use_Event },
	{ PERSIST_REQ_EV, PersistentRequest_Event },
	{ START_EV, MPI_Persistent_req_use_Event },
	{ STARTALL_EV, MPI_Persistent_req_use_Event },
	{ REQUEST_FREE_EV, MPI_Persistent_req_use_Event },
	{ COMM_RANK_EV, MPI_Common_Event },
	{ COMM_SIZE_EV, MPI_Common_Event },
	{ IPROBE_COUNTER_EV, NULL }, /* Software counters are unimplemented in TRF */
	{ TIME_OUTSIDE_IPROBES_EV, NULL },
	{ TEST_COUNTER_EV, NULL },
	{ FILE_OPEN_EV, NULL }, /* IO is unimplemented in TRF */
	{ FILE_CLOSE_EV, NULL },
	{ FILE_READ_EV, NULL },
	{ FILE_READ_ALL_EV, NULL },
	{ FILE_WRITE_EV, NULL },
	{ FILE_WRITE_ALL_EV, NULL },
	{ FILE_READ_AT_EV, NULL },
	{ FILE_READ_AT_ALL_EV, NULL },
	{ FILE_WRITE_AT_EV, NULL },
	{ FILE_WRITE_AT_ALL_EV, NULL },
	{ NULL_EV, NULL }
};

