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

#include "file_set.h"
#include "object_tree.h"
#include "mpi2out.h"
#include "events.h"
#include "trace_to_prv.h"
#include "HardwareCounters.h"
#include "mpi_prv_events.h"
#include "semantics.h"
#include "mpi_prv_semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "timesync.h"
#include "communication_queues.h"
#include "trace_communication.h"
#include "options.h"
#include "intercommunicators.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

#ifndef HAVE_MPI_H
# define MPI_PROC_NULL (-1)
#else
# include <mpi.h>
#endif

//#define DEBUG

static int Get_State (unsigned int EvType)
{
	int state = 0;
	
	switch (EvType)
	{
		case MPI_INIT_EV:
		case MPI_FINALIZE_EV:
			state = STATE_INITFINI;
		break;
		case MPI_FILE_OPEN_EV:
		case MPI_FILE_CLOSE_EV:
		case MPI_FILE_READ_EV:
		case MPI_FILE_READ_ALL_EV:
		case MPI_FILE_WRITE_EV:
		case MPI_FILE_WRITE_ALL_EV:
		case MPI_FILE_READ_AT_EV:
		case MPI_FILE_READ_AT_ALL_EV:
		case MPI_FILE_WRITE_AT_EV:
		case MPI_FILE_WRITE_AT_ALL_EV:
			state = STATE_IO;
		break;
		case MPI_WIN_FREE_EV:
		case MPI_WIN_START_EV:
		case MPI_REQUEST_FREE_EV:
		case MPI_COMM_RANK_EV:
		case MPI_COMM_SIZE_EV:
		case MPI_COMM_CREATE_EV:
		case MPI_COMM_FREE_EV:
		case MPI_COMM_DUP_EV:
		case MPI_COMM_SPLIT_EV:
		case MPI_COMM_SPAWN_EV:
		case MPI_COMM_SPAWN_MULTIPLE_EV:
		case MPI_CART_CREATE_EV:
		case MPI_CART_SUB_EV:
		case MPI_CANCEL_EV:
		case MPI_REQUEST_GET_STATUS_EV:
		case MPI_INTERCOMM_CREATE_EV:
		case MPI_INTERCOMM_MERGE_EV:
		case MPI_WIN_POST_EV:
		case MPI_WIN_COMPLETE_EV:
			state = STATE_MIXED;
		break;
		case MPI_PROBE_EV:
		case MPI_IPROBE_EV:
			state = STATE_PROBE;
		break;
		case MPI_TEST_EV:
		case MPI_TESTALL_EV:
		case MPI_TESTSOME_EV:
		case MPI_TESTANY_EV:
		case MPI_WAIT_EV:
		case MPI_WAITALL_EV:
		case MPI_WAITSOME_EV:
		case MPI_WAITANY_EV:
		case MPI_WIN_WAIT_EV:
			state = STATE_TWRECV;
		break;
		case MPI_SEND_EV:
		case MPI_RSEND_EV:
		case MPI_SSEND_EV:
		case MPI_BSEND_EV:
			state = STATE_SEND;
		break;
		case MPI_ISEND_EV:
		case MPI_IRSEND_EV:
		case MPI_ISSEND_EV:
		case MPI_IBSEND_EV:
			state = STATE_ISEND;
		break;
		case MPI_BARRIER_EV:
		case MPI_IBARRIER_EV:
			state = STATE_BARRIER;
		break;
		case MPI_REDUCE_EV:
		case MPI_ALLREDUCE_EV:
		case MPI_BCAST_EV:
		case MPI_ALLTOALL_EV:
		case MPI_ALLTOALLV_EV:
		case MPI_ALLGATHER_EV:
		case MPI_ALLGATHERV_EV:
		case MPI_GATHER_EV:
		case MPI_GATHERV_EV:
		case MPI_SCATTER_EV:
		case MPI_SCATTERV_EV:
		case MPI_REDUCESCAT_EV:
		case MPI_SCAN_EV:
		case MPI_IREDUCE_EV:
		case MPI_IALLREDUCE_EV:
		case MPI_IBCAST_EV:
		case MPI_IALLTOALL_EV:
		case MPI_IALLTOALLV_EV:
		case MPI_IALLGATHER_EV:
		case MPI_IALLGATHERV_EV:
		case MPI_IGATHER_EV:
		case MPI_IGATHERV_EV:
		case MPI_ISCATTER_EV:
		case MPI_ISCATTERV_EV:
		case MPI_IREDUCESCAT_EV:
		case MPI_ISCAN_EV:
			state = STATE_BCAST;
		break;
		case MPI_WIN_FENCE_EV:
		case MPI_GET_EV:
		case MPI_PUT_EV:
			state = STATE_MEMORY_XFER;
		break;
		default:
			fprintf (stderr, "mpi2prv: Error! Unknown MPI event %d parsed at %s (%s:%d)\n",
			  EvType, __func__, __FILE__, __LINE__);
			fflush (stderr);
			exit (-1);
		break;
	}
	return state;
}

/******************************************************************************
 ***  Any_Send_Event
 ******************************************************************************/

static int Any_Send_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned recv_thread, recv_vthread, EvType, EvValue;
	thread_t *thread_info;
	task_t *task_info, *task_info_partner;
	event_t * recv_begin, * recv_end;
	int EvComm;

	EvType  = Get_EvEvent(current_event);
	EvValue = Get_EvValue(current_event);
	EvComm  = Get_EvComm (current_event);

	Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	switch (EvValue)
	{
		case EVT_BEGIN:
			thread_info->Send_Rec = current_event;
		break;
		case EVT_END:
			if (MatchComms_Enabled(ptask, task))
			{
				if (MPI_PROC_NULL != Get_EvTarget (current_event))
				{
					int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

					if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(current_event)))
					{
#if defined(DEBUG)
						fprintf (stderr, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", EvType, current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
						task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(current_event)+1);
						CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;

#if defined(DEBUG)
							fprintf (stderr, "SEND_CMD(%u) DID NOT find receiver\n", EvType);
#endif
							position = WriteFileBuffer_getPosition (thread_info->file->wfb);
							CommunicationQueues_QueueSend (task_info->send_queue, thread_info->Send_Rec, current_event, position, thread, thread_info->virtual_thread, Get_EvTarget(current_event), Get_EvTag(current_event), 0);
							trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, target_ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
						}
						else
						{
#if defined(DEBUG)
							fprintf (stderr, "SEND_CMD(%u) find receiver\n", EvType);
#endif
							trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, target_ptask, 1+Get_EvTarget(current_event), recv_thread, recv_vthread, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
						}
					}
#if defined(PARALLEL_MERGE)
					else
					{
#if defined(DEBUG)
						fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d >> PENDING\n", Get_EvEvent(current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
						trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, target_ptask, Get_EvTarget (current_event));
					}
#endif
				}
			}
		break;
	}
	return 0;
}


/******************************************************************************
 ***  SendRecv_Event
 ******************************************************************************/

static int SendRecv_Event (event_t * current_event, 
	unsigned long long current_time, unsigned int cpu, unsigned int ptask, 
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	thread_t *thread_info;
	task_t *task_info, *task_info_partner;
#if !defined(AVOID_SENDRECV)
	unsigned recv_thread, send_thread, recv_vthread, send_vthread;
	event_t *recv_begin, *recv_end, *send_begin, *send_end;
	off_t send_position;
#endif
	int EvComm = Get_EvComm (current_event);

	Switch_State (STATE_SENDRECVOP, (Get_EvValue(current_event) == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, Get_EvEvent(current_event), Get_EvValue(current_event));

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	if (!get_option_merge_SkipSendRecvComms())
	{
		if (Get_EvValue (current_event) == EVT_BEGIN)
		{
			thread_info->Send_Rec = current_event;

			/* Treat the send part */
			if (MatchComms_Enabled(ptask, task))
				if (MPI_PROC_NULL != Get_EvTarget (thread_info->Send_Rec))
				{
					int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

					if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(thread_info->Send_Rec)))
					{
#if defined(DEBUG)
						fprintf (stderr, "SENDRECV/SEND: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(thread_info->Send_Rec), task-1, Get_EvTarget(thread_info->Send_Rec), Get_EvTag(thread_info->Send_Rec));
#endif
						task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(thread_info->Send_Rec)+1);

						CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (thread_info->Send_Rec), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;

#if defined(DEBUG)
							fprintf (stderr, "SENDRECV/SEND DID NOT find partner\n");
#endif
							position = WriteFileBuffer_getPosition (thread_info->file->wfb);
							CommunicationQueues_QueueSend (task_info->send_queue, thread_info->Send_Rec, current_event, position, thread, thread_info->virtual_thread, Get_EvTarget(thread_info->Send_Rec), Get_EvTag(thread_info->Send_Rec), 0);
							trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, target_ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
						}
						else if (recv_begin != NULL && recv_end != NULL)
						{
#if defined(DEBUG)
							fprintf (stderr, "SENDRECV/SEND found partner\n");
#endif
							trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, target_ptask, 1+Get_EvTarget(thread_info->Send_Rec), recv_thread, recv_vthread, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractRecv returned recv_begin = %p and recv_end = %p\n", recv_begin, recv_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
#if defined(DEBUG)
						fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d >> PENDING\n", Get_EvEvent(current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
						trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, target_ptask, Get_EvTarget (thread_info->Send_Rec));
					}
#endif /* PARALLEL_MERGE */
					}

		}
		else if (Get_EvValue (current_event) == EVT_END)
		{
			thread_info->Recv_Rec = current_event;

			/* Treat the receive part */
			if (MatchComms_Enabled(ptask, task))
				if (MPI_PROC_NULL != Get_EvTarget (thread_info->Recv_Rec))
				{
					int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

					if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(thread_info->Recv_Rec)))
					{
#if defined(DEBUG)
						fprintf (stderr, "SENDRECV/RECV: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(thread_info->Recv_Rec), task-1, Get_EvTarget(thread_info->Recv_Rec), Get_EvTag(thread_info->Recv_Rec));
#endif

						task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(thread_info->Recv_Rec)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (thread_info->Recv_Rec), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin && NULL == send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "SENDRECV/RECV DID NOT find partner\n");
#endif
							CommunicationQueues_QueueRecv (task_info->recv_queue, thread_info->Send_Rec, current_event, thread, thread_info->virtual_thread, Get_EvTarget(thread_info->Recv_Rec), Get_EvTag(thread_info->Recv_Rec), 0);
						}
						else if (NULL != send_begin && NULL != send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "SENDRECV/RECV found partner\n");
#endif
							trace_communicationAt (target_ptask, 1+Get_EvTarget(thread_info->Recv_Rec), send_thread, send_vthread, ptask, task, thread, thread_info->virtual_thread, send_begin, send_end, thread_info->Send_Rec, thread_info->Recv_Rec, TRUE, send_position);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (ptask-1, task-1, Get_EvTime(thread_info->Send_Rec));
						phy_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
						AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), ptask-1, task-1, thread-1,
							thread_info->virtual_thread-1, target_ptask-1, Get_EvTarget(current_event), fset, MatchComms_GetZone(ptask, task));
					}
#endif /* PARALLEL_MERGE */
				}
		}
	}

	return 0;
}

static unsigned int Get_GlobalOP_CommID (event_t *current)
{
	return Get_EvComm(current);
}

static unsigned int Get_GlobalOP_isRoot (event_t *current, int task)
{
	unsigned int res = FALSE;
	switch (Get_EvEvent(current))
	{
		case MPI_REDUCE_EV:
		case MPI_IREDUCE_EV:
			res = Get_EvAux(current) == Get_EvTag(current);
		break;
		case MPI_BCAST_EV:
		case MPI_IBCAST_EV:
			res = Get_EvTarget(current) == Get_EvTag(current);
		break;
		case MPI_GATHER_EV:
		case MPI_IGATHER_EV:
		case MPI_GATHERV_EV:
		case MPI_IGATHERV_EV:
		case MPI_SCATTER_EV:
		case MPI_ISCATTER_EV:
		case MPI_SCATTERV_EV:
		case MPI_ISCATTERV_EV:
			res = Get_EvTarget(current) == task-1;
		break;
		case MPI_REDUCESCAT_EV:
		case MPI_IREDUCESCAT_EV:
			res = Get_EvTarget(current) == 0;
		break;
	}
	return res;
}

static unsigned int Get_GlobalOP_SendSize (event_t *current, int is_root)
{
	unsigned int res = 0;
	switch (Get_EvEvent(current))
	{
		case MPI_BARRIER_EV:
		case MPI_IBARRIER_EV:
			res = 0;
		break;
		case MPI_REDUCE_EV:
		case MPI_IREDUCE_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case MPI_BCAST_EV:
		case MPI_IBCAST_EV:
			res = (!is_root)?0:Get_EvSize(current);
		break;
		default:
			res = Get_EvSize(current);
		break;
	}
	return res;
}

static unsigned int Get_GlobalOP_RecvSize (event_t *current, int is_root)
{
	unsigned int res = 0;
	switch (Get_EvEvent(current))
	{
		case MPI_BARRIER_EV:
		case MPI_IBARRIER_EV:
			res = 0;
		break;
		case MPI_REDUCE_EV:
		case MPI_IREDUCE_EV:
			res = (!is_root)?0:Get_EvSize(current);
		break;
		case MPI_BCAST_EV:
		case MPI_IBCAST_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case MPI_REDUCESCAT_EV:
		case MPI_IREDUCESCAT_EV:
			res = (is_root)?Get_EvSize(current):Get_EvAux(current);
		break;
		case MPI_SCAN_EV:
		case MPI_ISCAN_EV:
		case MPI_ALLREDUCE_EV:
		case MPI_IALLREDUCE_EV:
			res = Get_EvSize(current);
		break;
		default:
			res = Get_EvAux(current);
		break;
	}
	return res;
}

/******************************************************************************
 *** GlobalOP_event
 ******************************************************************************/

static int GlobalOP_event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	unsigned int comm_id, send_size, receive_size, is_root;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	/* First global operation found, start matching communications from now on */
	if ((tracingCircularBuffer())                                  && /* Circular buffer is enabled */ 
            (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES) && /* The buffer behavior is to skip matches */
            (!MatchComms_Enabled(ptask, task))                         && /* Not matching already */
            (EvValue == EVT_END)                                       && /* End of the collective */
            (Get_EvSize(current_event) == GET_NUM_TASKS(ptask))           /* World collective */
            /* (getTagForCircularBuffer() == Get_EvAux(current_event)) */)
	{
		MatchComms_On(ptask, task);
	}

	Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	if (EVT_BEGIN == EvValue)
	{
		comm_id = Get_GlobalOP_CommID (current_event);
		is_root = Get_GlobalOP_isRoot (current_event, task);
		send_size = Get_GlobalOP_SendSize (current_event, is_root);
		receive_size = Get_GlobalOP_RecvSize (current_event, is_root);
		trace_enter_global_op (cpu, ptask, task, thread, current_time, comm_id,
		  send_size, receive_size, is_root?1:0);

		Enable_MPI_Soft_Counter (EvType);
	}

	return 0;
}

/******************************************************************************
 ***  Other_MPI_Event:
 ******************************************************************************/

static int Other_MPI_Event (event_t * current_event, 
	unsigned long long current_time, unsigned int cpu, unsigned int ptask, 
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

	/* XXX: Workaround to set the state to NOT_TRACING after the MPI_Init when using circular buffer.
     * We should definitely do this another way round. 
	 */ 
	if ((EvType == MPI_INIT_EV) && (EvValue == EVT_END))
	{
		if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES))
		{
			/* The first event beyond the MPI_Init will remove the STATE_NOT_TRACING (see Push_State) */
			Push_State (STATE_NOT_TRACING, ptask, task, thread);
		}
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	if (EvType == MPI_INIT_EV && EvValue == EVT_BEGIN)
	{
		UINT32 PID = Get_EvTarget (current_event);
		UINT32 PPID = Get_EvSize (current_event);
		UINT32 Depth = Get_EvTag (current_event);
		trace_paraver_event (cpu, ptask, task, thread, current_time, PID_EV, PID);
		trace_paraver_event (cpu, ptask, task, thread, current_time, PPID_EV, PPID);
		trace_paraver_event (cpu, ptask, task, thread, current_time, FORK_DEPTH_EV, Depth);
	}

	return 0;
}

/******************************************************************************
 ***  MPIIO_Event:
 ******************************************************************************/

static int MPIIO_Event (event_t * current_event,
        unsigned long long current_time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
        unsigned int EvType, EvValue;
        UNREFERENCED_PARAMETER(fset);

        EvType  = Get_EvEvent (current_event);
        EvValue = Get_EvValue (current_event);

        Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

        trace_paraver_state (cpu, ptask, task, thread, current_time);
        trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
        trace_paraver_event (cpu, ptask, task, thread, current_time, MPI_IO_SIZE_EV, Get_EvSize( current_event ));

        Enable_MPI_Soft_Counter (EvType);

	return 0;
}

/******************************************************************************
 ***  Recv_Event
 ******************************************************************************/

static int Recv_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned EvType, EvValue, send_thread, send_vthread;
	thread_t *thread_info;
	task_t *task_info, *task_info_partner;
	int EvComm;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);
	EvComm  = Get_EvComm  (current_event);

	Switch_State (STATE_WAITMESS, (EvValue == EVT_BEGIN), ptask, task, thread);

	if (EvValue == EVT_BEGIN)
	{
		thread_info->Recv_Rec = current_event;
	}
	else
	{
		if (MatchComms_Enabled(ptask, task))
		{
			if (MPI_PROC_NULL != Get_EvTarget(current_event))
			{
				int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

				if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(current_event)))
				{
#if defined(DEBUG)
					fprintf (stderr, "RECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
					task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(current_event)+1);

					CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

					if (NULL == send_begin || NULL == send_end)
					{
#if defined(DEBUG)
						fprintf (stderr, "RECV_CMD DID NOT find partner\n");
#endif
						CommunicationQueues_QueueRecv (task_info->recv_queue, thread_info->Recv_Rec, current_event, thread, thread_info->virtual_thread, Get_EvTarget(current_event), Get_EvTag(current_event), 0);
					}
					else if (NULL != send_begin && NULL != send_end)
					{
#if defined(DEBUG)
						fprintf (stderr, "RECV_CMD find partner\n");
#endif
						trace_communicationAt (target_ptask, 1+Get_EvTarget(current_event), send_thread, send_vthread, ptask, task, thread, thread_info->virtual_thread, send_begin, send_end, thread_info->Recv_Rec, current_event, TRUE, send_position);
					}
					else
						fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
				}
#if defined(PARALLEL_MERGE)
				else
				{
					UINT64 log_r, phy_r;

					log_r = TIMESYNC (ptask-1, task-1, Get_EvTime(thread_info->Recv_Rec));
					phy_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
					AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), ptask-1, task-1, thread-1,
						thread_info->virtual_thread-1, target_ptask-1, Get_EvTarget(current_event), fset, MatchComms_GetZone(ptask, task));
				}
#endif
			}
		}
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  IRecv_Event
 ******************************************************************************/

static int IRecv_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned EvType, EvValue, send_thread, send_vthread;
	thread_t *thread_info;
	task_t *task_info, *task_info_partner;
	int EvComm;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);
	EvComm  = Get_EvComm  (current_event);

	Switch_State (STATE_IWAITMESS, (EvValue == EVT_BEGIN), ptask, task, thread);

	if (EvValue == EVT_END)
	{
		if (MatchComms_Enabled(ptask, task))
		{
			event_t *receive = Search_MPI_IRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (NULL != receive)
			{
				if (MPI_PROC_NULL != Get_EvTarget(receive))
				{
					int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

					if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(receive)))
					{
#if defined(DEBUG)
						fprintf (stderr, "IRECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(receive), Get_EvTag(receive));
#endif
						task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(receive)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin || NULL == send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "IRECV_CMD DID NOT find COMM\n");
#endif
							CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, receive, thread, thread_info->virtual_thread, Get_EvTarget(receive), Get_EvTag(receive), 0);
						}
						else if (NULL != send_begin && NULL != send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "IRECV_CMD find COMM (partner times = %lld/%lld)\n", Get_EvTime(send_begin), Get_EvTime(send_end));
#endif
							trace_communicationAt (target_ptask, 1+Get_EvTarget(receive), send_thread, send_vthread, ptask, task, thread, thread_info->virtual_thread, send_begin, send_end, current_event, receive, TRUE, send_position);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (ptask-1, task-1, Get_EvTime(receive));
						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), ptask-1, task-1, thread-1,
							thread_info->virtual_thread-1, target_ptask-1, Get_EvTarget(receive), fset, MatchComms_GetZone(ptask, task));
					}
#endif
				}
			}
		}
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  MPI_PersistentRequest_Init_Event
 ******************************************************************************/

int MPI_PersistentRequest_Init_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (((EvType == MPI_RECV_INIT_EV) ? STATE_IRECV : STATE_ISEND), (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}


/******************************************************************************
 ***  MPI_PersistentRequest_Event
 ******************************************************************************/

int MPI_PersistentRequest_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	thread_t *thread_info;
	task_t *task_info, *task_info_partner;
	event_t *recv_begin, *recv_end;
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned recv_thread, send_thread, recv_vthread, send_vthread;
	int EvComm; 

	EvComm = Get_EvComm( current_event );

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);
	trace_paraver_state (cpu, ptask, task, thread, current_time);

	/* If this is a send, look for the receive */
	if (Get_EvValue (current_event) == MPI_ISEND_EV)
	{
		thread_info->Send_Rec = current_event;

		if (MatchComms_Enabled(ptask, task))
		{
			if (MPI_PROC_NULL != Get_EvTarget (current_event))
			{
				int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

				if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(current_event)))
				{
#if defined(DEBUG)
					fprintf (stderr, "PERS_REQ_ISEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", Get_EvValue (current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif

					task_info_partner = GET_TASK_INFO(target_ptask, Get_EvTarget(current_event)+1);

					CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

					if (recv_begin == NULL || recv_end == NULL)
					{
						off_t position;
#if defined(DEBUG)
						fprintf (stderr, "PER_REQ_ISEND_CMD DID NOT find a partner\n");
#endif
						position = WriteFileBuffer_getPosition (thread_info->file->wfb);
						CommunicationQueues_QueueSend (task_info->send_queue, current_event, current_event, position, thread, thread_info->virtual_thread, Get_EvTarget(current_event), Get_EvTag(current_event), 0);
						trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, target_ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
					}
					else
					{
#if defined(DEBUG)
						fprintf (stderr, "PER_REQ_ISEND_CMD DID NOT find a partner\n");
#endif
						trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, target_ptask, 1+Get_EvTarget(current_event), recv_thread, recv_vthread, current_event, current_event, recv_begin, recv_end, FALSE, 0);
					}

				}
#if defined(PARALLEL_MERGE)
				else
				{
#if defined(DEBUG)
					fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d >> PENDING\n", Get_EvEvent(current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
					trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, target_ptask, Get_EvTarget (current_event));
				}
#endif
			}
		}
	}

	/* If this is a receive, look for the send */
	if (Get_EvValue(current_event) == MPI_IRECV_EV)
	{
		thread_info->Recv_Rec = current_event;

		if (MatchComms_Enabled(ptask, task))
		{
			event_t *receive = Search_MPI_IRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (NULL != receive)
			{
				int target_ptask = intercommunicators_get_target_ptask( ptask, task, EvComm );

				if (MPI_PROC_NULL != Get_EvTarget(receive))
				{
					if (isTaskInMyGroup (fset, target_ptask-1, Get_EvTarget(receive)))
					{
#if defined(DEBUG)
						fprintf (stderr, "PERS_REQ_IRECV_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", Get_EvValue (current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(receive), Get_EvTag(receive));
#endif

						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(receive)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin || NULL == send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "PER_REQ_IRECV_CMD DID NOT find a partner\n");
#endif
							CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, receive, thread, thread_info->virtual_thread, Get_EvTarget(current_event), Get_EvTag(current_event), 0);
						}
						else if (NULL != send_begin && NULL != send_end)
						{
#if defined(DEBUG)
							fprintf (stderr, "PERS_REQ_IRECV_CMD find partner (send position = %llu)\n", (unsigned long long) send_position);
#endif
							trace_communicationAt (target_ptask, 1+Get_EvTarget(receive), send_thread, send_vthread, ptask, task, thread, thread_info->virtual_thread, send_begin, send_end, current_event, receive, TRUE, send_position);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (ptask-1, task-1, Get_EvTime(receive));
						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), ptask-1, task-1, thread-1,
							thread_info->virtual_thread-1, target_ptask-1, Get_EvTarget(receive), fset, MatchComms_GetZone(ptask, task));
					}
#endif
				}
			}
		}
	}

	return 0;
}

/******************************************************************************
 ***  MPI_PersistentRequest_Free_Event
 ******************************************************************************/

int MPI_PersistentRequest_Free_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_TWRECV, (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}


/******************************************************************************
 ***  MPI_Start_Event
 ******************************************************************************/

int MPI_Start_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	thread_t * thread_info;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_MIXED, (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	switch (EvValue)
	{
		/* We don't know if the start will issue a send or recv, so we store both.
		   This will be solved in MPI_PersistentRequest_Event */
		case EVT_BEGIN:
			thread_info->Send_Rec = current_event;
			thread_info->Recv_Rec = current_event;
		break;
		case EVT_END:
		break;
	}
	return 0;
}


/******************************************************************************
 ***  MPI_Request_get_status_SoftwareCounter_Event
 ******************************************************************************/

int MPI_Request_get_status_SoftwareCounter_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_MPI_Soft_Counter (EvType);

	return 0;
}

/******************************************************************************
 ***  MPI_ElapsedTimeOutsideRequest_get_status_Event
 ******************************************************************************/

int MPI_ElapsedTimeOutsideRequest_get_status_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType;
	unsigned long long EvValue;
	UINT64 elapsed_time;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	elapsed_time = Get_EvValue (current_event);
	EvValue = (unsigned long long) (elapsed_time);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_MPI_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  MPI_IProbeSoftwareCounter_Event
 ******************************************************************************/

int MPI_IProbeSoftwareCounter_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_MPI_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  MPI_ElapsedTimeOutsideIProbes_Event
 ******************************************************************************/

int MPI_ElapsedTimeOutsideIProbes_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType;
	unsigned long long EvValue;
	UINT64  elapsed_time;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	elapsed_time = Get_EvValue (current_event);
	EvValue = (unsigned long long) (elapsed_time);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_MPI_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  MPI_TestSoftwareCounter_Event
 ******************************************************************************/

int MPI_TestSoftwareCounter_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_MPI_Soft_Counter (EvType);

	return 0;
}

/******************************************************************************
 *** MPI_RMA_Event (Remote Memory Address)
 ******************************************************************************/
int MPI_RMA_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);

	Switch_State (Get_State(Get_EvEvent(current_event)),
		Get_EvValue(current_event) == EVT_BEGIN, ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time,
		Get_EvEvent(current_event), Get_EvValue (current_event));

	return 0;
}


/******************************************************************************
 ***  MPI_GenerateParaverTraces
 ******************************************************************************/

SingleEv_Handler_t PRV_MPI_Event_Handlers[] = {
	{ MPI_SEND_EV, Any_Send_Event },
	{ MPI_BSEND_EV, Any_Send_Event },
	{ MPI_SSEND_EV, Any_Send_Event },
	{ MPI_RSEND_EV, Any_Send_Event },
	{ MPI_IBSEND_EV, Any_Send_Event },
	{ MPI_ISSEND_EV, Any_Send_Event },
	{ MPI_IRSEND_EV, Any_Send_Event },
	{ MPI_ISEND_EV, Any_Send_Event },
	{ MPI_SENDRECV_EV, SendRecv_Event },
	{ MPI_SENDRECV_REPLACE_EV, SendRecv_Event },
	{ MPI_RECV_EV, Recv_Event },
	{ MPI_IRECV_EV, IRecv_Event },
	{ MPI_REDUCE_EV, GlobalOP_event },
	{ MPI_ALLREDUCE_EV, GlobalOP_event },
	{ MPI_PROBE_EV, Other_MPI_Event },
	{ MPI_REQUEST_GET_STATUS_EV, Other_MPI_Event },
	{ MPI_IPROBE_EV, Other_MPI_Event },
	{ MPI_BARRIER_EV, GlobalOP_event },
	{ MPI_CANCEL_EV, Other_MPI_Event },
	{ MPI_TEST_EV, Other_MPI_Event },
	{ MPI_TESTALL_EV, Other_MPI_Event },
	{ MPI_TESTANY_EV, Other_MPI_Event },
	{ MPI_TESTSOME_EV, Other_MPI_Event },
	{ MPI_WAIT_EV, Other_MPI_Event },
	{ MPI_WAITALL_EV, Other_MPI_Event },
	{ MPI_WAITANY_EV, Other_MPI_Event },
	{ MPI_WAITSOME_EV, Other_MPI_Event },
	{ MPI_IRECVED_EV, SkipHandler },
	{ MPI_BCAST_EV, GlobalOP_event },
	{ MPI_ALLTOALL_EV, GlobalOP_event },
	{ MPI_ALLTOALLV_EV, GlobalOP_event },
	{ MPI_ALLGATHER_EV, GlobalOP_event },
	{ MPI_ALLGATHERV_EV, GlobalOP_event },
	{ MPI_GATHER_EV, GlobalOP_event },
	{ MPI_GATHERV_EV, GlobalOP_event },
	{ MPI_SCATTER_EV, GlobalOP_event },
	{ MPI_SCATTERV_EV, GlobalOP_event },
	{ MPI_REDUCESCAT_EV, GlobalOP_event },
	{ MPI_SCAN_EV, GlobalOP_event },
	{ MPI_INIT_EV, Other_MPI_Event },
	{ MPI_FINALIZE_EV, Other_MPI_Event },
	{ MPI_RECV_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ MPI_SEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ MPI_BSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ MPI_RSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ MPI_SSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ MPI_PERSIST_REQ_EV, MPI_PersistentRequest_Event },
	{ MPI_START_EV, MPI_Start_Event },
	{ MPI_STARTALL_EV, MPI_Start_Event },
	{ MPI_REQUEST_FREE_EV, MPI_PersistentRequest_Free_Event },
	{ MPI_COMM_RANK_EV, Other_MPI_Event },
	{ MPI_COMM_SIZE_EV, Other_MPI_Event },
	{ MPI_COMM_CREATE_EV, Other_MPI_Event },
	{ MPI_COMM_FREE_EV, Other_MPI_Event },
	{ MPI_COMM_SPLIT_EV, Other_MPI_Event },
	{ MPI_COMM_SPAWN_EV, Other_MPI_Event },
	{ MPI_COMM_SPAWN_MULTIPLE_EV, Other_MPI_Event },
	{ MPI_COMM_DUP_EV, Other_MPI_Event },
	{ MPI_CART_CREATE_EV, Other_MPI_Event },
	{ MPI_CART_SUB_EV, Other_MPI_Event },
	{ MPI_INTERCOMM_CREATE_EV, Other_MPI_Event },
	{ MPI_INTERCOMM_MERGE_EV, Other_MPI_Event },
	{ MPI_REQUEST_GET_STATUS_COUNTER_EV, MPI_Request_get_status_SoftwareCounter_Event },
	{ MPI_TIME_OUTSIDE_MPI_REQUEST_GET_STATUS_EV, MPI_ElapsedTimeOutsideRequest_get_status_Event },
	{ MPI_IPROBE_COUNTER_EV, MPI_IProbeSoftwareCounter_Event },
	{ MPI_TIME_OUTSIDE_IPROBES_EV, MPI_ElapsedTimeOutsideIProbes_Event },
	{ MPI_TEST_COUNTER_EV, MPI_TestSoftwareCounter_Event },
	{ MPI_FILE_OPEN_EV, Other_MPI_Event },
	{ MPI_FILE_CLOSE_EV, Other_MPI_Event },
	{ MPI_FILE_READ_EV, MPIIO_Event },
	{ MPI_FILE_READ_ALL_EV, MPIIO_Event },
	{ MPI_FILE_WRITE_EV, MPIIO_Event },
	{ MPI_FILE_WRITE_ALL_EV, MPIIO_Event },
	{ MPI_FILE_READ_AT_EV, MPIIO_Event },
	{ MPI_FILE_READ_AT_ALL_EV, MPIIO_Event },
	{ MPI_FILE_WRITE_AT_EV, MPIIO_Event },
	{ MPI_FILE_WRITE_AT_ALL_EV, MPIIO_Event },
	{ MPI_PUT_EV, MPI_RMA_Event},
	{ MPI_GET_EV, MPI_RMA_Event},
	{ MPI_WIN_CREATE_EV, MPI_RMA_Event},
	{ MPI_WIN_FENCE_EV, MPI_RMA_Event},
	{ MPI_WIN_START_EV, MPI_RMA_Event},
	{ MPI_WIN_FREE_EV, MPI_RMA_Event},
	{ MPI_WIN_POST_EV, MPI_RMA_Event},
	{ MPI_WIN_COMPLETE_EV, MPI_RMA_Event},
	{ MPI_WIN_WAIT_EV, MPI_RMA_Event},
	{ MPI_IREDUCE_EV, GlobalOP_event},
	{ MPI_IALLREDUCE_EV, GlobalOP_event},
	{ MPI_IBARRIER_EV, GlobalOP_event},
	{ MPI_IBCAST_EV, GlobalOP_event},
	{ MPI_IALLTOALL_EV, GlobalOP_event},
	{ MPI_IALLTOALLV_EV, GlobalOP_event},
	{ MPI_IALLGATHER_EV, GlobalOP_event},
	{ MPI_IALLGATHERV_EV, GlobalOP_event},
	{ MPI_IGATHER_EV, GlobalOP_event},
	{ MPI_IGATHERV_EV, GlobalOP_event},
	{ MPI_ISCATTER_EV, GlobalOP_event},
	{ MPI_ISCATTERV_EV, GlobalOP_event},
	{ MPI_IREDUCESCAT_EV, GlobalOP_event},
	{ MPI_ISCAN_EV, GlobalOP_event},
	{ NULL_EV, NULL }
};
