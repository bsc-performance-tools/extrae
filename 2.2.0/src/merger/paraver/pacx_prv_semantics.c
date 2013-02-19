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
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "mpi2out.h"
#include "events.h"
#include "trace_to_prv.h"
#include "HardwareCounters.h"
#include "pacx_prv_events.h"
#include "semantics.h"
#include "pacx_prv_semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "timesync.h"
#include "communication_queues.h"
#include "trace_communication.h"
#include "options.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

#if 0
# include <pacx.h>
#endif

#define DEBUG

#define MPI_PROC_NULL (-1)

static int Get_State (unsigned int EvType)
{
	int state = 0;
	
	switch (EvType)
	{
		case PACX_INIT_EV:
		case PACX_FINALIZE_EV:
			state = STATE_INITFINI;
		break;
		case PACX_FILE_OPEN_EV:
		case PACX_FILE_CLOSE_EV:
		case PACX_FILE_READ_EV:
		case PACX_FILE_READ_ALL_EV:
		case PACX_FILE_WRITE_EV:
		case PACX_FILE_WRITE_ALL_EV:
		case PACX_FILE_READ_AT_EV:
		case PACX_FILE_READ_AT_ALL_EV:
		case PACX_FILE_WRITE_AT_EV:
		case PACX_FILE_WRITE_AT_ALL_EV:
			state = STATE_IO;
		break;
		case PACX_REQUEST_FREE_EV:
		case PACX_COMM_RANK_EV:
		case PACX_COMM_SIZE_EV:
		case PACX_CANCEL_EV:
			state = STATE_MIXED;
		break;
		case PACX_PROBE_EV:
		case PACX_IPROBE_EV:
			state = STATE_PROBE;
		break;
		case PACX_TEST_EV:
		case PACX_WAIT_EV:
		case PACX_WAITALL_EV:
		case PACX_WAITSOME_EV:
		case PACX_WAITANY_EV:
			state = STATE_TWRECV;
		break;
		case PACX_SEND_EV:
		case PACX_RSEND_EV:
		case PACX_SSEND_EV:
		case PACX_BSEND_EV:
			state = STATE_SEND;
		break;
		case PACX_ISEND_EV:
		case PACX_IRSEND_EV:
		case PACX_ISSEND_EV:
		case PACX_IBSEND_EV:
			state = STATE_ISEND;
		break;
		case PACX_BARRIER_EV:
			state = STATE_BARRIER;
		break;
		case PACX_REDUCE_EV:
		case PACX_ALLREDUCE_EV:
		case PACX_BCAST_EV:
		case PACX_ALLTOALL_EV:
		case PACX_ALLTOALLV_EV:
		case PACX_ALLGATHER_EV:
		case PACX_ALLGATHERV_EV:
		case PACX_GATHER_EV:
		case PACX_GATHERV_EV:
		case PACX_SCATTER_EV:
		case PACX_SCATTERV_EV:
		case PACX_REDUCESCAT_EV:
		case PACX_SCAN_EV:
			state = STATE_BCAST;
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
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;
	event_t * recv_begin, * recv_end;

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

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
			if (MatchComms_Enabled(ptask, task, thread))
				if (MPI_PROC_NULL != Get_EvTarget (current_event))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
					{
#if defined(DEBUG)
						fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", EvType, current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);
						CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;

#if defined(DEBUG)
							fprintf (stdout, "SEND_CMD(%u) DID NOT find receiver\n", EvType);
#endif
							position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
							CommunicationQueues_QueueSend (task_info->send_queue, thread_info->Send_Rec, current_event, position, thread, thread_info->virtual_thread, 0);
							trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
						}
						else
						{
#if defined(DEBUG)
							fprintf (stdout, "SEND_CMD(%u) find receiver\n", EvType);
#endif
							trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, 1+Get_EvTarget(current_event), recv_thread, recv_vthread, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
						}
					}
#if defined(PARALLEL_MERGE)
					else
						trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, Get_EvTarget (current_event));
#endif
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
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;
#if !defined(AVOID_SENDRECV)
	unsigned recv_thread, send_thread, recv_vthread, send_vthread;
	event_t *recv_begin, *recv_end, *send_begin, *send_end;
	off_t send_position;
#endif

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
			thread_info->Recv_Rec = current_event;
		}
		else if (Get_EvValue (current_event) == EVT_END)
		{
			/* Treat the send part */
			if (MatchComms_Enabled(ptask, task, thread))
				if (MPI_PROC_NULL != Get_EvTarget (thread_info->Send_Rec))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(thread_info->Send_Rec)))
					{
#if defined(DEBUG)
						fprintf (stdout, "SENDRECV/SEND: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(thread_info->Send_Rec)+1);

						CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (thread_info->Send_Rec), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;

#if defined(DEBUG)
							fprintf (stdout, "SENDRECV/SEND DID NOT find partner\n");
#endif
							position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
							CommunicationQueues_QueueSend (task_info->send_queue, thread_info->Send_Rec, current_event, position, thread, thread_info->virtual_thread, 0);
							trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
						}
						else if (recv_begin != NULL && recv_end != NULL)
						{
#if defined(DEBUG)
							fprintf (stdout, "SENDRECV/SEND find partner\n");
#endif
							trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, 1+Get_EvTarget(thread_info->Send_Rec), recv_thread, recv_vthread, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractRecv returned recv_begin = %p and recv_end = %p\n", recv_begin, recv_end);
					}
#if defined(PARALLEL_MERGE)
					else
						trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, Get_EvTarget (thread_info->Send_Rec));
#endif /* PARALLEL_MERGE */
					}

			/* Treat the receive part */
			if (MatchComms_Enabled(ptask, task, thread))
				if (MPI_PROC_NULL != Get_EvTarget (current_event))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
					{
#if defined(DEBUG)
						fprintf (stdout, "SENDRECV/RECV: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif

						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin && NULL == send_end)
						{
#if defined(DEBUG)
							fprintf (stdout, "SENDRECV/RECV DID NOT find partner\n");
#endif
							CommunicationQueues_QueueRecv (task_info->recv_queue, thread_info->Recv_Rec, current_event, thread, thread_info->virtual_thread, 0);
						}
						else if (NULL != send_begin && NULL != send_end)
						{
#if defined(DEBUG)
							fprintf (stdout, "SENDRECV/RECV find partner\n");
#endif
							trace_communicationAt (ptask, 1+Get_EvTarget(current_event), send_thread, send_vthread, task, thread, thread_info->virtual_thread, send_begin, send_end, thread_info->Recv_Rec, current_event, TRUE, send_position);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(thread_info->Recv_Rec));
						phy_r = TIMESYNC (task-1, Get_EvTime(current_event));
						AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), task-1, thread-1, thread_info->virtual_thread-1,
						  Get_EvTarget(current_event), fset);
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
		case PACX_REDUCE_EV:
			res = Get_EvAux(current) == Get_EvTag(current);
		break;
		case PACX_BCAST_EV:
			res = Get_EvTarget(current) == Get_EvTag(current);
		break;
		case PACX_GATHER_EV:
		case PACX_GATHERV_EV:
		case PACX_SCATTER_EV:
		case PACX_SCATTERV_EV:
			res = Get_EvTarget(current) == task-1;
		break;
	}
	return res;
}

static unsigned int Get_GlobalOP_SendSize (event_t *current, int is_root)
{
	unsigned int res = 0;
	switch (Get_EvEvent(current))
	{
		case PACX_BARRIER_EV:
			res = 0;
		break;
		case PACX_REDUCE_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case PACX_BCAST_EV:
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
		case PACX_BARRIER_EV:
			res = 0;
		break;
		case PACX_REDUCE_EV:
			res = (!is_root)?0:Get_EvSize(current);
		break;
		case PACX_BCAST_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case PACX_REDUCESCAT_EV:
		case PACX_SCAN_EV:
		case PACX_ALLREDUCE_EV:
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

	/* First global operation found, start matching communications from now on (if this is the behaviour for the circular buffer) */
	if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES) && (!MatchComms_Enabled(ptask, task, thread)) && (EvValue == EVT_BEGIN) && (getTagForCircularBuffer() == Get_EvAux(current_event)))
	{
		MatchComms_On(ptask, task, thread);
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
		  send_size, receive_size, is_root?1:0, FALSE);
	}

	return 0;
}

/******************************************************************************
 ***  Other_PACX_Event:
 ******************************************************************************/

static int Other_PACX_Event (event_t * current_event, 
	unsigned long long current_time, unsigned int cpu, unsigned int ptask, 
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

	/* XXX: Workaround to set the state to NOT_TRACING after the PACX_Init when using circular buffer.
     * We should definitely do this another way round. 
	 */ 
	if ((EvType == PACX_INIT_EV) && (EvValue == EVT_END))
	{
		if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES))
		{
			/* The first event beyond the PACX_Init will remove the STATE_NOT_TRACING (see Push_State) */
			Push_State (STATE_NOT_TRACING, ptask, task, thread);
		}
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

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
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_WAITMESS, (EvValue == EVT_BEGIN), ptask, task, thread);

	if (EvValue == EVT_BEGIN)
	{
		thread_info->Recv_Rec = current_event;
	}
	else
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			if (MPI_PROC_NULL != Get_EvTarget(current_event))
			{
				if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
				{
#if defined(DEBUG)
					fprintf (stdout, "RECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
					task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

					CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

					if (NULL == send_begin || NULL == send_end)
					{
#if defined(DEBUG)
						fprintf (stdout, "RECV_CMD DID NOT find partner\n");
#endif
						CommunicationQueues_QueueRecv (task_info->recv_queue, thread_info->Recv_Rec, current_event, thread, thread_info->virtual_thread, 0);
					}
					else if (NULL != send_begin && NULL != send_end)
					{
#if defined(DEBUG)
						fprintf (stdout, "RECV_CMD find partner\n");
#endif
						trace_communicationAt (ptask, 1+Get_EvTarget(current_event), send_thread, send_vthread, task, thread, thread_info->virtual_thread, send_begin, send_end, thread_info->Recv_Rec, current_event, TRUE, send_position);
					}
					else
						fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
				}
#if defined(PARALLEL_MERGE)
				else
				{
					UINT64 log_r, phy_r;

					log_r = TIMESYNC (task-1, Get_EvTime(thread_info->Recv_Rec));
					phy_r = TIMESYNC (task-1, Get_EvTime(current_event));
					AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), task-1, thread-1, thread_info->virtual_thread-1,
					  Get_EvTarget(current_event), fset);
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
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_info = GET_TASK_INFO(ptask, task);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_IWAITMESS, (EvValue == EVT_BEGIN), ptask, task, thread);

	if (EvValue == EVT_END)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			event_t *receive = Search_MPI_IRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (NULL != receive)
			{
				if (MPI_PROC_NULL != Get_EvTarget(receive))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(receive)))
					{
#if defined(DEBUG)
						fprintf (stdout, "IRECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(receive), Get_EvTag(receive));
#endif
						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(receive)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin || NULL == send_end)
						{
#if defined(DEBUG)
							fprintf (stdout, "IRECV_CMD DID NOT find COMM\n");
#endif
							CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, receive, thread, thread_info->virtual_thread, 0);
						}
						else if (NULL != send_begin && NULL != send_end)
						{
#if defined(DEBUG)
							fprintf (stdout, "IRECV_CMD find COMM (partner times = %lld/%lld)\n", Get_EvTime(send_begin), Get_EvTime(send_end));
#endif
							trace_communicationAt (ptask, 1+Get_EvTarget(receive), send_thread, send_vthread, task, thread, thread_info->virtual_thread, send_begin, send_end, current_event, receive, TRUE, send_position);
						}
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (task-1, Get_EvTime(receive));

						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), task-1, thread-1, thread_info->virtual_thread-1, Get_EvTarget(receive), fset);
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
 ***  PACX_PersistentRequest_Init_Event
 ******************************************************************************/

int PACX_PersistentRequest_Init_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (((EvType == PACX_RECV_INIT_EV) ? STATE_IRECV : STATE_ISEND), (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}


/******************************************************************************
 ***  PACX_PersistentRequest_Event
 ******************************************************************************/

int PACX_PersistentRequest_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;
	event_t *recv_begin, *recv_end;
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned recv_thread, send_thread, recv_vthread, send_vthread;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	trace_paraver_state (cpu, ptask, task, thread, current_time);

	/* If this is a send, look for the receive */
	if (Get_EvValue (current_event) == PACX_ISEND_EV)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			if (MPI_PROC_NULL != Get_EvTarget (current_event))
			{
				if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
				{
#if defined(DEBUG)
					fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", Get_EvValue(current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif

					task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

					CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, &recv_vthread, 0);

					if (recv_begin == NULL || recv_end == NULL)
					{
						off_t position;
						position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
						CommunicationQueues_QueueSend (task_info->send_queue, current_event, current_event, position, thread, thread_info->virtual_thread, 0);
						trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, ptask, Get_EvTarget(current_event)+1, 1, Get_EvSize(current_event), Get_EvTag(current_event));
					}
					else
						trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, 1+Get_EvTarget(current_event), recv_thread, recv_vthread, current_event, current_event, recv_begin, recv_end, FALSE, 0);

				}
#if defined(PARALLEL_MERGE)
				else
					trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, thread_info->Send_Rec, current_event, Get_EvTarget (current_event));
#endif
			}
		}
	}

	/* If this is a receive, look for the send */
	if (Get_EvValue(current_event) == PACX_IRECV_EV)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			event_t *receive = Search_PACX_IRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (MPI_PROC_NULL != Get_EvTarget(receive))
			{
				if (NULL != receive)
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(receive)))
					{
#if defined(DEBUG)
						fprintf (stdout, "IRECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif

						task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

						CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, 0);

						if (NULL == send_begin || NULL == send_end)
							CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, receive, thread, thread_info->virtual_thread, 0);
						else if (NULL != send_begin && NULL != send_end)
							trace_communicationAt (ptask, 1+Get_EvTarget(receive), send_thread, send_vthread, task, thread, thread_info->virtual_thread, send_begin, send_end, current_event, receive, TRUE, send_position);
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (task-1, Get_EvTime(receive));

						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), task-1, thread-1, thread_info->virtual_thread-1, Get_EvTarget(receive), fset);
					}
#endif
				}
			}
		}
	}

	return 0;
}

/******************************************************************************
 ***  PACX_PersistentRequest_Free_Event
 ******************************************************************************/

int PACX_PersistentRequest_Free_Event (event_t * current_event,
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
 ***  PACX_Start_Event
 ******************************************************************************/

int PACX_Start_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	struct thread_t * thread_info;
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
		   This will be solved in PACX_PersistentRequest_Event */
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
 ***  PACX_IProbeSoftwareCounter_Event
 ******************************************************************************/

int PACX_IProbeSoftwareCounter_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_PACX_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  PACX_ElapsedTimeOutsideIProbes_Event
 ******************************************************************************/

int PACX_ElapsedTimeOutsideIProbes_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType;
	unsigned long long EvValue;
	double elapsed_time;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	elapsed_time = Get_EvValue (current_event);
	EvValue = (unsigned long long) (elapsed_time);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_PACX_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  PACX_TestSoftwareCounter_Event
 ******************************************************************************/

int PACX_TestSoftwareCounter_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_PACX_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  PACX_GenerateParaverTraces
 ******************************************************************************/

SingleEv_Handler_t PRV_PACX_Event_Handlers[] = {
	{ PACX_SEND_EV, Any_Send_Event },
	{ PACX_BSEND_EV, Any_Send_Event },
	{ PACX_SSEND_EV, Any_Send_Event },
	{ PACX_RSEND_EV, Any_Send_Event },
	{ PACX_IBSEND_EV, Any_Send_Event },
	{ PACX_ISSEND_EV, Any_Send_Event },
	{ PACX_IRSEND_EV, Any_Send_Event },
	{ PACX_ISEND_EV, Any_Send_Event },
	{ PACX_SENDRECV_EV, SendRecv_Event },
	{ PACX_SENDRECV_REPLACE_EV, SendRecv_Event },
	{ PACX_RECV_EV, Recv_Event },
	{ PACX_IRECV_EV, IRecv_Event },
	{ PACX_REDUCE_EV, GlobalOP_event },
	{ PACX_ALLREDUCE_EV, GlobalOP_event },
	{ PACX_PROBE_EV, Other_PACX_Event },
	{ PACX_IPROBE_EV, Other_PACX_Event },
	{ PACX_BARRIER_EV, GlobalOP_event },
	{ PACX_CANCEL_EV, Other_PACX_Event },
	{ PACX_TEST_EV, Other_PACX_Event },
	{ PACX_WAIT_EV, Other_PACX_Event },
	{ PACX_WAITALL_EV, Other_PACX_Event },
	{ PACX_WAITANY_EV, Other_PACX_Event },
	{ PACX_WAITSOME_EV, Other_PACX_Event },
	{ PACX_IRECVED_EV, SkipHandler },
	{ PACX_BCAST_EV, GlobalOP_event },
	{ PACX_ALLTOALL_EV, GlobalOP_event },
	{ PACX_ALLTOALLV_EV, GlobalOP_event },
	{ PACX_ALLGATHER_EV, GlobalOP_event },
	{ PACX_ALLGATHERV_EV, GlobalOP_event },
	{ PACX_GATHER_EV, GlobalOP_event },
	{ PACX_GATHERV_EV, GlobalOP_event },
	{ PACX_SCATTER_EV, GlobalOP_event },
	{ PACX_SCATTERV_EV, GlobalOP_event },
	{ PACX_REDUCESCAT_EV, GlobalOP_event },
	{ PACX_SCAN_EV, GlobalOP_event },
	{ PACX_INIT_EV, Other_PACX_Event },
	{ PACX_FINALIZE_EV, Other_PACX_Event },
	{ PACX_RECV_INIT_EV, PACX_PersistentRequest_Init_Event },
	{ PACX_SEND_INIT_EV, PACX_PersistentRequest_Init_Event },
	{ PACX_BSEND_INIT_EV, PACX_PersistentRequest_Init_Event },
	{ PACX_RSEND_INIT_EV, PACX_PersistentRequest_Init_Event },
	{ PACX_SSEND_INIT_EV, PACX_PersistentRequest_Init_Event },
	{ PACX_PERSIST_REQ_EV, PACX_PersistentRequest_Event },
	{ PACX_START_EV, PACX_Start_Event },
	{ PACX_STARTALL_EV, PACX_Start_Event },
	{ PACX_REQUEST_FREE_EV, PACX_PersistentRequest_Free_Event },
	{ PACX_COMM_RANK_EV, Other_PACX_Event },
	{ PACX_COMM_SIZE_EV, Other_PACX_Event },
	{ PACX_IPROBE_COUNTER_EV, PACX_IProbeSoftwareCounter_Event },
	{ PACX_TIME_OUTSIDE_IPROBES_EV, PACX_ElapsedTimeOutsideIProbes_Event },
	{ PACX_TEST_COUNTER_EV, PACX_TestSoftwareCounter_Event },
	{ PACX_FILE_OPEN_EV, Other_PACX_Event },
	{ PACX_FILE_CLOSE_EV, Other_PACX_Event },
	{ PACX_FILE_READ_EV, Other_PACX_Event },
	{ PACX_FILE_READ_ALL_EV, Other_PACX_Event },
	{ PACX_FILE_WRITE_EV, Other_PACX_Event },
	{ PACX_FILE_WRITE_ALL_EV, Other_PACX_Event },
	{ PACX_FILE_READ_AT_EV, Other_PACX_Event },
	{ PACX_FILE_READ_AT_ALL_EV, Other_PACX_Event },
	{ PACX_FILE_WRITE_AT_EV, Other_PACX_Event },
	{ PACX_FILE_WRITE_AT_ALL_EV, Other_PACX_Event },
	{ NULL_EV, NULL }
};
