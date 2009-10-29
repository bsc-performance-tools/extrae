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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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
#include "mpi_prv_events.h"
#include "semantics.h"
#include "mpi_prv_semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "timesync.h"
#include "communication_queues.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

#ifndef HAVE_MPI_H
# define MPI_PROC_NULL (-1)
#else
# include <mpi.h>
#endif

/******************************************************************************
 ***  trace_communication
 ******************************************************************************/

static void trace_communicationAt (unsigned ptask, event_t *send_begin,
	event_t *send_end, event_t *recv_begin, event_t *recv_end, 
	int atposition, off_t position)
{
	struct thread_t *thread_r_info, *thread_s_info;
	unsigned long long log_s, log_r, phy_s, phy_r;
	unsigned cpu_r, cpu_s, task_r, task_s, thread_r, thread_s;

	/* Look for the receive partner ... in the sender events */
	task_r = Get_EvTarget (send_begin)+1;
	thread_r = 1;
	thread_r_info = GET_THREAD_INFO(ptask, task_r, thread_r);
	cpu_r = thread_r_info->cpu;

	/* Look for the sender partner ... in the receiver events */
	task_s = Get_EvTarget (recv_end)+1;
	thread_s = 1;
	thread_s_info = GET_THREAD_INFO(ptask, task_s, thread_s);
	cpu_s = thread_s_info->cpu;

	/* Synchronize event times */
	log_s = TIMESYNC(task_s-1, Get_EvTime (send_begin));
	phy_s = TIMESYNC(task_s-1, Get_EvTime (send_end));
	log_r = TIMESYNC(task_r-1, Get_EvTime (recv_begin));
	phy_r = TIMESYNC(task_r-1, Get_EvTime (recv_end));
	
	trace_paraver_communication (cpu_s, ptask, task_s, thread_s, log_s, phy_s,
	  cpu_r, ptask, task_r, thread_r, log_r, phy_r, Get_EvSize (recv_end),
		Get_EvTag (recv_end), atposition, position);
}

#if defined(PARALLEL_MERGE)
static int trace_pending_communication (unsigned int ptask, unsigned int task,
	unsigned int thread, event_t * begin_s, event_t * end_s, unsigned int recvr)
{
	unsigned long long log_s, phy_s;

	/* Synchronize event times */
	log_s = TIMESYNC (task-1, Get_EvTime (begin_s));
	phy_s = TIMESYNC (task-1, Get_EvTime (end_s));

	trace_paraver_pending_communication (task, ptask, task, thread, log_s,
		phy_s, recvr + 1, ptask, recvr + 1, thread, 0ULL, 0ULL, Get_EvSize (begin_s), Get_EvTag (begin_s));
  return 0;
}
#endif

#if defined(DEAD_CODE)
static void MatchComms (
	FileSet_t *fset,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	unsigned int receiver, unsigned int tag,
	event_t *send_begin, event_t *send_end)
{
	int pending = 0;
	unsigned int sender = task - 1;
	event_t *recv_begin = NULL, *recv_end = NULL;

	if (MatchComms_Enabled(ptask, task, thread))
	{
		pending = SearchRecvEvent_FS (fset, ptask, receiver, sender, tag, &recv_begin, &recv_end);
		if (!pending)
		{
			trace_communication (cpu, ptask, task, thread, send_begin, send_end, recv_begin, recv_end);
		}
#if defined(PARALLEL_MERGE)
		else
		{
			trace_pending_communication (ptask, task, thread, send_begin, send_end, receiver);
		}
#endif
	}
}
#endif

static int Get_State (unsigned int EvType)
{
	int state = 0;
	
	switch (EvType)
	{
		case MPIINIT_EV:
		case FINALIZE_EV:
			state = STATE_INITFINI;
		break;
		case FILE_OPEN_EV:
		case FILE_CLOSE_EV:
		case FILE_READ_EV:
		case FILE_READ_ALL_EV:
		case FILE_WRITE_EV:
		case FILE_WRITE_ALL_EV:
		case FILE_READ_AT_EV:
		case FILE_READ_AT_ALL_EV:
		case FILE_WRITE_AT_EV:
		case FILE_WRITE_AT_ALL_EV:
			state = STATE_IO;
		break;
		case REQUEST_FREE_EV:
		case COMM_RANK_EV:
		case COMM_SIZE_EV:
		case CANCEL_EV:
			state = STATE_MIXED;
		break;
		case PROBE_EV:
		case IPROBE_EV:
			state = STATE_PROBE;
		break;
		case TEST_EV:
		case WAIT_EV:
		case WAITALL_EV:
		case WAITSOME_EV:
		case WAITANY_EV:
			state = STATE_TWRECV;
		break;
		case SEND_EV:
		case RSEND_EV:
		case SSEND_EV:
		case BSEND_EV:
			state = STATE_SEND;
		break;
		case ISEND_EV:
		case IRSEND_EV:
		case ISSEND_EV:
		case IBSEND_EV:
			state = STATE_ISEND;
		break;
		case BARRIER_EV:
			state = STATE_BARRIER;
		break;
		case REDUCE_EV:
		case ALLREDUCE_EV:
		case BCAST_EV:
		case ALLTOALL_EV:
		case ALLTOALLV_EV:
		case ALLGATHER_EV:
		case ALLGATHERV_EV:
		case GATHER_EV:
		case GATHERV_EV:
		case SCATTER_EV:
		case SCATTERV_EV:
		case REDUCESCAT_EV:
		case SCAN_EV:
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
	unsigned recv_thread, EvType, EvValue;
	struct thread_t *thread_info, *thread_info_partner;
	event_t * recv_begin, * recv_end;

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	thread_info = GET_THREAD_INFO(ptask, task, thread);
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
						thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(current_event)].threads[0]);

						CommunicationQueues_ExtractRecv (thread_info_partner->file, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;
							position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
							CommunicationQueues_QueueSend (thread_info->file, thread_info->Send_Rec, current_event, position, thread);
							trace_paraver_unmatched_communication (ptask, task, thread);
						}
						else
							trace_communicationAt (ptask, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
					}
#if defined(PARALLEL_MERGE)
					else
						trace_pending_communication (ptask, task, thread, thread_info->Send_Rec, current_event, Get_EvTarget (current_event));
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
	struct thread_t *thread_info, *thread_info_partner;
#if !defined(AVOID_SENDRECV)
	unsigned recv_thread, send_thread;
	event_t *recv_begin, *recv_end, *send_begin, *send_end;
	off_t send_position;
#endif

	Switch_State (STATE_SENDRECVOP, (Get_EvValue(current_event) == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, Get_EvEvent(current_event), Get_EvValue(current_event));

	thread_info = GET_THREAD_INFO(ptask, task, thread);

	if (!option_SkipSendRecvComms)
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
						thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(thread_info->Send_Rec)].threads[0]);

						CommunicationQueues_ExtractRecv (thread_info_partner->file, task-1, Get_EvTag (thread_info->Send_Rec), &recv_begin, &recv_end, &recv_thread);

						if (recv_begin == NULL || recv_end == NULL)
						{
							off_t position;
							position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
							CommunicationQueues_QueueSend (thread_info->file, thread_info->Send_Rec, current_event, position, thread);
							trace_paraver_unmatched_communication (ptask, task, thread);
						}
						else if (recv_begin != NULL && recv_end != NULL)
							trace_communicationAt (ptask, thread_info->Send_Rec, current_event, recv_begin, recv_end, FALSE, 0);
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractRecv returned recv_begin = %p and recv_end = %p\n", recv_begin, recv_end);
					}
#if defined(PARALLEL_MERGE)
					else
						trace_pending_communication (ptask, task, thread, thread_info->Send_Rec, current_event, Get_EvTarget (thread_info->Send_Rec));
#endif /* PARALLEL_MERGE */
					}

			/* Treat the receive part */
			if (MatchComms_Enabled(ptask, task, thread))
				if (MPI_PROC_NULL != Get_EvTarget (current_event))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
					{
						thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(current_event)].threads[0]);

						CommunicationQueues_ExtractSend (thread_info_partner->file, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread);

						if (NULL == send_begin && NULL == send_end)
							CommunicationQueues_QueueRecv (thread_info->file, thread_info->Recv_Rec, current_event, thread);
						else if (NULL != send_begin && NULL != send_end)
							trace_communicationAt (ptask, send_begin, send_end, thread_info->Recv_Rec, current_event, TRUE, send_position);
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(thread_info->Recv_Rec));
						phy_r = TIMESYNC (task-1, Get_EvTime(current_event));
						AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), task-1,
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
		case REDUCE_EV:
			res = Get_EvAux(current) == Get_EvTag(current);
		break;
		case BCAST_EV:
			res = Get_EvTarget(current) == Get_EvTag(current);
		break;
		case GATHER_EV:
		case GATHERV_EV:
		case SCATTER_EV:
		case SCATTERV_EV:
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
		case BARRIER_EV:
			res = 0;
		break;
		case REDUCE_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case BCAST_EV:
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
		case BARRIER_EV:
			res = 0;
		break;
		case REDUCE_EV:
			res = (!is_root)?0:Get_EvSize(current);
		break;
		case BCAST_EV:
			res = (is_root)?0:Get_EvSize(current);
		break;
		case REDUCESCAT_EV:
		case SCAN_EV:
		case ALLREDUCE_EV:
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
		  send_size, receive_size, is_root?1:0);
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
	if ((EvType == MPIINIT_EV) && (EvValue == EVT_END))
	{
		if ((tracingCircularBuffer()) && (getBehaviourForCircularBuffer() == CIRCULAR_SKIP_MATCHES))
		{
			/* The first event beyond the MPI_Init will remove the STATE_NOT_TRACING (see Push_State) */
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
	unsigned EvType, EvValue, send_thread;
	struct thread_t *thread_info, *thread_info_partner;

	thread_info = &(obj_table[ptask - 1].tasks[task - 1].threads[thread - 1]);

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
					thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(current_event)].threads[0]);

					CommunicationQueues_ExtractSend (thread_info_partner->file, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread);

					if (NULL == send_begin || NULL == send_end)
						CommunicationQueues_QueueRecv (thread_info->file, thread_info->Recv_Rec, current_event, thread);
					else if (NULL != send_begin && NULL != send_end)
						trace_communicationAt (ptask, send_begin, send_end, thread_info->Recv_Rec, current_event, TRUE, send_position);
					else
						fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
				}
#if defined(PARALLEL_MERGE)
				else
				{
					UINT64 log_r, phy_r;

					log_r = TIMESYNC (task-1, Get_EvTime(thread_info->Recv_Rec));
					phy_r = TIMESYNC (task-1, Get_EvTime(current_event));
					AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), task-1,
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
	unsigned EvType, EvValue, send_thread;
	struct thread_t *thread_info, *thread_info_partner;

	thread_info = &(obj_table[ptask - 1].tasks[task - 1].threads[thread - 1]);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_IWAITMESS, (EvValue == EVT_BEGIN), ptask, task, thread);

	if (EvValue == EVT_END)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			event_t *receive = SearchIRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (NULL != receive)
			{
				if (MPI_PROC_NULL != Get_EvTarget(receive))
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(receive)))
					{
						thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(receive)].threads[0]);

						CommunicationQueues_ExtractSend (thread_info_partner->file, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread);

						if (NULL == send_begin || NULL == send_end)
							CommunicationQueues_QueueRecv (thread_info->file, current_event, receive, thread);
						else if (NULL != send_begin && NULL != send_end)
							trace_communicationAt (ptask, send_begin, send_end, current_event, receive, TRUE, send_position);
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (task-1, Get_EvTime(receive));

						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), task-1, Get_EvTarget(receive), fset);
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

	Switch_State (((EvType == RECV_INIT_EV) ? STATE_IRECV : STATE_ISEND), (EvValue == EVT_BEGIN), ptask, task, thread);

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
	struct thread_t *thread_info, *thread_info_partner;
	event_t *recv_begin, *recv_end;
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned recv_thread, send_thread;

	thread_info = GET_THREAD_INFO(ptask, task, thread);
	trace_paraver_state (cpu, ptask, task, thread, current_time);

	/* If this is a send, look for the receive */
	if (Get_EvValue (current_event) == ISEND_EV)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			if (MPI_PROC_NULL != Get_EvTarget (current_event))
			{
				if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
				{
					thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(current_event)].threads[0]);

					CommunicationQueues_ExtractRecv (thread_info_partner->file, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread);

					if (recv_begin == NULL || recv_end == NULL)
					{
						off_t position;
						position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
						CommunicationQueues_QueueSend (thread_info->file, current_event, current_event, position, thread);
						trace_paraver_unmatched_communication (ptask, task, thread);
					}
					else
						trace_communicationAt (ptask, current_event, current_event, recv_begin, recv_end, FALSE, 0);
				}
#if defined(PARALLEL_MERGE)
				else
					trace_pending_communication (ptask, task, thread, thread_info->Send_Rec, current_event, Get_EvTarget (current_event));
#endif
			}
		}
	}

	/* If this is a receive, look for the send */
	if (Get_EvValue(current_event) == IRECV_EV)
	{
		if (MatchComms_Enabled(ptask, task, thread))
		{
			event_t *receive = SearchIRECVED (current_event, Get_EvAux (current_event), thread_info->file);
			if (MPI_PROC_NULL != Get_EvTarget(receive))
			{
				if (NULL != receive)
				{
					if (isTaskInMyGroup (fset, Get_EvTarget(receive)))
					{
						thread_info_partner = &(obj_table[ptask-1].tasks[Get_EvTarget(receive)].threads[0]);

						CommunicationQueues_ExtractSend (thread_info_partner->file, task-1, Get_EvTag (receive), &send_begin, &send_end, &send_position, &send_thread);

						if (NULL == send_begin || NULL == send_end)
							CommunicationQueues_QueueRecv (thread_info->file, current_event, receive, thread);
						else if (NULL != send_begin && NULL != send_end)
							trace_communicationAt (ptask, send_begin, send_end, current_event, receive, TRUE, send_position);
						else
							fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
					}
#if defined(PARALLEL_MERGE)
					else
					{
						UINT64 log_r, phy_r;

						log_r = TIMESYNC (task-1, Get_EvTime(current_event));
						phy_r = TIMESYNC (task-1, Get_EvTime(receive));

						AddForeignRecv (phy_r, log_r, Get_EvTag(receive), task-1, Get_EvTarget(receive), fset);
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

	Enable_Soft_Counter (EvType);

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
	double elapsed_time;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	elapsed_time = Get_EvValue (current_event);
	EvValue = (unsigned long long) (elapsed_time);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	Enable_Soft_Counter (EvType);

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

	Enable_Soft_Counter (EvType);

	return 0;
}


/******************************************************************************
 ***  MPI_GenerateParaverTraces
 ******************************************************************************/

SingleEv_Handler_t PRV_MPI_Event_Handlers[] = {
	{ SEND_EV, Any_Send_Event },
	{ BSEND_EV, Any_Send_Event },
	{ SSEND_EV, Any_Send_Event },
	{ RSEND_EV, Any_Send_Event },
	{ IBSEND_EV, Any_Send_Event },
	{ ISSEND_EV, Any_Send_Event },
	{ IRSEND_EV, Any_Send_Event },
	{ ISEND_EV, Any_Send_Event },
	{ SENDRECV_EV, SendRecv_Event },
	{ SENDRECV_REPLACE_EV, SendRecv_Event },
	{ RECV_EV, Recv_Event },
	{ IRECV_EV, IRecv_Event },
	{ REDUCE_EV, GlobalOP_event },
	{ ALLREDUCE_EV, GlobalOP_event },
	{ PROBE_EV, Other_MPI_Event },
	{ IPROBE_EV, Other_MPI_Event },
	{ BARRIER_EV, GlobalOP_event },
	{ CANCEL_EV, Other_MPI_Event },
	{ TEST_EV, Other_MPI_Event },
	{ WAIT_EV, Other_MPI_Event },
	{ WAITALL_EV, Other_MPI_Event },
	{ WAITANY_EV, Other_MPI_Event },
	{ WAITSOME_EV, Other_MPI_Event },
	{ IRECVED_EV, SkipHandler },
	{ BCAST_EV, GlobalOP_event },
	{ ALLTOALL_EV, GlobalOP_event },
	{ ALLTOALLV_EV, GlobalOP_event },
	{ ALLGATHER_EV, GlobalOP_event },
	{ ALLGATHERV_EV, GlobalOP_event },
	{ GATHER_EV, GlobalOP_event },
	{ GATHERV_EV, GlobalOP_event },
	{ SCATTER_EV, GlobalOP_event },
	{ SCATTERV_EV, GlobalOP_event },
	{ REDUCESCAT_EV, GlobalOP_event },
	{ SCAN_EV, GlobalOP_event },
	{ MPIINIT_EV, Other_MPI_Event },
	{ FINALIZE_EV, Other_MPI_Event },
	{ RECV_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ SEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ BSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ RSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ SSEND_INIT_EV, MPI_PersistentRequest_Init_Event },
	{ PERSIST_REQ_EV, MPI_PersistentRequest_Event },
	{ START_EV, MPI_Start_Event },
	{ STARTALL_EV, MPI_Start_Event },
	{ REQUEST_FREE_EV, MPI_PersistentRequest_Free_Event },
	{ COMM_RANK_EV, Other_MPI_Event },
	{ COMM_SIZE_EV, Other_MPI_Event },
	{ IPROBE_COUNTER_EV, MPI_IProbeSoftwareCounter_Event },
	{ TIME_OUTSIDE_IPROBES_EV, MPI_ElapsedTimeOutsideIProbes_Event },
	{ TEST_COUNTER_EV, MPI_TestSoftwareCounter_Event },
	{ FILE_OPEN_EV, Other_MPI_Event },
	{ FILE_CLOSE_EV, Other_MPI_Event },
	{ FILE_READ_EV, Other_MPI_Event },
	{ FILE_READ_ALL_EV, Other_MPI_Event },
	{ FILE_WRITE_EV, Other_MPI_Event },
	{ FILE_WRITE_ALL_EV, Other_MPI_Event },
	{ FILE_READ_AT_EV, Other_MPI_Event },
	{ FILE_READ_AT_ALL_EV, Other_MPI_Event },
	{ FILE_WRITE_AT_EV, Other_MPI_Event },
	{ FILE_WRITE_AT_ALL_EV, Other_MPI_Event },
	{ NULL_EV, NULL }
};
