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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/communication_queues.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 14:40:44 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: communication_queues.c,v 1.2 2009/05/28 14:40:44 harald Exp $";

#include "communication_queues.h"
#include "file_set.h"

/**************************************************************************
*** SEND PART
**************************************************************************/

typedef struct
{
	event_t *send_begin;
	event_t *send_end;
	off_t send_position;
	unsigned thread;
} 
SendData_t;

typedef struct
{
	int tag;
	int target;
}
SendDataReference_t;

void CommunicationQueues_QueueSend (FileItem_t *fsend, event_t *send_begin,
	event_t *send_end, off_t send_position, unsigned thread)
{
	SendData_t tmp;

	tmp.send_begin = send_begin;
	tmp.send_end = send_end;
	tmp.send_position = send_position;
	tmp.thread = thread;

	NewQueue_add (fsend->send_queue, &tmp);
}

static int CompareSend_cbk (void *reference, void *data)
{
	SendData_t *d = (SendData_t*) data;
	SendDataReference_t *ref = (SendDataReference_t*) reference;

	return ref->tag == Get_EvTag(d->send_begin) && 
	  ref->target == Get_EvTarget(d->send_begin);
}

void CommunicationQueues_ExtractSend (FileItem_t *fsend, int receiver,
	int tag, event_t **send_begin, event_t **send_end,
	off_t *send_position, unsigned *thread)
{
	SendData_t *res;
	SendDataReference_t reference;
	reference.tag = tag;
	reference.target = receiver;

	res = (SendData_t*) NewQueue_search (fsend->send_queue, &reference, CompareSend_cbk);

	if (NULL != res)
	{
		*send_begin = res->send_begin;
		*send_end = res->send_end;
		*send_position = res->send_position;
		*thread = res->thread;
		NewQueue_delete (fsend->send_queue, res);
	}
	else
	{
		*send_begin = NULL;
		*send_end = NULL;
		*send_position = 0;
	}
}

/**************************************************************************
*** RECEIVE PART
**************************************************************************/

typedef struct
{
	event_t *recv_begin;
	event_t *recv_end;
	unsigned thread;
} 
RecvData_t;

typedef struct
{
	int tag;
	int target;
}
RecvDataReference_t;

void CommunicationQueues_QueueRecv (FileItem_t *freceive, event_t *recv_begin,
	event_t *recv_end, unsigned thread)
{
	SendData_t tmp;

	tmp.send_begin = recv_begin;
	tmp.send_end = recv_end;
	tmp.thread = thread;

	NewQueue_add (freceive->recv_queue, &tmp);
}

static int CompareRecv_cbk (void *reference, void *data)
{
	RecvData_t *d = (RecvData_t*) data;
	RecvDataReference_t *ref = (RecvDataReference_t*) reference;

	return ref->tag == Get_EvTag(d->recv_end) && 
	  ref->target == Get_EvTarget(d->recv_end);
}

void CommunicationQueues_ExtractRecv (FileItem_t *freceive, int sender,
	int tag, event_t **recv_begin, event_t **recv_end, unsigned *thread)
{
	RecvData_t *res;
	RecvDataReference_t reference;
	reference.tag = tag;
	reference.target = sender;

	res = (RecvData_t*) NewQueue_search (freceive->recv_queue, &reference, CompareRecv_cbk);

	if (NULL != res)
	{
		*recv_begin = res->recv_begin;
		*recv_end = res->recv_end;
		*thread = res->thread;
		NewQueue_delete (freceive->recv_queue, res);
	}
	else
	{
		*recv_begin = NULL;
		*recv_end = NULL;
	}
}

/**********************************************************************
*** INITIALIZATION 
**********************************************************************/

void CommunicationQueues_Init (NewQueue_t **fsend, NewQueue_t **freceive)
{
	/* Initialize queues. Allocate 1024 entries by default. */
	*fsend = NewQueue_create (sizeof(SendData_t), 1024);
	*freceive = NewQueue_create (sizeof(RecvData_t), 1024);
}
