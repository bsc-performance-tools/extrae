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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/recv_queue.h,v $
 | 
 | @last_commit: $Date: 2009/05/25 10:31:02 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _RECV_QUEUE_H
#define _RECV_QUEUE_H

#include "queue.h"
#include "record.h"

typedef struct RecvQ_type
{
#define RECV_BEGIN_RECORD 0
#define RECV_END_RECORD 1
  event_t *Recv[2];             /* Receive record file pointers :
                                 * Recv[0] points to RECV_EV  - EVT_BEGIN
                                 * Recv[1] points to RECV_EV  - EVT_END
                                 */
  struct RecvQ_type *next;
  struct RecvQ_type *prev;
}
RecvQ_t;

#define GetRecv_RecordQ(item,type)      ( (item)->Recv[(type)] )
#define SetRecv_RecordQ(item,nou,type)  ( (item)->Recv[(type)] = (nou) )

#define RECVQ_SIZE  sizeof(RecvQ_t)

void Init_RecvQ (RecvQ_t * RecvQ);

void Queue_RecvQ (RecvQ_t * queue, RecvQ_t * new);

void Remove_RecvQ (RecvQ_t * new);


RecvQ_t *Alloc_RecvQ_Item ();

RecvQ_t *RecvQueueSearch (void *freceive, RecvQ_t * queue, unsigned int tag,
	unsigned int sender);

#endif
