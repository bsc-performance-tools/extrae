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

#ifndef _SEND_QUEUE_H
#define _SEND_QUEUE_H

#include <config.h>

#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "queue.h"
#include "record.h"

typedef struct SendQ_type
{
  struct SendQ_type *next;
  struct SendQ_type *prev;

#define SEND_BEGIN_RECORD 0
#define SEND_END_RECORD 1
  event_t *Send[2];             /* Receive record file pointers :
                                 * Recv[0] points to SEND_EV  - EVT_BEGIN
                                 * Recv[1] points to SEND_EV  - EVT_END
                                 */
	off_t position;
}
SendQ_t;

#define GetSend_RecordQ(item,type)      ( (item)->Send[(type)] )
#define GetSend_PositionQ(item)         ( (item)->position )
#define SetSend_RecordQ(item,nou,type)  ( (item)->Send[(type)] = (nou) )
#define SetSend_PositionQ(item,npos)    ( (item)->position = (npos) )

#define SENDQ_SIZE  sizeof(RecvQ_t)

void Init_SendQ (SendQ_t * SendQ);

void Queue_SendQ (SendQ_t * queue, SendQ_t * new);

void Remove_SendQ (SendQ_t * new);

SendQ_t *Alloc_SendQ_Item (void);

SendQ_t *SendQueueSearch (void *freceive, SendQ_t * queue, unsigned int tag,
	unsigned int sender);

#endif
