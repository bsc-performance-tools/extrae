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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/recv_queue.c,v $
 | 
 | @last_commit: $Date: 2009/05/25 10:31:02 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED  rcsid[] = "$Id: recv_queue.c,v 1.4 2009/05/25 10:31:02 harald Exp $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "queue.h"
#include "file_set.h"
#include "recv_queue.h"

RecvQ_t *free_RecvQ = NULL;

/******************************************************************************
 ***  QueueSearch : Searches in a Receive record queue for the receive item
 ***                with tag "tag" and sender "sender".
 ******************************************************************************/

RecvQ_t *RecvQueueSearch (void *freceive, RecvQ_t * queue, unsigned int tag,
	unsigned int sender)
{
  event_t *current;
  RecvQ_t *ptmp;

  if (freceive == NULL)
  {
    fprintf (stderr, "mpi2prv: Error! QueueSearch receive freceive null!!!\n");
    return NULL;
  }

  for (ptmp = (queue)->next; ptmp != (queue); ptmp = ptmp->next)
  {
    current = GetRecv_RecordQ (ptmp, RECV_END_RECORD);
    if ((Get_EvTarget (current) == sender) && (Get_EvTag (current) == tag))
      break;
  }
  return (ptmp == (queue)) ? NULL : ptmp;
}

/******************************************************************************
 ***  Init_RecvQ
 ******************************************************************************/

void Init_RecvQ (RecvQ_t * RecvQ)
{

  INIT_QUEUE (RecvQ);
}

/******************************************************************************
 ***  Alloc_RecvQ_Item
 ******************************************************************************/

RecvQ_t *Alloc_RecvQ_Item ()
{
  RecvQ_t *nou = NULL;

/* return((RecvQ_t *)malloc(sizeof(RecvQ_t)));*/
  ALLOC_NEW_ITEM (free_RecvQ, RECVQ_SIZE, nou, "Alloc RecvQ Item");

  return nou;
}

/******************************************************************************
 ***  Queue_RecvQ
 ******************************************************************************/

void Queue_RecvQ (RecvQ_t * queue, RecvQ_t * new)
{
  ENQUEUE_ITEM (queue, new);
}

/******************************************************************************
 ***  Remove_RecvQ
 ******************************************************************************/

void Remove_RecvQ (RecvQ_t * new)
{
  REMOVE_ITEM (new);
}
