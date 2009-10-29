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

#include "queue.h"
#include "file_set.h"
#include "send_queue.h"

SendQ_t *free_SendQ = NULL;

/******************************************************************************
 ***  QueueSearch : Searches in a Receive record queue for the receive item
 ***                with tag "tag" and sender "sender".
 ******************************************************************************/

SendQ_t *SendQueueSearch (void *fsend, SendQ_t * queue, unsigned int tag,
	unsigned int receiver)
{
  event_t *current;
  SendQ_t *ptmp;

  if (fsend == NULL)
  {
    fprintf (stderr, "fsend null!!!\n");
    return NULL;
  }

  for (ptmp = (queue)->next; ptmp != (queue); ptmp = ptmp->next)
  {
    current = GetSend_RecordQ (ptmp, SEND_END_RECORD);
    if ((Get_EvTarget (current) == receiver) && (Get_EvTag (current) == tag))
      break;
  }
  return (ptmp == (queue)) ? NULL : ptmp;
}

/******************************************************************************
 ***  Init_SendQ
 ******************************************************************************/

void Init_SendQ (SendQ_t * SendQ)
{
  INIT_QUEUE (SendQ);
}

/******************************************************************************
 ***  Alloc_SendQ_Item
 ******************************************************************************/

SendQ_t *Alloc_SendQ_Item (void)
{
  SendQ_t *nou = NULL;
  ALLOC_NEW_ITEM (free_SendQ, SENDQ_SIZE, nou, "Alloc SendQ Item");
  return nou;
}

/******************************************************************************
 ***  Queue_SendQ
 ******************************************************************************/

void Queue_SendQ (SendQ_t * queue, SendQ_t * new)
{
  ENQUEUE_ITEM (queue, new);
}

/******************************************************************************
 ***  Remove_SendQ
 ******************************************************************************/

void Remove_SendQ (SendQ_t * new)
{
  REMOVE_ITEM (new);
}
