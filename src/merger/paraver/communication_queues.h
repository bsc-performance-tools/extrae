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

#ifndef _COMMUNICATION_QUEUES_H_
#define _COMMUNICATION_QUEUES_H_

#include <config.h>

#include "file_set.h"

void CommunicationQueues_Init (NewQueue_t **fsend, NewQueue_t **freceive);

void CommunicationQueues_QueueSend (FileItem_t *fsend, event_t *send_begin,
	event_t *send_end, off_t send_position, unsigned thread);
void CommunicationQueues_QueueRecv (FileItem_t *freceive, event_t *recv_begin,
	event_t *recv_end, unsigned thread);

void CommunicationQueues_ExtractRecv (FileItem_t *freceive, int sender,
	int tag, event_t **recv_begin, event_t **recv_end, unsigned *thread);
void CommunicationQueues_ExtractSend (FileItem_t *fsend, int receiver,
	int tag, event_t **send_begin, event_t **send_end,
	off_t *send_position, unsigned *thread);

#endif
