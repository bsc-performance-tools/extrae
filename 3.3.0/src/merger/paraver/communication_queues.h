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

#ifndef _COMMUNICATION_QUEUES_H_
#define _COMMUNICATION_QUEUES_H_

#include <config.h>
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "new-queue.h"
#include "record.h"

void CommunicationQueues_Init (NewQueue_t **fsend, NewQueue_t **freceive);

void CommunicationQueues_Clear (NewQueue_t *queue);

void CommunicationQueues_QueueSend (NewQueue_t *qsend, event_t *send_begin,
	event_t *send_end, off_t send_position, unsigned thread,
	unsigned vthread, unsigned partner, unsigned tag, long long key);
void CommunicationQueues_QueueRecv (NewQueue_t *qreceive, event_t *recv_begin,
	event_t *recv_end, unsigned thread, unsigned vthread,
	unsigned partner, unsigned tag, long long key);

void CommunicationQueues_ExtractRecv (NewQueue_t *qreceive, int sender,
	int tag, event_t **recv_begin, event_t **recv_end, unsigned *thread,
	unsigned *vthread, long long key);
void CommunicationQueues_ExtractSend (NewQueue_t *qsend, int receiver,
	int tag, event_t **send_begin, event_t **send_end,
	off_t *send_position, unsigned *thread, unsigned *vthread, long long key);

#endif
