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

#ifndef TRACE_COMMUNICATION_H_INCLUDED
#define TRACE_COMMUNICATION_H_INCLUDED

void trace_communicationAt (unsigned ptask_s, unsigned task_s, unsigned thread_s, unsigned vthread_s,
	unsigned ptask_r, unsigned task_r, unsigned thread_r, unsigned vthread_r, event_t *send_begin,
	event_t *send_end, event_t *recv_begin, event_t *recv_end, 
	int atposition, off_t position);

#if defined(PARALLEL_MERGE)
int trace_pending_communication (unsigned int ptask_s, unsigned int task_s,
	unsigned int thread_s, unsigned vthread_s, event_t * begin_s, event_t * end_s, unsigned int ptask_r, unsigned int task_r);
#endif /* PARALLEL_MERGE */

#endif 
