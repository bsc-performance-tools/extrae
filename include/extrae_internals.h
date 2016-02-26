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

#ifndef __EXTRAE_INTERNALS_H_INCLUDED__
#define __EXTRAE_INTERNALS_H_INCLUDED__

#include "extrae_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void Extrae_set_threadid_function (unsigned (*threadid_function)(void));
void Extrae_set_numthreads_function (unsigned (*numthreads_function)(void));

void Extrae_set_taskid_function (unsigned (*taskid_function)(void));
void Extrae_set_numtasks_function (unsigned (*numtasks_function)(void));
void Extrae_set_barrier_tasks_function (void (*barriertasks_function)(void));
void Extrae_set_finalize_task_function (void (*finalizetask_function)(void));

void Extrae_set_thread_name (unsigned thread, char *name);
void Extrae_function_from_address (extrae_type_t type, void *address);

#ifdef __cplusplus
}
#endif

#endif /* __EXTRAE_INTERNALS_H_INCLUDED__ */
