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

#ifndef __TASKID_H__
#define __TASKID_H__

#if defined(__cplusplus)
extern "C" {
#endif

void Extrae_set_taskid_function (unsigned (*taskid_function)(void));
void Extrae_set_numtasks_function (unsigned (*numtasks_function)(void));
void Extrae_set_barrier_tasks_function (void (*barriertasks_function)(void));
void Extrae_set_finalize_task_function (void (*finalizetask_function)(void));

/* Internal routines */

unsigned Extrae_get_task_number (void);
unsigned Extrae_get_num_tasks (void);
void Extrae_barrier_tasks (void);
void Extrae_finalize_task (void);

unsigned Extrae_get_initial_TASKID (void);
void Extrae_set_initial_TASKID (unsigned u);

#if defined(__cplusplus)
}
#endif

#define TASKID Extrae_get_task_number()

#endif /* __TASKID_H__ */

