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

#include "common.h"

#include "taskid.h"

/*
   Default routines
   1 taskid in total, task id is always 0 and barrier does nothing
*/

static unsigned Extrae_taskid_default_function (void)
{ return 0; }

static unsigned Extrae_num_tasks_default_function (void)
{ return 1; }

static void Extrae_callback_routine_do_nothing (void)
{ return; }

/* Callback definitions and API */

static unsigned (*get_task_num) (void) = Extrae_taskid_default_function;
static unsigned (*get_num_tasks) (void) = Extrae_num_tasks_default_function;
static void (*barrier_tasks) (void) = Extrae_callback_routine_do_nothing;
static void (*finalize_task) (void) = Extrae_callback_routine_do_nothing;

void Extrae_set_taskid_function (unsigned (*taskid_function)(void))
{
	get_task_num = taskid_function;
}

void Extrae_set_numtasks_function (unsigned (*numtasks_function)(void))
{
	get_num_tasks = numtasks_function;
}

void Extrae_set_barrier_tasks_function (void (*barriertasks_function)(void))
{
	barrier_tasks = barriertasks_function;
}

void Extrae_set_finalize_task_function (void (*finalizetask_function)(void))
{
	finalize_task = finalizetask_function;
}

/* Internal routines */

unsigned Extrae_get_task_number (void)
{
	return get_task_num();
}

unsigned Extrae_get_num_tasks (void)
{
	return get_num_tasks();
}

void Extrae_barrier_tasks (void)
{
	barrier_tasks();
}

void Extrae_finalize_task (void)
{
	finalize_task();
}

/******************************************************************************
 *** Store the first taskid 
 ******************************************************************************/
static unsigned Extrae_Initial_TASKID = 0;

unsigned Extrae_get_initial_TASKID (void)
{
        return Extrae_Initial_TASKID;
}

void Extrae_set_initial_TASKID (unsigned u)
{
        Extrae_Initial_TASKID = u;
}

