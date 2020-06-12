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

#include <stdlib.h>
#include <stdio.h>

#include "common.h"

#include "taskid.h"

/*
   Default routines
   1 taskid in total, task id is always 0 and barrier does nothing
*/

/* Runtime rank environment variables */
/* SLURM_PROCID                       */
/* OMPI_COMM_WORLD_RANK               */
/* MV2_COMM_WORLD_RANK                */
/* PMI_RANK                           */
/* MPI_RANKID                         */
/* MP_CHILD                           */
/* EC_FARM_ID                         */
/* EC_FARM_LOCALENT                   */
/* ALPS_APP_PE                        */
/**************************************/
/* Change to read envvars, save value to a var and change callback to getter */
static unsigned xtr_taskid = 0;
static unsigned xtr_num_tasks = 1;

static unsigned (*get_task_num) (void);
static unsigned (*get_num_tasks) (void);

static unsigned xtr_get_taskid ()
{
	return xtr_taskid;
}

static unsigned xtr_set_taskid ()
{
	unsigned int NUM_ENVVARS = 9;
	char *envvars[] =
	{
		"SLURM_PROCID",         // SLURM
		"EC_FARM_ID",           // PBS
		"EC_FARM_LOCALENT",     // PBS
		"ALPS_APP_PE",          // ALPS
		"OMPI_COMM_WORLD_RANK", // OpenMPI
		"MV2_COMM_WORLD_RANK",  // MVAPICH
		"PMI_RANK",             // MPICH, Intel
		"MPI_RANKID",           // Platform MPI
		"MP_CHILD"              // Spectrum MPI
	};

	unsigned int i = 0;
	char *envread = NULL;
	unsigned int localid = 0;

	while ((i < NUM_ENVVARS) && (xtr_taskid == 0))
	{
		envread = getenv(envvars[i]);

		if ((envread != NULL) && ((localid = (unsigned int)strtoul(envread, NULL, 10)) != 0))
		{
			xtr_taskid = localid;
#if defined(DEBUG)
			fprintf (stdout, PACKAGE_NAME": Task %u got TASKID from %s\n", xtr_taskid, envvars[i]);
#endif
		}

		i++;
	}

	if (xtr_taskid >= xtr_num_tasks) xtr_num_tasks = xtr_taskid + 1;

	get_task_num = xtr_get_taskid;

	return xtr_taskid;
}

static unsigned xtr_get_num_tasks (void)
{
	return xtr_num_tasks;
}

static void Extrae_callback_routine_do_nothing (void)
{ return; }


/* Callback definitions and API */

static unsigned (*get_task_num) (void) = xtr_set_taskid;
static unsigned (*get_num_tasks) (void) = xtr_get_num_tasks;
static void (*barrier_tasks) (void) = Extrae_callback_routine_do_nothing;

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

