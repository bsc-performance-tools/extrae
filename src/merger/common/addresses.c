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

#include <common.h>

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#if defined(PARALLEL_MERGE)
# include <mpi.h>
#endif

#include "addresses.h"

#define AC_ALLOC_CHUNK 256

void AddressCollector_Initialize (struct address_collector_t *ac)
{
	ac->allocated = 0;
	ac->count = 0;
	ac->addresses = NULL;
	ac->types = NULL;
	ac->tasks = NULL;
	ac->ptasks = NULL;
}

static int AddresCollector_Search (struct address_collector_t *ac,
	UINT64 address, int type)
{
	unsigned i;
	for (i = 0; i < ac->count; i++)
		if (ac->addresses[i] == address && ac->types[i] == type)
			return TRUE;

	return FALSE;
}

void AddressCollector_Add (struct address_collector_t *ac, unsigned ptask,
	unsigned task, UINT64 address, int type)
{
	if (!AddresCollector_Search (ac, address, type))
	{
		if (ac->allocated == ac->count)
		{
			ac->addresses = (UINT64*) realloc (ac->addresses, (ac->count + AC_ALLOC_CHUNK)*sizeof(UINT64));
			if (ac->addresses == NULL)
			{
				fprintf (stderr, "mpi2prv: Error when reallocating address_collector_t in AdressCollector_Add\n");
				exit (-1);
			}
			ac->types = (int*) realloc (ac->types, (ac->count + AC_ALLOC_CHUNK)*sizeof(int));
			if (ac->types == NULL)
			{
				fprintf (stderr, "mpi2prv: Error when reallocating address_collector_t in AdressCollector_Add\n");
				exit (-1);
			}
			ac->ptasks = (unsigned*) realloc (ac->ptasks, (ac->count + AC_ALLOC_CHUNK)*sizeof(unsigned));
			if (ac->ptasks == NULL)
			{
				fprintf (stderr, "mpi2prv: Error when reallocating address_collector_t in AdressCollector_Add\n");
				exit (-1);
			}
			ac->tasks = (unsigned*) realloc (ac->tasks, (ac->count + AC_ALLOC_CHUNK)*sizeof(unsigned));
			if (ac->tasks == NULL)
			{
				fprintf (stderr, "mpi2prv: Error when reallocating address_collector_t in AdressCollector_Add\n");
				exit (-1);
			}
			ac->allocated += AC_ALLOC_CHUNK;
		}
		ac->ptasks[ac->count] = ptask;
		ac->tasks[ac->count] = task;
		ac->addresses[ac->count] = address;
		ac->types[ac->count] = type;
		ac->count++;
	}
}

unsigned AddressCollector_Count (struct address_collector_t *ac)
{
	return ac->count;
}

UINT64* AddressCollector_GetAllAddresses (struct address_collector_t *ac)
{
	return ac->addresses;
}

int* AddressCollector_GetAllTypes (struct address_collector_t *ac)
{
	return ac->types;
}

unsigned* AddressCollector_GetAllPtasks (struct address_collector_t *ac)
{
	return ac->ptasks;
}

unsigned* AddressCollector_GetAllTasks (struct address_collector_t *ac)
{
	return ac->tasks;
}
#if defined(PARALLEL_MERGE)

#include "mpi-tags.h"
#include "mpi-aux.h"

void AddressCollector_GatherAddresses (int numtasks, int taskid,
	struct address_collector_t *ac)
{
	MPI_Status s;
	int res;
	if (numtasks > 1)
	{
		if (taskid == 0)
		{
			char tmp;
			int task;
			unsigned num_addresses;

			fprintf (stdout, "mpi2prv: Gathering addresses across processors... ");
			fflush (stdout);

			for (task = 1; task < numtasks; task++)
			{
				res = MPI_Send (&tmp, 1, MPI_CHAR, task, ADDRESSCOLLECTOR_ASK_TAG,
					MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed ask for collected addresses");
				res = MPI_Recv (&num_addresses, 1, MPI_UNSIGNED, task,
					ADDRESSCOLLECTOR_NUM_TAG, MPI_COMM_WORLD, &s);
				MPI_CHECK(res, MPI_Recv, "Failed receiving number of collected addresses");
				if (num_addresses > 0)
				{
					unsigned i;
					UINT64 buffer_addresses[num_addresses];
					int buffer_types[num_addresses];
					unsigned buffer_ptasks[num_addresses];
					unsigned buffer_tasks[num_addresses];

					res = MPI_Recv (&buffer_addresses, num_addresses, MPI_LONG_LONG, task,
						ADDRESSCOLLECTOR_ADDRESSES_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");
					res = MPI_Recv (&buffer_types, num_addresses, MPI_INT, task,
						ADDRESSCOLLECTOR_TYPES_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");
					res = MPI_Recv (&buffer_ptasks, num_addresses, MPI_INT, task,
						ADDRESSCOLLECTOR_PTASKS_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");
					res = MPI_Recv (&buffer_tasks, num_addresses, MPI_INT, task,
						ADDRESSCOLLECTOR_TASKS_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");

					for (i = 0; i < num_addresses; i++)
						AddressCollector_Add (ac, buffer_ptasks[i], buffer_tasks[i],
						  buffer_addresses[i], buffer_types[i]);
				}
			}

			fprintf (stdout, "done\n");
			fflush (stdout);
		}
		else
		{
			unsigned num_addresses = AddressCollector_Count(ac);
			char tmp;

			res = MPI_Recv (&tmp, 1, MPI_CHAR, 0, ADDRESSCOLLECTOR_ASK_TAG,
				MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Failed waiting for master to ask for collected addresses");
			res = MPI_Send (&num_addresses, 1, MPI_UNSIGNED, 0, ADDRESSCOLLECTOR_NUM_TAG,
				MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Send, "Failed sending number of collected addresses");
			if (num_addresses > 0)
			{
				UINT64 *buffer_addresses = AddressCollector_GetAllAddresses (ac);
				int *buffer_events = AddressCollector_GetAllTypes (ac);
				unsigned *buffer_ptasks = AddressCollector_GetAllPtasks (ac);
				unsigned *buffer_tasks = AddressCollector_GetAllTasks (ac);

				res = MPI_Send (buffer_addresses, num_addresses, MPI_LONG_LONG, 0,
					ADDRESSCOLLECTOR_ADDRESSES_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
				res = MPI_Send (buffer_events, num_addresses, MPI_INT, 0,
					ADDRESSCOLLECTOR_TYPES_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
				res = MPI_Send (buffer_ptasks, num_addresses, MPI_UNSIGNED, 0,
					ADDRESSCOLLECTOR_PTASKS_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
				res = MPI_Send (buffer_tasks, num_addresses, MPI_UNSIGNED, 0,
					ADDRESSCOLLECTOR_TASKS_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
			}
		}
	}
}
#endif
