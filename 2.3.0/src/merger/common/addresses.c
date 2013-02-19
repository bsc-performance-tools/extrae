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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

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

static char UNUSED rcsid[] = "$Id$";

void AddressCollector_Initialize (struct address_collector_t *ac)
{
	ac->allocated = 0;
	ac->count = 0;
	ac->types = NULL;
	ac->addresses = NULL;
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

void AddressCollector_Add (struct address_collector_t *ac, UINT64 address, int type)
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
			ac->allocated += AC_ALLOC_CHUNK;
		}
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

					res = MPI_Recv (&buffer_addresses, num_addresses, MPI_LONG_LONG, task,
						ADDRESSCOLLECTOR_ADDRESSES_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");
					res = MPI_Recv (&buffer_types, num_addresses, MPI_LONG_LONG, task,
						ADDRESSCOLLECTOR_TYPES_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Failed receiving collected addresses");

					for (i = 0; i < num_addresses; i++)
						AddressCollector_Add (ac, buffer_addresses[i], buffer_types[i]);
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

				res = MPI_Send (buffer_addresses, num_addresses, MPI_LONG_LONG, 0,
					ADDRESSCOLLECTOR_ADDRESSES_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
				res = MPI_Send (buffer_events, num_addresses, MPI_INT, 0,
					ADDRESSCOLLECTOR_TYPES_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Failed sending collected addresses");
			}
		}
	}
}
#endif
