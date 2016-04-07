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

#include "address_space.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "object_tree.h"

struct AddressSpaceRegion_st
{
	uint64_t AddressBegin;
	uint64_t AddressEnd;
	uint64_t CallerAddresses[MAX_CALLERS];
	uint32_t CallerType;
	int in_use;
};

struct AddressSpace_st
{
	struct AddressSpaceRegion_st *Regions;
	uint32_t nRegions;  /* number of regions */
	uint32_t aRegions;  /* number of allocated regions */
};

#define ADDRESS_SPACE_ALLOC_SIZE 256

struct AddressSpace_st* AddressSpace_create (void)
{
	struct AddressSpace_st * as = (struct AddressSpace_st*) malloc (
	  sizeof(struct AddressSpace_st));
	if (NULL == as)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate memory to allocate address space!\n");
		exit (-1);
	}
	as->Regions = NULL;
	as->nRegions = as->aRegions = 0;
	return as;
}

void AddressSpace_add (struct AddressSpace_st *as, uint64_t AddressBegin,
	uint64_t AddressEnd, uint64_t *CallerAddresses,
	uint32_t CallerType)
{
	unsigned u;
	unsigned v;

	if (as->nRegions == as->aRegions)
	{
		as->Regions = (struct AddressSpaceRegion_st *) realloc (as->Regions,
		  (as->nRegions+ADDRESS_SPACE_ALLOC_SIZE)*sizeof(struct AddressSpaceRegion_st));
		if (NULL == as->Regions)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate memory to allocate address space!\n");
			exit (-1);
		}

		for (u = as->aRegions; u < as->aRegions+ADDRESS_SPACE_ALLOC_SIZE; u++)
			as->Regions[u].in_use = FALSE;
		as->aRegions += ADDRESS_SPACE_ALLOC_SIZE;
	}

	for (u = 0; u < as->aRegions; u++)
		if (!as->Regions[u].in_use)
		{
			as->Regions[u].AddressBegin = AddressBegin;
			as->Regions[u].AddressEnd = AddressEnd;
			as->Regions[u].CallerType = CallerType;
			for (v = 0; v < MAX_CALLERS; v++)
				as->Regions[u].CallerAddresses[v] = CallerAddresses[v];
			as->Regions[u].in_use = TRUE;
			as->nRegions++;
			break;
		}
}

void AddressSpace_remove (struct AddressSpace_st *as, uint64_t AddressBegin)
{
	unsigned u;
	unsigned v;

	for (u = 0; u < as->aRegions; u++)
		if (as->Regions[u].in_use && as->Regions[u].AddressBegin == AddressBegin)
		{
			as->Regions[u].in_use = FALSE;
			as->Regions[u].AddressBegin = 0;
			as->Regions[u].AddressEnd = 0;
			as->Regions[u].CallerType = 0;
			for (v = 0; v < MAX_CALLERS; v++)
				as->Regions[u].CallerAddresses[v] = 0;
			as->nRegions--;
			break;
		}
}

int AddressSpace_search (struct AddressSpace_st *as, uint64_t Address,
	uint64_t **CallerAddresses, uint32_t *CallerType)
{
	unsigned u;

	for (u = 0; u < as->aRegions; u++)
		if (as->Regions[u].in_use)
			if (as->Regions[u].AddressBegin <= Address &&
			    Address <= as->Regions[u].AddressEnd)
		{
			if (CallerAddresses)
				*CallerAddresses = as->Regions[u].CallerAddresses;
			if (CallerType)
				*CallerType = as->Regions[u].CallerType;
			return TRUE;
		}
	return FALSE;
}
