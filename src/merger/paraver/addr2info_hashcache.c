/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/mpitrace/fusion/trunk/src/merger/paraver/addr2info.c $
 | @last_commit: $Date: 2010-05-13 11:54:20 +0200 (dj, 13 mai 2010) $
 | @version:     $Revision: 257 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"
#include <stdio.h>

#define CACHE_SIZE 32768
#define CACHE_MASK 0x7fff

struct Addr2Info_HashCache_Entry
{
	UINT64 address;
	int line_number;
	int function_number;
};

static struct Addr2Info_HashCache_Entry Addr2Info_HashCache[CACHE_SIZE];
static int Addr2Info_HashCache_Hits;
static int Addr2Info_HashCache_Misses;
static int Addr2Info_HashCache_Replacements;

static int Addr2Info_HashCache_HashFunction (UINT64 address)
{
	return address & CACHE_MASK;
}

void Addr2Info_HashCache_Initialize (void)
{
	int i;

	for (i = 0; i < CACHE_SIZE; i++)
		Addr2Info_HashCache[i].address = 0;
	Addr2Info_HashCache_Hits =
	  Addr2Info_HashCache_Misses =
	  Addr2Info_HashCache_Replacements = 0;
}

int Addr2Info_HashCache_Search (UINT64 address, int *line, int *function)
{
	int index;

	index = Addr2Info_HashCache_HashFunction (address);
	if (Addr2Info_HashCache[index].address == address)
	{
		Addr2Info_HashCache_Hits++;
		*line = Addr2Info_HashCache[index].line_number;
		*function = Addr2Info_HashCache[index].function_number;
		return TRUE;
	}
	else
	{
		Addr2Info_HashCache_Misses++;
		return FALSE;
	}
}

int Addr2Info_HashCache_Insert (UINT64 address, int line, int function)
{
	int index;

	index = Addr2Info_HashCache_HashFunction (address);
	if (Addr2Info_HashCache[index].address != address)
	{
		Addr2Info_HashCache_Replacements++;
		Addr2Info_HashCache[index].address = address;
		Addr2Info_HashCache[index].line_number = line;
		Addr2Info_HashCache[index].function_number = function;
	}
}

int Addr2Info_HashCache_ShowStatistics (void)
{
	fprintf (stdout, "mpi2prv: Addr2Info Hash Cache statistics:\n"
	                 "mpi2prv: Number of searches : %d\n"
	                 "mpi2prv: Number of hits : %d\n"
	                 "mpi2prv: Number of misses : %d\n"
	                 "mpi2prv: Number of replacements : %d\n",
	  Addr2Info_HashCache_Hits+Addr2Info_HashCache_Misses,
	  Addr2Info_HashCache_Hits,
	  Addr2Info_HashCache_Misses,
	  Addr2Info_HashCache_Replacements);
}
