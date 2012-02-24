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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "cpunode.h"

#define FLOG(x) ((x)<10?1:(x)<100?2:(x)<1000?3:(x)<10000?4:(x)<100000?5:(x)<1000000?6:(x)<10000000?7:8)

int ComparaTraces (struct input_t *t1, struct input_t *t2)
{
	if (t1->ptask < t2->ptask)
		return -1;
	else if (t1->ptask > t2->ptask)
		return 1;
	else
	{
		if (t1->task < t2->task)
			return -1;
		else if (t1->task > t2->task)
			return 1;
		else
		{
			if (t1->thread < t2->thread)
				return -1;
			else if (t1->thread > t2->thread)
				return 1;
			else
				return 0;
		}
	}
}

int SortByHost (const void *t1, const void *t2)
{
	struct input_t *trace1 = (struct input_t*) t1;
	struct input_t *trace2 = (struct input_t*) t2;

	if (trace1->node != NULL && trace2->node != NULL)
	{
		int resultat = strcmp (trace1->node, trace2->node);

		if (resultat == 0)
			return ComparaTraces (trace1, trace2);
		else
			return resultat;
	}
	/* This cannot happen! */
	else if (trace1->node == NULL && trace2->node != NULL)
		return -1;
	else if (trace1->node != NULL && trace2->node == NULL)
		return 1;
	else 
		return ComparaTraces (trace1, trace2);
}

int SortByOrder (const void *t1, const void *t2)
{
	struct input_t *trace1 = (struct input_t*) t1;
	struct input_t *trace2 = (struct input_t*) t2;

	if (trace1->order < trace2->order)
		return -1;
	else if (trace1->order > trace2->order)
		return 1;
	else
		return 0;
}

int SortBySize (const void *t1, const void *t2)
{
	struct input_t *trace1 = (struct input_t*) t1;
	struct input_t *trace2 = (struct input_t*) t2;

	if (trace1->filesize < trace2->filesize)
		return -1;
	else if (trace1->filesize > trace2->filesize)
		return 1;
	else
		return 0;
}

/***
  AssignCPUNode
***/

struct Pair_NodeCPU *AssignCPUNode(int nfiles, struct input_t *files)
{
	int i;
	int NodeCount;
	int NodeID;
	struct Pair_NodeCPU *result;
	char *previousNode = "";

	/* Sort MPIT files per host basis */
	qsort (files, nfiles, sizeof(input_t), SortByHost);

	NodeCount = 0;

	/* Assign CPU and a NodeID to each MPIT file */
	for (i = 0; i < nfiles; i++)
	{
		files[i].cpu = i+1;

		if (strcmp (previousNode, files[i].node) != 0)
		{
			previousNode = files[i].node;
			files[i].nodeid = ++NodeCount;
		}
		else
			files[i].nodeid = NodeCount;
	}

	/* Allocate output information */
	result = (struct Pair_NodeCPU*) malloc ((NodeCount+1) * sizeof(struct Pair_NodeCPU));
	if (result == NULL)
	{
		fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold Node-CPU information\n");
		exit (0);
	}

	previousNode = "";
	NodeID = 0;

	/* Copy file information into Node information */
	for (i = 0; i < nfiles; i++)
	{	
		if (strcmp (previousNode, files[i].node) != 0)
		{
			result[NodeID].CPUs = 1;
			result[NodeID].files = (struct input_t**) malloc (1 * sizeof(struct input_t*));
			if (result[NodeID].files == NULL)
			{
				fprintf (stderr, "mpi2prv: Error! Cannot allocate memory to hold Node-CPU information\n");
				exit (0);
			}
			result[NodeID].files[0] = (struct input_t*) malloc(sizeof(struct input_t));
			memcpy(result[NodeID].files[0], &files[i], sizeof(struct input_t));

			previousNode = files[i].node;
			NodeID++;
		}
		else
		{
			int prevNode = NodeID - 1;

			result[prevNode].CPUs++;
			result[prevNode].files = (struct input_t**) realloc (
			  result[prevNode].files, result[prevNode].CPUs * sizeof(struct input_t*));
			if (result[prevNode].files == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot re-allocate memory to hold Node-CPU information\n");
				exit (0);
			}
			result[prevNode].files[result[prevNode].CPUs-1] = (struct input_t*) malloc(sizeof(struct input_t));
			memcpy(result[prevNode].files[result[prevNode].CPUs-1], &files[i], sizeof(struct input_t));

		}
	}

	/* The last node will have 0 CPUs and will be named 'null' */
	result[NodeCount].CPUs = 0;
	result[NodeCount].files = NULL;

	/* ReSort MPIT files per "original" basis" */
	qsort (files, nfiles, sizeof(input_t), SortByOrder);

	return result;
}

/***
  GenerateROWfile
  Creates a .ROW file containing in which nodes were running (if some input has NODE info).
***/
int GenerateROWfile (char *name, struct Pair_NodeCPU *info)
{
	int i, j, k;
	int numNodes;
	int numCPUs;
	char FORMAT[128];
	FILE *fd;

	/* Compute how many CPUs and NODEs */
	numNodes = numCPUs = 0;
	while (info[numNodes].files != NULL)
	{
		numCPUs += info[numNodes].CPUs;
		numNodes ++;
	}

	/* This will provide %04d.%s pex */
	sprintf (FORMAT, "%%0%dd.%%s", FLOG(numCPUs));

#if HAVE_FOPEN64
	fd = fopen64 (name, "w");
#else
	fd = fopen (name, "w");
#endif
	fprintf (fd, "LEVEL CPU SIZE %d\n", numCPUs);

	/* K will be our "Global CPU" counter */	
	k = 1;
	for (i = 0; i < numNodes; i++)
	{
		char *node = info[i].files[0]->node;
		for (j = 0; j < info[i].CPUs; j++)
		{
			fprintf (fd, FORMAT, k, node);
			fprintf (fd, "\n");
			k++;
		}
	}

	fprintf (fd, "\nLEVEL NODE SIZE %d\n", numNodes);
	for (i = 0; i < numNodes; i++)
		fprintf (fd, "%s\n", info[i].files[0]->node);

	fprintf (fd, "\nLEVEL THREAD SIZE %d\n", numCPUs);
	for (i = 0; i < numNodes; i++)
		for (j = 0; j < info[i].CPUs; j++) 
			fprintf (fd, "%s\n", info[i].files[j]->threadname);

	fclose (fd);

	return 0;
}
