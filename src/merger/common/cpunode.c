/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/common/cpunode.c,v $
 | 
 | @last_commit: $Date: 2009/03/18 16:35:39 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: cpunode.c,v 1.5 2009/03/18 16:35:39 harald Exp $";

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

	for (i = 0; i < nfiles; i++)
	{	
		if (strcmp (previousNode, files[i].node) != 0)
		{
			result[NodeID].NodeName = (char*) malloc ((strlen(files[i].node)+1)*sizeof(char));
			if (result[NodeID].NodeName == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold Node-CPU information\n");
				exit (0);
			}
			strncpy (result[NodeID].NodeName, files[i].node, strlen(files[i].node));
			result[NodeID].NodeName[strlen(files[i].node)] = (char) 0;
			result[NodeID].CPUs = 1;

			NodeID++;
			previousNode = files[i].node;
		}
		else
			result[NodeID-1].CPUs++;
	}

	/* The last node will have 0 CPUs and will be named 'null' */
	result[NodeCount].CPUs = 0;
	result[NodeCount].NodeName = NULL;

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
	while (info[numNodes].NodeName != NULL)
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
		for (j = 0; j < info[i].CPUs; j++)
		{
			fprintf (fd, FORMAT, k, info[i].NodeName);
			fprintf (fd, "\n");
			k++;
		}

	fprintf (fd, "\n\nLEVEL NODE SIZE %d\n", numNodes);
	for (i = 0; i < numNodes; i++)
		fprintf (fd, "%s\n", info[i].NodeName);
	fclose (fd);

	return 0;
}
