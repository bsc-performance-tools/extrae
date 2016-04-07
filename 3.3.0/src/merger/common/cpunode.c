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

#include "options.h"

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

int SortByObject (const void *t1, const void *t2)
{
	struct input_t *trace1 = (struct input_t*) t1;
	struct input_t *trace2 = (struct input_t*) t2;

	if (trace1->ptask > trace2->ptask)
	{
		return 1;
	}
	else if (trace1->ptask == trace2->ptask)
	{
		if (trace1->task > trace2->task)
		{
			return 1;
		}
		else if (trace1->task == trace2->task)
		{
			if (trace1->thread > trace2->thread)
				return 1;
			else if (trace1->thread == trace2->thread)
				return 0;
			else
				return -1;
		}
		else
			return -1;
	}
	else
		return -1;
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

struct Pair_NodeCPU *AssignCPUNode (unsigned nfiles, struct input_t *files)
{
	struct Pair_NodeCPU *result;
	unsigned **nodefiles = NULL;
	unsigned *nodecount = NULL;
	char **nodenames = NULL;
	unsigned numnodes = 0;
	unsigned i, j, found, found_pos, total_cpus;

	for (i = 0; i < nfiles; i++)
	{
		/* Has the node already appeared? */
		for (found_pos = 0, found = FALSE, j = 0; j < numnodes && !found; j++)
		{
			found = strcmp (nodenames[j], files[i].node) == 0;
			if (found)
				found_pos = j;
		}

#if defined(DEBUG)
		fprintf (stdout, "Checking for node %s - found? %d - position? %d\n", files[i].node, found, found_pos);
#endif

		/* If didn't appear, allocate it */
		if (!found)
		{
			nodenames = (char**) realloc (nodenames, (numnodes+1)*sizeof(char*));
			if (nodenames == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold nodenames information\n");
				exit (0);
			}
			nodenames[numnodes] = files[i].node;
			nodecount = (unsigned*) realloc (nodecount, (numnodes+1)*sizeof(char*));
			if (nodecount == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold nodecount information\n");
				exit (0);
			}
			nodecount[numnodes] = 1;
			nodefiles = (unsigned **) realloc (nodefiles, (numnodes+1)*sizeof(unsigned*));
			if (nodefiles == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold nodefiles information\n");
				exit (0);
			}
			nodefiles[numnodes] = (unsigned*) malloc (nodecount[numnodes]*sizeof(unsigned));
			if (nodefiles[numnodes] == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold nodefiles[%d] information (1)\n", numnodes);
				exit (0);
			}
			nodefiles[numnodes][nodecount[numnodes]-1] = i;
			numnodes++;

#if defined(DEBUG)
		fprintf (stdout, "Node %s (in position %d) -> occurrences = %d\n", files[i].node, numnodes-1, nodecount[numnodes-1]);
#endif
		}
		else
		{
			/* Found node, stored in position found_pos: increase the node count usage,
			   and allocate the referred file */
			nodecount[found_pos]++;
			nodefiles[found_pos] = (unsigned*) realloc (nodefiles[found_pos], nodecount[found_pos]*sizeof(unsigned));
			if (nodefiles[found_pos] == NULL)
			{
				fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold nodefiles[%d] information (2)\n", numnodes);
				exit (0);
			}
			nodefiles[found_pos][nodecount[found_pos]-1] = i;

#if defined(DEBUG)
		fprintf (stdout, "Node %s (in position %d) -> occurrences = %d\n", files[i].node, found_pos, nodecount[found_pos]);
#endif
		}

	}

	/* Allocate output information */
	result = (struct Pair_NodeCPU*) malloc ((numnodes+1) * sizeof(struct Pair_NodeCPU));
	if (result == NULL)
	{
		fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold Node-CPU information\n");
		exit (0);
	}

	/* Prepare the resulting output and modify file->cpu and file->nodeid */
	for (total_cpus = 0, i = 0; i < numnodes; i++)
	{
		result[i].CPUs = nodecount[i];
#if defined(DEBUG)
		fprintf (stdout, "NodeInfo::result[%d].CPUs = %d\n", i, result[i].CPUs);
#endif
		result[i].files = (struct input_t **) malloc (result[i].CPUs*sizeof(struct input_t*));
		if (result[i].files == NULL)
		{
			fprintf (stderr, "mpi2prv: Error cannot allocate memory to hold cpu node information\n");
			exit (0);
		}
		
		for (j = 0; j < nodecount[i]; j++)
		{
			/* Fill CPU and NODEID within the file_t structure */
			files[nodefiles[i][j]].cpu = ++total_cpus;
			files[nodefiles[i][j]].nodeid = i+1; /* Number of node starts at 1 */

			/* Fill result */
			result[i].files[j] = &files[nodefiles[i][j]];
		}
	}

	/* Last entry should be 0,NULL */
	result[numnodes].CPUs = 0;
	result[numnodes].files = NULL;

	/* Free memory */
	if (numnodes > 0)
	{
		for (i = 0; i < numnodes; i++)
			free (nodefiles[i]);
		free (nodefiles);
		free (nodenames);
		free (nodecount);
	}

	return result;
}

/***
  GenerateROWfile
  Creates a .ROW file containing in which nodes were running (if some input has NODE info).
***/
int GenerateROWfile (char *name, struct Pair_NodeCPU *info, int nfiles, struct input_t *files)
{
	int i, j, k;
	int numNodes;
	int numCPUs;
	char FORMAT[128];
	FILE *fd;

	/* Compute how many CPUs and NODEs */
	numNodes = numCPUs = 0;
	while (info[numNodes].CPUs > 0)
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

	if (!get_option_merge_NanosTaskView())
	{
		qsort (files, nfiles, sizeof(input_t), SortByObject);
		fprintf (fd, "\nLEVEL THREAD SIZE %d\n", numCPUs);
		for (i = 0; i < nfiles; i++)
			fprintf (fd, "%s\n", files[i].threadname);
		qsort (files, nfiles, sizeof(input_t), SortByOrder);
	}
	else
	{
		/* What naming scheme should we follow in Nanos Task View?
		   While undecided, keep this clear. Paraver will handle it. */
	}

	fclose (fd);

	return 0;
}
