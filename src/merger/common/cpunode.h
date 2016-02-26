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

#ifndef CPUNODE_H
#define CPUNODE_H

#include <config.h>

#include "mpi2out.h" /* per input_t */

struct Pair_NodeCPU
{
	struct input_t **files;
	int CPUs;
};

struct Pair_NodeCPU *AssignCPUNode(unsigned nfiles, struct input_t *files);
int GenerateROWfile (char *name, struct Pair_NodeCPU *info, int nfiles, struct input_t *files);

int ComparaTraces (struct input_t *t1, struct input_t *t2);
int SortByHost (const void *t1, const void *t2);
int SortByOrder (const void *t1, const void *t2);
int SortBySize (const void *t1, const void *t2);
int SortByObject (const void *t1, const void *t2);


#endif
