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

#ifndef TRACE_TO_PRV_H
#define TRACE_TO_PRV_H

#include "mpi2out.h"
#ifdef HAVE_ZLIB
# include "zlib.h"
#endif

#include "vector.h"
#include "extrae_vector.h"
#include "addresses.h"
#include "cpunode.h"
#include "fdz.h"

int Paraver_ProcessTraceFiles (unsigned long nfiles,
	struct input_t *files, unsigned int num_appl,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int idtask);

extern int **EnabledTasks;
extern unsigned long long **EnabledTasks_time;
extern struct address_collector_t CollectedAddresses;

extern mpi2prv_vector_t *RegisteredStackValues;
extern Extrae_Vector_t RegisteredCodeLocationTypes;

#endif /* __TRACE_TO_PRV_H__ */
