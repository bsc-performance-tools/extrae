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

#ifndef DIMEMAS_GENERATOR_H_DEFINED
#define DIMEMAS_GENERATOR_H_DEFINED

#include "file_set.h"
#include "trace_to_prv.h"

int Dimemas_NX_Generic_Send( FILE *fd, int task, int thread, int task_r, int commid, int size, UINT64 tag, int synchronism );
int Dimemas_NX_Send( FILE *fd, int task, int thread, int task_r, int commid, int size, UINT64 tag );
int Dimemas_NX_ImmediateSend( FILE *fd, int task, int thread, int task_r, int commid, int size, UINT64 tag );
int Dimemas_NX_BlockingSend( FILE *fd, int task, int thread, int task_r, int commid, int size, UINT64 tag );
int Dimemas_NX_Generic_Recv( FILE *fd, int task, int thread, int task_s, int commid, int size, UINT64 tag, int type );
int Dimemas_NX_Recv( FILE *fd, int task, int thread, int task_s, int commid, int size, UINT64 tag ); 
int Dimemas_NX_Irecv( FILE *fd, int task, int thread, int task_s, int commid, int size, UINT64 tag );
int Dimemas_NX_Wait( FILE *fd, int task, int thread, int task_s, int commid, int size, UINT64 tag );
int Dimemas_Communicator_Definition( FILE *fd, long long commid, int Ntasks, int *TaskList );
int Dimemas_CPU_Burst( FILE *fd, int task, int thread, double burst_time );
int Dimemas_User_Event( FILE *fd, int task, int thread, UINT64 type, UINT64 value );

#if defined(DEAD_CODE)
int Dimemas_Block_Definition( FILE *fd, UINT64 block, char *Label );
int Dimemas_Block_Begin( FILE *fd, int task, int thread, UINT64 block );
int Dimemas_Block_End( FILE *fd, int task, int thread, UINT64 block );
#endif

int Dimemas_Global_OP( FILE *fd, int task, int thread, int opid, int commid, int root_rank, int root_thd, UINT64 sendsize, UINT64 recvsize );

#if defined(DEAD_CODE)
int Dimemas_User_EventType_Definition( FILE *fd, UINT64 type, char *Label, int color );
int Dimemas_User_EventValue_Definition( FILE *fd, UINT64 type, long64_t value, char *Label );
#endif

int Dimemas_WriteHeader (unsigned num_appl, FILE *trf_fd,
	struct Pair_NodeCPU *info, char *outName);
int Dimemas_WriteOffsets (unsigned num_appl, FILE *trf_fd, char *outName,
	unsigned long long offset_position, unsigned int numfiles,
	unsigned long long *offsets);

#endif /* DIMEMAS_GENERATOR_H_DEFINED */
