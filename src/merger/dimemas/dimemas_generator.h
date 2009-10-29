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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef DIMEMAS_GENERATOR_H_DEFINED
#define DIMEMAS_GENERATOR_H_DEFINED

#include "file_set.h"
#include "trace_to_prv.h"

int DimemasHeader_WriteHeader( FILE *fd );
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

int Dimemas_WriteHeader (FILE *trf_fd, struct Pair_NodeCPU *info,
	char *outName);
int Dimemas_WriteOffsets (FILE *trf_fd, char *outName,
	unsigned long long offset_position, unsigned int numfiles,
	unsigned long long *offsets);

#endif /* DIMEMAS_GENERATOR_H_DEFINED */
