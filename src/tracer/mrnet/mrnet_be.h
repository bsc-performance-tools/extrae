/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/mrnet_be.h,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MRNET_BE_H__
#define __MRNET_BE_H__

#include "trace_buffers.h"
#include "mrnet_buffers.h"
#include "signals.h"

extern Condition_t SyncFlush_Completion;

/* Prototypes */
#if defined(__cplusplus)
extern "C" {
#endif

void Enable_MRNet();
int MRNet_isEnabled();
int Join_MRNet(int rank);
int Connect_Backend(int rank, char *be_topology_file);
void Register_Streams(int rank);
void * Commands_Handler (void * rank_ptr);
void Quit_MRNet(int rank);
int MRNet_Sync_Flush (Buffer_t *buffer);
int MRNet_Notify_Flush (Buffer_t *buffer);
void MRN_CloseFiles ();

#if defined(__cplusplus)
}
#endif

#endif /* __MRNET_BE_H__ */
