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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/timesync.h,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __TIMESYNC_H__
#define __TIMESYNC_H__

#include "types.h"

typedef struct
{
   int init;
   UINT64 init_time;
   UINT64 sync_time;
   int node_id;
} SyncInfo_t;

enum
{
   TS_NODE,
   TS_TASK,
   TS_DEFAULT,
   TS_NOSYNC
};

#ifdef __cplusplus
extern "C" {
#endif
int TimeSync_Initialize (int num_tasks);
int TimeSync_SetInitialTime (int task, UINT64 init_time, UINT64 sync_time, char *node);
int TimeSync_CalculateLatencies (int sync_strategy);
UINT64 TimeSync (int task, UINT64 time);
UINT64 TimeDesync (int task, UINT64 time);
#ifdef __cplusplus
}
#endif

#define TIMESYNC(task, time) TimeSync(task, time)

#define TIMEDESYNC(task, time) TimeDesync(task, time)

#endif /* __TIMESYNC_H__ */
