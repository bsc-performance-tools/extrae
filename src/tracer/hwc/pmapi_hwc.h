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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/hwc/pmapi_hwc.h,v $
 | 
 | @last_commit: $Date: 2009/01/12 16:16:36 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __PMAPI_HWC_H__
#define __PMAPI_HWC_H__

/*------------------------------------------------ Prototypes ---------------*/

void HWCBE_PMAPI_Initialize (int TRCOptions);
int HWCBE_PMAPI_Init_Thread (UINT64 time, int threadid);

int HWCBE_PMAPI_Start_Set (UINT64 time, int numset, int threadid);
int HWCBE_PMAPI_Stop_Set (int numset, int threadid);
int HWCBE_PMAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                       char *change_at_globalops, char *change_at_time, int num_overflows, 
                       char **overflow_counters, unsigned long long *overflow_values);

int HWCBE_PMAPI_Read (unsigned int tid, long long *store_buffer);

#endif /* __PMAPI_HWC_H__ */
