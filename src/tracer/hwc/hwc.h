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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/hwc/hwc.h,v $
 | 
 | @last_commit: $Date: 2009/10/29 10:10:19 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __HWC_H__
#define __HWC_H__

/*------------------------------------------------ Global Variables ---------*/

extern int HWCEnabled;
#if defined(SAMPLING_SUPPORT)
extern int SamplingSupport;
extern int EnabledSampling;
#endif

/*------------------------------------------------ Structures ---------------*/

enum ChangeType_t {
  CHANGE_NEVER=0,
  CHANGE_GLOPS,
  CHANGE_TIME
};

/*------------------------------------------------ Prototypes ---------------*/

#ifdef __cplusplus
extern "C" {
#endif
void HWC_Initialize (int options);
void HWC_Start_Counters (int num_threads);
void HWC_Restart_Counters (int old_num_threads, int new_num_threads);

void HWC_Start_Next_Set (UINT64 time, int thread_id);
void HWC_Start_Previous_Set (UINT64 time, int thread_id);
int HWC_Check_Pending_Set_Change (UINT64 time, enum ChangeType_t type, int thread_id);
int HWC_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                 char *change_at_globalops, char *change_at_time, int num_overflows, 
                 char **overflow_counters, unsigned long long *overflow_values);

int HWC_Get_Current_Set ();
int HWC_Get_Num_Sets ();
int HWC_Get_Set_Counters_Ids (int set_id, int **io_HWCIds);
int HWC_Get_Set_Counters_ParaverIds (int set_id, int **io_HWCParaverIds);
int HWC_Read (unsigned int tid, UINT64 time, long long *store_buffer);
int HWC_Reset (unsigned int tid);
int HWC_Resetting ();
int HWC_Accum (unsigned int tid, UINT64 time);
int HWC_Accum_Reset (unsigned int tid);
int HWC_Accum_Valid_Values (unsigned int tid);
int HWC_Accum_Copy_Here (unsigned int tid, long long *store_buffer);
int HWC_Accum_Add_Here (unsigned int tid, long long *store_buffer);

void HWC_Parse_XML_Config (int task_id, int num_tasks, char *distribution);
void HWC_Parse_Env_Config (int task_id);

void HWC_Set_ChangeAtTime_Frequency (int set, unsigned long long ns);

#ifdef __cplusplus
}
#endif

#endif /* __HWC_H__ */

