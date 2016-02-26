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

#ifndef __HWC_H__
#define __HWC_H__

/*------------------------------------------------ Global Variables ---------*/

extern int HWCEnabled;
#if defined(SAMPLING_SUPPORT)
extern int SamplingSupport;
#endif

/*------------------------------------------------ Structures ---------------*/

enum ChangeType_t {
  CHANGE_NEVER=0,
  CHANGE_GLOPS,
  CHANGE_TIME
};

enum ChangeTo_t {
  CHANGE_SEQUENTIAL=0,
  CHANGE_RANDOM
};

/*------------------------------------------------ Prototypes ---------------*/

#ifdef __cplusplus
extern "C" {
#endif
int HWC_IsEnabled();
void HWC_Initialize (int options);
void HWC_CleanUp (unsigned nthreads);
void HWC_Start_Counters (int num_threads, UINT64 time, int forked);
void HWC_Restart_Counters (int old_num_threads, int new_num_threads);

void HWC_Start_Next_Set (UINT64 glops, UINT64 time, int thread_id);
void HWC_Start_Previous_Set (UINT64 glops, UINT64 time, int thread_id);
int HWC_Check_Pending_Set_Change (UINT64 countglops, UINT64 time, int thread_id);
int HWC_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                 char *change_at_globalops, char *change_at_time, int num_overflows, 
                 char **overflow_counters, unsigned long long *overflow_values);
void HWC_Start_Current_Set (UINT64 countglops, UINT64 time, int thread_id);
void HWC_Stop_Current_Set (UINT64 time, int thread_id);
int HWC_Get_Current_Set (int threadid);
int HWC_Get_Num_Sets ();
int HWC_Get_Set_Counters_Ids (int set_id, int **io_HWCIds);
int HWC_Get_Set_Counters_ParaverIds (int set_id, int **io_HWCParaverIds);
int HWC_Get_Position_In_Set (int set_id, int hwc_id);
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

int HWC_IsCommonToAllSets(int set_id, int hwc_index);
int HWC_GetNumberOfCommonCounters(void);

#ifdef __cplusplus
}
#endif

#endif /* __HWC_H__ */

