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

#ifndef __PMAPI_HWC_H__
#define __PMAPI_HWC_H__

/*------------------------------------------------ Prototypes ---------------*/

void HWCBE_PMAPI_Initialize (int TRCOptions);
int HWCBE_PMAPI_Init_Thread (UINT64 time, int threadid, int forked);
void HWCBE_PMAPI_CleanUp (void);

int HWCBE_PMAPI_Start_Set (UINT64 countglops, UINT64 time, int numset, int threadid);
int HWCBE_PMAPI_Stop_Set (UINT64 time, int numset, int threadid);
int HWCBE_PMAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                       char *change_at_globalops, char *change_at_time, int num_overflows, 
                       char **overflow_counters, unsigned long long *overflow_values);

int HWCBE_PMAPI_Read (unsigned int tid, long long *store_buffer);

void HWCBE_PMAPI_CleanUp (unsigned nthreads);

HWC_Definition_t *HWCBE_PMAPI_GetCounterDefinitions(unsigned *count);

#endif /* __PMAPI_HWC_H__ */
