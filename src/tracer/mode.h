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

#ifndef __MODE_H__
#define __MODE_H__

#include "clock.h"
#include "trace_mode.h"

extern int *MPI_Deepness;
extern int *Current_Trace_Mode;
extern int *Pending_Trace_Mode_Change;

#define CURRENT_TRACE_MODE(tid) Current_Trace_Mode[tid]
#define PENDING_TRACE_MODE_CHANGE(tid) Pending_Trace_Mode_Change[tid]

#define INCREASE_MPI_DEEPNESS(tid) (MPI_Deepness[tid]++)
#define DECREASE_MPI_DEEPNESS(tid) (MPI_Deepness[tid]--)
#define MPI_IS_NOT_STACKED(tid) (MPI_Deepness[tid] == 0)

void TMODE_setInitial (int mode);
int Trace_Mode_Initialize (int num_threads);
int Trace_Mode_reInitialize (int old_num_threads, int new_num_threads);
void Trace_Mode_Change (int tid, iotimer_t time);
void Trace_Mode_CleanUp (void);
void Trace_mode_switch (void);
int Trace_Mode_FirstMode (unsigned thread);

/* Bursts mode specific */

extern unsigned long long BurstsMode_Threshold;
extern int BurstsMode_MPI_Stats;

#define MINIMUM_BURST_DURATION (BurstsMode_Threshold)
#define TRACING_MPI_STATISTICS (BurstsMode_MPI_Stats)

void TMODE_setBurstsThreshold  (unsigned long long threshold);
void TMODE_setBurstsStatistics (int status);

#endif /* __MODE_H__ */
