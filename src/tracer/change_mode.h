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
#include "trace_buffers.h"

#define CURRENT_TRACE_MODE(tid) (get_thread_trace_data(tid)->current_trace_mode)
#define PENDING_TRACE_MODE_CHANGE(tid) (get_thread_trace_data(tid)->pending_trace_mode_change)

#define INCREASE_MPI_DEEPNESS(tid) (get_thread_trace_data(tid)->mpi_deepness++)
#define DECREASE_MPI_DEEPNESS(tid) (get_thread_trace_data(tid)->mpi_deepness--)
#define MPI_IS_NOT_STACKED(tid) (get_thread_trace_data(tid)->mpi_deepness == 0)


void TMODE_setInitial (int mode);
void TMODE_setCurrent (unsigned long long burst_threshold);
int Trace_Mode_Initialize (int num_threads);
int Trace_Mode_reInitialize (int old_num_threads, int new_num_threads);
void Trace_Mode_Change (int tid, iotimer_t time);
void Trace_mode_switch (void);
int Trace_Mode_FirstMode (unsigned thread);

/* Bursts mode specific */

extern unsigned long long BurstMode_Threshold;
extern int BurstMode_MPI_Stats;
extern int BurstMode_OMP_Stats;
extern int BurstMode_OMP_Summarization;

#define MINIMUM_BURST_DURATION (BurstMode_Threshold)
#define TRACING_MPI_STATISTICS (BurstMode_MPI_Stats)
#define TRACING_OMP_STATISTICS (BurstMode_OMP_Stats)
#define OMP_SUMMARIZATION_ENABLED (BurstMode_OMP_Summarization)


void TMODE_setBurstThreshold  (unsigned long long threshold);
void TMODE_setBurstStatistics (int type,int status);
void TMODE_setBurstOMPSummarization (int status);

#endif /* __MODE_H__ */
