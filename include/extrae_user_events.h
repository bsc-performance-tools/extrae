/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/mpitrace/fusion/trunk/include/mpitrace_user_events.h $
 | @last_commit: $Date: 2010-02-04 18:22:43 +0100 (dj, 04 feb 2010) $
 | @version:     $Revision: 160 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MPITRACE_USER_EVENTS
#define MPITRACE_USER_EVENTS

#ifdef __cplusplus
extern "C" {
#endif

void Extrae_init (void);
void OMPItrace_init (void);
void MPItrace_init (void);
void OMPtrace_init (void);
void SEQtrace_init (void);

void Extrae_fini (void);
void OMPItrace_fini (void);
void MPItrace_fini (void);
void OMPtrace_fini (void);
void SEQtrace_fini (void);

void Extrae_user_function (unsigned enter);
void OMPItrace_user_function (unsigned enter);
void MPItrace_user_function (unsigned enter);
void OMPtrace_user_function (unsigned enter);
void SEQtrace_user_function (unsigned enter);

void Extrae_event (unsigned int type, unsigned int value);
void OMPItrace_event (unsigned int type, unsigned int value);
void MPItrace_event (unsigned int type, unsigned int value);
void OMPtrace_event (unsigned int type, unsigned int value);
void SEQtrace_event (unsigned int type, unsigned int value);

void Extrae_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPItrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void MPItrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPtrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void SEQtrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);

void Extrae_shutdown (void);
void MPItrace_shutdown (void);
void OMPItrace_shutdown (void);
void OMPtrace_shutdown (void);
void SEQtrace_shutdown (void);

void Extrae_restart (void);
void MPItrace_restart (void);
void OMPItrace_restart (void);
void OMPtrace_restart (void);
void SEQtrace_restart (void);

void Extrae_counters (void);
void MPItrace_counters (void);
void OMPItrace_counters (void);
void OMPtrace_counters (void);
void SEQtrace_counters (void);

void Extrae_previous_hwc_set (void);
void MPItrace_previous_hwc_set (void);
void OMPItrace_previous_hwc_set (void);
void OMPtrace_previous_hwc_set (void);
void SEQtrace_previous_hwc_set (void);

void Extrae_next_hwc_set (void);
void MPItrace_next_hwc_set (void);
void OMPItrace_next_hwc_set (void);
void OMPtrace_next_hwc_set (void);
void SEQtrace_next_hwc_set (void);

void Extrae_eventandcounters (int type, int value);
void MPItrace_eventandcounters (int type, int value);
void OMPItrace_eventandcounters (int type, int value);
void OMPtrace_eventandcounters (int type, int value);
void SEQtrace_eventandcounters (int type, int value);

void Extrae_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPItrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void MPItrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPtrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void SEQtrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);

void Extrae_set_tracing_tasks (int from, int to);
void OMPtrace_set_tracing_tasks (int from, int to);
void MPItrace_set_tracing_tasks (int from, int to);
void OMPItrace_set_tracing_tasks (int from, int to);

#define EXTRAE_DISABLE_ALL_OPTIONS      0
#define MPITRACE_DISABLE_ALL_OPTIONS	  EXTRAE_DISABLE_ALL_OPTIONS
#define EXTRAE_CALLER_OPTION            1
#define MPITRACE_CALLER_OPTION          EXTRAE_CALLER_OPTION
#define EXTRAE_HWC_OPTION               2
#define MPITRACE_HWC_OPTION             EXTRAE_HWC_OPTION
#define EXTRAE_MPI_HWC_OPTION           4
#define MPITRACE_MPI_HWC_OPTION         EXTRAE_MPI_HWC_OPTION
#define EXTRAE_MPI_OPTION               8 
#define MPITRACE_MPI_OPTION             EXTRAE_MPI_OPTION
#define EXTRAE_OMP_OPTION               16
#define MPITRACE_OMP_OPTION             EXTRAE_OMP_OPTION
#define EXTRAE_OMP_HWC_OPTION           32 
#define MPITRACE_OMP_HWC_OPTION         EXTRAE_OMP_HWC_OPTION
#define EXTRAE_UF_HWC_OPTION            64
#define MPITRACE_UF_HWC_OPTION          EXTRAE_UF_HWC_OPTION
#define EXTRAE_SAMPLING_OPTION          128
#define MPITRACE_SAMPLING_OPTION        EXTRAE_UF_HWC_OPTION

#define EXTRAE_ENABLE_ALL_OPTIONS \
  (EXTRAE_CALLER_OPTION | \
   EXTRAE_HWC_OPTION | \
   EXTRAE_MPI_HWC_OPTION | \
   EXTRAE_MPI_OPTION | \
   EXTRAE_OMP_OPTION | \
   EXTRAE_OMP_HWC_OPTION | \
   EXTRAE_UF_HWC_OPTION | \
   EXTRAE_SAMPLING_OPTION)
#define MPITRACE_ENABLE_ALL_OPTIONS EXTRAE_ENABLE_ALL_OPTIONS

void Extrae_set_options (int options);
void MPItrace_set_options (int options);
void OMPtrace_set_options (int options);
void OMPItrace_set_options (int options);
void SEQtrace_set_options (int options);

void Extrae_network_counters (void);
void OMPItrace_network_counters (void);
void MPItrace_network_counters (void);
void OMPtrace_network_counters (void);
void SEQtrace_network_counters (void);

void Extrae_network_routes (int mpi_rank);
void OMPItrace_network_routes (int mpi_rank);
void MPItrace_network_routes (int mpi_rank);
void OMPtrace_network_routes (int mpi_rank);
void SEQtrace_network_routes (int mpi_rank);

#ifdef __cplusplus
}
#endif

#endif
