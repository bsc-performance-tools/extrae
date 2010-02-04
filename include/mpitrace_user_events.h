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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MPITRACE_USER_EVENTS
#define MPITRACE_USER_EVENTS

#ifdef __cplusplus
extern "C" {
#endif

void OMPItrace_init (void);
void MPItrace_init (void);
void OMPtrace_init (void);
void SEQtrace_init (void);

void OMPItrace_fini (void);
void MPItrace_fini (void);
void OMPtrace_fini (void);
void SEQtrace_fini (void);

void OMPItrace_user_function (unsigned enter);
void MPItrace_user_function (unsigned enter);
void OMPtrace_user_function (unsigned enter);
void SEQtrace_user_function (unsigned enter);

void OMPItrace_event (unsigned int type, unsigned int value);
void MPItrace_event (unsigned int type, unsigned int value);
void OMPtrace_event (unsigned int type, unsigned int value);
void SEQtrace_event (unsigned int type, unsigned int value);

void OMPItrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void MPItrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPtrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);
void SEQtrace_nevent (unsigned int count, unsigned int *types,
	unsigned int *values);

void MPItrace_shutdown (void);
void OMPItrace_shutdown (void);
void OMPtrace_shutdown (void);
void SEQtrace_shutdown (void);

void MPItrace_restart (void);
void OMPItrace_restart (void);
void OMPtrace_restart (void);
void SEQtrace_restart (void);

void MPItrace_counters (void);
void OMPItrace_counters (void);
void OMPtrace_counters (void);
void SEQtrace_counters (void);

void MPItrace_previous_hwc_set (void);
void OMPItrace_previous_hwc_set (void);
void OMPtrace_previous_hwc_set (void);
void SEQtrace_previous_hwc_set (void);

void MPItrace_next_hwc_set (void);
void OMPItrace_next_hwc_set (void);
void OMPtrace_next_hwc_set (void);
void SEQtrace_next_hwc_set (void);

void MPItrace_eventandcounters (int type, int value);
void OMPItrace_eventandcounters (int type, int value);
void OMPtrace_eventandcounters (int type, int value);
void SEQtrace_eventandcounters (int type, int value);

void OMPItrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void MPItrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void OMPtrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);
void SEQtrace_neventandcounters (unsigned int count, unsigned int *types,
	unsigned int *values);

void OMPtrace_set_tracing_tasks (int from, int to);
void MPItrace_set_tracing_tasks (int from, int to);
void OMPItrace_set_tracing_tasks (int from, int to);

#define MPITRACE_DISABLE_ALL_OPTIONS	  0
#define MPITRACE_CALLER_OPTION          1
#define MPITRACE_HWC_OPTION             2
#define MPITRACE_MPI_HWC_OPTION         4
#define MPITRACE_MPI_OPTION             8 
#define MPITRACE_OMP_OPTION             16
#define MPITRACE_OMP_HWC_OPTION         32 
#define MPITRACE_UF_HWC_OPTION          64

#define MPITRACE_ENABLE_ALL_OPTIONS \
  (MPITRACE_CALLER_OPTION | \
   MPITRACE_HWC_OPTION | \
   MPITRACE_MPI_HWC_OPTION | \
   MPITRACE_MPI_OPTION | \
   MPITRACE_OMP_OPTION | \
   MPITRACE_OMP_HWC_OPTION | \
   MPITRACE_UF_HWC_OPTION)

void MPItrace_set_options (int options);
void OMPtrace_set_options (int options);
void OMPItrace_set_options (int options);

void OMPItrace_network_counters (void);
void MPItrace_network_counters (void);
void OMPtrace_network_counters (void);
void SEQtrace_network_counters (void);

void OMPItrace_network_routes (int mpi_rank);
void MPItrace_network_routes (int mpi_rank);
void OMPtrace_network_routes (int mpi_rank);
void SEQtrace_network_routes (int mpi_rank);

#ifdef __cplusplus
}
#endif

#endif
