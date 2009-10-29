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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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

void OMPItrace_event (unsigned int type, unsigned int value);
void MPItrace_event (unsigned int type, unsigned int value);
void OMPtrace_event (unsigned int type, unsigned int value);
void SEQtrace_event (unsigned int type, unsigned int value);

void OMPItrace_Nevent (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void MPItrace_Nevent (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void OMPtrace_Nevent (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void SEQtrace_Nevent (unsigned int count, unsigned int *tipus,
	unsigned int *valors);

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

void MPItrace_eventandcounters (int Type, int Value);
void OMPItrace_eventandcounters (int Type, int Value);
void OMPtrace_eventandcounters (int Type, int Value);
void SEQtrace_eventandcounters (int Type, int Value);

void OMPItrace_Neventandcounters (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void MPItrace_Neventandcounters (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void OMPtrace_Neventandcounters (unsigned int count, unsigned int *tipus,
	unsigned int *valors);
void SEQtrace_Neventandcounters (unsigned int count, unsigned int *tipus,
	unsigned int *valors);

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
