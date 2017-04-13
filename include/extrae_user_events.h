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

#ifndef MPITRACE_USER_EVENTS
#define MPITRACE_USER_EVENTS

#include "extrae_types.h"
#include "extrae_internals.h"

#ifdef __cplusplus
extern "C" {
#endif

void Extrae_init (void);
void OMPItrace_init (void);
void MPItrace_init (void);
void OMPtrace_init (void);
void SEQtrace_init (void);

extrae_init_type_t Extrae_is_initialized (void);
extrae_init_type_t OMPItrace_is_initialized (void);
extrae_init_type_t MPItrace_is_initialized (void);
extrae_init_type_t OMPtrace_is_initialized (void);
extrae_init_type_t SEQtrace_is_initialized (void);

void Extrae_fini (void);
void OMPItrace_fini (void);
void MPItrace_fini (void);
void OMPtrace_fini (void);
void SEQtrace_fini (void);

void Extrae_flush (void);
void OMPItrace_flush (void);
void MPItrace_flush (void);
void OMPtrace_flush (void);
void SEQtrace_flush (void);

unsigned long long Extrae_user_function (unsigned enter);
unsigned long long OMPItrace_user_function (unsigned enter);
unsigned long long MPItrace_user_function (unsigned enter);
unsigned long long OMPtrace_user_function (unsigned enter);
unsigned long long SEQtrace_user_function (unsigned enter);

void Extrae_event (extrae_type_t type, extrae_value_t value);
void OMPItrace_event (extrae_type_t type, extrae_value_t value);
void MPItrace_event (extrae_type_t type, extrae_value_t value);
void OMPtrace_event (extrae_type_t type, extrae_value_t value);
void SEQtrace_event (extrae_type_t type, extrae_value_t value);

void Extrae_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values);
void OMPItrace_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values);
void MPItrace_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values);
void OMPtrace_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values);
void SEQtrace_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values);

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

void Extrae_eventandcounters (extrae_type_t type, extrae_value_t value);
void MPItrace_eventandcounters (extrae_type_t type, extrae_value_t value);
void OMPItrace_eventandcounters (extrae_type_t type, extrae_value_t value);
void OMPtrace_eventandcounters (extrae_type_t type, extrae_value_t value);
void SEQtrace_eventandcounters (extrae_type_t type, extrae_value_t value);

void Extrae_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values);
void OMPItrace_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values);
void MPItrace_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values);
void OMPtrace_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values);
void SEQtrace_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values);

void Extrae_define_event_type (extrae_type_t *type, char *type_description, unsigned *nvalues, extrae_value_t *values, char **values_description);

void Extrae_set_tracing_tasks (unsigned from, unsigned to);
void OMPtrace_set_tracing_tasks (unsigned from, unsigned to);
void MPItrace_set_tracing_tasks (unsigned from, unsigned to);
void OMPItrace_set_tracing_tasks (unsigned from, unsigned to);

#define EXTRAE_DISABLE_ALL_OPTIONS      (0)
#define MPITRACE_DISABLE_ALL_OPTIONS	  EXTRAE_DISABLE_ALL_OPTIONS
#define EXTRAE_CALLER_OPTION            (1<<0)
#define MPITRACE_CALLER_OPTION          EXTRAE_CALLER_OPTION
#define EXTRAE_HWC_OPTION               (1<<1)
#define MPITRACE_HWC_OPTION             EXTRAE_HWC_OPTION
#define EXTRAE_MPI_HWC_OPTION           (1<<2)
#define MPITRACE_MPI_HWC_OPTION         EXTRAE_MPI_HWC_OPTION
#define EXTRAE_MPI_OPTION               (1<<3)
#define MPITRACE_MPI_OPTION             EXTRAE_MPI_OPTION
#define EXTRAE_OMP_OPTION               (1<<4)
#define MPITRACE_OMP_OPTION             EXTRAE_OMP_OPTION
#define EXTRAE_OMP_HWC_OPTION           (1<<5)
#define MPITRACE_OMP_HWC_OPTION         EXTRAE_OMP_HWC_OPTION
#define EXTRAE_UF_HWC_OPTION            (1<<6)
#define MPITRACE_UF_HWC_OPTION          EXTRAE_UF_HWC_OPTION
#define EXTRAE_PTHREAD_OPTION           (1<<7)
#define MPITRACE_PTHREAD_OPTION         EXTRAE_PTHREAD_OPTION
#define EXTRAE_PTHREAD_HWC_OPTION       (1<<8)
#define MPITRACE_PTHREAD_HWC_OPTION     EXTRAE_PTHREAD_HWC_OPTION
#define EXTRAE_SAMPLING_OPTION          (1<<9)
#define MPITRACE_SAMPLING_OPTION        EXTRAE_SAMPLING_HWC_OPTION

#define EXTRAE_ENABLE_ALL_OPTIONS \
  (EXTRAE_CALLER_OPTION | \
   EXTRAE_HWC_OPTION | \
   EXTRAE_MPI_HWC_OPTION | \
   EXTRAE_MPI_OPTION | \
   EXTRAE_OMP_OPTION | \
   EXTRAE_OMP_HWC_OPTION | \
   EXTRAE_UF_HWC_OPTION | \
   EXTRAE_PTHREAD_OPTION | \
   EXTRAE_PTHREAD_HWC_OPTION | \
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

void Extrae_init_UserCommunication (struct extrae_UserCommunication *);
void OMPItrace_init_UserCommunication (struct extrae_UserCommunication *);
void MPItrace_init_UserCommunication (struct extrae_UserCommunication *);
void OMPtrace_init_UserCommunication (struct extrae_UserCommunication *);
void SEQtrace_init_UserCommunication (struct extrae_UserCommunication *);

void Extrae_init_CombinedEvents (struct extrae_CombinedEvents *);
void OMPItrace_init_CombinedEvents (struct extrae_CombinedEvents *);
void MPItrace_init_CombinedEvents (struct extrae_CombinedEvents *);
void OMPtrace_init_CombinedEvents (struct extrae_CombinedEvents *);
void SEQtrace_init_CombinedEvents (struct extrae_CombinedEvents *);

void Extrae_emit_CombinedEvents (struct extrae_CombinedEvents *);
void OMPItrace_emit_CombinedEvents (struct extrae_CombinedEvents *);
void MPItrace_emit_CombinedEvents (struct extrae_CombinedEvents *);
void OMPtrace_emit_CombinedEvents (struct extrae_CombinedEvents *);
void SEQtrace_emit_CombinedEvents (struct extrae_CombinedEvents *);

void Extrae_resume_virtual_thread (unsigned vthread);
void Extrae_suspend_virtual_thread (void);
void Extrae_register_stacked_type (extrae_type_t type);

void Extrae_get_version (unsigned *major, unsigned *minor, unsigned *revision);
void Extrae_register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char* s1, const char *s2);
void Extrae_register_function_address (void *ptr, const char *funcname, const char *modname, unsigned line);

#ifdef __cplusplus
}
#endif

#endif
