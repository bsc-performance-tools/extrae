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

#ifndef MISC_WRAPPER_DEFINED
#define MISC_WRAPPER_DEFINED

#include "clock.h"
#include "extrae_types.h"

void Extrae_init_Wrapper(void);
void Extrae_init_tracing (int forked);
void Extrae_fini_Wrapper(void);
void Extrae_fini_last_chance_Wrapper(void);
void Extrae_shutdown_Wrapper (void);
void Extrae_restart_Wrapper (void);

void Extrae_N_Event_Wrapper (unsigned *count, extrae_type_t *tipus, extrae_value_t *valors);
void Extrae_N_Eventsandcounters_Wrapper (unsigned *count, extrae_type_t *tipus, extrae_value_t *valors);
void Extrae_counters_Wrapper (void);
void Extrae_counters_at_Time_Wrapper (UINT64 time);
void Extrae_setcounters_Wrapper (int *evc1, int *evc2);
void Extrae_set_options_Wrapper (int options);
void Extrae_getrusage_set_to_0_Wrapper (UINT64 time);
void Extrae_getrusage_Wrapper (void);
void Extrae_memusage_Wrapper (void);
UINT64 Extrae_user_function_Wrapper (unsigned enter);
void Extrae_function_from_address_Wrapper (extrae_type_t type, void *address);

void Extrae_next_hwc_set_Wrapper (void);
void Extrae_previous_hwc_set_Wrapper (void);

void Extrae_notify_new_pthread (void);

void Extrae_init_UserCommunication_Wrapper (struct extrae_UserCommunication *ptr);
void Extrae_init_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr);
void Extrae_emit_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr);

void Extrae_Resume_virtual_thread_Wrapper (unsigned u);
void Extrae_Suspend_virtual_thread_Wrapper (void);
void Extrae_register_stacked_type_Wrapper (extrae_type_t type);
void Extrae_register_codelocation_type_Wrapper (extrae_type_t type_function,
	extrae_type_t type_file_line, const char *description_function,
	const char *description_file_line);
void Extrae_register_function_address_Wrapper (void *ptr, const char *funcname, 
	const char *modname, unsigned line);
void Extrae_define_event_type_Wrapper (extrae_type_t type, char *description,
	unsigned nvalues, extrae_value_t *values, char **description_values);
void Extrae_get_version_Wrapper (unsigned *major, unsigned *minor,
  unsigned *revision);

void Extrae_change_number_of_threads_Wrapper (unsigned nthreads);

void Extrae_flush_manual_Wrapper ();

#endif
