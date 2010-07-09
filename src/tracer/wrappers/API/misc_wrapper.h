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

#ifndef MISC_WRAPPER_DEFINED
#define MISC_WRAPPER_DEFINED

#include "clock.h"
#include "extrae_types.h"

void MPItrace_init_Wrapper(void);
void MPItrace_fini_Wrapper(void);
void MPItrace_shutdown_Wrapper (void);
void MPItrace_restart_Wrapper (void);

void MPItrace_Event_Wrapper (unsigned int *tipus, unsigned int *valor);
void MPItrace_N_Event_Wrapper (unsigned int *count, unsigned int *tipus,
  unsigned int *valors);
void MPItrace_Eventandcounters_Wrapper (int *Type, int *Value);
void MPItrace_N_Eventsandcounters_Wrapper (unsigned int *count, 
  unsigned int *tipus, unsigned int *valors);
void MPItrace_counters_Wrapper ();
void MPItrace_setcounters_Wrapper (int *evc1, int *evc2);
void MPItrace_set_options_Wrapper (int options);
void MPItrace_getrusage_Wrapper (iotimer_t timestamp);
void MPItrace_memusage_Wrapper (iotimer_t timestamp);
void MPItrace_user_function_Wrapper (int enter);
void MPItrace_function_from_address_Wrapper (int type, void *address);

void MPItrace_next_hwc_set_Wrapper (void);
void MPItrace_previous_hwc_set_Wrapper (void);

void MPItrace_notify_new_pthread (void);

void Extrae_init_UserCommunication_Wrapper (struct extrae_UserCommunication *ptr);
void Extrae_init_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr);
void Extrae_emit_CombinedEvents_Wrapper (struct extrae_CombinedEvents *ptr);

#endif
