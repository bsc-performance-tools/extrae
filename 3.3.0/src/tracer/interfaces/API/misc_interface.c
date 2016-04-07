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

#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif

#include "misc_interface.h"
#include "misc_wrapper.h"
#include "wrapper.h"

#if !defined(HAVE_ALIAS_ATTRIBUTE)

/*** FORTRAN BINDINGS + non alias routine duplication ****/

#define apifTRACE_INIT(x) \
	void CtoF77(x##_init) (void) \
	{ \
		Extrae_init_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_INIT)

#define apifTRACE_IS_INITIALIZED(x) \
	void CtoF77(x##_is_initialized) (unsigned *res) \
	{ \
		*res = Extrae_is_initialized_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_IS_INITIALIZED)

#define apifTRACE_FINI(x) \
	void CtoF77(x##_fini) (void) \
	{ \
		if (mpitrace_on) \
			Extrae_fini_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_FINI)

#define apifTRACE_EVENT(x) \
	void CtoF77(x##_event) (extrae_type_t *type, extrae_value_t *value) \
	{ \
		if (mpitrace_on) \
		{ \
			unsigned one = 1; \
			Backend_Enter_Instrumentation (1); \
			Extrae_N_Event_Wrapper (&one, type, value); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENT)

#define apifTRACE_NEVENT(x) \
	void CtoF77(x##_nevent) (unsigned *count, extrae_type_t *types, extrae_value_t *values) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (*count); \
			Extrae_N_Event_Wrapper (count, types, values); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENT)

#define apifTRACE_SHUTDOWN(x) \
	void CtoF77(x##_shutdown) (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_shutdown_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SHUTDOWN)

#define apifTRACE_EVENTANDCOUNTERS(x) \
	void CtoF77(x##_eventandcounters) (extrae_type_t *type, extrae_value_t *value) \
	{ \
		if (mpitrace_on) \
		{ \
			unsigned one = 1; \
			Backend_Enter_Instrumentation (1); \
			Extrae_N_Eventsandcounters_Wrapper (&one, type, value); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENTANDCOUNTERS)

#define apifTRACE_NEVENTANDCOUNTERS(x) \
	void CtoF77(x##_neventandcounters) (unsigned *count, extrae_type_t *types, extrae_value_t *values) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (*count); \
			Extrae_N_Eventsandcounters_Wrapper (count, types, values); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENTANDCOUNTERS)

#define apifTRACE_RESTART(x) \
	void CtoF77(x##_restart) (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_restart_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_RESTART)

#define apifTRACE_COUNTERS(x) \
	void CtoF77(x##_counters) (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_counters_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_COUNTERS)

#define apifTRACE_PREVIOUS_HWC_SET(x) \
	void CtoF77(x##_previous_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_previous_hwc_set_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_PREVIOUS_HWC_SET)

#define apifTRACE_NEXT_HWC_SET(x) \
	void CtoF77(x##_next_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_next_hwc_set_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEXT_HWC_SET)

#define apifTRACE_SETOPTIONS(x) \
	void CtoF77(x##_set_options) (int *options) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_set_options_Wrapper (*options); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SETOPTIONS)

#define apifTRACE_USER_FUNCTION(x) \
	void CtoF77(x##_user_function) (unsigned *enter) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_user_function_Wrapper (*enter); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION);

#define apifTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void CtoF77(x##_function_from_address) (extrae_type_t *type, extrae_value_t *address) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_function_from_address_Wrapper (*type, address); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION_FROM_ADDRESS);

#if defined(PTHREAD_SUPPORT)

# define apifTRACE_NOTIFY_NEW_PTHREAD(x) \
	void CtoF77(x##_notify_new_pthread) (void) \
	{ \
		if (mpitrace_on) \
			Backend_NotifyNewPthread (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NOTIFY_NEW_PTHREAD);

# define apifTRACE_SET_NUM_TENTATIVE_THREADS(x) \
	void CtoF77(x##_set_num_tentative_threads) (int *numthreads) \
	{ \
		if (mpitrace_on) \
			Backend_setNumTentativeThreads (*numthreads); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SET_NUM_TENTATIVE_THREADS);

#endif /* PTHREAD_SUPPORT */

# define apifDEFINE_EVENT_TYPE(x) \
	void CtoF77(x##_define_event_type) (extrae_type_t *type, char *description, unsigned *nvalues, extrae_value_t *values, char **description_values) \
	{ \
		Extrae_define_event_type_Wrapper (*type, description, *nvalues, values, description_values); \
	}
  EXPAND_F_ROUTINE_WITH_PREFIXES(apifDEFINE_EVENT_TYPE);

#define apifTRACE_GET_VERSION(x) \
	void CtoF77(x##_get_version) (unsigned *M, unsigned *m, unsigned *r) \
	{ \
		Extrae_get_version_Wrapper (M, m, r); \
	}
  EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_GET_VERSION);

#define apifCHANGE_NUM_THREADS(x) \
	void CtoF77(x##_change_num_threads) (unsigned *n) \
	{ \
		Extrae_change_number_of_threads_Wrapper (*n); \
	}
  EXPAND_F_ROUTINE_WITH_PREFIXES(apifCHANGE_NUM_THREADS);

#define apifTRACE_FLUSH(x) \
	void CtoF77(x##_flush) (void) \
	{ \
		Extrae_flush_manual_Wrapper (); \
	}
  EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_FLUSH)

/*** C BINDINGS + non alias routine duplication ****/

#define apiTRACE_INIT(x) \
	void x##_init (void) \
	{ \
		Extrae_init_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_INIT)

#define apiTRACE_IS_INITIALIZED(x) \
	extrae_init_type_t x##_is_initialized (void) \
	{ \
		return Extrae_is_initialized_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_IS_INITIALIZED)

#define apiTRACE_FINI(x) \
	void x##_fini (void) \
	{ \
		if (mpitrace_on) \
			Extrae_fini_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_FINI)

#define apiTRACE_EVENT(x) \
	void x##_event (extrae_type_t type, extrae_value_t value) \
	{ \
		if (mpitrace_on) \
		{ \
			unsigned one = 1; \
			Backend_Enter_Instrumentation (1); \
			Extrae_N_Event_Wrapper (&one, &type, &value); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENT)

#define apiTRACE_NEVENT(x) \
	void x##_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (count); \
			Extrae_N_Event_Wrapper (&count, types, values); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENT)

#define apiTRACE_EVENTANDCOUNTERS(x) \
	void x##_eventandcounters (extrae_type_t type, extrae_value_t value) \
	{ \
		if (mpitrace_on) \
		{ \
			unsigned one = 1; \
			Backend_Enter_Instrumentation (1); \
			Extrae_N_Eventsandcounters_Wrapper (&one, &type, &value); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENTANDCOUNTERS)

#define apiTRACE_NEVENTANDCOUNTERS(x) \
	void x##_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (count); \
			Extrae_N_Eventsandcounters_Wrapper (&count, types, values); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENTANDCOUNTERS)

#define apiTRACE_SHUTDOWN(x) \
	void x##_shutdown (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_shutdown_Wrapper(); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SHUTDOWN)

#define apiTRACE_RESTART(x) \
	void x##_restart (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_restart_Wrapper(); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_RESTART);

#define apiTRACE_COUNTERS(x) \
	void x##_counters(void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_counters_Wrapper(); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_COUNTERS);

#define apiTRACE_PREVIOUS_HWC_SET(x) \
	void x##_previous_hwc_set (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_previous_hwc_set_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_PREVIOUS_HWC_SET)

#define apiTRACE_NEXT_HWC_SET(x) \
	void x##_next_hwc_set (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_next_hwc_set_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEXT_HWC_SET)

#define apiTRACE_SETOPTIONS(x) \
	void x##_set_options (int options) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_set_options_Wrapper (options); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SETOPTIONS);

#define apiTRACE_USER_FUNCTION(x) \
	UINT64 x##_user_function (unsigned enter) \
	{ \
		UINT64 r = 0; \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			r = Extrae_user_function_Wrapper (enter); \
			Backend_Leave_Instrumentation (); \
		} \
		return r; \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_USER_FUNCTION);

#define apiTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void x##_function_from_address (extrae_type_t type, extrae_value_t *address) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_function_from_address_Wrapper (type, address); \
			Backend_Leave_Instrumentation (); \
		} \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_USER_FUNCTION_FROM_ADDRESS);

#if defined(PTHREAD_SUPPORT)
# define apiTRACE_NOTIFY_NEW_PTHREAD(x) \
	void x##_notify_new_pthread (void) \
	{ \
		if (mpitrace_on) \
			Backend_NotifyNewPthread (); \
	}
 EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NOTIFY_NEW_PTHREAD);

# define apiTRACE_SET_NUM_TENTATIVE_THREADS(x) \
	void x##_set_num_tentative_threads (int numthreads) \
	{ \
		if (mpitrace_on) \
			Backend_setNumTentativeThreads (numthreads); \
	}
 EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SET_NUM_TENTATIVE_THREADS);

#endif /* PTHREAD_SUPPORT */

# define apiDEFINE_EVENT_TYPE(x) \
	void x##_define_event_type (extrae_type_t *type, char *description, unsigned *nvalues, extrae_value_t *values, char **description_values) \
	{ \
		Extrae_define_event_type_Wrapper (*type, description, *nvalues, values, description_values); \
	}
 EXPAND_ROUTINE_WITH_PREFIXES(apiDEFINE_EVENT_TYPE);

#define apiTRACE_INIT_USERCOMMUNICATION(x) \
	void x##_init_UserCommunication (struct extrae_UserCommunication *ptr) \
	{ \
		Extrae_init_UserCommunication_Wrapper (ptr); \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_INIT_USERCOMMUNICATION);

#define apiTRACE_INIT_COMBINEDEVENTS(x) \
	void x##_init_CombinedEvents (struct extrae_UserCommunication *ptr) \
	{ \
		Extrae_init_CombinedEvents_Wrapper (ptr); \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_INIT_COMBINEDEVENTS);

#define apiTRACE_EMIT_COMBINEDEVENTS(x) \
	void x##_emit_CombinedEvents (struct extrae_UserCommunication *ptr) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_emit_CombinedEvents_Wrapper (ptr); \
			Backend_Leave_Instrumentation (); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EMIT_COMBINEDEVENTS);

#define apiTRACE_RESUME_VIRTUAL_THREAD(x) \
	void x##_resume_virtual_thread (unsigned u) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_Resume_virtual_thread_Wrapper (u); \
			Backend_Leave_Instrumentation (); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_RESUME_VIRTUAL_THREAD);

#define apiTRACE_SUSPEND_VIRTUAL_THREAD(x) \
	void x##_suspend_virtual_thread (void) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_Suspend_virtual_thread_Wrapper (); \
			Backend_Leave_Instrumentation (); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SUSPEND_VIRTUAL_THREAD);

#define apiTRACE_REGISTER_STACKED_TYPE(x) \
	void x##register_stacked_type (extrae_type_t t) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_register_stacked_type_Wrapper (t); \
			Backend_Leave_Instrumentation (); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_REGISTER_STACKED_TYPE);

#define apiTRACE_GET_VERSION(x) \
	void x##_get_version (unsigned *M, unsigned *m, unsigned *r) \
	{ \
		Extrae_get_version_Wrapper (M, m, r); \
	}
  EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_GET_VERSION);

#define apiTRACE_REGISTER_CODELOCATION_TYPE(x) \
	void x##register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char *s1, const char *s2) \
	{ \
		if (mpitrace_on) \
		{ \
			Backend_Enter_Instrumentation (1); \
			Extrae_register_codelocation_type_Wrapper (t1, t2, s1, s2); \
			Backend_Leave_Instrumentation (); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_REGISTER_CODELOCATION_TYPE);

#define apiTRACE_REGISTER_FUNCTION_ADDRESS(x) \
	void x##register_function_address (void *ptr, const char *funcname, const char *modname, unsigned line) \
	{ \
		if (mpitrace_on) \
		{ \
				Extrae_register_function_address_Wrapper (ptr, funcname, modname, line); \
		} \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_REGISTER_FUNCTION_ADDRESS);

#define apiCHANGE_NUM_THREADS(x) \
	void x##_change_num_threads (unsigned n) \
	{ \
		Extrae_change_number_of_threads_Wrapper (n); \
	}
  EXPAND_ROUTINE_WITH_PREFIXES(apiCHANGE_NUM_THREADS);
	
#else /* HAVE_WEAK_ALIAS_ATTRIBUTE */

/** C BINDINGS **/

INTERFACE_ALIASES_C(_init,Extrae_init,(void),void)
void Extrae_init (void)
{
	Extrae_init_Wrapper ();
}

INTERFACE_ALIASES_C(_is_initialized,Extrae_is_initialized,(void),extrae_init_type_t)
extrae_init_type_t Extrae_is_initialized (void)
{
	return Extrae_is_initialized_Wrapper();
}

INTERFACE_ALIASES_C(_fini,Extrae_fini,(void),void)
void Extrae_fini (void)
{
	if (mpitrace_on)
		Extrae_fini_Wrapper ();
}

INTERFACE_ALIASES_C(_event, Extrae_event, (extrae_type_t type, extrae_value_t value),void)
void Extrae_event (extrae_type_t type, extrae_value_t value)
{
	if (mpitrace_on)
	{
		unsigned one = 1;
		Backend_Enter_Instrumentation (1);
		Extrae_N_Event_Wrapper (&one, &type, &value);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_nevent, Extrae_nevent, (unsigned count, extrae_type_t *types, extrae_value_t *values),void)
void Extrae_nevent (unsigned count, extrae_type_t *types, extrae_value_t *values)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (count);
		Extrae_N_Event_Wrapper (&count, types, values);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_eventandcounters, Extrae_eventandcounters, (extrae_type_t type, extrae_value_t value),void)
void Extrae_eventandcounters (extrae_type_t type, extrae_value_t value)
{
	if (mpitrace_on)
	{
		unsigned one = 1;
		Backend_Enter_Instrumentation (1);
		Extrae_N_Eventsandcounters_Wrapper (&one, &type, &value);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_neventandcounters, Extrae_neventandcounters, (unsigned count, extrae_type_t *types, extrae_value_t *values),void)
void Extrae_neventandcounters (unsigned count, extrae_type_t *types, extrae_value_t *values)
{
 	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (count);
		Extrae_N_Eventsandcounters_Wrapper (&count, types, values);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_shutdown, Extrae_shutdown,(void),void)
void Extrae_shutdown (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_shutdown_Wrapper();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_restart, Extrae_restart,(void),void)
void Extrae_restart (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_restart_Wrapper();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_counters, Extrae_counters,(void),void)
void Extrae_counters(void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_counters_Wrapper();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_next_hwc_set, Extrae_next_hwc_set,(void),void)
void Extrae_next_hwc_set (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_next_hwc_set_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_previous_hwc_set, Extrae_previous_hwc_set,(void),void)
void Extrae_previous_hwc_set (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_previous_hwc_set_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_set_options, Extrae_set_options,(int options),void)
void Extrae_set_options (int options)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_set_options_Wrapper (options);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_user_function, Extrae_user_function,(unsigned enter),UINT64)
UINT64 Extrae_user_function (unsigned enter)
{
	UINT64 r = 0;
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		r = Extrae_user_function_Wrapper (enter);
		Backend_Leave_Instrumentation ();
	}
	return r;
}

INTERFACE_ALIASES_C(_function_from_address,Extrae_function_from_address,(extrae_type_t type, void *address),void)
void Extrae_function_from_address (extrae_type_t type, void *address)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_function_from_address_Wrapper (type, address);
		Backend_Leave_Instrumentation ();
	}
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_C(_notify_new_pthread,Extrae_notify_new_pthread,(void),void)
void Extrae_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_C(_set_num_tentative_threads,Extrae_set_num_tentative_threads,(int numthreads),void)
void Extrae_set_num_tentative_threads (int numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (numthreads);
}

#endif /* PTHREAD_SUPPORT */

INTERFACE_ALIASES_C(_define_event_type,Extrae_define_event_type,(extrae_type_t *type, char *description, unsigned *nvalues, extrae_value_t *values, char **description_values),void)
void Extrae_define_event_type (extrae_type_t *type, char *description, unsigned *nvalues, extrae_value_t *values, char **description_values)
{
	Extrae_define_event_type_Wrapper (*type, description, *nvalues, values, description_values);
}

INTERFACE_ALIASES_C(_init_UserCommunication,Extrae_init_UserCommunication,(struct extrae_UserCommunication *ptr),void)
void Extrae_init_UserCommunication (struct extrae_UserCommunication *ptr)
{
	Extrae_init_UserCommunication_Wrapper (ptr);
}

INTERFACE_ALIASES_C(_init_CombinedEvents,Extrae_init_CombinedEvents,(struct extrae_UserCommunication *ptr),void)
void Extrae_init_CombinedEvents (struct extrae_CombinedEvents *ptr)
{
	Extrae_init_CombinedEvents_Wrapper (ptr);
}

INTERFACE_ALIASES_C(_emit_CombinedEvents,Extrae_emit_CombinedEvents,(struct extrae_UserCommunication *ptr),void)
void Extrae_emit_CombinedEvents (struct extrae_CombinedEvents *ptr)
{
	unsigned nevents;

	if (mpitrace_on)
	{
		nevents = ptr->nEvents+ptr->nCommunications;
		nevents += (ptr->UserFunction!=EXTRAE_USER_FUNCTION_NONE)?1:0;
		nevents += (ptr->Callers)?Caller_Count[CALLER_MPI]:0;

		Backend_Enter_Instrumentation (nevents);
		Extrae_emit_CombinedEvents_Wrapper (ptr);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_resume_virtual_thread,Extrae_resume_virtual_thread,(unsigned u),void)
void Extrae_resume_virtual_thread (unsigned u)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_Resume_virtual_thread_Wrapper (u);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_suspend_virtual_thread,Extrae_suspend_virtual_thread,(void),void)
void Extrae_suspend_virtual_thread (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_Suspend_virtual_thread_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_register_stacked_type,Extrae_register_stacked_type,(extrae_type_t),void)
void Extrae_register_stacked_type (extrae_type_t t)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_register_stacked_type_Wrapper (t);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_C(_get_version,Extrae_get_version,(unsigned*,unsigned*,unsigned*),void)
void Extrae_get_version (unsigned *M, unsigned *m, unsigned *r)
{
	Extrae_get_version_Wrapper (M, m, r);
}

INTERFACE_ALIASES_C(_register_codelocation_type,Extrae_register_codelocation_type,(extrae_type_t,const char*, const char*),void)
void Extrae_register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char* s1, const char *s2)
{
	/* Does not need to check whether mpitrace_on is enabled */
	Extrae_register_codelocation_type_Wrapper (t1, t2, s1, s2);
}

INTERFACE_ALIASES_C(_register_function_address,Extrae_register_function_address,(void*,const char*,const char*,unsigned),void)
void Extrae_register_function_address (void *ptr,const char *funcname,const char *modname, unsigned line)
{
	/* Does not need to check whether mpitrace_on is enabled */
	Extrae_register_function_address_Wrapper (ptr, funcname, modname, line);
}

INTERFACE_ALIASES_C(_change_num_threads,Extrae_change_num_threads,(unsigned),void)
void Extrae_change_num_threads (unsigned n)
{
	Extrae_change_number_of_threads_Wrapper (n);
}

INTERFACE_ALIASES_C(_flush,Extrae_flush,(void),void)
void Extrae_flush (void)
{
	Extrae_flush_manual_Wrapper ();
}

/** FORTRAN BINDINGS **/

INTERFACE_ALIASES_F(_init,_INIT,extrae_init,(void),void)
void extrae_init (void)
{
	Extrae_init_Wrapper ();
}

INTERFACE_ALIASES_F(_is_initialized,_IS_INITIALIZE,extrae_is_initialized,(unsigned *res),void)
void extrae_is_initialized (unsigned *res)
{
	*res = Extrae_is_initialized_Wrapper();
}

INTERFACE_ALIASES_F(_fini,_FINI,extrae_fini,(void),void)
void extrae_fini (void)
{
	if (mpitrace_on)
		Extrae_fini_Wrapper ();
}

INTERFACE_ALIASES_F(_event,_EVENT,extrae_event,(extrae_type_t *type, extrae_value_t *value),void)
void extrae_event (extrae_type_t *type, extrae_value_t *value)
{
	if (mpitrace_on)
	{
		unsigned one = 1;
		Backend_Enter_Instrumentation (1);
		Extrae_N_Event_Wrapper (&one, type, value);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_nevent,_NEVENT,extrae_nevent,(unsigned *count, extrae_type_t *types, extrae_value_t *values),void)
void extrae_nevent (unsigned *count, extrae_type_t *types, extrae_value_t *values)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_N_Event_Wrapper (count, types, values);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_shutdown,_SHUTDOWN,extrae_shutdown,(void),void)
void extrae_shutdown (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_shutdown_Wrapper();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_restart,_RESTART,extrae_restart,(void),void)
void extrae_restart (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_restart_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_eventandcounters,_EVENTANDCOUNTERS,extrae_eventandcounters, (extrae_type_t *type, extrae_value_t *value),void)
void extrae_eventandcounters (extrae_type_t *type, extrae_value_t *value)
{
	if (mpitrace_on)
	{
		unsigned one = 1;
		Backend_Enter_Instrumentation (1);
		Extrae_N_Eventsandcounters_Wrapper (&one, type, value);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_neventandcounters,_NEVENTANDCOUNTERS,extrae_neventandcounters, (unsigned *count, extrae_type_t *types, extrae_value_t *values),void)
void extrae_neventandcounters (unsigned *count, extrae_type_t *types, extrae_value_t *values)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (*count);
		Extrae_N_Eventsandcounters_Wrapper (count, types, values);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_counters,_COUNTERS,extrae_counters, (void),void)
void extrae_counters (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_counters_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_next_hwc_set,_NEXT_HWC_SET,extrae_next_hwc_set,(void),void)
void extrae_next_hwc_set (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_next_hwc_set_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_previous_hwc_set,_PREVIOUS_HWC_SET,extrae_previous_hwc_set,(void),void)
void extrae_previous_hwc_set (void)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_previous_hwc_set_Wrapper ();
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_set_options,_SET_OPTIONS,extrae_set_options,(int *options),void)
void extrae_set_options (int *options)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_set_options_Wrapper (*options);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_user_function,_USER_FUNCTION,extrae_user_function,(unsigned *enter),void)
void extrae_user_function (unsigned *enter)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_user_function_Wrapper (*enter);
		Backend_Leave_Instrumentation ();
	}
}

INTERFACE_ALIASES_F(_function_from_address,_USER_FUNCTION_FROM_ADDRESS,extrae_function_from_address,(extrae_type_t *type, void *address),void)
void extrae_function_from_address (extrae_type_t *type, void *address)
{
	if (mpitrace_on)
	{
		Backend_Enter_Instrumentation (1);
		Extrae_function_from_address_Wrapper (*type, address);
		Backend_Leave_Instrumentation ();
	}
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_F(_notify_new_pthread,_NOTIFY_NEW_PTHREAD,extrae_notify_new_pthread,(void), void)
void extrae_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_F(_set_num_tentative_threads,_SET_NUM_TENTATIVE_THREADS,extrae_set_num_tentative_threads,(int *numthreads),void)
void extrae_set_num_tentative_threads (int *numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (*numthreads);
}

#endif /* PTHREAD_SUPPORT */

INTERFACE_ALIASES_F_REUSE_C(_define_event_type,_DEFINE_EVENT_TYPE,Extrae_define_event_type,(extrae_type_t *type, char *description, unsigned *nvalues, extrae_value_t *values, char **description_values),void)
/* This extrae_define_event_type calls automatically to the C version */

INTERFACE_ALIASES_F_REUSE_C(_get_version,_GET_VERSION,Extrae_get_version,(unsigned*,unsigned*,unsigned*),void)
/* This extrae_get_version calls automatically to the C version */

INTERFACE_ALIASES_F_REUSE_C(_change_num_threads,_CHANGE_NUM_THREADS,Extrae_change_num_threads,(unsigned),void)
/* This extrae_change_num_threads calls automatically to the C version */

INTERFACE_ALIASES_F(_flush,_FLUSH,extrae_flush,(void), void)
void extrae_flush (void)
{
	Extrae_flush_manual_Wrapper();
}

#endif
