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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

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
		MPItrace_init_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_INIT)

#define apifTRACE_FINI(x) \
	void CtoF77(x##_fini) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_fini_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_FINI)

#define apifTRACE_EVENT(x) \
	void CtoF77(x##_event) (unsigned *tipus, unsigned *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Event_Wrapper (tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENT)

#define apifTRACE_NEVENT(x) \
	void CtoF77(x##_nevent) (unsigned *count, unsigned *tipus, unsigned *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Event_Wrapper (count, tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENT)

#define apifTRACE_SHUTDOWN(x) \
	void CtoF77(x##_shutdown) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_shutdown_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SHUTDOWN)

#define apifTRACE_EVENTANDCOUNTERS(x) \
	void CtoF77(x##_eventandcounters) (unsigned *tipus, unsigned *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Eventandcounters_Wrapper (tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENTANDCOUNTERS)

#define apifTRACE_NEVENTANDCOUNTERS(x) \
	void CtoF77(x##_neventandcounters) (unsigned *count, unsigned *tipus, unsigned *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Eventsandcounters_Wrapper (count, tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENTANDCOUNTERS)

#define apifTRACE_RESTART(x) \
	void CtoF77(x##_restart) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_restart_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_RESTART)

#define apifTRACE_COUNTERS(x) \
	void CtoF77(x##_counters) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_counters_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_COUNTERS)

#define apifTRACE_PREVIOUS_HWC_SET(x) \
	void CtoF77(x##_previous_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_previous_hwc_set_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_PREVIOUS_HWC_SET)

#define apifTRACE_NEXT_HWC_SET(x) \
	void CtoF77(x##_next_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_next_hwc_set_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEXT_HWC_SET)

#define apifTRACE_SETOPTIONS(x) \
	void CtoF77(x##_set_options) (int *options) \
	{ \
		if (mpitrace_on) \
			MPItrace_set_options_Wrapper (*options); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SETOPTIONS)

#define apifTRACE_USER_FUNCTION(x) \
	void CtoF77(x##_user_function) (int *enter) \
	{ \
		if (mpitrace_on) \
			MPItrace_user_function_Wrapper (*enter); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION);

#define apifTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void CtoF77(x##_function_from_address) (int *type, void *address) \
	{ \
		if (mpitrace_on) \
			MPItrace_function_from_address_Wrapper (*type, address); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION_FROM_ADDRESS);

#if defined(PTHREAD_SUPPORT)

# define apifTRACE_NOTIFY_NEW_PTHREAD(x) \
	void x##_notify_new_pthread (void) \
	{ \
		if (mpitrace_on) \
			Backend_NotifyNewPthread (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NOTIFY_NEW_PTHREAD);

# define apifTRACE_SET_NUM_TENTATIVE_THREADS(x) \
	void x##_set_num_tentative_threads (int *numthreads) \
	{ \
		if (mpitrace_on) \
			Backend_setNumTentativeThreads (*numthreads); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SET_NUM_TENTATIVE_THREADS);

#endif /* PTHREAD_SUPPORT */

/*** C BINDINGS + non alias routine duplication ****/

#define apiTRACE_INIT(x) \
	void CtoF77(x##_init) (void) \
	{ \
		MPItrace_init_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_INIT)

#define apiTRACE_FINI(x) \
	void CtoF77(x##_fini) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_fini_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_FINI)

#define apiTRACE_EVENT(x) \
	void x##_event (unsigned tipus, unsigned valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Event_Wrapper (&tipus, &valor); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENT)

#define apiTRACE_NEVENT(x) \
	void x##_nevent (unsigned count, unsigned *tipus, unsigned *valors) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Event_Wrapper (&count, tipus, valors); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENT)

#define apiTRACE_EVENTANDCOUNTERS(x) \
	void x##_eventandcounters (unsigned tipus, unsigned valor) \
	{ \
  		if (mpitrace_on) \
    			MPItrace_Eventandcounters_Wrapper (&tipus, &valor); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENTANDCOUNTERS)

#define apiTRACE_NEVENTANDCOUNTERS(x) \
	void x##_neventandcounters (unsigned count, unsigned *tipus, unsigned *valors) \
	{ \
  		if (mpitrace_on) \
    			MPItrace_N_Eventsandcounters_Wrapper (&count, tipus, valors); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENTANDCOUNTERS)

#define apiTRACE_SHUTDOWN(x) \
	void x##_shutdown (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_shutdown_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SHUTDOWN)

#define apiTRACE_RESTART(x) \
	void x##_restart (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_restart_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_RESTART);

#define apiTRACE_COUNTERS(x) \
	void x##_counters(void) \
	{ \
		if (mpitrace_on) \
			MPItrace_counters_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_COUNTERS);

#define apiTRACE_PREVIOUS_HWC_SET(x) \
	void x##_previous_hwc_set (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_previous_hwc_set_Wrapper (); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_PREVIOUS_HWC_SET)

#define apiTRACE_NEXT_HWC_SET(x) \
	void x##_next_hwc_set (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_next_hwc_set_Wrapper (); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEXT_HWC_SET)

#define apiTRACE_SETOPTIONS(x) \
	void x##_set_options (int options) \
	{ \
		if (mpitrace_on) \
			MPItrace_set_options_Wrapper (options); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SETOPTIONS);

#define apiTRACE_USER_FUNCTION(x) \
	void x##_user_function (int enter) \
	{ \
		if (mpitrace_on) \
			MPItrace_user_function_Wrapper (enter); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_USER_FUNCTION);

#define apiTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void x##_function_from_address (int type, void *address) \
	{ \
		if (mpitrace_on) \
			MPItrace_function_from_address_Wrapper (type, address); \
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

#else /* HAVE_WEAK_ALIAS_ATTRIBUTE */

/** C BINDINGS **/

INTERFACE_ALIASES_C(_init,Extrae_init,(void))
void Extrae_init (void)
{
	MPItrace_init_Wrapper ();
}

INTERFACE_ALIASES_C(_fini,Extrae_fini,(void))
void Extrae_fini (void)
{
	if (mpitrace_on)
		MPItrace_fini_Wrapper ();
}

INTERFACE_ALIASES_C(_event, Extrae_event, (unsigned tipus, unsigned valors))
void Extrae_event (unsigned tipus, unsigned valors)
{
	if (mpitrace_on)
		MPItrace_Event_Wrapper (&tipus, &valors);
}

INTERFACE_ALIASES_C(_nevent, Extrae_nevent, (unsigned count, unsigned *tipus, unsigned *valors))
void Extrae_nevent (unsigned count, unsigned *tipus, unsigned *valors)
{
	if (mpitrace_on)
		MPItrace_N_Event_Wrapper (&count, tipus, valors);
}

INTERFACE_ALIASES_C(_eventandcounters, Extrae_eventandcounters, (unsigned tipus, unsigned valor))
void Extrae_eventandcounters (unsigned tipus, unsigned valor)
{
	if (mpitrace_on)
		MPItrace_Eventandcounters_Wrapper (&tipus, &valor);
}

INTERFACE_ALIASES_C(_neventandcounters, Extrae_neventandcounters, (unsigned count, unsigned *tipus, unsigned *valors))
void Extrae_neventandcounters (unsigned count, unsigned *tipus, unsigned *valors)
{
 	if (mpitrace_on)
		MPItrace_N_Eventsandcounters_Wrapper (&count, tipus, valors);
}

INTERFACE_ALIASES_C(_shutdown, Extrae_shutdown, (void))
void Extrae_shutdown (void)
{
	if (mpitrace_on)
		MPItrace_shutdown_Wrapper();
}

INTERFACE_ALIASES_C(_restart, Extrae_restart, (void))
void Extrae_restart (void)
{
	if (mpitrace_on)
		MPItrace_restart_Wrapper();
}

INTERFACE_ALIASES_C(_counters, Extrae_counters, (void))
void Extrae_counters(void)
{
	if (mpitrace_on)
		MPItrace_counters_Wrapper();
}

INTERFACE_ALIASES_C(_next_hwc_set, Extrae_next_hwc_set, (void))
void Extrae_next_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_next_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_C(_previous_hwc_set, Extrae_previous_hwc_set, (void))
void Extrae_previous_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_previous_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_C(_set_options, Extrae_set_options, (int options))
void Extrae_set_options (int options)
{
	if (mpitrace_on)
		MPItrace_set_options_Wrapper (options);
}

INTERFACE_ALIASES_C(_user_function, Extrae_user_function, (int enter))
void Extrae_user_function (int enter)
{
	if (mpitrace_on)
		MPItrace_user_function_Wrapper (enter);
}

INTERFACE_ALIASES_C(_function_from_address,Extrae_function_from_address, (int type, void *address))
void Extrae_function_from_address (int type, void *address)
{
	if (mpitrace_on)
		MPItrace_function_from_address_Wrapper (type, address);
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_C(_notify_new_pthread,Extrae_notify_new_pthread, (void))
void Extrae_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_C(_set_num_tentative_threads,Extrae_set_num_tentative_threads, (int numthreads))
void Extrae_set_num_tentative_threads (int numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (numthreads);
}
#endif

/** FORTRAN BINDINGS **/

INTERFACE_ALIASES_F(_init,_INIT,extrae_init,(void))
void extrae_init (void)
{
	MPItrace_init_Wrapper ();
}

INTERFACE_ALIASES_F(_fini,_FINI,extrae_fini,(void))
void extrae_fini (void)
{
	if (mpitrace_on)
		MPItrace_fini_Wrapper ();
}

INTERFACE_ALIASES_F(_event,_EVENT,extrae_event,(unsigned *tipus, unsigned *valor))
void extrae_event (unsigned *tipus, unsigned *valor)
{
	if (mpitrace_on)
		MPItrace_Event_Wrapper (tipus, valor);
}

INTERFACE_ALIASES_F(_nevent,_NEVENT,extrae_nevent,(unsigned *count, unsigned *tipus, unsigned *valor))
void extrae_nevent (unsigned *count, unsigned *tipus, unsigned *valor)
{
	if (mpitrace_on)
		MPItrace_N_Event_Wrapper (count, tipus, valor);
}

INTERFACE_ALIASES_F(_shutdown,_SHUTDOWN,extrae_shutdown,(void))
void extrae_shutdown (void)
{
	if (mpitrace_on)
		MPItrace_shutdown_Wrapper();
}

INTERFACE_ALIASES_F(_restart,_RESTART,extrae_restart,(void))
void extrae_restart (void)
{
	if (mpitrace_on)
		MPItrace_restart_Wrapper ();
}

INTERFACE_ALIASES_F(_eventandcounters,_EVENTANDCOUNTERS,extrae_eventandcounters, (unsigned *tipus, unsigned *valor))
void extrae_eventandcounters (unsigned *tipus, unsigned *valor)
{
	if (mpitrace_on)
		MPItrace_Eventandcounters_Wrapper (tipus, valor);
}

INTERFACE_ALIASES_F(_neventandcounters,_NEVENTANDCOUNTERS,extrae_neventandcounters, (unsigned *count, unsigned *tipus, unsigned *valor))
void extrae_neventandcounters (unsigned *count, unsigned *tipus, unsigned *valor)
{
	if (mpitrace_on)
		MPItrace_N_Eventsandcounters_Wrapper (count, tipus, valor);
}

INTERFACE_ALIASES_F(_counters,_COUNTERS,extrae_counters, (void))
void extrae_counters (void)
{
	if (mpitrace_on)
		MPItrace_counters_Wrapper ();
}

INTERFACE_ALIASES_F(_next_hwc_set,_NEXT_HWC_SET,extrae_next_hwc_set,(void))
void extrae_next_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_next_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_F(_previous_hwc_set,_PREVIOUS_HWC_SET,extrae_previous_hwc_set,(void))
void extrae_previous_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_previous_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_F(_set_options,_SET_OPTIONS,extrae_set_options,(int *options))
void extrae_set_options (int *options)
{
	if (mpitrace_on)
		MPItrace_set_options_Wrapper (*options);
}

INTERFACE_ALIASES_F(_user_function,_USER_FUNCTION,extrae_user_function,(int *enter))
void extrae_user_function (int *enter)
{
	if (mpitrace_on)
		MPItrace_user_function_Wrapper (*enter);
}

INTERFACE_ALIASES_F(_function_from_address,_USER_FUNCTION_FROM_ADDRESS,extrae_function_from_address, (int *type, void *address))
void extrae_function_from_address (int *type, void *address)
{
	if (mpitrace_on)
		MPItrace_function_from_address_Wrapper (type, address);
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_F(_notify_new_pthread,_NOTIFY_NEW_PTHREAD,extrae_notify_new_pthread, (void))
void extrae_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_F(_set_num_tentative_threads,_SET_NUM_TENTATIVE_THREADS,extrae_set_num_tentative_threads, (int *numthreads))
void extrae_set_num_tentative_threads (int *numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (*numthreads);
}
#endif

#endif
