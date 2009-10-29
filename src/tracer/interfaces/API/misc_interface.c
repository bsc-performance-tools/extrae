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
	void CtoF77(x##trace_init) (void) \
	{ \
		MPItrace_init_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_INIT)

#define apifTRACE_FINI(x) \
	void CtoF77(x##trace_fini) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_fini_Wrapper ();\
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_FINI)

#define apifTRACE_EVENT(x) \
	void CtoF77(x##trace_event) (unsigned int *tipus, unsigned int *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Event_Wrapper (tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENT)

#define apifTRACE_NEVENT(x) \
	void CtoF77(x##trace_nevent) (unsigned int *count, unsigned int *tipus, unsigned int *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Event_Wrapper (count, tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENT)

#define apifTRACE_SHUTDOWN(x) \
	void CtoF77(x##trace_shutdown) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_shutdown_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SHUTDOWN)

#define apifTRACE_EVENTANDCOUNTERS(x) \
	void CtoF77(x##trace_eventandcounters) (unsigned int *tipus, unsigned int *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Eventandcounters_Wrapper (tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_EVENTANDCOUNTERS)

#define apifTRACE_NEVENTANDCOUNTERS(x) \
	void CtoF77(x##trace_neventandcounters) (unsigned int *count, unsigned int *tipus, unsigned *valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Eventsandcounters_Wrapper (count, tipus, valor); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEVENTANDCOUNTERS)

#define apifTRACE_RESTART(x) \
	void CtoF77(x##trace_restart) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_restart_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_RESTART)

#define apifTRACE_COUNTERS(x) \
	void CtoF77(x##trace_counters) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_counters_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_COUNTERS)

#define apifTRACE_PREVIOUS_HWC_SET(x) \
	void CtoF77(x##trace_previous_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_previous_hwc_set_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_PREVIOUS_HWC_SET)

#define apifTRACE_NEXT_HWC_SET(x) \
	void CtoF77(x##trace_next_hwc_set) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_next_hwc_set_Wrapper (); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_NEXT_HWC_SET)

#define apifTRACE_SETOPTIONS(x) \
	void CtoF77(x##trace_set_options) (int *options) \
	{ \
		if (mpitrace_on) \
			MPItrace_set_options_Wrapper (*options); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SETOPTIONS)

#define apifTRACE_USER_FUNCTION(x) \
	void CtoF77(x##trace_user_function) (int *enter) \
	{ \
		if (mpitrace_on) \
			MPItrace_user_function_Wrapper (*enter); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION);

#define apifTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void CtoF77(x##trace_function_from_address) (int *type, void *address) \
	{ \
		if (mpitrace_on) \
			MPItrace_function_from_address_Wrapper (*type, address); \
	}
EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_USER_FUNCTION_FROM_ADDRESS);

#if defined(PTHREAD_SUPPORT)

# define apifTRACE_NOTIFY_NEW_PTHREAD(x) \
	void x##trace_notify_new_pthread (void) \
	{ \
		if (mpitrace_on) \
			Backend_NotifyNewPthread (); \
	}
 EXPAND_F_ROUTINE_WITH_PREFIXES(apiTRACE_NOTIFY_NEW_PTHREAD);

# define apifTRACE_SET_NUM_TENTATIVE_THREADS(x) \
	void x##trace_set_num_tentative_threads (int *numthreads) \
	{ \
		if (mpitrace_on) \
			Backend_setNumTentativeThreads (*numthreads); \
	}
	EXPAND_F_ROUTINE_WITH_PREFIXES(apifTRACE_SET_NUM_TENTATIVE_THREADS);

#endif /* PTHREAD_SUPPORT */

/*** C BINDINGS + non alias routine duplication ****/

#define apiTRACE_INIT(x) \
	void CtoF77(x##trace_init) (void) \
	{ \
		MPItrace_init_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_INIT)

#define apiTRACE_FINI(x) \
	void CtoF77(x##trace_fini) (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_fini_Wrapper ();\
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_FINI)

#define apiTRACE_EVENT(x) \
	void x##trace_event (unsigned int tipus, unsigned int valor) \
	{ \
		if (mpitrace_on) \
			MPItrace_Event_Wrapper (&tipus, &valor); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENT)

#define apiTRACE_NEVENT(x) \
	void x##trace_nevent (unsigned int count, unsigned int *tipus, unsigned int *valors) \
	{ \
		if (mpitrace_on) \
			MPItrace_N_Event_Wrapper (&count, tipus, valors); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENT)

#define apiTRACE_EVENTANDCOUNTERS(x) \
	void x##trace_eventandcounters (unsigned int tipus, unsigned int valor) \
	{ \
  		if (mpitrace_on) \
    			MPItrace_Eventandcounters_Wrapper (&tipus, &valor); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_EVENTANDCOUNTERS)

#define apiTRACE_NEVENTANDCOUNTERS(x) \
	void x##trace_neventandcounters (unsigned int count, unsigned int *tipus, unsigned int *valors) \
	{ \
  		if (mpitrace_on) \
    			MPItrace_N_Eventsandcounters_Wrapper (&count, tipus, valors); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEVENTANDCOUNTERS)

#define apiTRACE_SHUTDOWN(x) \
	void x##trace_shutdown (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_shutdown_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SHUTDOWN)

#define apiTRACE_RESTART(x) \
	void x##trace_restart (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_restart_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_RESTART);

#define apiTRACE_COUNTERS(x) \
	void x##trace_counters(void) \
	{ \
		if (mpitrace_on) \
			MPItrace_counters_Wrapper(); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_COUNTERS);

#define apiTRACE_PREVIOUS_HWC_SET(x) \
	void x##trace_previous_hwc_set (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_previous_hwc_set_Wrapper (); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_PREVIOUS_HWC_SET)

#define apiTRACE_NEXT_HWC_SET(x) \
	void x##trace_next_hwc_set (void) \
	{ \
		if (mpitrace_on) \
			MPItrace_next_hwc_set_Wrapper (); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NEXT_HWC_SET)

#define apiTRACE_SETOPTIONS(x) \
	void x##trace_set_options (int options) \
	{ \
		if (mpitrace_on) \
			MPItrace_set_options_Wrapper (options); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SETOPTIONS);

#define apiTRACE_USER_FUNCTION(x) \
	void x##trace_user_function (int enter) \
	{ \
		if (mpitrace_on) \
			MPItrace_user_function_Wrapper (enter); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_USER_FUNCTION);

#define apiTRACE_USER_FUNCTION_FROM_ADDRESS(x) \
	void x##trace_function_from_address (int type, void *address) \
	{ \
		if (mpitrace_on) \
			MPItrace_function_from_address_Wrapper (type, address); \
	}
EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_USER_FUNCTION_FROM_ADDRESS);

#if defined(PTHREAD_SUPPORT)
# define apiTRACE_NOTIFY_NEW_PTHREAD(x) \
	void x##trace_notify_new_pthread (void) \
	{ \
		if (mpitrace_on) \
			Backend_NotifyNewPthread (); \
	}
 EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_NOTIFY_NEW_PTHREAD);

# define apiTRACE_SET_NUM_TENTATIVE_THREADS(x) \
	void x##trace_set_num_tentative_threads (int numthreads) \
	{ \
		if (mpitrace_on) \
			Backend_setNumTentativeThreads (numthreads); \
	}
	EXPAND_ROUTINE_WITH_PREFIXES(apiTRACE_SET_NUM_TENTATIVE_THREADS);
#endif /* PTHREAD_SUPPORT */

#else /* HAVE_WEAK_ALIAS_ATTRIBUTE */

/** C BINDINGS **/

INTERFACE_ALIASES_C(trace_init,MPItrace_init,(void))
void MPItrace_init (void)
{
	MPItrace_init_Wrapper ();
}

INTERFACE_ALIASES_C(trace_fini,MPItrace_fini,(void))
void MPItrace_fini (void)
{
	if (mpitrace_on)
		MPItrace_fini_Wrapper ();
}

INTERFACE_ALIASES_C(trace_event, MPItrace_event, (unsigned int tipus, unsigned int valors))
void MPItrace_event (unsigned int tipus, unsigned int valors)
{
	if (mpitrace_on)
		MPItrace_Event_Wrapper (&tipus, &valors);
}

INTERFACE_ALIASES_C(trace_nevent, MPItrace_nevent, (unsigned int count, unsigned int *tipus, unsigned int *valors))
void MPItrace_nevent (unsigned int count, unsigned int *tipus, unsigned int *valors)
{
	if (mpitrace_on)
		MPItrace_N_Event_Wrapper (&count, tipus, valors);
}

INTERFACE_ALIASES_C(trace_eventandcounters, MPItrace_eventandcounters, (unsigned int tipus, unsigned int valor))
void MPItrace_eventandcounters (unsigned int tipus, unsigned int valor)
{
	if (mpitrace_on)
		MPItrace_Eventandcounters_Wrapper (&tipus, &valor);
}

INTERFACE_ALIASES_C(trace_neventandcounters, MPItrace_neventandcounters, (unsigned int count, unsigned int *tipus, unsigned int *valors))
void MPItrace_neventandcounters (unsigned int count, unsigned int *tipus, unsigned int *valors)
{
 	if (mpitrace_on)
		MPItrace_N_Eventsandcounters_Wrapper (&count, tipus, valors);
}

INTERFACE_ALIASES_C(trace_shutdown, MPItrace_shutdown, (void))
void MPItrace_shutdown (void)
{
	if (mpitrace_on)
		MPItrace_shutdown_Wrapper();
}

INTERFACE_ALIASES_C(trace_restart, MPItrace_restart, (void))
void MPItrace_restart (void)
{
	if (mpitrace_on)
		MPItrace_restart_Wrapper();
}

INTERFACE_ALIASES_C(trace_counters, MPItrace_counters, (void))
void MPItrace_counters(void)
{
	if (mpitrace_on)
		MPItrace_counters_Wrapper();
}

INTERFACE_ALIASES_C(trace_next_hwc_set, MPItrace_next_hwc_set, (void))
void MPItrace_next_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_next_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_C(trace_previous_hwc_set, MPItrace_previous_hwc_set, (void))
void MPItrace_previous_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_previous_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_C(trace_set_options, MPItrace_set_options, (int options))
void MPItrace_set_options (int options)
{
	if (mpitrace_on)
		MPItrace_set_options_Wrapper (options);
}

INTERFACE_ALIASES_C(trace_user_function, MPItrace_user_function, (int enter))
void MPItrace_user_function (int enter)
{
	if (mpitrace_on)
		MPItrace_user_function_Wrapper (enter);
}

INTERFACE_ALIASES_C(trace_function_from_address, MPItrace_function_from_address, (int type, void *address))
void MPItrace_function_from_address (int type, void *address)
{
	if (mpitrace_on)
		MPItrace_function_from_address_Wrapper (type, address);
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_C(trace_notify_new_pthread,MPItrace_notify_new_pthread, (void))
void MPItrace_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_C(trace_set_num_tentative_threads,MPItrace_set_num_tentative_threads, (int numthreads))
void MPItrace_set_num_tentative_threads (int numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (numthreads);
}
#endif

/** FORTRAN BINDINGS **/

INTERFACE_ALIASES_F(trace_init,TRACE_INIT,mpitrace_init,(void))
void mpitrace_init (void)
{
	MPItrace_init_Wrapper ();
}

INTERFACE_ALIASES_F(trace_fini,TRACE_FINI,mpitrace_fini,(void))
void mpitrace_fini (void)
{
	if (mpitrace_on)
		MPItrace_fini_Wrapper ();
}

INTERFACE_ALIASES_F(trace_event,TRACE_EVENT,mpitrace_event,(unsigned int *tipus, unsigned int *valor))
void mpitrace_event (unsigned int *tipus, unsigned int *valor)
{
	if (mpitrace_on)
		MPItrace_Event_Wrapper (tipus, valor);
}

INTERFACE_ALIASES_F(trace_nevent,TRACE_NEVENT,mpitrace_nevent,(unsigned int *count, unsigned int *tipus, unsigned int *valor))
void mpitrace_nevent (unsigned int *count, unsigned int *tipus, unsigned int *valor)
{
	if (mpitrace_on)
		MPItrace_N_Event_Wrapper (count, tipus, valor);
}

INTERFACE_ALIASES_F(trace_shutdown,TRACE_SHUTDOWN,mpitrace_shutdown,(void))
void mpitrace_shutdown (void)
{
	if (mpitrace_on)
		MPItrace_shutdown_Wrapper();
}

INTERFACE_ALIASES_F(trace_restart,TRACE_RESTART,mpitrace_restart,(void))
void mpitrace_restart (void)
{
	if (mpitrace_on)
		MPItrace_restart_Wrapper ();
}

INTERFACE_ALIASES_F(trace_eventandcounters,TRACE_EVENTANDCOUNTERS,mpitrace_eventandcounters, (unsigned int *tipus, unsigned int *valor))
void mpitrace_eventandcounters (unsigned int *tipus, unsigned int *valor)
{
	if (mpitrace_on)
		MPItrace_Eventandcounters_Wrapper (tipus, valor);
}

INTERFACE_ALIASES_F(trace_neventandcounters,TRACE_NEVENTANDCOUNTERS,mpitrace_neventandcounters, (unsigned int *count, unsigned int *tipus, unsigned *valor))
void mpitrace_neventandcounters (unsigned int *count, unsigned int *tipus, unsigned *valor)
{
	if (mpitrace_on)
		MPItrace_N_Eventsandcounters_Wrapper (count, tipus, valor);
}

INTERFACE_ALIASES_F(trace_counters,TRACE_COUNTERS,mpitrace_counters, (void))
void mpitrace_counters (void)
{
	if (mpitrace_on)
		MPItrace_counters_Wrapper ();
}

INTERFACE_ALIASES_F(trace_next_hwc_set, TRACE_NEXT_HWC_SET, mpitrace_next_hwc_set, (void))
void mpitrace_next_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_next_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_F(trace_previous_hwc_set, TRACE_PREVIOUS_HWC_SET, mpitrace_previous_hwc_set, (void))
void mpitrace_previous_hwc_set (void)
{
	if (mpitrace_on)
		MPItrace_previous_hwc_set_Wrapper ();
}

INTERFACE_ALIASES_F(trace_set_options,TRACE_SET_OPTIONS,mpitrace_set_options, (int *options))
void mpitrace_set_options (int *options)
{
	if (mpitrace_on)
		MPItrace_set_options_Wrapper (*options);
}

INTERFACE_ALIASES_F(trace_user_function,TRACE_USER_FUNCTION,mpitrace_user_function, (int *enter))
void mpitrace_user_function (int *enter)
{
	if (mpitrace_on)
		MPItrace_user_function_Wrapper (*enter);
}

INTERFACE_ALIASES_F(trace_function_from_address,TRACE_USER_FUNCTION_FROM_ADDRESS,mpitrace_function_from_address, (int *type, void *address))
void mpitrace_function_from_address (int *type, void *address)
{
	if (mpitrace_on)
		MPItrace_function_from_address_Wrapper (type, address);
}

#if defined(PTHREAD_SUPPORT)
INTERFACE_ALIASES_F(trace_notify_new_pthread,TRACE_NOTIFY_NEW_PTHREAD,mpitrace_notify_new_pthread, (void))
void mpitrace_notify_new_pthread (void)
{
	if (mpitrace_on)
		Backend_NotifyNewPthread ();
}

INTERFACE_ALIASES_F(trace_set_num_tentative_threads,TRACE_SET_NUM_TENTATIVE_THREADS,mpitrace_set_num_tentative_threads, (int *numthreads))
void mpitrace_set_num_tentative_threads (int *numthreads)
{
	if (mpitrace_on)
		Backend_setNumTentativeThreads (*numthreads);
}
#endif

#endif
