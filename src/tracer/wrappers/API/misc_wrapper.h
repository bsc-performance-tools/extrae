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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/wrappers/API/misc_wrapper.h,v $
 | 
 | @last_commit: $Date: 2009/01/05 16:14:54 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MISC_WRAPPER_DEFINED
#define MISC_WRAPPER_DEFINED

#include "clock.h"

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
void MPItrace_user_function_Wrapper (int enter);
void MPItrace_function_from_address_Wrapper (int type, void *address);

void MPItrace_next_hwc_set_Wrapper (void);
void MPItrace_previous_hwc_set_Wrapper (void);

void MPItrace_notify_new_pthread (void);

#endif
