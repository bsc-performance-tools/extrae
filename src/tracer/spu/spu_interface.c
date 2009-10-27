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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/spu/spu_interface.c,v $
 | 
 | @last_commit: $Date: 2009/01/12 16:13:21 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: spu_interface.c,v 1.3 2009/01/12 16:13:21 gllort Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "spu_wrapper.h"

extern int mpitrace_on;

void SPUtrace_event (unsigned int tipus, unsigned int valor)
{
  if (mpitrace_on)
    Trace_Event_C_Wrapper (tipus, valor);
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void MPItrace_event (unsigned int tipus, unsigned int valor) __attribute__ ((alias ("SPUtrace_event")));
void MPItrace_event (unsigned int tipus, unsigned int valor) __attribute__ ((deprecated));
#endif

void SPUtrace_Nevent (int count, unsigned int *tipus, unsigned int *valors)
{
	if (mpitrace_on)
		Trace_MultipleEvent_C_Wrapper (count, tipus, valors);
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void MPItrace_Nevent (int count, unsigned int *tipus, unsigned int *valors) __attribute__ ((alias ("SPUtrace_Nevent")));
void MPItrace_Nevent (int count, unsigned int *tipus, unsigned int *valors) __attribute__ ((deprecated));
#endif

void SPUtrace_shutdown (void)
{
  if (mpitrace_on)
    mpitrace_shutdown_Wrapper ();
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void MPItrace_shutdown (void) __attribute__ ((alias ("SPUtrace_shutdown")));
void MPItrace_shutdown (void) __attribute__ ((deprecated));
#endif

void SPUtrace_restart (void)
{
  if (mpitrace_on)
    mpitrace_restart_Wrapper ();
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void MPItrace_restart (void) __attribute__ ((alias ("SPUtrace_restart")));
void MPItrace_restart (void) __attribute__ ((deprecated));
#endif

