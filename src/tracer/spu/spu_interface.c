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

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "spu_wrapper.h"

extern int mpitrace_on;

void Extrae_event (unsigned int tipus, unsigned int valor)
{
  if (mpitrace_on)
    Trace_Event_C_Wrapper (tipus, valor);
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void SPUtrace_event (unsigned int tipus, unsigned int valor) __attribute__ ((alias ("Extrae_event")));
#endif

void Extrae_Nevent (int count, unsigned int *tipus, unsigned int *valors)
{
	if (mpitrace_on)
		Trace_MultipleEvent_C_Wrapper (count, tipus, valors);
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void SPUtrace_Nevent (int count, unsigned int *tipus, unsigned int *valors) __attribute__ ((alias ("Extrae_Nevent")));
#endif

void Extrae_shutdown (void)
{
  if (mpitrace_on)
    mpitrace_shutdown_Wrapper ();
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void SPUtrace_shutdown (void) __attribute__ ((alias ("Extrae_shutdown")));
#endif

void Extrae_restart (void)
{
  if (mpitrace_on)
    mpitrace_restart_Wrapper ();
}
#if defined(HAVE_ALIAS_ATTRIBUTE)
void SPUtrace_restart (void) __attribute__ ((alias ("Extrae_restart")));
#endif

