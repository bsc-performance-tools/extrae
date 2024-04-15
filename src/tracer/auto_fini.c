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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#include <misc_wrapper.h>
#include "auto_fini.h"

__attribute__((destructor))
void Extrae_auto_library_fini (void)
{
	/*
	 * In MN5 we observed that PMPI_Init calls fork and create processes that end before returning from the PMPI. 
	 * The forked processes also enter our destructor and try to finalize trace files that don't exist and crash. 
	 * The following check prevents the forked processes to enter the destructor.
	 * BEWARE! If we add instrumentation for fork + exec, this will be problematic. We will probably need to track
	 * the pid of the user's forks (disregarding MPI internals), and allow those to go through the destructor.
	 */
	if (pid_at_constructor == getpid())
	{
		Extrae_fini_Wrapper ();
	}
}

