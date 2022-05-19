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

#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include "gnu_libgomp.h"
#include "omp_common.h"
#include "pdebug.h"

int xtr_OMP_tracing_config = 0;

/******************************************************************************\
 *                                                                            * 
 *                             INITIALIZATIONS                                * 
 *                                                                            * 
\******************************************************************************/

/**
 * Extrae_OpenMP_init
 *
 * Main initialization for the OpenMP instrumentation module.
 * Detects the runtime that is present (IBM, Intel, GNU...) and loads
 * specific hooks for the present runtime.
 * Also loads some common hooks for basic OpenMP routines available in
 * all runtimes.
 */
void Extrae_OpenMP_init(int me)
{
	UNREFERENCED_PARAMETER(me);

#if defined(PIC)
	int intel_hooked = FALSE;
	int gnu_hooked = FALSE;
	int hooked = 0;

// Support for Intel OpenMP not available for the new implementation of the GOMP runtime
# if 0 && defined(INTEL_OPENMP)
	intel_hooked = xtr_OMP_KMPC_init();
# endif /* INTEL_OPENMP */

# if defined(GNU_OPENMP)
	gnu_hooked = xtr_OMP_GOMP_init();
# endif /* GNU_OPENMP */

	hooked = intel_hooked + gnu_hooked;

	if (hooked > 0) 
	{
		MASTER_OUT("Detected and hooked OpenMP runtime:%s%s\n",
		           intel_hooked ? " [Intel KMPC]" : "",
		           gnu_hooked ? " [GNU GOMP]" : "");
	}
	else
	{
		MASTER_WARN("You have loaded an OpenMP tracing library but the application seems not to be linked with an OpenMP runtime. Did you compile the application with the proper flags (-fopenmp, -openmp...)?\n");
	}

#else  /* PIC */
	MASTER_WARN("OpenMP instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

