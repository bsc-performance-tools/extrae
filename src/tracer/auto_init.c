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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.3/src/tracer/auto_init_fini.c $
 | @last_commit: $Date: 2012-10-19 12:52:48 +0200 (dv, 19 oct 2012) $
 | @version:     $Revision: 1276 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: auto_init_fini.c 1276 2012-10-19 10:52:48Z harald $";

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif

#include <misc_interface.h>

#include "auto_fini.h"

static int Extrae_automatically_loaded = FALSE;

__attribute__((constructor))
void Extrae_auto_library_init (void)
{
	if (!Extrae_automatically_loaded)
	{
		/* Do not automatically load if DynInst is orchestrating the tracing */
		if (getenv("EXTRAE_DYNINST_RUN") != NULL)
			if (strcmp (getenv("EXTRAE_DYNINST_RUN"), "yes") == 0)
				return;
		Extrae_init();
		Extrae_automatically_loaded = TRUE;

		/* We have experienced issues with __attribute__(destructor).
		   If it fails, give another chance to close instrumentation
		   through atexit(3) */
		atexit (Extrae_auto_library_fini);
	}
}

