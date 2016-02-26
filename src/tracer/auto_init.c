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

#include <misc_interface.h>

#include "auto_fini.h"

static int Extrae_automatically_loaded = FALSE;

__attribute__((destructor))
void Gateway_to_Extrae_auto_library_fini (void)
{
  Extrae_auto_library_fini();
}

__attribute__((constructor))
void Extrae_auto_library_init (void)
{
	int skip_auto_library_init = FALSE;
	char *skip_envvar = getenv ("EXTRAE_SKIP_AUTO_LIBRARY_INITIALIZE");
	if (skip_envvar != NULL)
		if (strncasecmp (skip_envvar, "yes", strlen("yes")) == 0 ||
		    strncasecmp (skip_envvar, "true", strlen("true")) == 0 ||
		    strncmp (skip_envvar, "1", strlen ("1")) == 0)
			skip_auto_library_init = TRUE;

	if (!Extrae_automatically_loaded && !skip_auto_library_init)
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
		atexit (Gateway_to_Extrae_auto_library_fini);
	}
}

