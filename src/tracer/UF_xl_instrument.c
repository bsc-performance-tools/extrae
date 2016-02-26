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

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "wrapper.h"
#include "utils.h"
#include "UF_gcc_instrument.h"

static int LookForUF (const char *fname);
static int UF_names_allocated = 0;
static int UF_names_count = 0;
static char **UF_names = NULL;

/***
  __func_trace_enter, __func_trace_exit
  these routines are callback functions to instrument routines which have
  been compiled with -finstrument-functions (GCC)
***/

void __func_trace_enter (const char *const function_name,
	const char *const file_name, int line_number, void **const user_data)
{
	UNREFERENCED_PARAMETER (file_name);
	UNREFERENCED_PARAMETER (line_number);
	UNREFERENCED_PARAMETER (user_data);

	if (mpitrace_on && UF_names_count > 0)
	{
		if (LookForUF (function_name))
		{
			UINT64 ip = Extrae_get_caller(3);
			TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, ip, tracejant_hwc_uf);
		}
	}
}

void __func_trace_exit (const char *const function_name,
	const char *const file_name, int line_number, void **const user_data)
{
	UNREFERENCED_PARAMETER (file_name);
	UNREFERENCED_PARAMETER (line_number);
	UNREFERENCED_PARAMETER (user_data);

	if (mpitrace_on && UF_names_count > 0)
	{
		if (LookForUF (function_name))
		{
			TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, 0, tracejant_hwc_uf);
		}
	}
}

static void AddUFtoInstrument (char *fname)
{
	if (UF_names_count == UF_names_allocated)
	{
		UF_names_allocated += 128;
		UF_names = (char**) realloc (UF_names, sizeof(char*)*UF_names_allocated);
		if (UF_names == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot reallocate UF_names buffer\n");
			exit (0);	
		}
	}

	UF_names[UF_names_count] = strdup (fname);
	if (UF_names[UF_names_count] == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot duplicate function name in AddUFtoInstrument\n");
		exit (0);	
	}
	UF_names_count++;
}

static int LookForUF (const char *fname)
{
	int i;

	for (i = 0; i < UF_names_count; i++)
		if (strcmp (UF_names[i], fname) == 0)
			return TRUE;

	return FALSE;
}

void InstrumentUFroutines_XL_CleanUp (void)
{
	int i;

	for (i = 0; i < UF_names_count; i++)
		xfree (UF_names[i]);
	xfree (UF_names);
}

void InstrumentUFroutines_XL (int rank, char *filename)
{
	FILE *f = fopen (filename, "r");
	if (f != NULL)
	{
		char buffer[1024];

		if (fgets (buffer, sizeof(buffer), f) != NULL)
			while (!feof(f))
			{
				if (strlen(buffer) > 1)
					buffer[strlen(buffer)-1] = (char) 0;
				AddUFtoInstrument (buffer);
				if (fgets (buffer, sizeof(buffer), f) == NULL)
					break;
			}
		fclose (f);

		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": Number of user functions traced (XL runtime): %u\n", UF_names_count);
	}
	else
	{
		if (strlen(filename) > 0 && rank == 0)
			fprintf (stderr, PACKAGE_NAME": Warning! Cannot open %s file\n", filename);
	}
}

