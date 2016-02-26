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

#include "wrapper.h"
#include "UF_gcc_instrument.h"

/* #define DEBUG */

/* Configure the hash so it uses up to 1 Mbyte */
#if SIZEOF_VOIDP == 8
# define MAX_UFs_l2  (17)
#elif SIZEOF_VOIDP == 4
# define MAX_UFs_l2  (18)
#else
# error "Error! Unknown SIZEOF_VOIDP value!"
#endif

#define MAX_UFs      (1<<MAX_UFs_l2)
#define MAX_UFs_mask ((1<<MAX_UFs_l2)-1)
#define UF_lookahead (64)

static void *UF_addresses[MAX_UFs];
static unsigned int UF_collisions, UF_count, UF_distance;

#if SIZEOF_VOIDP == 8
# define HASH(address) ((address>>3) & MAX_UFs_mask)
#elif SIZEOF_VOIDP == 4
# define HASH(address) ((address>>2) & MAX_UFs_mask)
#endif

static int UF_tracing_enabled = FALSE;
static int LookForUFaddress (void *address);

/***
  __cyg_profile_func_enter, __cyg_profile_func_exit
  these routines are callback functions to instrument routines which have
  been compiled with -finstrument-functions (GCC)
***/

void __cyg_profile_func_enter (void *this_fn, void *call_site)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG TID %d __cyg_profile_func_enter (%p, %p)\n", THREADID, this_fn, call_site);
#else
	UNREFERENCED_PARAMETER (call_site);
#endif

	if (mpitrace_on && UF_tracing_enabled)
	{
		if (LookForUFaddress (this_fn))
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": DEBUG TID %d LookForUFaddress (%p) == TRUE\n", THREADID, this_fn);
#endif
			TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, (uintptr_t) this_fn, TRACING_HWC_UF);
		}
		else
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": DEBUG TID %d LookForUFaddress (%p) == FALSE\n", THREADID, this_fn);
#endif
		}
	}
}

void __cyg_profile_func_exit (void *this_fn, void *call_site)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG TID %d __cyg_profile_func_exit (%p, %p)\n", THREADID, this_fn, call_site);
#else
	UNREFERENCED_PARAMETER (call_site);
#endif

	if (mpitrace_on && UF_tracing_enabled)
	{
		if (LookForUFaddress (this_fn))
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": DEBUG TID %d LookForUFaddress (%p) == TRUE\n", THREADID, this_fn);
#endif
			TRACE_EVENTANDCOUNTERS (TIME, USRFUNC_EV, EVT_END, TRACING_HWC_UF);
		}
		else
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": DEBUG TID %d LookForUFaddress (%p) == FALSE\n", THREADID, this_fn);
#endif
		}
	}
}

static void AddUFtoInstrument (void *address)
{
	int i = HASH((long)address);

	if (UF_addresses[i] == NULL)
	{
		UF_addresses[i] = address;
		UF_count++;
	}
	else
	{
		int count = 1;
		while (UF_addresses[(i+count)%MAX_UFs] != NULL && count < UF_lookahead)
			count++;

		if (UF_addresses[(i+count)%MAX_UFs] == NULL)
		{
			UF_addresses[(i+count)%MAX_UFs] = address;
			UF_collisions++;
			UF_count++;
			UF_distance += count;
		}
		else
			fprintf (stderr, PACKAGE_NAME": Cannot add UF %p\n", address);
	}
}

static int LookForUFaddress (void *address)
{
	int i = HASH((long)address);
	int count = 0;

	while (UF_addresses[(i+count)%MAX_UFs] != address && 
		UF_addresses[(i+count)%MAX_UFs] != NULL &&
		count < UF_lookahead)
	{
		count++;
	}

	return UF_addresses[(i+count)%MAX_UFs] == address;
}

static void ResetUFtoInstrument (void)
{
	int i;
	for (i = 0; i < MAX_UFs; i++)
		UF_addresses[i] = NULL;
	UF_distance = UF_count = UF_collisions = 0;
}

void InstrumentUFroutines_GCC_CleanUp (void)
{
}

void InstrumentUFroutines_GCC (int rank, char *filename)
{
	FILE *f = fopen (filename, "r");
	if (f != NULL)
	{
		char buffer[1024], fname[1024];
		unsigned long address;

		ResetUFtoInstrument ();

		if (fgets (buffer, sizeof(buffer), f) != NULL)
			while (!feof(f))
			{
				if (sscanf (buffer, "%lx # %s", &address, fname) == 2)
					AddUFtoInstrument ((void*) address);
				if (fgets (buffer, sizeof(buffer), f) == NULL)
					break;
			}
		fclose (f);

		if (rank == 0)
		{
			if (UF_collisions > 0)
				fprintf (stdout, PACKAGE_NAME": Number of user functions traced (GCC runtime): %u (collisions: %u, avg distance = %u)\n",
    	    UF_count, UF_collisions, UF_distance/UF_collisions);
			else
				fprintf (stdout, PACKAGE_NAME": Number of user functions traced (GCC runtime): %u\n", UF_count);
		}
	}
	else
	{
		if (strlen(filename) > 0 && rank == 0)
			fprintf (stderr, PACKAGE_NAME": Warning! Cannot open %s file\n", filename);
	}

	if (UF_count > 0)
		UF_tracing_enabled = TRUE;
}

