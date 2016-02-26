
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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "utils.h"
#include "threadinfo.h"

static Extrae_thread_info_t *thread_info = NULL;
static unsigned thread_info_nthreads = 0;

void Extrae_allocate_thread_CleanUp (void)
{
	xfree (thread_info);
}

void Extrae_allocate_thread_info (unsigned nthreads)
{
	unsigned u;

	thread_info = (Extrae_thread_info_t*) realloc (thread_info, nthreads*sizeof (Extrae_thread_info_t));

	for (u = 0; u < nthreads; u++)
		Extrae_set_thread_name (u, "");

	thread_info_nthreads = nthreads;
}

void Extrae_reallocate_thread_info (unsigned prevnthreads, unsigned nthreads)
{
	unsigned u;

	thread_info = (Extrae_thread_info_t*) realloc (thread_info, nthreads*sizeof (Extrae_thread_info_t));

	for (u = prevnthreads; u < nthreads; u++)
		Extrae_set_thread_name (u, "");

	thread_info_nthreads = nthreads;
}

void Extrae_set_thread_name (unsigned thread, char *name)
{
	size_t i;

	/* Clear space */
	memset (thread_info[thread].ThreadName, 0, THREAD_INFO_NAME_LEN);

	/* Copy name */
	snprintf (thread_info[thread].ThreadName, THREAD_INFO_NAME_LEN, "%s", name);

	/* Remove spaces */
	for (i = 0; i < strlen(thread_info[thread].ThreadName); i++)
		if (thread_info[thread].ThreadName[i] == ' ')
			thread_info[thread].ThreadName[i] = '_';

	/* Set last char to empty */
	thread_info[thread].ThreadName[THREAD_INFO_NAME_LEN-1] = (char) 0;
}

char *Extrae_get_thread_name (unsigned thread)
{
	return thread_info[thread].ThreadName;
}

unsigned Extrae_search_thread_name (char *name, int *found)
{
	unsigned u;

	*found = FALSE;
	for (u = 0; u < thread_info_nthreads; u++)
		if (strcmp (name, Extrae_get_thread_name(u)) == 0)
		{
			*found = TRUE;
			return u;
		}

	return 0;
}

