
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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/wrappers/API/wrapper.c $
 | @last_commit: $Date: 2011-10-20 10:49:48 +0200 (dj, 20 oct 2011) $
 | @version:     $Revision: 795 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: wrapper.c 795 2011-10-20 08:49:48Z harald $";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "threadinfo.h"

static Extrae_thread_info_t *thread_info = NULL;

void Extrae_allocate_thread_info (unsigned nthreads)
{
	unsigned u;

	thread_info = (Extrae_thread_info_t*) realloc (thread_info, nthreads*sizeof (Extrae_thread_info_t));

	for (u = 0; u < nthreads; u++)
		Extrae_set_thread_name (u, "");
}

void Extrae_reallocate_thread_info (unsigned prevnthreads, unsigned nthreads)
{
	unsigned u;

	thread_info = (Extrae_thread_info_t*) realloc (thread_info, nthreads*sizeof (Extrae_thread_info_t));

	for (u = prevnthreads; u < nthreads; u++)
		Extrae_set_thread_name (u, "");
}

void Extrae_set_thread_name (unsigned thread, char *name)
{
	/* Clear space */
	memset (thread_info[thread].ThreadName, 0, THREAD_INFO_NAME_LEN);

	/* Copy name */
	snprintf (thread_info[thread].ThreadName, THREAD_INFO_NAME_LEN, name);

	/* Set last char to empty */
	thread_info[thread].ThreadName[THREAD_INFO_NAME_LEN-1] = (char) 0;
}

char *Extrae_get_thread_name (unsigned thread)
{
	return thread_info[thread].ThreadName;
}
