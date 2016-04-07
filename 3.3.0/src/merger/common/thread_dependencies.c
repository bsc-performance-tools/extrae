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
#include "thread_dependencies.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

struct ThreadDependency_st
{
	void *predecessor_data;
	void *dependency_data;
	int in_use;
};

struct ThreadDependencies_st
{
	struct ThreadDependency_st *Dependencies;
	unsigned nDependencies; /* number of dependencies */
	unsigned aDependencies; /* number of allocated dependencies */
};

#define THREAD_DEPENDENCY_ALLOC_SIZE	256

struct ThreadDependencies_st * ThreadDependency_create (void)
{
	struct ThreadDependencies_st * td = (struct ThreadDependencies_st*)
	  malloc (sizeof(struct ThreadDependencies_st));
	if (NULL == td)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate memory to allocate thread dependencies!\n");
		exit (-1);
	}
	td->Dependencies = NULL;
	td->nDependencies = td->aDependencies = 0;
	return td;
}

void ThreadDependency_add (struct ThreadDependencies_st *td,
	const void *dependency_data)
{
	unsigned u;
	if (td->nDependencies == td->aDependencies)
	{
		td->Dependencies = (struct ThreadDependency_st*) realloc (td->Dependencies,
		  (td->aDependencies+THREAD_DEPENDENCY_ALLOC_SIZE)
		  *sizeof(struct ThreadDependency_st));
		if (NULL == td->Dependencies)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate memory to allocate thread dependencies!\n");
			exit (-1);
		}

		for (u = td->aDependencies;
		     u < td->aDependencies+THREAD_DEPENDENCY_ALLOC_SIZE;
		     u++)
			td->Dependencies[u].in_use = FALSE;
		td->aDependencies += THREAD_DEPENDENCY_ALLOC_SIZE;
	}

	for (u = 0; u < td->aDependencies; u++)
		if (!td->Dependencies[u].in_use)
		{
			td->Dependencies[u].dependency_data = dependency_data;
			td->Dependencies[u].predecessor_data = NULL;
			td->Dependencies[u].in_use = TRUE;
			td->nDependencies++;
			break;
		}
}

void ThreadDependency_processAll_ifMatchDelete (struct ThreadDependencies_st *td,
	ThreadDepedendencyProcessor_ifMatchDelete cb, const void *userdata)
{
	unsigned u;
	for (u = 0; u < td->aDependencies; u++)
	{
		if (td->Dependencies[u].in_use && td->Dependencies[u].predecessor_data != NULL)
			if (cb (td->Dependencies[u].dependency_data,
			        td->Dependencies[u].predecessor_data, userdata))
			{
				td->Dependencies[u].in_use = FALSE;
				if (td->Dependencies[u].predecessor_data)
					free (td->Dependencies[u].predecessor_data);
				td->Dependencies[u].predecessor_data = NULL;
				td->nDependencies--;
			}
	}
}

void ThreadDependency_processAll_ifMatchSetPredecessor (struct ThreadDependencies_st *td,
	ThreadDepedendencyProcessor_ifMatchSetPredecessor cb, void *user_data)
{
	unsigned u;
	for (u = 0; u < td->aDependencies; u++)
		if (td->Dependencies[u].in_use)
		{
			void *pdata = NULL;
			if (cb (td->Dependencies[u].dependency_data, user_data, &pdata))
				td->Dependencies[u].predecessor_data = pdata;
		}
}
