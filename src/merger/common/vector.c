
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
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "vector.h"

#define ALLOC_SIZE 32

mpi2prv_vector_t * Vector_Init (void)
{
	mpi2prv_vector_t *tmp = (mpi2prv_vector_t*) malloc (sizeof(mpi2prv_vector_t));

	if (tmp == NULL)
	{
		fprintf (stderr, "mpi2prv: Error! Cannot allocate memory for vector!\n");
		exit (0);
	}

	tmp->count = tmp->allocated = 0;
	tmp->data = NULL;

	return tmp;
}

int Vector_Search (mpi2prv_vector_t *vec, unsigned long long v)
{
	unsigned u;

	for (u = 0; u < vec->count; u++)
		if (vec->data[u] == v)
			return TRUE;

	return FALSE;
}

void Vector_Add (mpi2prv_vector_t *vec, unsigned long long v)
{
	if (!Vector_Search(vec, v))
	{
		if (vec->data == NULL || vec->count+1 >= vec->allocated)
		{
			vec->data = realloc (vec->data, (vec->allocated + ALLOC_SIZE)*sizeof(unsigned long long));
			if (vec->data == NULL)
			{
				fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for vector!\n");
				exit (0);
			}
			vec->allocated += ALLOC_SIZE;
		}
		vec->data[vec->count] = v;
		vec->count++;
	}
}

unsigned Vector_Count (mpi2prv_vector_t *vec)
{
	return vec->count;
}
