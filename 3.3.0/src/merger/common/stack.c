
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

#include "stack.h"

#define ALLOC_SIZE 32

mpi2prv_stack_t * Stack_Init (void)
{
	mpi2prv_stack_t *tmp = (mpi2prv_stack_t*) malloc (sizeof(mpi2prv_stack_t));

	if (tmp == NULL)
	{
		fprintf (stderr, "mpi2prv: Error! Cannot allocate memory for stack!\n");
		exit (0);
	}

	tmp->count = tmp->allocated = 0;
	tmp->data = NULL;

	return tmp;
}

void Stack_Push (mpi2prv_stack_t *s, unsigned long long v)
{
	if (s->data == NULL || s->count+1 >= s->allocated)
	{
		s->data = realloc (s->data, (s->allocated + ALLOC_SIZE)*sizeof(unsigned long long));
		if (s->data == NULL)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for stack!\n");
			exit (0);
		}
		s->allocated += ALLOC_SIZE;
	}

	s->data[s->count] = v;
	s->count++;
}

void Stack_Pop (mpi2prv_stack_t *s)
{
	if (s->count > 0)
	{
		s->count--;

		/* If we pop the whole stack, free the allocated memory */
		if (s->count == 0)
		{
			free (s->data);
			s->data = NULL;
			s->allocated = 0;
		}
	}
}

unsigned Stack_Depth (mpi2prv_stack_t *s)
{
	return s->count;
}

unsigned long long Stack_ValueAt (mpi2prv_stack_t *s, unsigned pos)
{
	if (pos < s->count)
		return s->data[pos];
	else
		return 0;
}

unsigned long long Stack_Top (mpi2prv_stack_t *s)
{
	return Stack_ValueAt (s, s->count-1);
}

