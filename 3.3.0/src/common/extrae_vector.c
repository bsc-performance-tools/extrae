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
#include "extrae_vector.h"

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "debug.h"

#define EXTRAE_VECTOR_ALLOC_SIZE 32

void Extrae_Vector_Init (Extrae_Vector_t *v)
{
	v->data = NULL;
	v->count = v->allocated = 0;
}

void Extrae_Vector_Destroy (Extrae_Vector_t *v)
{
	if (v->data != NULL)
		free (v->data);
	v->data = NULL;
	v->count = v->allocated = 0;
}

void Extrae_Vector_Append (Extrae_Vector_t *v, void *element)
{
	if (v->count == v->allocated)
	{
		v->data = (void**) realloc (
		  v->data, (v->allocated+EXTRAE_VECTOR_ALLOC_SIZE)*sizeof(void*));
		if (v->data == NULL)
		{
			fprintf (stderr, "Extrae (%s,%d): Fatal error! Cannot allocate memory for Extrae_Vector_Append\n", __FILE__, __LINE__);
			exit (-1);
		}
		v->allocated += EXTRAE_VECTOR_ALLOC_SIZE;
	}
	v->data[v->count] = element;
	v->count++;
}

unsigned Extrae_Vector_Count (Extrae_Vector_t *v)
{
	return v->count;
}

void * Extrae_Vector_Get (Extrae_Vector_t *v, unsigned position)
{
	ASSERT(position<v->count, "Out Of Bounds access to Extrae_Vector_Get")
	return v->data[position];
}

int Extrae_Vector_Search (Extrae_Vector_t *v, const void *element,
	int(*comparison)(const void *, const void *))
{
	unsigned u;
	for (u = 0; u < v->count; u++)
		if (comparison (element, v->data[u]))
			return TRUE;
	return FALSE;
}
