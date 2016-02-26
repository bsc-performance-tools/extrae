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
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "new-queue.h"

NewQueue_t * NewQueue_create (size_t SizeOfElement, int ElementsPerAllocation)
{
	NewQueue_t *tmp;

#if defined(DEBUG)
	fprintf (stderr, "NewQueue_create (SizeOfElement = %d, ElementsPerAllocation = %d)\n", SizeOfElement, ElementsPerAllocation);
#endif

	tmp = (NewQueue_t*) malloc (sizeof(NewQueue_t));
	if (NULL == tmp)
	{
		fprintf (stderr, "mpi2prv: Failed to allocate the new queue!\n");
		exit (-1);
	}

	tmp->ElementsAllocated = 0;
	tmp->NumOfElements = 0;
	tmp->Data = NULL;
	tmp->SizeOfElement = SizeOfElement;
	tmp->ElementsPerAllocation = ElementsPerAllocation;

	return tmp;
}

void NewQueue_clear (NewQueue_t *q)
{
	if (q != NULL)
	{
		q->NumOfElements = 0;
	}
}

void NewQueue_add (NewQueue_t *q, void *data)
{
	size_t offset;

#if defined(DEBUG)
	fprintf (stderr, "NewQueue_add (q=%p, data=%p)\n", q, data);
#endif

	if (q->NumOfElements == q->ElementsAllocated)
	{
		q->Data = realloc (q->Data, (q->ElementsAllocated+q->ElementsPerAllocation) * q->SizeOfElement);
		if (NULL == q->Data)
		{
			fprintf (stderr, "mpi2prv: Failed to reallocate the new queue!\n");
			exit (-1);
		}
		q->ElementsAllocated = q->ElementsAllocated+q->ElementsPerAllocation;
	}

	offset = q->NumOfElements*q->SizeOfElement;
	memcpy ((((char*)q->Data)+offset), data, q->SizeOfElement);
	q->NumOfElements++;
}

void *NewQueue_search (NewQueue_t *q, void *reference, int(*compare)(void *, void*))
{
	int i = 0;
	size_t offset = 0;
	int found = FALSE;

#if defined(DEBUG)
	fprintf (stderr, "NewQueue_search (q=%p, reference=%p, compare=%p)\n", q, reference, compare);
#endif

	if (q->NumOfElements > 0)
	{
		while (i < q->NumOfElements)
		{
			found = (*compare) (reference, ((char*)q->Data)+offset);
			if (found)
				break;
			offset += q->SizeOfElement;
			i++;
		}
	}

	if (found)
		return ((char*)q->Data)+offset;
	else
		return NULL;
}

void NewQueue_delete (NewQueue_t *q, void *data)
{
	int i = 0;
	size_t offset = 0;
	int found = FALSE;

#if defined(DEBUG)
	fprintf (stderr, "NewQueue_delete (q=%p, data=%p)\n", q, data);
#endif

	if (q->NumOfElements > 0)
	{
		while (i < q->NumOfElements && !found)
		{
			found = (char*)data == ((char*)q->Data)+offset;
			if (found)
				break;
			offset += q->SizeOfElement;
			i++;
		}
	}

	if (found)
	{
		if (i < q->NumOfElements-1)
		{
			memcpy (((char*)q->Data)+offset, ((char*)q->Data)+offset+q->SizeOfElement, q->SizeOfElement);
			while (i < q->NumOfElements-1)
			{
				offset += q->SizeOfElement;
				i++;
				memcpy (((char*)q->Data)+offset, ((char*)q->Data)+offset+q->SizeOfElement, q->SizeOfElement);
			}
		}
	}
	q->NumOfElements--;
}

void NewQueue_dump(NewQueue_t *q, void(*printer)(void *))
{
        int i = 0;
        size_t offset = 0;

        if (q->NumOfElements > 0)
        {
                while (i < q->NumOfElements)
                {
                        (*printer) (((char*)q->Data)+offset);
                        offset += q->SizeOfElement;
                        i++;
                }
        }
}

