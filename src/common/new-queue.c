/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/new-queue.c,v $
 | 
 | @last_commit: $Date: 2009/05/25 10:31:02 $
 | @version:     $Revision: 1.1 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: new-queue.c,v 1.1 2009/05/25 10:31:02 harald Exp $";

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

