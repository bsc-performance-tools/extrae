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

#ifndef _NEW_QUEUE_H_
#define _NEW_QUEUE_H_

#include <config.h>
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

typedef struct {
	void *Data;
	size_t SizeOfElement;
	int NumOfElements;
	int ElementsPerAllocation;
	int ElementsAllocated;
} NewQueue_t;

NewQueue_t * NewQueue_create (size_t SizeOfElement, int ElementsPerAllocation);
void NewQueue_clear (NewQueue_t *q);
void NewQueue_add (NewQueue_t *q, void *data);
void* NewQueue_search (NewQueue_t *q, void *reference, int(*compare)(void *, void*));
void NewQueue_delete (NewQueue_t *q, void *data);
void NewQueue_dump(NewQueue_t *q, void(*printer)(void *));

#endif

