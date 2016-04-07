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

#ifndef _EXTRAE_VECTOR_H_

#define _EXTRAE_VECTOR_H_

typedef struct Extrae_Vector_st
{
	void **data;
	unsigned count;
	unsigned allocated;
} Extrae_Vector_t;

/* Initialize vector structure */
void Extrae_Vector_Init (Extrae_Vector_t *v);

/* Destroy vector structure */
void Extrae_Vector_Destroy (Extrae_Vector_t *v);

/* Add a new element to the structure */
void Extrae_Vector_Append (Extrae_Vector_t *v, void *element);

/* Get the number of elements within the structure */
unsigned Extrae_Vector_Count (Extrae_Vector_t *v);

/* Get the element at requested position */
void * Extrae_Vector_Get (Extrae_Vector_t *v, unsigned position);

/* Search for a given element which is found via callback */
int Extrae_Vector_Search (Extrae_Vector_t *v, const void *element,
	int(*comparison)(const void *, const void *));

#endif /* _EXTRAE_VECTOR_H_ */

