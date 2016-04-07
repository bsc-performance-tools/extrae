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

#ifndef MPI2PRV_VECTOR_H_INCLUDED
#define MPI2PRV_VECTOR_H_INCLUDED

typedef struct mpi2prvvector_st
{
	unsigned long long *data;
	unsigned count;
	unsigned allocated;
} mpi2prv_vector_t;


/* Initialize vector, return the new structure */
mpi2prv_vector_t * Vector_Init (void);

/* Search within vec the element v, return TRUE if found, else FALSE */
int Vector_Search (mpi2prv_vector_t *vec, unsigned long long v);

/* Add v into the vector. We don't check for duplicates. */
void Vector_Add (mpi2prv_vector_t *vec, unsigned long long v);

/* Number of elements within the vector vec */
unsigned Vector_Count (mpi2prv_vector_t *vec);

#endif /* MPI2PRV_VECTOR_H_INCLUDED */
