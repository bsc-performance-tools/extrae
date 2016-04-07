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

#ifndef MPI2PRV_STACK_H_INCLUDED
#define MPI2PRV_STACK_H_INCLUDED

typedef struct mpi2prv_stack_st
{
	unsigned long long *data;
	unsigned count;
	unsigned allocated;
} mpi2prv_stack_t;

mpi2prv_stack_t * Stack_Init (void);
void Stack_Push (mpi2prv_stack_t *s, unsigned long long v);
void Stack_Pop (mpi2prv_stack_t *s);
unsigned Stack_Depth (mpi2prv_stack_t *s);
unsigned long long Stack_ValueAt (mpi2prv_stack_t *s, unsigned pos);
unsigned long long Stack_Top (mpi2prv_stack_t *s);

#endif /* MPI2PRV_STACK_H_INCLUDED */
