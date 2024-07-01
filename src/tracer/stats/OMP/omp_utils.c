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

#include "omp_utils.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_STACK_SIZE 8

/* -1 IS AN SPECIAL RETURN VALUE, THAT INDICATES THE STACK IS EMPTY, IT CANNOT BE STORED IN IT 
    THIS IS AN STACK OF UNSIGED INTEGERS */

// Data structure to represent a stack
struct stack
{
    int maxsize;  // capacity of the stack
    int top;
    unsigned int *items;
};
 
struct stack* newStack( void )
{
    struct stack *pt = (struct stack*)malloc(sizeof(struct stack));
 
    pt->maxsize = MAX_STACK_SIZE;
    pt->top = -1;
    pt->items = (unsigned int*)malloc(sizeof(unsigned int) * MAX_STACK_SIZE);
 
    return pt;
}

void deleteStack(struct stack *pt) {
  free(pt->items);
  free(pt);
}

int size(struct stack *pt) {
    return pt->top + 1;
}
 
int isEmpty(struct stack *pt) {
    return pt->top == -1;
}
 
int isFull(struct stack *pt) {
    return pt->top == pt->maxsize - 1;
}

void push(struct stack *pt, unsigned int x)
{
  if (isFull(pt))
  {
    pt->items = realloc(pt->items, pt->maxsize * 2);
  }

  pt->items[++pt->top] = x;
}
 
// Utility function to return the top element of the stack
unsigned int peek(struct stack *pt)
{
    // check for an empty stack
    if (!isEmpty(pt)) {
        return pt->items[pt->top];
    }

  return -1;
}

unsigned int pop(struct stack *pt)
{
    if (!isEmpty(pt))
    {
      return pt->items[pt->top--];
    }
  return -1;
}