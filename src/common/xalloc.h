/*H*****************************************************************************
* FILENAME : xmalloc.h             
*
* DESCRIPTION :
*     Memory allocation routines that bypass our instrumentation wrappers
*     and perform some extra checks on the results.
*
* PUBLIC FUNCTIONS :
*     void * xmalloc(size);
*     void * xmalloc_and_zero(size)
*     void * xrealloc(ptr, size);
*     void * xrealloc_and_zero(ptr, size);
*     void * xmemset(s, c, n);
*     void   xfree(ptr);
*
* NOTES :
*
* CHANGES :
*     DATE    WHO     DETAIL
*     03Apr20 GL      Initial implementation
*
*H*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*** Defines ***/

/**
 * xmalloc, xrealloc, xfree
 *
 * Depending on whether xalloc.c is compiled, uses our x-calls to bypass 
 * the instrumentation wrappers, or direct calls to the real symbols 
 * (that may go through our instrumentation wrappers) 
 */
#define xmalloc(size)                            \
({                                               \
  void *new_ptr = NULL;                          \
  if (_xmalloc) new_ptr = _xmalloc(size);        \
  else new_ptr = malloc(size);                   \
  if ((new_ptr == NULL) && (size > 0))           \
  {                                              \
    fprintf(stderr,"xmalloc: Virtual memory exhausted at %s (%s, %d)\n", __FUNCTION__, __FILE__, __LINE__);\
    perror("malloc");                            \
    exit(EXIT_FAILURE);                          \
  }                                              \
  new_ptr;                                       \
})

#define xrealloc(ptr, size)                      \
({                                               \
  void *new_ptr = NULL;                          \
  if (_xrealloc) new_ptr = _xrealloc(ptr, size); \
  else new_ptr = realloc(ptr, size);             \
  if ((new_ptr == NULL) && (size > 0))           \
  {                                              \
    fprintf(stderr,"xrealloc: Virtual memory exhausted at %s (%s, %d)\n", __FUNCTION__, __FILE__, __LINE__);\
    perror("realloc");                            \
    exit(EXIT_FAILURE);                          \
  }                                              \
  new_ptr;                                       \
})

#define xfree(ptr)                               \
{                                                \
  if (_xfree) _xfree(ptr);                       \
  else free(ptr);                                \
  ptr = NULL;                                    \
}

/**
 * xmemset, xmalloc_and_zero, xrealloc_and_zero
 *
 * Currently memset is not instrumented so no need a bypass wrapper for it 
 */
#define xmemset(s, c, n) memset(s, c, n)
#define xmalloc_and_zero(size) xmemset(xmalloc(size), 0, size)
#define xrealloc_and_zero(ptr, size) xmemset(xrealloc(ptr, size), 0, size)


/*** Prototypes ***/

void *_xmalloc(size_t size) __attribute__((weak));
void *_xrealloc(void *ptr, size_t size) __attribute__((weak));
void _xfree(void *) __attribute__((weak));

