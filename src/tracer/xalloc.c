#include "symptr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xalloc.h"

/*** Global variables ***/

// Pointers to the real malloc(), realloc(), and free() functions
static void *(*malloc_real)(size_t)          = NULL;
static void *(*realloc_real)(void *, size_t) = NULL;
static void  (*free_real)(void *)            = NULL;


/**
 * xalloc_init
 *
 * Initializes the pointers to the real symbols malloc(), realloc(), free(), etc.
 */
static void xalloc_init(void)
{
	// Retrieve the real pointers to from the hook table
	malloc_real  = XTR_FIND_SYMBOL_OR_DIE("malloc");
	realloc_real = XTR_FIND_SYMBOL_OR_DIE("realloc");
	free_real    = XTR_FIND_SYMBOL_OR_DIE("free");
}


/**
 * _xmalloc
 * 
 * Calls to real malloc() bypassing the instrumentation wrapper.
 * 
 * @param size Number of bytes to allocate.
 * @return a pointer to the allocated memory.
 */
void * _xmalloc(size_t size)
{
	if (!malloc_real) xalloc_init();

	return malloc_real(size);
}


/**
 * _xrealloc
 *
 * Calls to real realloc() bypassing the instrumentation wrapper.
 *
 * @param ptr Pointer to the memory block to change its size.
 * @param size New size in bytes of the memory block.
 * @return a pointer to the new memory block.
 */
void * _xrealloc(void *ptr, size_t size)
{
	if (!realloc_real) xalloc_init();
	    
	return realloc_real(ptr, size);
}


/**
 * _xfree
 * 
 * Calls to real free() bypassing the instrumentation wrapper.
 *
 * @param ptr Pointer to memory space returned by a previous call to malloc(), calloc(), or realloc()
 */
void _xfree(void *ptr)  
{
	if (!free_real) xalloc_init();
	
	free_real(ptr);
}

