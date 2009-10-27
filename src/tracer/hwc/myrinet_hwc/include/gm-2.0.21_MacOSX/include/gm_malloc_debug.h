#ifndef gm_mdebug_h
#define gm_mdebug_h

#include "gm_debug_malloc.h"

#if !GM_DEBUG_MALLOC

#define GM_MDEBUG_RECORD_PTR(ptr,my_free) GM_SUCCESS
#define GM_MDEBUG_REMOVE_PTR(ptr,my_free) 

#else  /* GM_DEBUG_MALLOC */

#include "gm_compiler.h"	/* for GM_RETURN_ADDR() */

#define GM_MDEBUG_RECORD_PTR(ptr,my_free)				\
  _gm_mdebug_record_ptr ((void *) (ptr), #my_free, GM_CALLER (0))
#define GM_MDEBUG_REMOVE_PTR(ptr,my_free)				\
  _gm_mdebug_remove_ptr ((ptr), #my_free, GM_CALLER (0))

#endif /* GM_DEBUG_MALLOC */

GM_ENTRY_POINT gm_status_t _gm_mdebug_record_ptr (void *, char *, char *);
GM_ENTRY_POINT void _gm_mdebug_remove_ptr (void *, char *, char *);
void _gm_report_memory_leaks (void);
int _gm_allocated (void *ptr);
void *_gm_malloc (unsigned long len);
void _gm_free (void *ptr);

#endif				/* gm_mdebug_h */
