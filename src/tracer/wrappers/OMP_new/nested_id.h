#pragma once

#include "utils.h"
#include "xalloc.h"
// #include "omp_common.h"

/*** Defines ***/

#define OUT_OF_SCOPE -1
#define INFINITY     -8

// Possible values: 1 (nesting not supported), >1 (maximum number of levels supported), INFINITY (dynamic levels supported) 
#define XTR_MAX_NESTING_LEVEL  3 

/*** Types, structures & macros ***/

typedef struct xtr_nested_id_t xtr_nested_id_t;

/**
 * xtr_nested_id_t
 *
 * This structure contains a thread ID list per nesting level. 
 * (level == 0) => identifies the master thread & array 'id' is empty (no parallelism)
 * (level  > 0) => array 'id' contains the list of thread ID per level
 * (level  < 0) => thread belongs to a nesting level higher than supported
 * 
 * This is currently only used for debugging purposes.
 */
struct xtr_nested_id_t
{
#if XTR_MAX_NESTING_LEVEL == INFINITY
	int *id;
#else /* XTR_MAX_NESTING_LEVEL != INFINITY */
	int id[XTR_MAX_NESTING_LEVEL];
#endif /* XTR_MAX_NESTING_LEVEL */
	int level;
};

#if XTR_MAX_NESTING_LEVEL == INFINITY

// Nesting levels are allocated dynamically and we don't need to check for valid boundaries
#  define XTR_NESTED_ID_ALLOCATE(tid) tid.id = (int *)xmalloc(tid.level * sizeof(int))
#  define XTR_NESTED_ID_VALIDATE_SCOPE(tid)
#  define XTR_NESTED_ID_FREE(tid) xfree(tid.id)

#else /* XTR_MAX_NESTING_LEVEL != INFINITY */

// Nesting levels are allocated statically and we have to check for valid boundaries
#  define XTR_NESTED_ID_ALLOCATE(tid)
#  define XTR_NESTED_ID_VALIDATE_SCOPE(tid) if (tid.level > XTR_MAX_NESTING_LEVEL) { tid.level = OUT_OF_SCOPE; }
#  define XTR_NESTED_ID_FREE(tid) 

#endif /* XTR_MAX_NESTING_LEVEL */

/**
 * XTR_NESTED_ID_NEW
 *
 * Builds a new xtr_nested_id_t object.
 *
 * @param tid[out]            The current thread identifier.
 * @param get_level_fn        Callback to the runtime function to identify the current nesting level (e.g. int omp_get_level(void)).
 * @param get_id_per_level_fn Callback to the runtime function to identify the thread (e.g. int omp_get_ancestor_thread_num(int)).
 * @param ...                 Extra arguments that the function 'get_id_per_level_fn' may receive.
 */
# define XTR_NESTED_ID_NEW(tid, get_level_fn, get_id_per_level_fn, ...) \
{                                                                       \
  tid.level = get_level_fn();                                           \
  XTR_NESTED_ID_ALLOCATE(tid);                                          \
  XTR_NESTED_ID_VALIDATE_SCOPE(tid);                                    \
  for (int i=1; i<=tid.level; i++)                                      \
  {                                                                     \
    tid.id[i-1] = get_id_per_level_fn(i, ##__VA_ARGS__);                \
  }                                                                     \
} 

/**
 * XTR_NESTED_ID_INITIALIZER
 *
 * Static initializer for a xtr_nested_id_t object.
 * Master thread     => XTR_NESTED_ID_INITIALIZER(tid, 0)
 * 1st nested level  => XTR_NESTED_ID_INITIALIZER(tid, 1, 4)     // Thread 4 in level 1
 * 2nd+ nested level => XTR_NESTED_ID_INITIALIZER(tid, 2, 3, 5)) // Thread 3 in level 1, and 5 in level 2
 *
 * @param tid[out] The thread identifier.
 * @param nlevels  The number of nested levels.
 * @param ...      A list of thread identifiers for each of the levels specified with 'nlevels'.
 */
#define BRACKETIZE(...) {__VA_ARGS__}
#define XTR_NESTED_ID_INITIALIZER(tid, nlevels, ...)                    \
{                                                                       \
  int id[nlevels] = BRACKETIZE(__VA_ARGS__);                            \
  tid.level = nlevels;                                                  \
  XTR_NESTED_ID_ALLOCATE(tid);                                          \
  XTR_NESTED_ID_VALIDATE_SCOPE(tid);                                    \
  for (int i=0; i<tid.level; i++)                                       \
  {                                                                     \
    tid.id[i] = id[i];                                                  \
  }                                                                     \
}

/**
 * XTR_NESTED_ID_OUT_OF_SCOPE
 *
 * @param tid A valid thread identifier object 'xtr_nested_id_t'.
 * @return whether the nested level of the given thread is above the maximum supported.
 */
#define XTR_NESTED_ID_OUT_OF_SCOPE(tid) (tid.level < 0)

/**
 * XTR_NESTED_ID_CHILDREN_OUT_OF_SCOPE
 *
 * @param tid A valid thread identifier object 'xtr_nested_id_t'.
 * @return whether the children that the given thread may open would be above the maximum level of nested parallelism supported.
 */
#define XTR_NESTED_ID_CHILDREN_OUT_OF_SCOPE(tid)                                     \
  ( XTR_NESTED_ID_OUT_OF_SCOPE(tid) ||                                               \
    ((tid.level + 1 > XTR_MAX_NESTING_LEVEL) && (XTR_MAX_NESTING_LEVEL != INFINITY)) \
  )                                                                                  \

/**
 * XTR_NESTED_ID_IS_MASTER
 * 
 * @param tid A valid thread identifier object 'xtr_nested_id_t'.
 * @return whether the given thread is the master thread.
 */
#define XTR_NESTED_ID_IS_MASTER(tid) xtr_nested_id_is_master(&tid)


/*** Prototypes ***/

// Query methods for xtr_nested_id_t object (debug only)
int xtr_nested_id_is_master(xtr_nested_id_t *tid);
char * xtr_nested_id_tostr(xtr_nested_id_t *tid);

// Query methods for nesting level directly to the runtime
int I_am_master_in_nested();
int xtr_omp_get_level();