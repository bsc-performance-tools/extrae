#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "nested_id.h"
#include "pdebug.h"

/**
 * xtr_nested_id_is_master
 * 
 * @returns whether the given thread identifier 'tid' corresponds to the master thread (id 0) in all nested levels.
 */
int xtr_nested_id_is_master(xtr_nested_id_t *tid)
{
	int i = 0, id = 0;

	if (tid->level == OUT_OF_SCOPE) return 0;

	for (i = 0; i < tid->level; i ++)
	{
	    id += tid->id[i];
	}
	return (id == 0);
}


/**
 * xtr_nested_id_tostr
 *
 * @param tid The nested thread identifier.
 *
 * @returns a string in the form "%d-%d-%d..." indicating the thread id in each nesting level, 
 *          or an empty string if the level exceeds XTR_MAX_NESTING_LEVEL.
 */
char *xtr_nested_id_tostr(xtr_nested_id_t *tid)
{
	int  i = 0;
	char tid_str[1024];
	char id_str[512];

	tid_str[0] = id_str[0] = '\0';

	for (i = 0; i < tid->level; i ++)
	{
		char current_lvl_tid[16];
		snprintf(current_lvl_tid, sizeof(current_lvl_tid), "%s%d", (i == 0 ? "" : "-"), tid->id[i]);

		if (strlen(id_str) + strlen(current_lvl_tid) + 1 > sizeof(id_str))
		{
			id_str[strlen(id_str)]   = '.';
			id_str[strlen(id_str)-1] = '.';
			id_str[strlen(id_str)-2] = '.';
			break;
		}
		else strcat(id_str, current_lvl_tid);
	}
	snprintf(tid_str, sizeof(tid_str), "%s", id_str);
	return strdup(tid_str);
}


/* I_am_master_in_nested
 *
 * Checks for nesting levels 2 or higher, if the calling thread is 
 * always the master thread. Example for 3-levels of nested parallelism:
 *
 * Thread ID @ Lv1  Lv2  Lv3
 * -------------------------
 *              0 -> 0 -> 0 (traced)
 *              0 -> 0 -> 1 (not traced)
 *              0 -> 1 -> 0 (not traced)
 *              0 -> 1 -> 1 (not traced)
 *              1 -> 0 -> 0 (traced)
 *              1 -> 0 -> 1 (not traced)
 *              1 -> 1 -> 0 (not traced)
 *              1 -> 1 -> 1 (not traced)
 *                   ^^^^^^
 *                   trace only those that are 0 in all these levels, i.e.
 *                   always 0 at levels 2, 3, etc.
 */
int I_am_master_in_nested (void)
{
  int i = 0;

  for (i = omp_get_level(); i >= 2; i --)
  {
    if (omp_get_ancestor_thread_num(i) != 0) return 0;
  }
  return 1;
}


/**
 * xtr_omp_get_level
 *
 * @returns The current OpenMP nesting level as long as we don't exceed XTR_MAX_NESTING_LEVEL, halts the execution with an error otherwise.  
 */
int xtr_omp_get_level()
{
	int level = omp_get_level();

	if (((XTR_MAX_NESTING_LEVEL != INFINITY) && (level < 0)) || (level > XTR_MAX_NESTING_LEVEL))
	{
		THREAD_ERROR("Current nesting level (%d) is out of bounds (maximum supported is %d). "
		             "Please recompile "PACKAGE_NAME" increasing the value of XTR_MAX_NESTING_LEVEL "
		             "at src/tracer/wrappers/OMP/omp_common.h\n",
		             level, XTR_MAX_NESTING_LEVEL);
		exit (-1);
	}
	return level;
}

