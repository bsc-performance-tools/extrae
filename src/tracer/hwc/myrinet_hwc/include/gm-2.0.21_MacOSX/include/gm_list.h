/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_hist_h_
#define _gm_hist_h_

/* set this to perform costly linear-time list debugging */
#define GM_DEBUG_LISTS 0

/****************************************************************
 * List checking macros
 ****************************************************************/

/* Cause a failed assertion if the list has a cycle */

#if !GM_DEBUG_LISTS
#define GM_ASSERT_NO_CYCLE(type, _list, next)
#else /* GM_DEBUG_LISTS */
#define GM_ASSERT_NO_CYCLE(type, _list, next) do {			\
  type *GM_ASSERT_NO_CYCLE_a, *GM_ASSERT_NO_CYCLE_b;			\
									\
  GM_ASSERT_NO_CYCLE_a = GM_ASSERT_NO_CYCLE_b = (_list);		\
									\
  /* prevent loop when not debugging. */				\
  if (!GM_DEBUG_LISTS)							\
    break;								\
									\
  if (!GM_ASSERT_NO_CYCLE_a)						\
    break;								\
  while (1)								\
    {									\
      GM_ASSERT_NO_CYCLE_b = GM_ASSERT_NO_CYCLE_b->next;		\
      if (!GM_ASSERT_NO_CYCLE_b)					\
	break;								\
      gm_always_assert (GM_ASSERT_NO_CYCLE_b != GM_ASSERT_NO_CYCLE_a);	\
									\
      GM_ASSERT_NO_CYCLE_b = GM_ASSERT_NO_CYCLE_b->next;		\
      if (!GM_ASSERT_NO_CYCLE_b)					\
	break;								\
      gm_always_assert (GM_ASSERT_NO_CYCLE_b != GM_ASSERT_NO_CYCLE_a);	\
									\
      GM_ASSERT_NO_CYCLE_a = GM_ASSERT_NO_CYCLE_a->next;		\
    }									\
} while (0)
#endif /* GM_DEBUG_LISTS */

/* Cause a failed assertion if OBJECT is in LIST */

#if !GM_DEBUG_LISTS
#define GM_ASSERT_NOT_IN_LIST(type, _list, _object, next)
#else /* GM_DEBUG_LISTS */
#define GM_ASSERT_NOT_IN_LIST(type, _list, _object, next) do {		\
  type *GM_ASSERT_NOT_IN_LIST_list;					\
  type *GM_ASSERT_NOT_IN_LIST_object;					\
									\
  GM_ASSERT_NOT_IN_LIST_list = (_list);					\
  GM_ASSERT_NOT_IN_LIST_object = (_object);				\
									\
  /* prevent loop when not debugging. */				\
  if (!GM_DEBUG_LISTS)							\
    break;								\
									\
  GM_ASSERT_NO_CYCLE (type, GM_ASSERT_NOT_IN_LIST_list, next);		\
									\
  while (GM_ASSERT_NOT_IN_LIST_list)					\
    {									\
      gm_always_assert (GM_ASSERT_NOT_IN_LIST_list			\
		       != GM_ASSERT_NOT_IN_LIST_object);		\
      GM_ASSERT_NOT_IN_LIST_list = GM_ASSERT_NOT_IN_LIST_list->next;	\
    }									\
} while (0)
#endif /* GM_DEBUG_LISTS */

/* Cause a failed assertion if OBJECT is not in LIST */

#if !GM_DEBUG_LISTS
#define GM_ASSERT_IN_LIST(type, _list, _object, next)
#else /* GM_DEBUG_LISTS */
#define GM_ASSERT_IN_LIST(type, _list, _object, next) do {		\
  type *GM_ASSERT_IN_LIST_list;						\
  type *GM_ASSERT_IN_LIST_object;					\
									\
  GM_ASSERT_IN_LIST_list = (_list);					\
  GM_ASSERT_IN_LIST_object = (_object);					\
									\
  /* prevent loop when not debugging. */				\
  if (!GM_DEBUG_LISTS)							\
    break;								\
									\
  GM_ASSERT_NO_CYCLE (type, GM_ASSERT_IN_LIST_list, next);		\
									\
  while (GM_ASSERT_IN_LIST_list)					\
    {									\
      gm_always_assert (GM_ASSERT_IN_LIST_list);			\
      GM_ASSERT_IN_LIST_list = GM_ASSERT_IN_LIST_list->next;		\
    }									\
} while (0)
#endif /* GM_DEBUG_LISTS */

#if !GM_DEBUG_LISTS
#define GM_PRINT_LIST_CNT(type, _list, next, _name)
#else /* GM_DEBUG_LISTS */
#define GM_PRINT_LIST_CNT(type, _list, next, _name) do {		\
  type *GM_DEBUG_LIST_CNT_list;						\
  const char *GM_DEBUG_LIST_CNT_name;					\
  unsigned int GM_DEBUG_LIST_CNT_cnt = 0;				\
									\
  GM_DEBUG_LIST_CNT_list = (_list);					\
  GM_DEBUG_LIST_CNT_name = (_name);					\
									\
  /* prevent loop when not debugging */					\
  if (!GM_DEBUG_LISTS)							\
    break;								\
									\
  GM_ASSERT_NO_CYCLE (type, _list, next);				\
  while (GM_DEBUG_LIST_CNT_list)					\
    {									\
      GM_DEBUG_LIST_CNT_cnt++;						\
      GM_DEBUG_LIST_CNT_list = GM_DEBUG_LIST_CNT_list->next;		\
    }									\
  GM_PRINT (GM_DEBUG_LISTS,						\
	    ("list \"%s\" has %u entries\n",				\
	     GM_DEBUG_LIST_CNT_name, GM_DEBUG_LIST_CNT_cnt));		\
} while (0)
#endif /* GM_DEBUG_LISTS */

/* Remove "_entry" from a linked list starting at "first" of elements
   of "type" linked with the field "next". */

#if !GM_DEBUG_LISTS
#define GM_LIST_REMOVE(typ, next, first, _entry)
#else /* GM_DEBUG_LISTS */
#define GM_LIST_REMOVE(typ, next, first, _entry) do {			\
  typ **GM_LIST_REMOVE_where;						\
  typ *GM_LIST_REMOVE_entry;						\
									\
  GM_LIST_REMOVE_entry = (_entry);					\
  for (GM_LIST_REMOVE_where = (first);					\
       *GM_LIST_REMOVE_where;						\
       GM_LIST_REMOVE_where = &(*GM_LIST_REMOVE_where)->next)		\
    {									\
      if (*GM_LIST_REMOVE_where == GM_LIST_REMOVE_entry)		\
	{								\
	  *GM_LIST_REMOVE_where = GM_LIST_REMOVE_entry->next;		\
	  break;							\
	}								\
    }									\
} while (0)
#endif /* GM_DEBUG_LISTS */

#endif /* _gm_list_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
