/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file implements the GM structure locking interface. */

#ifndef _gm_struct_lock_h_
#define _gm_struct_lock_h_

#include "gm.h"
#include "gm_cpp.h"

/****************************************************************
 * Interface
 ****************************************************************/

#if 1
#define GM_STRUCT_LOCK_DECL
#define GM_STRUCT_LOCK(s)
#define GM_STRUCT_UNLOCK(s)
#else  /* 0 */

/* Append this to the struct you want to check.  Only the bytes before
   the lock will be checked. */

#define GM_STRUCT_LOCK_DECL unsigned long __gm_struct_lock;

/* Lock a struct.  This simply stores a CRC for the protected data. */

#define GM_STRUCT_LOCK(s) do {				\
  if (GM_DEBUG)						\
    {							\
      (s).__gm_struct_lock = __GM_STRUCT_CRC (s);	\
    }							\
} while (0)

/* Unlock a struct.  This simply verifies the lock CRC. */

#define GM_STRUCT_UNLOCK(s) do {					\
  if (GM_DEBUG								\
      && (s).__gm_struct_lock != __GM_STRUCT_CRC (s))			\
    {									\
      __gm_struct_lock_report_client_corruption				\
	(__GM_WHERE__, (void *) &(s), &(s).__gm_struct_lock + 1);	\
    }									\
} while (0)

#endif /* 0 */

/****************************************************************
 * Support Macros
 ****************************************************************/

/* Compute CRC over protected data in struct S. */

#define __GM_STRUCT_CRC(s)						\
  gm_crc (&(s), (char *) &(s).__gm_struct_lock - (char *) &(s))

/****************************************************************
 * Prototypes
 ****************************************************************/

void __gm_struct_lock_report_client_corruption (char *where, void *start,
						void *limit);

#endif /* _gm_struct_lock_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
