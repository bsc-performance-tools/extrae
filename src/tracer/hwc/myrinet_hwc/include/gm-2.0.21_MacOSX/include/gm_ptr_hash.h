/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#include "gm.h"

struct gm_ptr_hash;

struct gm_ptr_hash *gm_create_ptr_hash (gm_size_t min_cnt);
void gm_destroy_ptr_hash (struct gm_ptr_hash *hash);
gm_status_t gm_ptr_hash_insert (struct gm_ptr_hash *hash,
				void *key,
				void *data);
void *gm_ptr_hash_remove (struct gm_ptr_hash *hash, void *key);
void *gm_ptr_hash_find (struct gm_ptr_hash *hash, void *key);
const char *gm_ptr_hash_func_description (void);
