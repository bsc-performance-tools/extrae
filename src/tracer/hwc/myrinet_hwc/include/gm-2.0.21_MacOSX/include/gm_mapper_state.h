/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2003 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_mapper_state_h_
#define _gm_mapper_state_h_

#define GM_MAPPER_STATE_FIELDS(_n_t)					\
  gm_u8 ## _n_t mapper_mac[6];						\
  gm_u8 ## _n_t pad_after_mac[2];					\
  gm_u32 ## _n_t map_version;						\
  gm_u32 ## _n_t num_hosts;						\
  gm_u32 ## _n_t network_configured;	/* Boolean */			\
  gm_u32 ## _n_t routes_valid; /* Boolean */				\
  gm_u32 ## _n_t level;	/* mapper level */				\
  gm_u32 ## _n_t flags;	/* mapper flags */

struct gm_mapper_state 
{
  GM_MAPPER_STATE_FIELDS (_t)
};
typedef struct gm_mapper_state gm_mapper_state_t;

struct gm_lanai_mapper_state
{
  GM_MAPPER_STATE_FIELDS (_n_t)
};
typedef struct gm_lanai_mapper_state gm_lanai_mapper_state_t;

#endif /* _gm_mapper_state_h_ */
