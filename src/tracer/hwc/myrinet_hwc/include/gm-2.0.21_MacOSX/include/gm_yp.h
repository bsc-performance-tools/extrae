/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_yp_h_
#define _gm_yp_h_

#include "gm.h"
#include "gm_constants.h"
#include "gm_debug_yp.h"
#include "gm_mpd_types.h"
#include "gm_simple_types.h"

GM_TYPEDEF_LP_N_T (struct, gm_yp_port_delayed_query);

/* Per-port YP state.  Each port is allowed one YP query at a time. */

typedef struct gm_yp_query_context
{
  /* The time at which the current query should time out. */
  gm_s32_n_t query_timeout_time;
  /* A unique ID for the query in progress, or 0 if no query is in progress.
     This allows us to ignore duplicate replies. */
  gm_u32_n_t query_id;
  /* 8 */
  /* At the start of a query, this is the node ID to query, or 0 for
     broadcast queries.  After the query, this is the node ID that
     answered, if any. */
  gm_u32_n_t node_id;
  gm_u32_n_t next_node_id_for_broadcast;
  /* 8 */
  /* Before the query, this holds the concatenated key and value
     for the query, with a NULL character in between.  After the query,
     this holds the matching value. */
  gm_u8_n_t in_out[2 * GM_MAX_YP_STRING_LEN];
  /* After the query, this is the status of the query. */
  gm_u32_n_t status;
  /* 8 */
  /* Holder for the single query packet descriptor when not in use. */
  gm_myrinet_packet_descriptor_free_list_t free_queries;
  gm_yp_port_delayed_query_lp_t delayed_query;
} gm_yp_query_context_t;

/* Linked-list entry for our simple YP database. */

struct gm_yp_key_value
{
  struct gm_yp_key_value *next;
  const char *key;
  const char *value;
  int spliced;			/* spliced into the YP database */
};

/* Create a YP entry for key and value.  Value may point to a
   dynamicly updated string, but must be 8-byte aligned. */

#define GM_YP(_key, _value) do {					\
  static struct gm_yp_key_value kv;					\
  const char *GM_YP_key, *GM_YP_value;					\
  GM_YP_key = (_key);							\
  GM_YP_value = (_value);						\
									\
  gm_assert (gm_strlen (GM_YP_key) < GM_MAX_YP_STRING_LEN);		\
  gm_assert (gm_strlen (GM_YP_value) < GM_MAX_YP_STRING_LEN);		\
									\
  /* Update the key and value, even if initialized, to allow the	\
     values to be updated. */						\
  kv.key = GM_YP_key;							\
  kv.value = GM_YP_value;						\
									\
  /* Splice in the entry. */						\
  if (kv.spliced)							\
    break;								\
  kv.next = __gm_yp_first_key_value;					\
  __gm_yp_first_key_value = &kv;					\
  kv.spliced = 1;							\
} while (0)

extern struct gm_yp_key_value *__gm_yp_first_key_value;

struct gm_port_protected_lanai_side;

/* Initialization */

gm_status_t gm_yp_init (void);
gm_status_t gm_yp_port_init (struct gm_port_protected_lanai_side *port);

/* Start a YP query in gm_firmware_command.c */

gm_status_t gm_yp_port_start_query (struct gm_port_protected_lanai_side *port);
gm_status_t gm_yp_port_cancel_query (struct gm_port_protected_lanai_side
				     *port);

/* For gm_rdma.c */

void gm_yp_handle_query_packet (gm_myrinet_packet_descriptor_lp_t d);
void gm_yp_handle_reply_packet (gm_myrinet_packet_descriptor_lp_t d);

#endif /* _gm_yp_h_ */
