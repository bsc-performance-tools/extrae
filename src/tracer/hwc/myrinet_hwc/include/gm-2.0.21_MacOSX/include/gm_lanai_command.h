/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_lanai_command_h_
#define _gm_lanai_command_h_

#include "gm.h"
#include "gm_ether.h"

#define GM_MAX_LANAI_DEREG_LEN (4096 * GM_PAGE_LEN)

enum gm_lanai_command {
  GM_LANAI_COMMAND_NONE,
  
  GM_LANAI_COMMAND_CLOSE,
  GM_LANAI_COMMAND_DEREGISTER,
  GM_LANAI_COMMAND_OPEN,
  GM_LANAI_COMMAND_PAUSE,
  GM_LANAI_COMMAND_SET_ETHERNET_MAC,
  GM_LANAI_COMMAND_SET_ROUTE,
  GM_LANAI_COMMAND_UNCACHE_PORT_PTES,
  GM_LANAI_COMMAND_UNPAUSE,
  GM_LANAI_COMMAND_YP,
  GM_LANAI_COMMAND_YP_CANCEL,
  GM_LANAI_COMMAND_CLEAR_COUNTERS
};

struct gm_instance_state;
struct gm_port_state;

gm_status_t gm_lanai_command (struct gm_instance_state *is,
			      enum gm_lanai_command cmd);
gm_status_t gm_lanai_command_close (struct gm_instance_state *is,
				    unsigned int port_num);
gm_status_t gm_lanai_command_deregister (struct gm_instance_state *is,
					 unsigned int port_num,
					 gm_up_t uvma, gm_up_t len);
gm_status_t gm_lanai_command_open (struct gm_instance_state *is,
				   unsigned int port_num);
gm_status_t gm_lanai_command_set_ethernet_mac (struct gm_instance_state *is,
					       gm_ethernet_mac_addr_t mac);
gm_status_t gm_lanai_command_set_route (struct gm_instance_state *is,
					gm_u32_t global_id,
					gm_u8_n_t route[],
					unsigned int route_len);
gm_status_t gm_lanai_command_uncache_port_ptes (struct gm_instance_state *is,
						unsigned int port_num);
gm_status_t gm_lanai_command_yp (struct gm_port_state *ps,
				 unsigned int timeout_usecs,
				 const char *key,
				 const char *value,
				 unsigned int *node_id,
				 char (*answer)[GM_MAX_YP_STRING_LEN]);

#endif /* _gm_lanai_command_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
