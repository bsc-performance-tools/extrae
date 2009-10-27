/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_register_recv_queue_h_
#define _gm_register_recv_queue_h_

#error

#include "gm_internal.h"

void gm_deregister_recv_queue (struct gm_port_state *ps);
void gm_catch_recv_queue_deregistration (struct gm_port_state *ps, gm_up_t up);

#endif /* _gm_register_recv_queue_h_ */
