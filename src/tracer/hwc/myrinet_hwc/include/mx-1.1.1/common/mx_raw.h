/*************************************************************************
 * The contents of this file are subject to the MYRICOM MYRINET          *
 * EXPRESS (MX) NETWORKING SOFTWARE AND DOCUMENTATION LICENSE (the       *
 * "License"); User may not use this file except in compliance with the  *
 * License.  The full text of the License can found in LICENSE.TXT       *
 *                                                                       *
 * Software distributed under the License is distributed on an "AS IS"   *
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied.  See  *
 * the License for the specific language governing rights and            *
 * limitations under the License.                                        *
 *                                                                       *
 * Copyright 2005 by Myricom, Inc.  All rights reserved.                 *
 *************************************************************************/

#ifndef MX_RAW_H
#define MX_RAW_H

#include "mx_int.h"
#include "mx_io.h"

#define MX_RAW_NO_EVENT      0
#define MX_RAW_SEND_COMPLETE 1
#define MX_RAW_RECV_COMPLETE 2
						     

#ifdef __cplusplus
extern "C"
{
#if 0
}
#endif
#endif

typedef struct mx_raw_endpoint * mx_raw_endpoint_t;

/****************************************/
/* mx_raw_open_endpoint:  */
/****************************************/
/*
 Give the handle of the raw endpoint for use with
    mx__driver_interface.h
*/

MX_FUNC(mx_endpt_handle_t) mx_raw_handle(mx_raw_endpoint_t ep);

/****************************************/
/* mx_raw_open_endpoint:  */
/****************************************/

/*
 Opens the raw endpoint.  There is one raw endpoint per instance,
 and by default only a priviledged application may open it. 
*/

MX_FUNC(mx_return_t) mx_raw_open_endpoint(uint32_t board_number,
					  mx_param_t *params_array,
					  uint32_t params_count,
					  mx_raw_endpoint_t *endpoint);       

/****************************************/
/* mx_raw_close_endpoint:  */
/****************************************/
/*
 Closes a raw endpoint 
*/
MX_FUNC(mx_return_t) mx_raw_close_endpoint(mx_raw_endpoint_t endpoint);       

/****************************************/
/* mx_raw_send:                         */
/****************************************/

/* 
  Sends a raw message of length buffer_length from route_pointer out
  physical_port, using the route specified by route_pointer and
  route_length.  The sum of route_length and buffer_length must
  not exceed MX_RAW_MTU (currently 1024 bytes).

  All messages are buffered, so the send_buffer can be recycled
  by the caller immediately after this function completes.

  The amount of buffer space is finite. There may be at most
  MX_RAW_NUM_TRANSMITS (currently 64) transmits outstanding at one
  time.  Buffer space is reclaimed when a send completion event
  is received via mx_raw_next_event().

  Tag may be used to identify which send completed.

*/

MX_FUNC(mx_return_t) mx_raw_send(mx_raw_endpoint_t endpoint,
				 uint32_t physical_port,
				 void *route_pointer,
				 uint32_t route_length,
				 void *send_buffer,
				 uint32_t buffer_length,
				 void *context);

/****************************************/
/* mx_raw_next_event:                   */
/****************************************/

typedef int mx_raw_status_t;

/* 
   Obtains the next raw event (received message or
   send completion), or reports there is no event ready.
   The caller should set recv_bytes to the maximum length
   receive he is prepared to handle.

   Blocks for up to timeout_ms (could be much more if
   the high resolution timer has not been enabled) waiting
   for an event.   If timeout_ms is zero, then it just
   obtains the next raw event, returning immediately
   even if no events are pending.

   The following are modified after a receive completion:
       incoming_port, recv_buffer, recv_bytes

   A send completion implicitly returns a transmit token
   to the caller.  Sends are recycled in order.

   mx_raw_status_t is used to determine the type of 
   event (if any) which occured.
*/

MX_FUNC(mx_return_t) mx_raw_next_event(mx_raw_endpoint_t endpoint,
				       uint32_t *incoming_port,
				       void **context,
				       void *recv_buffer,
				       uint32_t *recv_bytes,
				       uint32_t timeout_ms,
				       mx_raw_status_t *status);

/****************************************/
/* mx_raw_enable_hires_timer:           */
/* mx_raw_disable_hires_timer:          */
/****************************************/

/* Enable and disable the high resolution timer.  The
   high resolution timer is resource intensive and should
   be disabled whenever possible
 */
MX_FUNC(mx_return_t) mx_raw_enable_hires_timer(mx_raw_endpoint_t endpoint);

MX_FUNC(mx_return_t) mx_raw_disable_hires_timer(mx_raw_endpoint_t endpoint);

/****************************************/
/* mx_raw_set_route_begin:              */
/****************************************/
/* Start assigning a new set of routes.  This just prepares to record
   all assigned routes. */

MX_FUNC(mx_return_t) mx_raw_set_route_begin(mx_raw_endpoint_t endpoint);
/****************************************/
/* mx_raw_set_route_end:                */
/****************************************/
/* Called when finished assigning a new set of routes.  This
   pushes the new routes to the firmware */

MX_FUNC(mx_return_t) mx_raw_set_route_end(mx_raw_endpoint_t endpoint);

/****************************************/
/* mx_raw_set_route:                    */
/****************************************/
/* Sets a route of route_length to destination_id's input_port from
   this instance's output port.
   
   Multiple routes may be specified by making multiple calls to
   mx_raw_set_route.

   host_type specifies whether the destination is a GM, XM or MX host.

   This implicity adds a peer table entry to the destination
   if one does not already exist.

   mx_raw_set_route() must be bracketed by calls
   to mx_raw_set_route_begin() and mx_raw_set_route_end()
*/

typedef enum {
  MX_HOST_GM = 1,
  MX_HOST_XM = 2,
  MX_HOST_MX = 3
} mx_host_type_t;

MX_FUNC(mx_return_t) mx_raw_set_route(mx_raw_endpoint_t endpoint,
				      uint64_t destination_id,
				      void *route,
				      uint32_t route_length,
				      uint32_t input_port,
				      uint32_t output_port,
				      mx_host_type_t host_type);

/****************************************/
/* mx_raw_clear_routes:   */
/****************************************/

/* Clears all routes to the destination, but
   it remains a valid entry in the peer table */

MX_FUNC(mx_return_t) mx_raw_clear_routes(mx_raw_endpoint_t endpoint,
					 uint64_t destination_id,
					 uint32_t port);

/****************************************/
/* mx_raw_remove_peer:   */
/****************************************/

/* removes the destination from the peer table.  This
   also deletes any routing state associated with this
   destination */
   
MX_FUNC(mx_return_t) mx_raw_remove_peer(mx_raw_endpoint_t endpoint,
					uint64_t destination_id);

/****************************************/
/* mx_raw_set_map_version:  */
/****************************************/

/* publishes the map version, and other internal mapper
   information to other applications for the physical_port.

   currently, mx node name resolution is begun
   when a mapper calls mx_raw_set_map_version() with
   mapping_complete == 1.
*/

MX_FUNC(mx_return_t) mx_raw_set_map_version(mx_raw_endpoint_t endpoint,
					    uint32_t physical_port,
					    uint64_t mapper_id,
					    uint32_t map_version,
					    uint32_t num_nodes,
					    uint32_t mapping_complete);

/****************************************/
/* mx_raw_num_ports:  			*/
/****************************************/

/* Determines the number of physical ports on an interface */

MX_FUNC(mx_return_t) mx_raw_num_ports(mx_raw_endpoint_t endpoint,
				      uint32_t *num_ports);

MX_FUNC(mx_return_t) mx_raw_set_hostname(mx_raw_endpoint_t endpoint,
					 char *hostname);



#endif /* MX_RAW_H*/
