/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file simply lists all of the debug counters.  It is used any
   place one needs the list.  */

#ifndef _gm_debug_counters_h_
#define _gm_debug_counters_h_

#define GM_DEBUG_COUNTERS						\
 GM_DEBUG_CNT (completed_datagram_send_cnt,				\
	       "completed datagram send count")				\
 GM_DEBUG_CNT (completed_reliable_send_cnt,				\
	       "completed reliable send count")				\
 GM_DEBUG_CNT (dispatch_cnt,						\
	       "dispatch count")					\
 GM_DEBUG_CNT (enqueued_ethernet_packets,				\
	       "enqueued ethernet packets")				\
 GM_DEBUG_CNT (hashed_token_cnt,					\
	       "number of tokens reserved for receives in progress")	\
 GM_DEBUG_CNT (hit_cnt,							\
	       "page table cache hit count")				\
 GM_DEBUG_CNT (miss_cnt,						\
	       "page table cache miss count")				\
 GM_DEBUG_CNT (outstanding_dma_cnt,					\
	       "number of outstanding DMAs")				\
 GM_DEBUG_CNT (outstanding_hardware_send_cnt,				\
	       "number sends enqueued in LANai hardware queues")	\
 GM_DEBUG_CNT (queued_datagram_send_cnt,				\
	       "number of datagram sends enqueued")			\
 GM_DEBUG_CNT (queued_reliable_send_cnt,				\
	       "number of reliable sends enqueued")			\
 GM_DEBUG_CNT (queued_send_token_cnt,					\
	       "number of sends MCP has received from the user")	\
 GM_DEBUG_CNT (sends_in_send_queue_cnt,					\
	       "number of sends in MCP fair send queue")		\
 GM_DEBUG_CNT (sent_tokens_queued_for_host_cnt,				\
	       "number os send tokens passed by the MCP to the user")

/* define GM_NUM_ERROR_COUNTERS */

#define GM_DEBUG_CNT(name, description) _gm_ ## name,
enum
{
  GM_DEBUG_COUNTERS GM_NUM_DEBUG_COUNTERS
};
#undef GM_DEBUG_CNT

#endif /* _gm_debug_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
