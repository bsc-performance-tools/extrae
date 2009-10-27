/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_error_counters_h_
#define _gm_error_counters_h_

/* This file simply lists all of the error counters.  It is used any
   place one needs the list.  */

#if defined GM_ERROR_CNT
#error You must not define GM_ERROR_CNT before including this file.
#endif

/* The following must be in ASCIIbetical order. */

#define GM_ERROR_COUNTERS						\
     GM_ERROR_CNT (badcrc_cnt, "")					\
     GM_ERROR_CNT (badcrc__invalid_crc8_cnt, "")			\
     GM_ERROR_CNT (badcrc__unstripped_route_cnt, "")			\
     GM_ERROR_CNT (badcrc__misaligned_crc32_cnt, "")			\
     GM_ERROR_CNT (badcrc__invalid_crc32_cnt, "")			\
     GM_ERROR_CNT (drop_cnt, "")					\
     GM_ERROR_CNT (drop_cnt__badcrc, "")				\
     GM_ERROR_CNT (drop_cnt__bad_length, "")				\
     GM_ERROR_CNT (drop_cnt__ethernet_packet_too_long, "")		\
     GM_ERROR_CNT (drop_cnt__ethernet_mac_mismatch, "")			\
     GM_ERROR_CNT (drop_cnt__ignored_nack_close_connection, "")		\
     GM_ERROR_CNT (drop_cnt__ignored_nack_open_connection, "")		\
     GM_ERROR_CNT (drop_cnt__ignored_nack, "")				\
     GM_ERROR_CNT (drop_cnt__misrouted, "")				\
     GM_ERROR_CNT (drop_cnt__no_match_for_datagram_recv, "")		\
     GM_ERROR_CNT (drop_cnt__no_match_for_ether_recv, "")		\
     GM_ERROR_CNT (drop_cnt__no_match_for_reliable_recv, "")		\
     GM_ERROR_CNT (drop_cnt__no_match_for_raw_recv, "")			\
     GM_ERROR_CNT (drop_cnt__out_of_sequence, "")			\
     GM_ERROR_CNT (drop_cnt__raw_packet_too_long, "")			\
     GM_ERROR_CNT (drop_cnt__short_mapper_packet, "")			\
     GM_ERROR_CNT (drop_cnt__short_packet, "")				\
     GM_ERROR_CNT (drop_cnt__yp_no_return_route, "")			\
     GM_ERROR_CNT (drop_cnt__yp_no_reply_buffer, "")			\
     GM_ERROR_CNT (handle_connection_reset_request_cnt, "")		\
     GM_ERROR_CNT (nack_cnt, "")					\
     GM_ERROR_CNT (nack_down_recv_cnt, "")				\
     GM_ERROR_CNT (nack_down_send_cnt, "")				\
     GM_ERROR_CNT (nack_normal_cnt, "number of normal NACKs processed") \
     GM_ERROR_CNT (nack_receive_close_connection_cnt, "")		\
     GM_ERROR_CNT (nack_receive_open_connection_cnt, "")		\
     GM_ERROR_CNT (nack_received_cnt, "")				\
     GM_ERROR_CNT (nack_reject_cnt, "")					\
     GM_ERROR_CNT (nack_send_close_connection_cnt, "")			\
     GM_ERROR_CNT (nack_send_nothing1_cnt, "")				\
     GM_ERROR_CNT (nack_send_nothing2_cnt, "")				\
     GM_ERROR_CNT (nack_send_open_connection_cnt, "")			\
     GM_ERROR_CNT (out_of_sequence_cnt, "")				\
     GM_ERROR_CNT (out_of_sequence_cnt__early, "")			\
     GM_ERROR_CNT (out_of_sequence_cnt__late, "")			\
     GM_ERROR_CNT (out_of_sequence_cnt__connection_reset, "")		\
     GM_ERROR_CNT (resend_cnt, "")					\
     GM_ERROR_CNT (timeout__no_match_for_recv, "")			\
     GM_ERROR_CNT (timeout__node_unreachable, "")			\
     GM_ERROR_CNT (timeout__port_closed, "")				\
     GM_ERROR_CNT (timeout__send_rejected, "")				\
     GM_ERROR_CNT (used_bogus_send_cnt, "")				\
     GM_ERROR_CNT (used_bogus_recv_cnt, "")

/* define GM_NUM_ERROR_COUNTERS */

#define GM_ERROR_CNT(name, description) _gm_ ## name,
enum
{
  GM_ERROR_COUNTERS GM_NUM_ERROR_COUNTERS
};
#undef GM_ERROR_CNT

#endif /* _gm_error_counters_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
