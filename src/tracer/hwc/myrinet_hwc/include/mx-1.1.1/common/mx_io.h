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
 * Copyright 2003 - 2004 by Myricom, Inc.  All rights reserved.          *
 *************************************************************************/

/* This file defines details of the interface between the MX library
   and driver. To be used with mx__driver_interface.h */

#ifndef _mx_io_h_
#define _mx_io_h_

#ifdef MX_KERNEL
typedef struct mx_endpt_state * mx_endpt_handle_t;
#define MX__INVALID_HANDLE 0
#elif MX_OS_WINNT
#include <windows.h>
typedef HANDLE mx_endpt_handle_t;
#define MX__INVALID_HANDLE INVALID_HANDLE_VALUE
#else
typedef int mx_endpt_handle_t;
#define MX__INVALID_HANDLE -1
#endif

/* Explicitly pad structs to 8 bytes if they contain uint64_t, 4 bytes
   for uint32_t and smaller. Also, remember to add a compile time assert
   in libmyriexpress/mx__assertions.c to make sure the struct is the
   size you think it is. */

/* driver/lib interface (includes mcp/lib as well for now) 
   the LSB increase for minor backwards compatible change,
   the MSB increase for incompatible change */
#define MX_DRIVER_API_MAGIC 0x201

typedef struct {
  uint32_t driver_api_magic;
  char version_str[60];
  char build_str[128];
} mx_get_version_t;

typedef struct {
  uint32_t pid;
  char user_info[36];
  char comm[32];
} mx_opener_t;

typedef struct {
  uint32_t board_number;
  uint32_t endpt_number;
  uint32_t closed;
  mx_opener_t opener;
} mx_get_opener_t;

typedef struct {
  uint32_t board_number;
  uint32_t sram_size;
  uint32_t isr;
  uint32_t pad;
  uint64_t sram;
} mx_crashdump_t;

typedef struct {
  uint32_t board_number;
  uint32_t pad;
  uint64_t nic_id;
} mx_get_nic_id_t;

typedef struct {
  int32_t endpoint;
  uint32_t session_id;
} mx_set_endpt_t;

typedef struct {
  /* each of these is mmaped independantly */
  uint32_t sendq_offset;
  uint32_t sendq_len;
  uint32_t recvq_offset;
  uint32_t recvq_len;
  uint32_t eventq_offset;
  uint32_t eventq_len;

  /* The library mmaps the sram */
  uint32_t user_mmapped_sram_offset;
  uint32_t user_mmapped_sram_len;

  /* And uses these offsets/lens to find the queues.  Note that these
     offsets are relative to the start of user_mmapped_sram, these are
     not to be mmapped separately.
  */
  uint32_t user_reqq_offset;
  uint32_t user_dataq_offset;
  uint32_t user_reqq_len;
  uint32_t user_dataq_len;
} mx_get_copyblock_t;

typedef struct {
  uint64_t vaddr;
  uint32_t len;
  uint32_t pad;
} mx_reg_seg_t;

typedef struct {
  uint32_t nsegs;
  uint32_t rdma_id;
  uint64_t memory_context;
  mx_reg_seg_t segs[1];
} mx_reg_t;

typedef struct {
  uint32_t count;
  uint32_t board_number;
  char label[1][MX_MAX_STR_LEN];
} mx_get_counters_strings_t;

typedef struct {
  uint32_t count;
  uint32_t board_number;
  uint8_t label[1][MX_MAX_STR_LEN];
} mx_get_logging_strings_t;

typedef struct {
  uint64_t nic_id;
  uint32_t board_number;
  uint32_t pad;
} mx_nic_id_to_board_num_t;

typedef struct {
  uint32_t board_number;
  uint32_t size;
  uint64_t buffer;
} mx_get_logging_t;

typedef struct {
  uint32_t board_number;
  uint32_t index;
  uint64_t nic_id;
} mx_lookup_peer_t;

typedef struct {
  uint32_t sizeof_peer_t;
  uint32_t offset_of_type;
  uint32_t offset_of_node_name;
} mx_get_peer_format_t;

typedef struct {
  uint32_t board_number;
  uint32_t pad;
  uint64_t routes;
} mx_get_route_table_t;

typedef struct {
  uint32_t offset;
  uint32_t len;
  uint64_t va;
  uint32_t requested_permissions;
  uint32_t pad;
} mx_mmap_t;

typedef struct {
  uint64_t nic_id;
  uint64_t va;
  uint32_t len;
  uint32_t pad;
} mx_nic_id_hostname_t;

typedef struct {
  uint32_t board_number;
  uint32_t len;
  uint64_t va;
} mx_set_hostname_t;

typedef struct {
  uint32_t board_number;
  uint32_t count;
  uint32_t events;
  uint32_t spurious;
} mx_irq_counters_t;

typedef struct {
  uint32_t board_number;
  uint8_t count;
  uint8_t log_size;
  uint16_t pad;
  uint32_t dma_read;
  uint32_t rtc_start;
  uint32_t rtc_end;
} mx_dmabench_t;

typedef struct {
  uint32_t board_number;
  uint8_t mapper_mac[6];
  uint16_t iport;
  uint32_t map_version;
  uint32_t num_hosts;
  uint32_t network_configured;
  uint32_t routes_valid;
  uint32_t level;
  uint32_t flags;
} mx_mapper_state_t;

typedef struct {
  uint32_t board_number;
  uint32_t delay;
} mx_intr_coal_t;

typedef struct {
  uint32_t timeout;
  uint32_t status;
  uint32_t mcp_wake_events;
  uint32_t pad;
#define MX_WAIT_STATUS_GOOD 0
#define MX_WAIT_PARITY_ERROR_DETECTED 1
#define MX_WAIT_PARITY_ERROR_CORRECTED 2
#define MX_WAIT_PARITY_ERROR_UNCORRECTABLE 3
#define MX_WAIT_ENDPT_ERROR 4
#define MX_WAIT_TIMEOUT_OR_INTR 5
#define MX_MAX_WAIT	 0xffffffff

} mx_wait_t;

typedef struct {
  uint64_t recv_buffer;
  uint64_t context;
  uint32_t incoming_port;
  uint32_t recv_bytes;
  uint32_t timeout;
  /* potential values for status: */
#define MX_KRAW_NO_EVENT      0
#define MX_KRAW_SEND_COMPLETE 1
#define MX_KRAW_RECV_COMPLETE 2
/* Reasons that an mcp may be marked dead */
#define MX_DEAD_RECOVERABLE_SRAM_PARITY_ERROR 10
#define	MX_DEAD_SRAM_PARITY_ERROR 	11
#define	MX_DEAD_WATCHDOG_TIMEOUT 	12
#define	MX_DEAD_COMMAND_TIMEOUT 	13
#define	MX_DEAD_ENDPOINT_CLOSE_TIMEOUT 	14
#define	MX_DEAD_ROUTE_UPDATE_TIMEOUT 	15
  uint32_t status;
} mx_raw_next_event_t;

typedef struct {
  uint64_t data_pointer;
  uint64_t route_pointer;
  uint64_t context;
  uint8_t  route_length;
  uint8_t  physical_port;
  uint16_t buffer_length;
  uint32_t pad;
} mx_raw_send_t;

typedef struct {
  uint64_t route_pointer;
  uint32_t mac_low32;
  uint16_t mac_high16;
  uint8_t  source_port;
  uint8_t  route_length;
  uint32_t host_type;
  uint32_t pad;
  
} mx_set_route_t;

typedef struct {
  uint32_t board_number;
  uint32_t raw_mtu;
  uint32_t raw_max_route;
  uint32_t raw_num_tx_bufs;
} mx_raw_params_t;

typedef struct {
  uint32_t mac_low32;
  uint16_t mac_high16;
} mx_raw_destination_t;

typedef struct {
  uint64_t dst_va;
  uint64_t src_va;
  uint32_t length;
  uint32_t src_board_num;
  uint32_t src_endpt;
  uint32_t src_session;
} mx_direct_get_t;

typedef struct {
  uint32_t endpt;
  uint32_t session;
} mx_wake_endpt_t;

typedef struct {
  uint32_t board_number;
  uint32_t pad;
  uint64_t buffer;
} mx_get_eeprom_string_t;

typedef struct {
  uint32_t board_number;
  uint32_t val;
} mx_get_board_val_t;

#endif /* _mx_io_h_ */
