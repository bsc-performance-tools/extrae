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

#ifndef _mcp_global_h_
#define _mcp_global_h_

#include "mcp_config.h"
#include "mcp_types.h"
#include "mcp_logging.h"

#define MX_MCP_STATUS_LOAD    0xBEEF1111
#define MX_MCP_STATUS_INIT    0xBEEF2222
#define MX_MCP_STATUS_RUN     0xBEFF3333
#define MX_MCP_STATUS_PARITY  0xBEEF4444
#define MX_MCP_STATUS_ERROR   0xBEEF5555

#define MX_MCP_PARITY_NONE    0
#define MX_MCP_PARITY_PANIC   0xCACA1111
#define MX_MCP_PARITY_REBOOT  0xCACA2222
#define MX_MCP_PARITY_IGNORE  0xCACA3333

#define MX_DMA_INVALID_ENTRY  0xFFFFFFFF

/* Commands */
typedef enum {
  MX_MCP_CMD_NONE = 0,
  MX_MCP_CMD_RESET_COUNTERS,
  MX_MCP_CMD_OPEN_ENDPOINT,
  MX_MCP_CMD_SET_ENDPOINT_SESSION,
  MX_MCP_CMD_GET_HOST_SENDQ_OFFSET,
  MX_MCP_CMD_GET_HOST_RECVQ_OFFSET,
  MX_MCP_CMD_GET_HOST_EVENTQ_OFFSET,
  MX_MCP_CMD_GET_USER_MMAP_OFFSET,
  MX_MCP_CMD_GET_RDMA_WINDOWS_OFFSET,
  MX_MCP_CMD_ENABLE_ENDPOINT,
  MX_MCP_CMD_CLOSE_ENDPOINT,
  MX_MCP_CMD_ADD_PEER,
  MX_MCP_CMD_REMOVE_PEER,
  MX_MCP_CMD_UPDATE_ROUTES_P0,
  MX_MCP_CMD_UPDATE_ROUTES_P1,
  MX_MCP_CMD_ETHERNET_UP,
  MX_MCP_CMD_ETHERNET_DOWN,
  MX_MCP_CMD_START_LOGGING,
  MX_MCP_CMD_START_DMABENCH_READ,
  MX_MCP_CMD_START_DMABENCH_WRITE,
  MX_MCP_CMD_CLEAR_RAW_STATE,
  MX_MCP_CMD_PAUSE
} mx_mcp_cmd_type_t;

typedef enum {
  MX_MCP_CMD_OK = 0,
  MX_MCP_CMD_UNKNOWN,
  MX_MCP_CMD_ERROR_RANGE,
  MX_MCP_CMD_ERROR_BUSY,
  MX_MCP_CMD_ERROR_EMPTY,
  MX_MCP_CMD_ERROR_CLOSED,
  MX_MCP_CMD_ERROR_HASH_ERROR,
  MX_MCP_CMD_ERROR_BAD_PORT,
  MX_MCP_CMD_ERROR_RESOURCES
} mx_mcp_cmd_status_t;

/* Interrupt */
typedef enum {
  MX_MCP_INTR_INIT_DONE = 1,
  MX_MCP_INTR_CMD_ACK,
  MX_MCP_INTR_CMD_NACK,
  MX_MCP_INTR_RAW_SEND,
  MX_MCP_INTR_RAW_RECV,
  MX_MCP_INTR_MAPPER_TICK,
  MX_MCP_INTR_ETHER_SEND_DONE,
  MX_MCP_INTR_ETHER_RECV_SMALL,
  MX_MCP_INTR_ETHER_RECV_BIG,
  MX_MCP_INTR_ROUTES,
  MX_MCP_INTR_QUERY,
  MX_MCP_INTR_LOGGING,
  MX_MCP_INTR_PRINT,
  MX_MCP_INTR_DMABENCH,
  MX_MCP_INTR_WAKE,
  MX_MCP_INTR_ENDPT_ERROR,
  MX_MCP_INTR_RDMAWIN_UPDATE,
  MX_MCP_INTR_ENDPT_CLOSED,
  MX_MCP_INTR_LINK_CHANGE
} mx_mcp_intr_type_t;

/* Query types */
typedef enum {
  MX_MCP_QUERY_HOSTNAME = 1
} mx_mcp_query_type_t;


typedef struct
{
  /* Misc */
  uint32_t sizeof_global;          /* Size of the global structure. */
  uint32_t mcp_version;            /* Version code of the MCP. */
  uint32_t driver_api_version;     /* Version code of the MCP/Driver API. */
  uint32_t mcp_status;             /* Current status of the MCP. */
  uint32_t parity_status;
  uint32_t reboot_status;
  uint32_t params_ready;
  
  /* Parameters */
  uint32_t nodes_cnt;              /* Number of NICs on the fabric. */
  uint32_t endpoints_cnt;          /* Number of user endpoints. */
  uint32_t send_handles_cnt;       /* Number of Send handles per endpoints. */
  uint32_t pull_handles_cnt;       /* Number of Pull handles per endpoints. */
  uint32_t push_handles_cnt;       /* Number of Push handles. */
  uint32_t rdma_windows_cnt;
  uint32_t peer_hash_size;         /* Size of the Peer hash table. */
  uint32_t intr_coal_delay;        /* Minimum delay between interrupts (us). */
  uint32_t mac_high32;
  uint16_t mac_low16;
  uint16_t pad1;
  uint16_t mac_high16;
  uint16_t pad2;
  uint32_t mac_low32;
  uint32_t ethernet_mtu;
  uint32_t ethernet_smallbuf;
  uint32_t ethernet_bigbuf;
  uint32_t random_seed;
  uint32_t endpt_recovery;

  char hostname[MX_MAX_STR_LEN];    /* Must be 8-byte aligned */
  
  /* Interface MCP/Driver */
  mcp_dma_addr_t host_intr_queue[2];
  mcp_dma_addr_t raw_recv_vpages[MX_MCP_RAW_RECV_VPAGES];
  mcp_dma_addr_t host_query_vpage;
  volatile mcp_slot_t command_queue[MX_MCP_COMMANDQ_SLOTS];
  uint32_t raw_recv_enabled;
  uint32_t raw_host_recv_offset;
  uint32_t mapping_in_progress;
  uint32_t kreqq_offset;

  /* ethernet */
  volatile uint8_t ether_tx_ring[MX_MCP_VPAGE_SIZE];
  volatile uint8_t ether_rx_small_ring[MX_MCP_VPAGE_SIZE];
  volatile uint8_t ether_rx_big_ring[MX_MCP_VPAGE_SIZE];
  volatile uint32_t ether_tx_cnt;
  volatile uint32_t ether_rx_small_cnt;
  volatile uint32_t ether_rx_big_cnt;

  uint32_t counters_offset;        /* The offset of the counters. */
  uint32_t logging_buffer;
  uint32_t logging_size;
  uint32_t print_buffer_addr;      /* Address of printf buffer */
  uint32_t print_buffer_pos;       /* Current index in printf buffer */
  volatile uint32_t print_buffer_limit;     /* Limit of printf buffer */
  uint32_t clock_freq;
  uint32_t local_peer_index;

} mx_mcp_public_global_t;

#endif  /* _mcp_global_h_ */
