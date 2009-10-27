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

#ifndef _mx_mcp_config_h_
#define _mx_mcp_config_h_

#include "mx_constants.h"

#define MX_MCP_RBUF_CNT    4
#define MX_MCP_BUF_SIZE    (64 * 1024)
#define MX_MCP_STOP_SIZE   (32 * 1024)
#define MX_MCP_MAX_PACKET  (16 * 1024)
#define MX_MCP_BLOCK_SIZE  128

#define MX_MCP_SEND_BUFFERS_SIZE 9032
#define MX_MCP_SEND_BUFFERS_CNT  3

#define MX_MCP_NODES_CNT         1024
#define MX_MCP_ENDPOINTS_CNT     4
#define MX_MCP_SEND_HANDLES_CNT  63
#define MX_MCP_PULL_HANDLES_CNT  63
#define MX_MCP_PUSH_HANDLES_CNT  256
#define MX_MCP_RDMA_WINDOWS_CNT  256
#define MX_MCP_PEER_HASH         (2 * MX_MCP_NODES_CNT)

#define MX_MCP_LIGHT_ENDPOINTS_MIN  16
#define MX_MCP_LIGHT_ENDPOINTS_MAX  (64 * 1024)

#define MX_MCP_PUSH_FRAMES_CNT   8
#define MX_MCP_PULL_FRAMES_CNT   32

#define MX_MCP_SENDQ_VPAGE_CNT   1024
#define MX_MCP_RECVQ_VPAGE_CNT   2048
#define MX_MCP_EVENTQ_VPAGE_CNT  32

#define MX_MCP_ETHER_MAX_SEND_FRAG  12
#define MX_MCP_ETHER_MIN_RECV_FRAG  8
#define MX_MCP_ETHER_PAD	    2

#define MX_MCP_KREQQ_CNT     64   /* 2KB */
#define MX_MCP_UREQQ_CNT     (MX_MCP_SEND_HANDLES_CNT + 1)
#define MX_MCP_UDATAQ_SIZE   (MX_MCP_UMMAP_SIZE - (MX_MCP_UREQQ_CNT * 32) - 4)

#define MX_MCP_PROGRESS_TIMER       20000
#define MX_MCP_ROUTE_DISPERSION_ALWAYS 0

#define MX_MCP_ROUTE_CNT_BLOCK   8
#define MX_MCP_ROUTE_MAX_LENGTH  7
#define MX_MCP_RAW_SEND_CNT      64
#define MX_MCP_RAW_MTU           1024
#define MX_MCP_RAW_RECV_VPAGES   32

#define MX_MCP_INTRQ_SLOTS     128
#define MX_MCP_COMMANDQ_SLOTS  32
#define MX_MCP_GLOBAL_OFFSET   2048
#define MX_MCP_LOGGING_SIZE    1024

#define MX_MCP_PRINT_BUFFER_SIZE (256 + (MX_DEBUG * 768))
#define MX_MCP_INTR_COAL_DELAY   10

#define MX_MCP_VERSION             0x0000010A
#define MX_MCP_DRIVER_API_VERSION  0x00000200

#define MX_MCP_UMMAP_SIZE          (16 * 1024)  /* 16 K pages for IA-64 */

#define MX_MCP_VPAGE_SHIFT         12 /* 4096 */
#define MX_MCP_VPAGE_SIZE          (1 << MX_MCP_VPAGE_SHIFT)
#define MX_MCP_VPAGE_MASK          (MX_MCP_VPAGE_SIZE - 1)

#if MX_DEVEL
#define MX_MCP_PACKET_LOSS    00   /* simul packets loss (%) */
#endif

#endif  /* _mx_mcp_config_h_ */
