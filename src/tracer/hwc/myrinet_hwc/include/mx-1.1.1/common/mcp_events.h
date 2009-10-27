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

#ifndef _mcp_events_h_
#define _mcp_events_h_

#include "mx_auto_config.h"
#include "myriexpress.h"


#define MX_MCP_RECV_INLINE_SIZE 43

typedef enum {
  MX_MCP_UEVT_DONE_SUCCESS = 1,
  MX_MCP_UEVT_DONE_ERROR,
  MX_MCP_UEVT_CONNECT,
  MX_MCP_UEVT_RECV_NULL,
  MX_MCP_UEVT_RECV_INLINE,
  MX_MCP_UEVT_RECV_COPY_SINGLE,
  MX_MCP_UEVT_RECV_COPY_MULTI,
  MX_MCP_UEVT_RECV_RNDV,
  MX_MCP_UEVT_RDMA_NOTIFY
} mcp_uevt_type_t;

/* 4 Bytes */
typedef struct
{
  uint16_t lib_cookie;
  uint8_t status;
  uint8_t type;
} mcp_uevt_done_t;

/* 20 Bytes */
typedef struct
{
  uint8_t source_endpt;
  uint8_t dest_endpt;
  uint16_t lib_seqnum;
  uint16_t source_peer_index;
  uint16_t pad;
  uint32_t session_id;
  uint32_t app_key;
  uint8_t is_reply;
  uint8_t connect_seqnum;
  uint8_t status_code;
  uint8_t type;
} mcp_uevt_connect_t;

/* 20 Bytes */
typedef struct
{
  uint8_t source_endpt;
  uint8_t dest_endpt;
  uint16_t lib_seqnum;
  uint16_t source_peer_index;
  uint16_t recvq_vpage_index;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t length;
  uint16_t pad;
} mcp_uevt_recv_t;

/* 64 Bytes */
typedef struct
{
  uint8_t source_endpt;
  uint8_t dest_endpt;
  uint16_t lib_seqnum;
  uint16_t source_peer_index;
  uint16_t recvq_vpage_index;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t length;
  uint16_t pad;
  uint8_t data[MX_MCP_RECV_INLINE_SIZE];
  uint8_t type;
} mcp_uevt_recv_inline_t;

/* 20 Bytes */
typedef struct
{
  uint8_t source_endpt;
  uint8_t dest_endpt;
  uint16_t lib_seqnum;
  uint16_t source_peer_index;
  uint16_t recvq_vpage_index;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t length;
  uint8_t pad;
  uint8_t type;
} mcp_uevt_recv_copy_single_t;

/* 24 Bytes */
typedef struct
{
  uint8_t source_endpt;
  uint8_t dest_endpt;
  uint16_t lib_seqnum;
  uint16_t source_peer_index;
  uint16_t recvq_vpage_index;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t length;
  uint8_t pipeline_log;
  uint8_t pad;
  uint16_t frame_length;
  uint8_t frame_seqnum;
  uint8_t type;
} mcp_uevt_recv_copy_multi_t;

typedef struct
{
  uint32_t length;
  uint8_t rdmawin_id;
  uint8_t rdmawin_seqnum;
  uint8_t pad;
  uint8_t type;
} mcp_uevt_rdma_notify_t;


/* 64 Bytes */
typedef union
{
  struct {
    uint8_t pad[64 - sizeof (mcp_uevt_done_t)];
    mcp_uevt_done_t uevt;
  } done;
  struct {
    uint8_t pad[64 - sizeof (mcp_uevt_connect_t)];
    mcp_uevt_connect_t uevt;
  } connect;
  struct {
    mcp_uevt_recv_inline_t uevt;
  } recv_inline;
  struct {
    uint8_t pad[64 - sizeof (mcp_uevt_recv_copy_single_t)];
    mcp_uevt_recv_copy_single_t uevt;
  } recv_copy_single;
  struct {
    uint8_t pad[64 - sizeof (mcp_uevt_recv_copy_multi_t)];
    mcp_uevt_recv_copy_multi_t uevt;
  } recv_copy_multi;
  struct {
    uint8_t pad[64 - sizeof (mcp_uevt_rdma_notify_t)];
    mcp_uevt_rdma_notify_t uevt;
  } rdma_notify;
  struct {
    uint8_t pad[63];
    uint8_t type;
  } basic;
} mcp_uevt_t;

#endif  /* _mcp_events_h_ */
