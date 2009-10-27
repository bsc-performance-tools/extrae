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

#ifndef _mcp_requests_h_
#define _mcp_requests_h_

#include "mx_int.h"

typedef enum {
  MX_MCP_UREQ_NONE = 0,
  MX_MCP_UREQ_SEND_NULL,
  MX_MCP_UREQ_SEND_SMALL,
  MX_MCP_UREQ_SEND_MEDIUM,
  MX_MCP_UREQ_SEND_MEDIUM_CONT,
  MX_MCP_UREQ_SEND_RNDV,
  MX_MCP_UREQ_SEND_PULL,
  MX_MCP_UREQ_SEND_CONNECT,
  MX_MCP_UREQ_POST_WAKE
} mcp_ureq_type_t;

#define MX_MCP_KREQ_RAW            1
#define MX_MCP_KREQ_QUERY          2

#define MX_MCP_ETHER_FLAGS_VALID   0x1
#define MX_MCP_ETHER_FLAGS_LAST    0x2
#define MX_MCP_ETHER_FLAGS_HEAD    0x4
#define MX_MCP_ETHER_FLAGS_CKSUM   0x8


/* 32 Bytes */
typedef struct
{
  uint32_t pad0[2];
  uint16_t dest_peer_index;
  uint8_t dest_endpt;
  uint8_t pad1;
  uint32_t dest_session;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t lib_seqnum;
  uint16_t lib_cookie;
  uint8_t pad2[3];
  uint8_t type;
} mcp_ureq_null_t;

/* 32 Bytes */
typedef struct
{
  uint32_t pad0;
  uint16_t dest_peer_index;
  uint8_t dest_endpt;
  uint8_t pad1;
  uint32_t dest_session;
  uint32_t length;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t lib_seqnum;
  uint16_t lib_cookie;
  uint16_t offset;
  uint8_t pad2;
  uint8_t type;
} mcp_ureq_small_t;

/* 32 Bytes */
typedef struct
{
  uint16_t dest_peer_index;
  uint8_t dest_endpt;
  uint8_t pad;
  uint32_t dest_session;
  uint32_t length;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t lib_seqnum;
  uint16_t lib_cookie;
  uint32_t credits;
  uint16_t sendq_index;
  uint8_t pipeline;
  uint8_t type;
} mcp_ureq_medium_t;

/* 32 Bytes */
typedef struct
{
  uint16_t dest_peer_index;
  uint8_t dest_endpt;
  uint8_t pad0;
  uint32_t dest_session;
  uint32_t length;
  uint32_t match_a;
  uint32_t match_b;
  uint16_t lib_seqnum;
  uint16_t lib_cookie;
  uint8_t rdmawin_id;
  uint8_t rdmawin_seqnum;
  uint16_t offset;
  uint8_t pad2[3];
  uint8_t type;
} mcp_ureq_rndv_t;

/* 32 Bytes */
typedef struct
{
  uint32_t pad0;
  uint16_t target_peer_index;
  uint8_t target_endpt;
  uint8_t pad1;
  uint32_t target_session;
  uint32_t length;
  uint8_t origin_rdmawin_id;
  uint8_t origin_rdmawin_seqnum;
  uint16_t origin_rdma_offset;
  uint8_t target_rdmawin_id;
  uint8_t target_rdmawin_seqnum;
  uint16_t target_rdma_offset;
  uint16_t origin_lib_cookie;
  uint16_t pad2[2];
  uint8_t pad3;
  uint8_t type;
} mcp_ureq_pull_t;

/* 32 Bytes */
typedef struct
{
  uint32_t pad0;
  uint16_t dest_peer_index;
  uint8_t dest_endpt;
  uint8_t pad1;
  uint32_t dest_session;
  uint32_t timeout;
  uint32_t app_key;
  uint8_t is_reply;
  uint8_t connect_seqnum;
  uint8_t status_code;
  uint8_t pad2;
  uint16_t lib_seqnum;
  uint16_t lib_cookie;
  uint8_t pad3[3];
  uint8_t type;
} mcp_ureq_connect_t;

/* 32 Bytes */
typedef struct
{
  uint32_t pad1[6];
  uint32_t eventq_flow;
  uint8_t pad2[3];
  uint8_t type;
} mcp_ureq_wake_t;

/* 32 Bytes */
typedef union
{
  mcp_ureq_null_t null;
  mcp_ureq_small_t small;
  mcp_ureq_medium_t medium;
  mcp_ureq_rndv_t rndv;
  mcp_ureq_pull_t pull;
  mcp_ureq_connect_t connect;
  mcp_ureq_wake_t wake;
  struct {
    uint8_t pad[31];
    uint8_t type;
  } basic;
  uint64_t int_array[4];
} mcp_ureq_t;


/* 24 Bytes */
typedef struct
{
  uint32_t addr_high;
  uint32_t addr_low;
  uint32_t context_low;
  uint32_t context_high;
  uint16_t msg_offset;
  uint16_t msg_length;
  uint8_t route_length;
  uint8_t port;
  uint8_t pad[1];
  uint8_t type;
} mcp_kreq_raw_t;

/* 4 Bytes */
typedef struct
{
  uint16_t peer_index;
  uint8_t query_type;
  uint8_t type;
} mcp_kreq_query_t;

/* 32 Bytes */
typedef union
{
  struct {
    uint8_t pad[32 - sizeof (mcp_kreq_raw_t)];
    mcp_kreq_raw_t req;
  } raw;
  struct {
    uint8_t pad[32 - sizeof (mcp_kreq_query_t)];
    mcp_kreq_query_t req;
  } query;
  struct {
    uint8_t pad[31];
    uint8_t type;
  } basic;
  uint64_t   int64_array[32/8];
} mcp_kreq_t;


/* 16 Bytes */
typedef union
{
  struct {
    uint32_t dest_low32;
    uint16_t dest_high16;
    uint16_t cksum_offset;      /* where to start computing cksum */
    uint16_t pseudo_hdr_offset; /* where to store cksum */
    uint8_t  pad[5];
    uint8_t  flags;
  } head;
  struct {
    uint32_t addr_high;
    uint32_t addr_low;
    uint16_t length;
    uint8_t  pad[5];
    uint8_t  flags;
  } frag;
} mcp_kreq_ether_send_t;

/* 8 Bytes */
typedef struct
{
  uint32_t addr_high;
  uint32_t addr_low;
} mcp_kreq_ether_recv_t;

#endif  /* _mcp_requests_h_ */
