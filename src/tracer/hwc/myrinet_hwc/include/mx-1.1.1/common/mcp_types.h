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

#ifndef _mcp_types_h_
#define _mcp_types_h_

#define MX_MCP_DMA_ADDR_SHIFT 3

/* 8 Bytes */
typedef struct
{
  uint32_t high;
  uint32_t low;
} mcp_dma_addr_t;

/* 32 Bytes */
typedef struct
{
  uint32_t length;
  uint32_t index;
  uint8_t seqnum;
  uint8_t refcnt;
  uint8_t pad[6];
  mcp_dma_addr_t addr;
  mcp_dma_addr_t next;
} mcp_rdma_win_t;

/* 16 Bytes */
typedef struct
{
  uint32_t data0;
  uint32_t data1;
  uint32_t seqnum;
  uint16_t index;
  uint8_t flag;
  uint8_t type;
} mcp_slot_t;

#endif /* _mcp_types_h_ */
