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

#ifndef _mcp_logging_h_
#define _mcp_logging_h_


#define LOG_TYPE_PCI        1
#define LOG_TYPE_LANAI      2
#define LOG_TYPE_SEND_P0    3
#define LOG_TYPE_SEND_P1    4
#define LOG_TYPE_RECV_P0    5
#define LOG_TYPE_RECV_P1    6

#define LOG_TYPE_SEND(port) (LOG_TYPE_SEND_P0 + port)
#define LOG_TYPE_RECV(port) (LOG_TYPE_RECV_P0 + port)

#define LOG_STATE_START     1
#define LOG_STATE_STOP      2

#define LOG_EVENT_SEND_WIRE_RAW     201
#define LOG_EVENT_SEND_WIRE_MX      202
#define LOG_EVENT_SEND_WIRE_ETHER   203
#define LOG_EVENT_SEND_FIFO         204
#define LOG_EVENT_RECV_WIRE_RAW     205
#define LOG_EVENT_RECV_WIRE_MX      206
#define LOG_EVENT_RECV_WIRE_ETHER   207
#define LOG_EVENT_RECV_FIFO         208
#define LOG_EVENT_DMA_FIFO          209

typedef struct
{
  uint32_t cpuc;
  uint32_t isr;
  uint8_t type;
  uint8_t state;
  uint8_t send_done[2];
  uint8_t recv_done[2];
  uint16_t pad;
  uint16_t data0;
  uint16_t data1;
  uint32_t data2;
} mcp_log_t;


#endif  /* _mcp_logging_h_ */
