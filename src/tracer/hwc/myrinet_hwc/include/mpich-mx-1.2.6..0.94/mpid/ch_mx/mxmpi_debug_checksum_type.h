/*************************************************************************
 * Myricom MPICH-MX ch_mx backend                                        *
 * Copyright (c) 2003 by Myricom, Inc.                                   *
 * All rights reserved.                                                  *
 *************************************************************************/

#ifndef _mxmpi_debug_checksum_type_h
#define _mxmpi_debug_checksum_type_h

struct pkt_cksum
{
  unsigned long sum;
  unsigned int len;
  unsigned int info;
};

#define MXMPI_DEBUG_CHECKSUM_SEND 16
#define MXMPI_DEBUG_CHECKSUM_SEND_QUEUED 32
#define MXMPI_DEBUG_CHECKSUM_QUEUE_REG 64

#endif

