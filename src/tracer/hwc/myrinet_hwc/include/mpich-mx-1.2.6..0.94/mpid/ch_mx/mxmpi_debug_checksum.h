/*************************************************************************
 * Myricom MPICH-MX ch_mx backend                                        *
 * Copyright (c) 2003 by Myricom, Inc.                                   *
 * All rights reserved.                                                  *
 *************************************************************************/

#ifndef _mxmpi_debug_checksum_h
#define _mxmpi_debug_checksum_h

#include "mxmpi_debug_checksum_type.h"

#if MXMPI_DEBUG_CHECKSUM 

void mxmpi_debug_checksum_compute(struct pkt_cksum *cksum_ptr,
				 void *buf,
				 unsigned int len);
void mxmpi_debug_checksum_check(char *msg, void *buf,
			       unsigned int from,
			       struct pkt_cksum *cksum_ptr);
void mxmpi_debug_checksum_copy(struct pkt_cksum *cksum_target,
			      struct pkt_cksum *cksum_source);
void mxmpi_debug_checksum_info(struct pkt_cksum * cksum_ptr,
			      unsigned int info);
     

#define MXMPI_DEBUG_CHECKSUM_SMALL struct pkt_cksum cksum_small;
#define MXMPI_DEBUG_CHECKSUM_LARGE struct pkt_cksum cksum_large;
#define MXMPI_DEBUG_CHECKSUM_COMPUTE(cksum_ptr,buf,len) \
mxmpi_debug_checksum_compute(cksum_ptr, buf, len)
#define MXMPI_DEBUG_CHECKSUM_CHECK(msg,buf,from,cksum_ptr) \
mxmpi_debug_checksum_check(msg, buf, from, cksum_ptr)
#define MXMPI_DEBUG_CHECKSUM_COPY(cksum_target,cksum_source) \
mxmpi_debug_checksum_copy(cksum_target, cksum_source)
#define MXMPI_DEBUG_CHECKSUM_INFO(cksum_ptr,info) \
mxmpi_debug_checksum_info(cksum_ptr, info)
#else
#define MXMPI_DEBUG_CHECKSUM_SMALL
#define MXMPI_DEBUG_CHECKSUM_LARGE
#define MXMPI_DEBUG_CHECKSUM_COMPUTE(cksum_ptr,buf,len)
#define MXMPI_DEBUG_CHECKSUM_CHECK(msg,buf,from,cksum_ptr)
#define MXMPI_DEBUG_CHECKSUM_COPY(cksum_target,cksum_source)
#define MXMPI_DEBUG_CHECKSUM_INFO(cksum_ptr,info)
#endif

#endif
