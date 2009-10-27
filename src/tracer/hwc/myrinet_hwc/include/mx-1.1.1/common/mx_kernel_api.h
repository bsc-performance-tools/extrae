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

/* This MX kernel lib code was originally contributed by
 * Brice.Goglin@ens-lyon.org (LIP/INRIA/ENS-Lyon) */

#ifndef MX_KERNEL_API_H
#define MX_KERNEL_API_H

#ifndef MYRIEXPRESS_H
#error myriexpress.h must be included before mx_kernel_api.h
#endif

#ifdef MX_KERNEL

#include "mx_pin.h"
/** Kernel API requires to pass the type of memory in the segments. */

typedef struct
{
  /** A reference to the beginning of the segment. */
  uint64_t segment_ptr;
  /** The length, in bytes, of the contiguous segment. */
  uint32_t segment_length;
  uint32_t pad;
}
mx_ksegment_t;

#define MX_U64_TO_KVA(x) ((void*)(uintptr_t)(x))
#define MX_U64_TO_UVA(x) ((mx_uaddr_t)(x))
#define MX_KVA_TO_U64(x) ((uint64_t)(uintptr_t)(x))
#define MX_UVA_TO_U64(x) ((uint64_t)(mx_uaddr_t)(x))
#define MX_PA_TO_U64(x) (x)
#define MX_U64_TO_PA(x) (x)

mx_return_t mx_kisend(mx_endpoint_t endpoint,
		      mx_ksegment_t *segments_list,
		      uint32_t segments_count,
		      mx_pin_type_t pin_type,
		      mx_endpoint_addr_t dest_endpoint,
		      uint64_t match_info,
		      void *context,
		      mx_request_t *request);

#define mx_isend mx_uisend

static inline mx_return_t
mx_uisend(struct mx_endpoint * ep,
	  mx_ksegment_t *segments_list, uint32_t segments_count,
	  mx_endpoint_addr_t dest_address, uint64_t match_info,
	  void *context, mx_request_t *request)
{
  return mx_kisend(ep, segments_list, segments_count, MX_PIN_KERNEL,
		   dest_address, match_info, context, request);
}

mx_return_t mx_kissend(mx_endpoint_t endpoint,
		       mx_ksegment_t *segments_list,
		       uint32_t segments_count,
		       mx_pin_type_t pin_type,
		       mx_endpoint_addr_t dest_endpoint,
		       uint64_t match_info,
		       void *context,
		       mx_request_t *request);

#define mx_issend mx_uissend

static inline mx_return_t
mx_uissend(mx_endpoint_t ep,
	   mx_ksegment_t *segments_list, uint32_t segments_count,
	   mx_endpoint_addr_t dest_endpoint, uint64_t match_info,
	   void *context, mx_request_t *request)
{
  return mx_kissend(ep, segments_list, segments_count, MX_PIN_KERNEL,
		    dest_endpoint, match_info, context, request);
}

mx_return_t mx_kirecv(mx_endpoint_t endpoint,
		      mx_ksegment_t *segments_list,
		      uint32_t segments_count,
		      mx_pin_type_t pin_type,
		      uint64_t match_info,
		      uint64_t match_mask,
		      void *context,
		      mx_request_t *request);

#define mx_irecv mx_uirecv

static inline mx_return_t
mx_uirecv(mx_endpoint_t ep,
	  mx_ksegment_t *segments_list, uint32_t segments_count,
	  uint64_t match_info, uint64_t match_mask,
	  void *context, mx_request_t *request)
{
  return mx_kirecv(ep, segments_list, segments_count, MX_PIN_KERNEL,
		   match_info, match_mask, context, request);
}

#else /* MX_KERNEL */
#error mx_kernel_api.h included without MX_KERNEL defined
#endif

#endif
