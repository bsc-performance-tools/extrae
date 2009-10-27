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

#ifndef MYRIEXPRESS_H
#define MYRIEXPRESS_H

/* MX API version number:
   the LSB increase for minor backwards compatible change,
   the MSB increase for incompatible change */
#define MX_API 0x300

#include "mx_int.h"
#include "mx_constants.h"

#ifdef __cplusplus
extern "C"
{
#if 0
}
#endif
#endif

typedef struct mx_endpoint * mx_endpoint_t;

/* MX Types and Objects */

#define MX_SIZEOF_ADDR 16

typedef struct {
  uint64_t	stuff[MX_SIZEOF_ADDR/sizeof(uint64_t)];
} mx_endpoint_addr_t;

#define MX_ANY_NIC 0xffffffffU
#define MX_ANY_ENDPOINT 0xffffffffU

typedef union mx_request * mx_request_t;

typedef void * mx_segment_ptr_t;

typedef struct
{
  mx_segment_ptr_t segment_ptr;
  uint32_t segment_length;
}
mx_segment_t;

/* define infinite timeout for mx_wait */

#define MX_INFINITE   0

/* Return codes returned by all of the MX functions. */

enum mx_return_code
{
  /* The operation completed successfully. */
  MX_SUCCESS = 0,
  /* Something really bad happened */
  MX_BAD_BAD_BAD = 1,
  MX_FAILURE = 2,
  /* The MX library was already initialized. */
  MX_ALREADY_INITIALIZED = 3,
  /* The MX library is not initialized. */
  MX_NOT_INITIALIZED = 4,
  /* There are no mx device entries */
  MX_NO_DEV = 5,
  /* Driver is not loaded */
  MX_NO_DRIVER = 6,
  /* Permission denied */
  MX_NO_PERM = 7,
  /* The board index specified in the call does not exist. */
  MX_BOARD_UNKNOWN = 8,
  /* The MX endpoint is not valid or not open. */
  MX_BAD_ENDPOINT = 9,
  /* The list of segments is NULL but the counts is not 0*/
  MX_BAD_SEG_LIST = 10,
  /* The memory described by one of the segments is invalid */
  MX_BAD_SEG_MEM = 11,
  /* The total number of segments exceeds the limit */
  MX_BAD_SEG_CNT = 12,
  /* The pointer to the MX request object is not valid. */
  MX_BAD_REQUEST = 13,
  /* The matching mask is not an authorized value. */
  /* The matching mask provided by the application is not one of the four
     authorized values (0xFFFFFFFF00000000, 0xFFFFFFFF0000FFFF,
     0xFFFFFFFFFFFF0000 and 0xFFFFFFFFFFFFFFFF).  */
  MX_BAD_MATCH_MASK = 14,
  /* MX was unable to perform the operation by lack of resource. */
  MX_NO_RESOURCES = 15,
  /* The list of MX addr is null */
  MX_BAD_ADDR_LIST = 16,
  /* The count of entries in the MX addr list is 0 */
  MX_BAD_ADDR_COUNT = 17,
  /* The index of the root of the broadcast is greater than the
     number of entries in the list of MX addr. */
  MX_BAD_ROOT = 18,
  /* One or more pending operations are not yet completed. */
  MX_NOT_COMPLETED = 19,
  /* This resource is busy. */
  MX_BUSY = 20,
  /* The key is not recognized */
  MX_BAD_INFO_KEY = 21,
  /* The pointer where the info is to be returned is invalid */
  MX_BAD_INFO_VAL = 22,
  /* The NIC identifier (MAC address) is not valid */
  MX_BAD_NIC = 23,
  /* The list of parameters is NULL but the count is not 0 */
  MX_BAD_PARAM_LIST = 24,
  /* The name of one of the parameters is not recognized */
  MX_BAD_PARAM_NAME = 25,
  /* The value of one of the parameters is not valid */
  MX_BAD_PARAM_VAL = 26,
  /* one of the arguments passed to mx_hostname_to_nic_id is not valid**/
  MX_BAD_HOSTNAME_ARGS = 27,
  /* hostname not found **/
  MX_HOST_NOT_FOUND = 28,
  /* The data associated with the request is not yet buffered */
  MX_REQUEST_PENDING = 29,
  /* The function returned because the timeout expired */
  MX_TIMEOUT = 30,
  /* No incoming message matches the matching information */
  MX_NO_MATCH = 31,
  /* An out-of-range endpoint ID was specified */
  MX_BAD_ENDPOINT_ID = 32,
  /* Connection refused -- no peer at this address */
  MX_CONNECTION_FAILED = 33,
  /* Connection denied -- bad key */
  MX_BAD_CONNECTION_KEY = 34,
  /* The length of the buffer for get_info is too small */
  MX_BAD_INFO_LENGTH = 35,
  /* The NIC was not found in the our network peer table */
  MX_NIC_NOT_FOUND = 36,
  /* mx library version is incompatible with kernel or mcp */
  MX_BAD_KERNEL_VERSION = 37,
  /* Application was compiled and linked with different mx versions */
  MX_BAD_LIB_VERSION = 38,
  /* The NIC has died */
  MX_NIC_DEAD = 39
};

/* Type of return codes. */

typedef enum mx_return_code mx_return_t;

enum mx_param_key
{
  MX_PARAM_ERROR_HANDLER = 0,
  MX_PARAM_UNEXP_QUEUE_MAX = 1
};

typedef enum mx_param_key mx_param_key_t;

typedef mx_return_t (*mx_error_handler_t)(char *str, mx_return_t ret);

typedef struct {
  mx_param_key_t key;
  union {
    mx_error_handler_t error_handler;
    uint32_t unexp_queue_max;
  } val;
} mx_param_t;

enum mx_status_code
{
  /* Successful completion */
  MX_STATUS_SUCCESS = 0,
  /* Request still pending */
  MX_STATUS_PENDING = 1,
  /* Request has been buffered, but still pending */
  MX_STATUS_BUFFERED = 2,
  /* Posted operation failed */
  MX_STATUS_REJECTED = 3,
  /* Posted operation timed out */
  MX_STATUS_TIMEOUT = 4,
  /* Operation completed, but data was truncated due to undersized buffer */
  MX_STATUS_TRUNCATED = 5,
  /* Pending receive was cancelled */
  MX_STATUS_CANCELLED = 6,
  /* Destination nic is unknown on the network fabric */
  MX_STATUS_ENDPOINT_UNKNOWN = 7,
  /* remoted endpoint is closed */
  MX_STATUS_ENDPOINT_CLOSED = 8,
  /* Connectivity is broken between the source and the destination */
  MX_STATUS_ENDPOINT_UNREACHABLE = 9,
  /* Bad session (no mx_connect done?) */
  MX_STATUS_BAD_SESSION = 10,
  /* Connect failed because of bad credentials */
  MX_STATUS_BAD_KEY = 11,
  /* Destination endpoint rank is out of range for the peer */
  MX_STATUS_BAD_ENDPOINT = 12,
  /* Invalid rdma window given to the mcp */
  MX_STATUS_BAD_RDMAWIN = 13,
  /* Operation aborted on peer nic */
  MX_STATUS_ABORTED = 14,
  /* Status internal to the lib/ never returned to user */
  MX_STATUS_EVENTQ_FULL = 15
};

/* Type for accessing status codes. */

typedef enum mx_status_code mx_status_code_t;

typedef struct mx_status {
  /* A code indicating status of the completion of this operation. */
  mx_status_code_t code;
  /* The endpoint of the sending endpoint for receive operations */
  mx_endpoint_addr_t source;
  /* The match data from the received message */
  uint64_t match_info;
  /* The original length of the message */
  uint32_t msg_length;
  /* The actual number of bytes transferred,  Note that for a send, this
     does not indicate the size of the buffer provided by the receiver */
  uint32_t xfer_length;
  void *context; 
} mx_status_t;

/* MX API function declarations */

/* Initializes the MX library. */

#define mx_init() mx__init_api(MX_API)
/* please use mx_init() in applications, don't use mx__init_api directly */
MX_FUNC(mx_return_t) mx__init_api(int); 

/* Cleans up any internal structures used by the MX library. */

MX_FUNC(mx_return_t) mx_finalize(void);

MX_VAR(extern const mx_error_handler_t) MX_ERRORS_ARE_FATAL;
MX_VAR(extern const mx_error_handler_t) MX_ERRORS_RETURN;

/* Set Global Error handling (can be overriden per endpoint), returns previous handler */
MX_FUNC(mx_error_handler_t) mx_set_error_handler(mx_error_handler_t);

/* Opens a MX endpoint. (@ref mx_open_endpoint "Details"). */

MX_FUNC(mx_return_t) mx_open_endpoint(uint32_t board_number,
				      uint32_t endpoint_id,
				      uint32_t endpoint_key,
				      mx_param_t *params_array,
				      uint32_t params_count,
				      mx_endpoint_t *endpoint);

/* Closes an opened MX endpoint. (@ref mx_close_endpoint "Details"). */

MX_FUNC(mx_return_t) mx_close_endpoint(mx_endpoint_t endpoint);

MX_FUNC(mx_return_t) mx_wakeup(mx_endpoint_t endpoint);

typedef void (*mx_matching_callback_t)(void *context, 
				       uint64_t match_value,
				       int length);

MX_FUNC(mx_return_t) mx_register_unexp_callback(mx_endpoint_t ep, 
						mx_matching_callback_t cb,
						void *ctxt);

#define MX_MAX_SEGMENTS 256 /* XXX */

/* Initiates an asynchronous send. */

MX_FUNC(mx_return_t) mx_isend(mx_endpoint_t endpoint,
			      mx_segment_t *segments_list,
			      uint32_t segments_count,
			      mx_endpoint_addr_t dest_endpoint,
			      uint64_t match_info,
			      void *context,
			      mx_request_t *request);

MX_FUNC(mx_return_t) mx_issend(mx_endpoint_t endpoint,
			       mx_segment_t *segments_list,
			       uint32_t segments_count,
			       mx_endpoint_addr_t dest_endpoint,
			       uint64_t match_info,
			       void *context,
			       mx_request_t *request);

/* Match mask values */
#define MX_MATCH_MASK_NONE (~(uint64_t)0)

/* Initiates an asynchronous receive. */

MX_FUNC(mx_return_t) mx_irecv(mx_endpoint_t endpoint,
			      mx_segment_t *segments_list,
			      uint32_t segments_count,
			      uint64_t match_info,
			      uint64_t match_mask,
			      void *context,
			      mx_request_t *request);



/* Tries to cancel a request previously posted. */

MX_FUNC(mx_return_t) mx_cancel(mx_endpoint_t endpoint,
			       mx_request_t *request,
			       uint32_t *result);

/* Checks the completion of a pending operation. */

MX_FUNC(mx_return_t) mx_test(mx_endpoint_t endpoint,
			     mx_request_t *request,
			     mx_status_t *status,
			     uint32_t *result);

/* Blocks until the completion of a pending operation. */

MX_FUNC(mx_return_t) mx_wait(mx_endpoint_t endpoint,
			     mx_request_t *request,
			     uint32_t timeout,
			     mx_status_t *status,
			     uint32_t *result);

/* Looks for a request ready for completion on the specified endpoint. */
MX_FUNC(mx_return_t) mx_ipeek(mx_endpoint_t endpoint,
			      mx_request_t *request,
			      uint32_t *result);

/* Blocks until a complete request is available. */
MX_FUNC(mx_return_t) mx_peek(mx_endpoint_t endpoint,
			     uint32_t timeout,
			     mx_request_t *request,
			     uint32_t *result);

/* Looks to see if a matching receive has arrived. */
MX_FUNC(mx_return_t) mx_iprobe(mx_endpoint_t endpoint,
			       uint64_t match_info,
			       uint64_t match_mask,
			       mx_status_t *status,
			       uint32_t *result);

MX_FUNC(mx_return_t) mx_probe(mx_endpoint_t endpoint,
			      uint32_t timeout,
			      uint64_t match_info,
			      uint64_t match_mask,
			      mx_status_t *status,
			      uint32_t *result);

/* Checks if the application can reuse the buffer(s) */
MX_FUNC(mx_return_t) mx_ibuffered(mx_endpoint_t endpoint,
				  mx_request_t *request,
				  uint32_t *result);

/* Blocks until a complete request is available */
MX_FUNC(mx_return_t) mx_buffered(mx_endpoint_t endpoint,
				 mx_request_t *request,
				 uint32_t timeout,
				 uint32_t *result);

MX_FUNC(mx_return_t) mx_context(mx_request_t *request, void **context);

/* Retrieves various information about MX. For now see the API
   documentations for the list of acceptable key values */

enum mx_get_info_key {
  MX_NIC_COUNT = 1,
  MX_NIC_IDS = 2,
  MX_MAX_NATIVE_ENDPOINTS = 3,
  MX_NATIVE_REQUESTS = 4,
  MX_COUNTERS_COUNT = 5,
  MX_COUNTERS_LABELS = 6,
  MX_COUNTERS_VALUES = 7,
  MX_PRODUCT_CODE = 8,
  MX_PART_NUMBER = 9,
  MX_SERIAL_NUMBER = 10,
  MX_PORT_COUNT = 11
};

typedef enum mx_get_info_key mx_get_info_key_t;

MX_FUNC(mx_return_t) mx_get_info(mx_endpoint_t endpoint,
				 mx_get_info_key_t key,
				 void *in_val,
				 uint32_t in_len,
				 void *out_val,
				 uint32_t out_len);



/*
 * Utility macros
 */

/* macros to help printing uint64_t's */
#define MX_U32(x) \
((sizeof (x) == 8) ? ((uint32_t)((uint64_t)(x) >> 32)) : ((void)(x),0))
#define MX_L32(x) ((uint32_t)(x))


/*
 * Utility functions
 */

#define MX_MAX_HOSTNAME_LEN  80
MX_FUNC(mx_return_t) mx_hostname_to_nic_id(char *hostname,
					   uint64_t *nic_id);

MX_FUNC(mx_return_t) mx_board_number_to_nic_id(uint32_t board_number,
					       uint64_t *nic_id);

MX_FUNC(mx_return_t) mx_nic_id_to_board_number(uint64_t nic_id,
					       uint32_t *board_number);

MX_FUNC(mx_return_t) mx_nic_id_to_hostname(uint64_t nic_id,
				    char *hostname);

MX_FUNC(mx_return_t) mx_connect(mx_endpoint_t endpoint,
				uint64_t nic_id,
				uint32_t endpoint_id,
				uint32_t key,
				uint32_t timeout,
				mx_endpoint_addr_t *addr);

MX_FUNC(mx_return_t) mx_decompose_endpoint_addr(mx_endpoint_addr_t endpoint_addr,
						uint64_t *nic_id,
				       uint32_t *endpoint_id);

MX_FUNC(mx_return_t) mx_get_endpoint_addr(mx_endpoint_t endpoint,
					  mx_endpoint_addr_t *endpoint_addr);

/* Returns a static string corresponding to a specified error code. */

MX_FUNC(const char *) mx_strerror(mx_return_t return_code);

/* Returns a static string corresponding to a specified status code. */

MX_FUNC(const char *) mx_strstatus(mx_status_code_t status);

#ifdef __cplusplus
#if 0
{
#endif
}
#endif

#ifdef MX_KERNEL
/* Some communication primitives have different prototypes in the kernel. */
#include "mx_kernel_api.h"
#endif

#endif /* MYRIEXPRESS_H */
