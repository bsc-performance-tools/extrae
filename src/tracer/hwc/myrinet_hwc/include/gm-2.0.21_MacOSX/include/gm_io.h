/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file defines details of the interface between the GM library
   and driver.  OS-specific overrides and extensions should go in
   the architecture's "gm_arch_io.h" file. */

#ifndef _gm_io_h_
#define _gm_io_h_

#include "gm.h"
#include "gm_config.h"
#include "gm_types.h"

/* Allow the architecture to include any requisite header file and
   override definitions here. */

#include "gm_arch_io.h"

#if HAVE_SYS_TYPES_H && !GM_MCP && !GM_KERNEL
#include <sys/types.h>
#endif

#ifdef WIN32
#if GM_KERNEL
#include <devioctl.h>
#else
#include <winioctl.h>
#endif
#endif

#define GM_NOBLOCK 1
#define GM_NDELAY 2

#ifndef GM_IO
#define GM_IO(command) (('G' << 8) + (command))
#endif

/* If you override any ioctls in your gm_arch_io.h you need to override all. */
/* Please keep libgm/_gm_ioctl_cmd_name.c in sync with this list: */
#ifndef GM_SET_FLAGS

#define GM_SET_FLAGS				GM_IO ( 0)
#define GM_GET_RQST				GM_IO ( 1)
#define GM_ACCESS_GRANTED			GM_IO ( 2)
#define GM_SLEEP				GM_IO ( 3)
#define GM_GET_MAPPING_SPECS			GM_IO ( 4)
#define GM_GET_LANAI_GLOBALS_PTR		GM_IO ( 5)
#define GM_GET_NODE_ID				GM_IO ( 6)
#define GM_RECV_QUEUE_UPDATE			GM_IO ( 7)
#define GM_MMAP					GM_IO ( 8)
#define GM_MAP_CONTROL				GM_IO ( 9)
#define GM_MAP_SPECIAL				GM_IO (10)
#define GM_ENABLE_RAW_RECEIVES			GM_IO (11)
#define GM_SET_ROUTE				GM_IO (12)
#define GM_GET_UNIQUE_BOARD_ID			GM_IO (13)
#define GM_SET_NODE_ID				GM_IO (14)
#define GM_SET_ACCEPTABLE_SIZES_LOW		GM_IO (15)
#define GM_SET_ACCEPTABLE_SIZES_HIGH		GM_IO (16)
#define GM_GET_PAGE_LEN				GM_IO (17)
#define GM_REGISTER_MEMORY			GM_IO (18)
#define GM_DEREGISTER_MEMORY			GM_IO (19)
#define GM_GET_EEPROM				GM_IO (20)
#define GM_GET_MAX_NODE_ID			GM_IO (21)
/* 22 */
#define GM_NODE_ID_TO_UNIQUE_ID			GM_IO (23)
#define GM_UNIQUE_ID_TO_NODE_ID			GM_IO (24)
#define GM_SET_HOST_NAME			GM_IO (25)
#define GM_GET_HOST_NAME			GM_IO (26)
#define GM_HOST_NAME_TO_NODE_ID			GM_IO (27)
#define GM_NODE_ID_TO_HOST_NAME			GM_IO (28)
#define GM_SET_UNIQUE_ID			GM_IO (29)
#define GM_GET_ROUTE				GM_IO (30)
#define GM_SET_PORT_NUM				GM_IO (31)
#define GM_SET_REGISTER_MEMORY_LENGTH_AND_PVMA	GM_IO (32)
#define GM_CLEAR_ALL_ROUTES			GM_IO (35)
#define GM_GET_KERNEL_BUILD_ID_LEN		GM_IO (36)
#define GM_GET_KERNEL_BUILD_ID			GM_IO (37)
#define GM_GET_GLOBALS				GM_IO (38)
#define GM_GET_MAX_NODE_ID_IN_USE		GM_IO (39)
#define GM_FINISH_MMAP				GM_IO (40)
#define GM_GET_GLOBALS_OFFSET			GM_IO (41)
#define GM_GET_DEV				GM_IO (42)
#define GM_GET_MAPPER_UNIQUE_ID			GM_IO (43)
#define GM_LINUX_DEBUG_MODULE			GM_IO (44)
#define GM_GET_KTRACE				GM_IO (45)
#define GM_GET_OPENER_PIDS			GM_IO (46)
#define GM_SET_OPENER_PID			GM_IO (47)
#define GM_REGISTER_MEMORY_BY_STRUCT		GM_IO (48)
#define GM_DEREGISTER_MEMORY_BY_STRUCT		GM_IO (49)
#define GM_SET_MAPPER_STATE			GM_IO (50)
#define GM_GET_MAPPER_STATE			GM_IO (51)
#define GM_GET_FIRMWARE_STRING			GM_IO (52)
#define GM_COP_WAKEUP             		GM_IO (53)
#define GM_COP_SEND             		GM_IO (54)
#define GM_COP_RECEIVE             		GM_IO (55)
#define GM_COP_END                              GM_IO (56)
#define GM_DIRECTCOPY_GET			GM_IO (57)
#define GM_GET_GLOBALS_BY_REQUEST		GM_IO (58)
#define GM_WRITE_LANAI_REGISTER		        GM_IO (59)
#define GM_GET_PAGE_HASH_CACHE_SIZE		GM_IO (60)
#define GM_SET_ENABLE_NACK_DOWN			GM_IO (61)
#define GM_WAIT                                 GM_IO (62)
#define GM_REGISTER_RECV_QUEUE			GM_IO (63)
#define GM_DMA_ADDR				GM_IO (64)
#define GM_MUNMAP                               GM_IO (65)
#define GM_SYNC_COPY_BLOCK			GM_IO (66)
#define GM_CONTIGUOUS_DMA_MALLOC		GM_IO (67)
#define GM_CONTIGUOUS_DMA_FREE			GM_IO (68)
#define GM_NODE_ID_TO_GLOBAL_ID			GM_IO (69)
#define GM_GET_BOARD_CONFIG_STRINGS		GM_IO (70)
#define GM_ETHERNET_SET_CREDIT_INCR		GM_IO (71)
#define GM_ETHERNET_SET_MAX_CREDITS		GM_IO (72)
#define GM_ETHERNET_SET_MAX_SDMAS		GM_IO (73)
#define GM_ETHERNET_GET_PARAMS			GM_IO (74)
#define GM_DUMP_MCP				GM_IO (75)
#define GM_CLEAR_COUNTERS			GM_IO (76)
#define GM_DISABLE_SOFTWARE_LOOPBACK_CMD	GM_IO (77)
#define GM_ENABLE_SOFTWARE_LOOPBACK_CMD		GM_IO (78)
#define GM_SET_NODE_TYPE			GM_IO (79)
#define GM_GET_NODE_TYPE			GM_IO (80)
#define GM_YP_IOCTL				GM_IO (81)
/* Don't forget to modify GM_NUM_IO if you add an ioctl!!! */
#define GM_NUM_IO 82
#endif /* GM_SET_FLAGS */

/* Flags for GM_GET_RQST */

#define GM_GET_RQST_NONBLOCKING 0x01

				/* ioctl requests */

/* This structure is shared between 64-bit kernels and 32-bit user
   programs, so it should use only sized types. */
/* The fact that this structure is not a multiple of 64-bit was causing 
   a problem for 32 bit apps on 64 bit kernels (specifically x86_64) 
   -- nelson */
typedef struct gm_off_len
{
  gm_s64_t offset;
  gm_u64_t len;
  gm_u32_t permissions;
  gm_u32_t padding; /* make this structure a multiple of 64 bits */
}
gm_off_len_t;

struct gm_off_len_uvma
{
  gm_s64_t offset;
  gm_u64_t len;
  gm_up_t uvma;
};

struct gm_off_len_uvma_pvma
{
  gm_s64_t offset;
  gm_u64_t len;
  gm_up_t uvma;
  gm_up_t pvma;
};

struct gm_len_pvma
{
  gm_u64_t len;
  gm_up_t pvma;
};

typedef struct gm_mapping_specs
{
  struct gm_off_len control_regs;
  struct gm_off_len special_regs;
  struct gm_off_len sram;
  struct gm_off_len hash_piece_ptrs;
  struct gm_off_len send_queue;
  struct gm_off_len copy_block;
  struct gm_off_len RTC;
}
gm_mapping_specs_t;

typedef struct gm_lanai_register_access
{
  gm_u32_t offset;
  gm_u32_t value;
} gm_lanai_register_access_t;

typedef struct gm_node_type_info
{
  gm_u32_t node_type;
  gm_u8_t mac_addr[6];
}
gm_node_type_info_t;

typedef struct gm_set_route_info
{
  gm_u8_t mac_addr[6];
  gm_u8_t length;
  gm_u8_t route[GM_MAX_NETWORK_DIAMETER];
}
gm_set_route_info_t;

typedef struct gm_route_info
{
  gm_u32_t target_node_id;
  gm_u8_t length;
  gm_u8_t route[GM_MAX_NETWORK_DIAMETER];
}
gm_route_info_t;

#if	GM_OS_AIX
#define GM_GLOBALS_REQUEST_BUFF_SIZE	4096
#else	/* GM_OS_AIX */
#define GM_GLOBALS_REQUEST_BUFF_SIZE	1024
#endif	/* GM_OS_AIX */
typedef struct gm_globals_request
{
  gm_u32_t offset;
  gm_u32_t len;
  gm_u8_t buffer[GM_GLOBALS_REQUEST_BUFF_SIZE];
}
gm_globals_request_t;

typedef struct gm_receive_queue_page_description
{
  gm_up_t user_pointer;
  gm_u32_t page_number;
}
gm_receive_queue_page_description_t;

/* YP ioctl I/O buffer format. */

typedef struct gm_yp_io
{
  unsigned int timeout_usecs;	/* in */
  unsigned int node_id;		/* in/out */
  char key[GM_MAX_YP_STRING_LEN]; /* in */
  char value[GM_MAX_YP_STRING_LEN]; /* in/out */
} gm_yp_io_t;

char *_gm_io_control_code_name (unsigned int);

#endif /* _gm_io_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
