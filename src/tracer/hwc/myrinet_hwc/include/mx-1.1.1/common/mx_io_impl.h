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

/* This file defines some details to implement the interface between the MX 
   library and driver.  OS-specific overrides and extensions should go in
   the architecture's "mx_arch_io.h" file. */

/* Allow the architecture to include any requisite header file and
   override definitions here. */

#ifndef _mx_io_impl_h_
#define _mx_io_impl_h_

#if defined HAVE_SYS_ERRNO_H && !defined MX_KERNEL && !defined MX_MCP
#include <sys/errno.h>
#endif

#include "mx_constants.h"
#include "mx_arch_io.h"
#include "mx_io.h"
#include "mcp_requests.h"

#if defined HAVE_SYS_TYPES_H && !defined MX_MCP
#include <sys/types.h>
#endif

#ifdef WIN32
#ifdef MX_KERNEL
#include <devioctl.h>
#else
#include <windows.h>
#include <winioctl.h>
#endif
#endif

#define MX_NOBLOCK 1
#define MX_NDELAY 2

#ifndef MX_IO
#define MX_IO(command) (('M' << 8) + (command))
#endif

/* when adding an ioctl, don't forget to update the
   MX_NUM_IOCTLS define! */

/* Please don't renumber ioctls.  This means don't add one in the
   middle, and if you remove one, mark it as MX_UNUSED_IOCTL */

#define MX_TEST_IOCTL                            MX_IO ( 1)
#define MX_SET_ENDPOINT                          MX_IO ( 2)
#define MX_WAIT                                  MX_IO ( 3)
#define MX_CRASHDUMP                             MX_IO ( 4)
#define MX_GET_COPYBLOCKS                        MX_IO ( 5)
#define MX_GET_INSTANCE_COUNT                    MX_IO ( 6)
#define MX_GET_MAX_INSTANCE                      MX_IO ( 7)
#define MX_GET_NIC_ID                            MX_IO ( 8)
#define MX__REGISTER                          	 MX_IO ( 9)
#define MX__DEREGISTER                           MX_IO (10)
#define MX_GET_MAX_SEND_HANDLES			 MX_IO (11)
#define MX_GET_COUNTERS				 MX_IO (12)
#define MX_CLEAR_COUNTERS			 MX_IO (13)
#define MX_PIN_SEND                              MX_IO (14)
#define MX_GET_COUNTERS_STRINGS                  MX_IO (15)
#define MX_NIC_ID_TO_BOARD_NUM                   MX_IO (16)
#define MX_GET_LOGGING                           MX_IO (17)
#define MX_GET_NUM_PORTS                         MX_IO (18)
#define MX_GET_MAX_ENDPOINTS                     MX_IO (19)
#define MX_GET_LOGGING_STRINGS                   MX_IO (20)
#define MX_NIC_ID_TO_PEER_INDEX                  MX_IO (21)
#define MX_GET_MAPPER_MSGBUF_SIZE                MX_IO (22)
#define MX_GET_MAPPER_MAPBUF_SIZE                MX_IO (23)
#define MX_GET_MAPPER_MSGBUF                     MX_IO (24)
#define MX_GET_MAPPER_MAPBUF                     MX_IO (25)
#define MX_GET_PEER_FORMAT                       MX_IO (26)
#define MX_GET_ROUTE_SIZE                        MX_IO (27)
#define MX_GET_PEER_TABLE                        MX_IO (28)
#define MX_GET_ROUTE_TABLE                       MX_IO (29)
#define MX_GET_MAX_PEERS                         MX_IO (30)
#define MX_PAUSE_MAPPER                          MX_IO (31)
#define MX_RESUME_MAPPER                         MX_IO (32)
#define MX_GET_SERIAL_NUMBER                     MX_IO (33)
#define MX_GET_OPENER                            MX_IO (34)
#define MX_MMAP					 MX_IO (35)
#define MX_MUNMAP				 MX_IO (36)
#define MX_NIC_ID_TO_HOSTNAME			 MX_IO (37)
#define MX_HOSTNAME_TO_NIC_ID			 MX_IO (38)
#define MX_CLEAR_PEER_NAMES			 MX_IO (39)
#define MX_SET_HOSTNAME				 MX_IO (40)
#define MX_GET_CACHELINE_SIZE			 MX_IO (41)
#define MX_GET_MAX_EVENT_SIZE      		 MX_IO (42)
#define MX_GET_SMALL_MESSAGE_THRESHOLD		 MX_IO (43)
#define MX_GET_LINK_STATE		 	 MX_IO (44)
#define MX_GET_MEDIUM_MESSAGE_THRESHOLD		 MX_IO (45)
#define MX_SET_ROUTE		 		 MX_IO (46)
#define MX_SET_ROUTE_BEGIN	 		 MX_IO (47)
#define MX_SET_ROUTE_END	 		 MX_IO (48)
#define MX_PIN_RECV                              MX_IO (49)
#define MX_GET_MAX_RDMA_WINDOWS			 MX_IO (50)
#define MX_GET_IRQ_COUNTERS			 MX_IO (51)
#define MX_RUN_DMABENCH			 	 MX_IO (52)
#define MX_CLEAR_WAIT			 	 MX_IO (53)
#define MX_WAKE			 	 	 MX_IO (54)
#define MX_RAW_GET_PARAMS	 	 	 MX_IO (55)
#define MX_RAW_CLEAR_ROUTES	 	 	 MX_IO (56) 
#define MX_GET_BOARD_STATUS 	 	 	 MX_IO (57)
#define MX_GET_CPU_FREQ 	 	 	 MX_IO (58)
#define MX_GET_PCI_FREQ 	 	 	 MX_IO (59)
#define MX_SET_RAW	 	 	 	 MX_IO (60)
#define MX_RAW_SEND	 	 	 	 MX_IO (61)
#define MX_RAW_GET_NEXT_EVENT 	 	 	 MX_IO (62)
#define MX_RAW_TICKS 	 	 	 	 MX_IO (63)
#define MX_SET_MAPPER_STATE			 MX_IO (64)
#define MX_GET_MAPPER_STATE			 MX_IO (65)
#define MX_GET_MIN_LIGHT_ENDPOINTS               MX_IO (66)
#define MX_GET_MAX_LIGHT_ENDPOINTS               MX_IO (67)
#define MX_GET_INTR_COAL			 MX_IO (68)
#define MX_SET_INTR_COAL			 MX_IO (69)
#define MX_RECOVER_ENDPOINT			 MX_IO (70)
#define MX_REMOVE_PEER				 MX_IO (71)
#define MX_PEER_INDEX_TO_NIC_ID			 MX_IO (72)
#define MX_DIRECT_GET     			 MX_IO (73)
#define MX_WAKE_ENDPOINT     			 MX_IO (74)
#define MX_GET_PRODUCT_CODE    			 MX_IO (75)
#define MX_GET_PART_NUMBER    			 MX_IO (76)
#define MX_APP_WAIT				 MX_IO (77)
#define MX_APP_WAKE				 MX_IO (78)
#define MX_GET_VERSION				 MX_IO (79)
#define MX_GET_SRAM_SIZE			 MX_IO (80)
#define MX_GET_DUMP_REG_COUNT			 MX_IO (81)
#define MX_NUM_IOCTLS                            81

#endif /* _mx_io_impl_h_ */
