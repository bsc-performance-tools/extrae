/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_arch_types_h_
#define _gm_arch_types_h_

/**********************************************************************
 * This file architecture-specific types and macros needed by gm_impl.h
 **********************************************************************/

#ifndef __KERNEL__
#define __KERNEL__ 1
#endif

/* minimum linux include files to have the required types declared here */
#include <linux/autoconf.h>
#include <linux/version.h>
#include <linux/stddef.h>
#include <linux/timer.h>
#include <asm/page.h>
#include <linux/wait.h>
#include <asm/semaphore.h>
#include <asm/atomic.h>
#include <linux/errno.h>
#include <linux/if_ether.h>
#include <linux/string.h>
#include <linux/netdevice.h>
#include <linux/devfs_fs_kernel.h>

#include "gm_ether.h"
#include "gm_arch_def.h"
#include "gm_types.h"


struct gm_instance_state;
struct gm_port_state;

typedef struct gm_arch_dma_region
{
  struct gm_instance_state *is;
  void *addr;			/* aligned address */
  int index;			/* for gm_arch_dma_region_advance */
  gm_dp_t *dma_list;
  unsigned int num_pages;
  unsigned int len;
  int flags;
  int region;
}
gm_arch_dma_region_t;


struct gm_port;

struct gm_linux_dma_info
{
  gm_ethernet_segment_descriptor_t desc[GM_MAX_ETHERNET_SCATTER_CNT];
  gm_dp_t dma[GM_MAX_ETHERNET_SCATTER_CNT];
  int scatter_cnt;
};


struct gm_linux_ether_pkt
{
  struct gm_linux_dma_info dma_info;
  struct sk_buff *skb;
};


typedef struct
{
  struct net_device *dev;	/* the network (ethernet driver) */
  struct gm_instance_state *is;
  struct gm_port *port;
  struct gm_linux_ether_pkt send_ring[GM_SEND_RING];
  struct gm_linux_ether_pkt recv_ring[GM_RECV_RING];
  int tx_full;
  struct net_device_stats stats;
  int sreq;
  int sreq_index;		
  int sdone;
  int sdone_index;
  unsigned rdone;
  int dead;			/* timeout, MCP may be dead */
  unsigned int mcp_flags;	/* ethernet flags for MCP */
}
gm_arch_net_info_t;

void gmip_finalize (struct gm_instance_state *is);
int gmip_init (struct gm_instance_state *is);

typedef struct gm_arch_instance_info
{
  unsigned long iomem_base;
  unsigned long phys_base_addr;
  struct gm_instance_state *next;
  unsigned long interrupt;	/* 1 if we are currently inside the
				   interrupt handler, 0 otherwise, 
				   this should be a long because of
				   test_and_set_bit */
  struct pci_dev *pci_dev;
  unsigned int irq;
  struct gm_linux_work_struct test_lanai_task;
  struct timer_list timer;
  gm_arch_net_info_t net;
  devfs_handle_t devfs_handle[2];

  /* Lock to hold during any send queue access on this board */
  spinlock_t ethernet_send_lock;

  atomic_t free_iommu_pages;
}
gm_arch_instance_info_t;


/************
 * GM synchronization types
 ************/


/*  quite the same than Solaris */
typedef struct gm_arch_sync
{
  /* fields for mutex aquire/release */
  struct semaphore mutex;
  /* fields for sleep/wake */
  struct semaphore wake_sem;
  /* fields for sleep/wake */
  atomic_t wake_cnt;
  wait_queue_head_t sleep_queue;
}
gm_arch_sync_t;

typedef struct gm_arch_port_info
{
  int ref_count;		/* open file descriptor + mapping count */
  struct mm_struct *mm;
  unsigned long send_queue_addr;
  gm_arch_sync_t sync;
}
gm_arch_port_info_t;


#if GM_CAN_REGISTER_MEMORY
typedef struct
{
  struct gm_port_state *ps;
  struct page *page;
  struct inode *inode;
  long vm_flags;
  gm_phys_t phys;
  int magic;
  long virt_pagenum;
  gm_dp_t dma_handle;
}
gm_arch_page_lock_t;
#else
typedef char gm_arch_page_lock_t;
#endif


struct gm_port_state;

typedef struct
{
  struct vm_operations_struct wrap;
  struct 
  {
    struct vm_operations_struct *ori;
    gm_u32_t magic;
    struct vm_area_struct *vma;
    struct vm_area_struct *vma_to_dewrap;
  } gm;
}
gm_linux_vm_ops_t;

typedef int gm_arch_minor_t;
typedef int gm_arch_ioctl_context_t;
typedef atomic_t gm_atomic_t;

/* Lock the ethernet send queue, so that it may be accessed both from
   a kernel thread and the interrupt handler.

   As a special pragmatic exception to the usual coding conventions,
   GM_ARCH_LOCK_ETHER_SEND_QUEUE may declare variables than may be
   referenced by GM_ARCH_UNLOCK_ETHERNET_SEND_QUEUE. */

#define GM_ARCH_LOCK_ETHER_SEND_QUEUE(port)				\
unsigned long flags;								\
spin_lock_irqsave							\
(&(port)->kernel_port_state->instance->arch.ethernet_send_lock, flags)

#define GM_ARCH_UNLOCK_ETHER_SEND_QUEUE(port)				\
spin_unlock_irqrestore							\
(&(port)->kernel_port_state->instance->arch.ethernet_send_lock, flags)

#endif /* _gm_arch_types_h_ */
