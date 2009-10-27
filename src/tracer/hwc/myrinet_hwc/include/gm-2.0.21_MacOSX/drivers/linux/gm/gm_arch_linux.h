#ifndef _gm_arch_linux_h
#define _gm_arch_linux_h


#ifndef LINUX_VERSION_CODE
#include <linux/version.h>
#endif

#ifndef VERSION_CODE
#define VERSION_CODE(vers,rel,seq) ( ((vers)<<16) | ((rel)<<8) | (seq) )
#endif

#if LINUX_VERSION_CODE < VERSION_CODE(2,2,0)
#warning "This driver does not support Linux 2.1.x or earlier."
#error "Please use Linux 2.4.x or 2.2.x."
#elif LINUX_VERSION_CODE < VERSION_CODE(2,3,0)
#define LINUX_22 1
#define LINUX_24 0
#define LINUX_XX 22
#elif LINUX_VERSION_CODE < VERSION_CODE(2,4,0)
#warning "The use of unstable kernel release is discouraged."
#warning "Myricom help line will not support such configuration."
#define LINUX_22 0
#define LINUX_24 1
#define LINUX_XX 24
#elif LINUX_VERSION_CODE < VERSION_CODE(2,5,0)
#define LINUX_22 0
#define LINUX_24 1
#define LINUX_XX 24
#elif LINUX_VERSION_CODE < VERSION_CODE(2,7,0)
#define LINUX_22 0
#define LINUX_24 1
#define LINUX_XX 26
#else
#error "Unsupported Linux kernel release."
#endif

#define REGISTER_SYMTAB(tab)
#include <linux/poll.h>
#include <linux/sched.h>
#include <asm/uaccess.h>
#include <asm/io.h>
#include <linux/types.h>

#ifdef __alpha__
#include <asm/machvec.h>
#if LINUX_22
#include <asm/core_pyxis.h>
#endif
#endif


/* 
 *
 *  Part I: Transition from 2.2 to 2.4
 *
 */
#if LINUX_VERSION_CODE >= VERSION_CODE(2,2,18)
#include <linux/spinlock.h>
#endif

#if LINUX_VERSION_CODE < VERSION_CODE(2,2,18)
typedef struct wait_queue *wait_queue_head_t;
#define init_waitqueue_head(a) init_waitqueue(a)
#define init_MUTEX(s) do { *(s) = MUTEX; } while (0)
#define DECLARE_WAITQUEUE(a,b) struct wait_queue a = {(b), 0}
#endif


#if LINUX_XX <= 22

/* definition implementing the 2.4 and later APIS on top of 2.2 APIS */
#include <linux/pci.h>

#define PCI_ANY_ID (~0)
#define VXX_MAX_DEV_PER_DRIVER 16

struct pci_device_id
{
  unsigned int vendor;		/* Vendor ID or PCI_ANY_ID */
  unsigned int device;		/* Device ID or PCI_ANY_ID */
  unsigned int subvendor;	/* Subsystem ID's or PCI_ANY_ID */
  unsigned int subdevice;	/* Subsystem ID's or PCI_ANY_ID */
  unsigned int class;		/* (class,subclass,prog-if) triplet */
  unsigned int class_mask;
  unsigned long driver_data;	/* Data private to the driver */
};

struct pci_driver
{
  char *name;
  const struct pci_device_id *id_table;
  int (*probe) (struct pci_dev * dev, const struct pci_device_id * id);
  void (*remove) (struct pci_dev * dev);
  struct pci_dev *devs[VXX_MAX_DEV_PER_DRIVER];
};


int pci_module_init (struct pci_driver *);
void pci_unregister_driver (struct pci_driver *);

#define module_init(x) int init_module(void) { return x(); }
#define module_exit(x) void cleanup_module(void) { x(); }
#define dev_kfree_skb_any(a) dev_kfree_skb(a)
#define dev_kfree_skb_irq(a) dev_kfree_skb(a)
#define vma_private_data(v) ((void*)(v)->vm_pte)
#define virt_to_page(a) (mem_map+MAP_NR(a))
#define dev_start(a,b) do { (a)->start = (b) ; } while (0)
#define vma_get_pgoff(vma) ((vma)->vm_offset / PAGE_SIZE)
#define netif_wake_queue(dev) \
do { (dev)->tbusy = 0; mark_bh(NET_BH); } while (0)
#define netif_start_queue(dev) do { (dev)->tbusy = 0; } while (0)
#define netif_stop_queue(dev) do { (dev)->tbusy = 1; } while (0)
#define netif_queue_stopped(dev) ((dev)->tbusy)

#define net_device_stats enet_statistics
#define net_device device

#define pci_resource_start(pdev,i) \
((pdev)->base_address[i] & PCI_BASE_ADDRESS_MEM_MASK)

struct net_device *gm_linux_dev_alloc (char *, int *, int num);
#define gm_linux_free_netdev kfree

#define pci_set_dma_mask(pdev,a) (0)

#define devfs_register(dir,name,flags,major,minor,mode,ops,data) 0
#define devfs_unregister(handle)
#define devfs_register_chrdev register_chrdev
#define devfs_unregister_chrdev unregister_chrdev
typedef void * devfs_handle_t;

#define THIS_MODULE (&__this_module)

extern inline 
int schedule_task(struct tq_struct *task)
{
  queue_task(task, &tq_scheduler);
  return 1;
}

#define gm_mmap_up_write(a)
#define gm_mmap_down_write(a)

#define gm_linux_page_inode(page) ((page)->inode)

#endif /* LINUX_22 */

#if LINUX_XX >= 24

/* The stuff here is mainly to support early 2.4 kernels while using
   some functionality introduced only later in the series. */

#include <linux/devfs_fs_kernel.h>
#include <asm/system.h>
#include <linux/module.h>
#include <linux/kdev_t.h>

#if LINUX_VERSION_CODE < VERSION_CODE(2,4,3)
#define pci_set_dma_mask(pdev,a) (0)
#endif

#define vma_get_pgoff(vma) ((vma)->vm_pgoff)
#define dev_start(a,b)
#if LINUX_XX >= 26
struct net_device *gm_linux_dev_alloc (char *, int *, int num);
#define gm_linux_free_netdev free_netdev
#else
#define gm_linux_dev_alloc(s,err,num) dev_alloc(s,err)
#define gm_linux_free_netdev kfree
#endif

#if defined(HAVE_MMAP_UP_WRITE)
#define gm_mmap_up_write mmap_up_write
#define gm_mmap_down_write mmap_down_write
#elif LINUX_VERSION_CODE < VERSION_CODE(2,4,3)
#define gm_mmap_up_write(a) up(&(a)->mmap_sem)
#define gm_mmap_down_write(a) down(&(a)->mmap_sem)
#elif defined __alpha__ && LINUX_VERSION_CODE < VERSION_CODE(2,4,7)
/* __down_write and __up_write not exported */
/* not getting the semaphore is still safe because we have the page_table_lock
   and the kernel_lock anyway */
#define gm_mmap_up_write(a)
#define gm_mmap_down_write(a)
#elif defined __powerpc__ && LINUX_VERSION_CODE == VERSION_CODE(2,4,4)
/* __down_write and __up_write not defined/exported */
/* not getting the semaphore is still safe for this version
   because we have the page_table_lock and the kernel_lock anyway */
#define gm_mmap_up_write(a)
#define gm_mmap_down_write(a)
#else
#define gm_mmap_up_write(a) up_write(&(a)->mmap_sem)
#define gm_mmap_down_write(a) down_write(&(a)->mmap_sem)
#endif

#ifndef read_cr4
#define read_cr4() ({				\
        unsigned int __dummy;			\
        __asm__(				\
                "movl %%cr4,%0\n\t"		\
                :"=r" (__dummy));		\
        __dummy;				\
})
#endif

struct inode;

static inline struct inode *
gm_linux_vma_inode(struct vm_area_struct *vma)
{
  if ((vma->vm_flags & VM_SHARED) && vma->vm_file 
      && vma->vm_file->f_dentry)
    {
      return vma->vm_file->f_dentry->d_inode;
    } else {
      return 0;
    }
}
#endif /* LINUX_XX >= 24 */

/* version independant definitions */

#ifndef MODULE_LICENSE
#define MODULE_LICENSE(license)
#endif

/*
 * Figure out what, if any, high-memory support routines to use
 */
#ifdef HAVE_PTE_KUNMAP /* suse version */

#define pte_unmap pte_kunmap
#define pte_unmap_nested pte_unmap
#define pte_offset_map pte_offset
#define pte_offset_map_nested pte_offset

#elif defined HAVE_PTE_OFFSET_MAP_NESTED /* linux 2.5 version */

/* nothing to do since we use this interface in our code */

#elif defined HAVE_PTE_OFFSET_MAP /* redhat version */

#define pte_offset_map_nested pte_offset_map2
#define pte_unmap_nested pte_unmap2

#else /* not using high pte stuff */

#define pte_offset_map pte_offset
#define pte_offset_map_nested pte_offset
#define pte_unmap(pte)
#define pte_unmap_nested(pte)

#endif /* >= 24 */

#ifdef HAVE_REMAP_PAGE_RANGE_5ARGS
#define gm_linux_remap_page_range(vma,start,phys,len,prot) \
   remap_page_range ((vma),(start),(phys),(len),(prot))
#define gm_linux_io_remap_page_range(vma,start,phys,len,prot) \
   io_remap_page_range ((vma),(start),(phys),(len),(prot))
#else
#define gm_linux_remap_page_range(vma,start,phys,len,prot) \
   remap_page_range ((start),(phys),(len),(prot))
#define gm_linux_io_remap_page_range(vma,start,phys,len,prot) \
   io_remap_page_range ((start),(phys),(len),(prot))
#endif

#ifdef HAVE_REMAP_PFN_RANGE
 #define gm_linux_remap_pfn_range remap_pfn_range
#else
 #define gm_linux_remap_pfn_range(vma,start,pfn,len,prot) \
          gm_linux_remap_page_range(vma,start,(pfn)<<PAGE_SHIFT,len,prot)
#endif

#if GM_CPU_powerpc64 && LINUX_XX >= 24
/* 
   this linux port forgot to export the io_remap_page_range definition ,
   we will use remap_page_range, but that requires to use *real*
   physical addresses that might be quite different from pci_resource_start
*/
#undef gm_linux_io_remap_page_range
#define gm_linux_io_remap_page_range gm_linux_remap_page_range
#endif

#ifndef minor
#define minor(a) MINOR(a)
#endif

#ifndef set_current_state
#define set_current_state(x) do { current->state = (x); } while (0)
#endif

/*
 *
 * Part 2 : Transition from 2.4=> 2.6
 *
 */


/* pci_get/set_drvdata functions */
#if LINUX_XX <= 22
#define pci_get_drvdata(pdev) ((void*)(pdev)->base_address[5])
#define pci_set_drvdata(pdev,d) (((void*)(pdev)->base_address[5]) = d)
#elif ! defined HAVE_PCI_SET_DRVDATA
#define pci_get_drvdata(pdev) ((pdev)->driver_data)
#define pci_set_drvdata(pdev,d) ((pdev)->driver_data = (d))
#endif

/*
 *devfs stuff */

#if LINUX_XX >= 26
typedef char * devfs_handle_t;
#endif

#if LINUX_XX <= 24
#define GM_MOD_INC_USE_COUNT MOD_INC_USE_COUNT
#define GM_MOD_DEC_USE_COUNT MOD_DEC_USE_COUNT
#else
#define GM_MOD_INC_USE_COUNT
#define GM_MOD_DEC_USE_COUNT
#endif

#include <linux/interrupt.h>
#ifndef IRQ_HANDLED
#define irqreturn_t void
#define IRQ_HANDLED
#define IRQ_NONE
#define IRQ_RETVAL(a)
#endif

/* work queue stuff */
#if LINUX_XX <= 24

#include <linux/tqueue.h>
#define gm_linux_work_struct tq_struct
#define GM_LINUX_INIT_WORK(w,f,d) \
             do { (w)->routine = f ; (w)->data = d; } while (0)
#define gm_linux_schedule_work schedule_task
#if LINUX_XX <= 22
#define gm_linux_flush_scheduled_work() schedule()
#else
#define gm_linux_flush_scheduled_work flush_scheduled_tasks
#endif

#else /* LINUX_XX <= 24 */
/* >= 2.6 */

#define gm_linux_work_struct work_struct
#define GM_LINUX_INIT_WORK INIT_WORK
#define gm_linux_schedule_work schedule_work
#define gm_linux_flush_scheduled_work flush_scheduled_work
#endif /* >= 26*/

#if LINUX_XX <= 24
#define gm_linux_module_get(a) MOD_INC_USE_COUNT
#define gm_linux_module_put(a) MOD_DEC_USE_COUNT
#else
#define gm_linux_module_get(a) __module_get(a)
#define gm_linux_module_put(a) module_put(a)
#endif

#ifndef TestSetPageLocked
#define TestSetPageLocked TryLockPage
#endif

#if LINUX_XX <= 24
#define wait_on_page_locked wait_on_page
#endif

#ifndef UnlockPage
#define UnlockPage unlock_page
#endif

#if LINUX_XX >= 26
#define gm_linux_page_count page_count
#else
#define gm_linux_page_count(page) atomic_read(&(page)->count)
#endif

#ifdef VM_HUGETLB
#define gm_linux_hugetlb(flags) ((flags) & VM_HUGETLB)
#else
#define gm_linux_hugetlb(flags) 0
#endif

#endif /* gm_arch_linux */

