#ifndef _gm_arch_def_h_
#define _gm_arch_def_h_

#include "gm.h"
#include "gm_config.h"
#include "gm_arch_linux.h"


/* this is hardcoded in the makefile */
#define GM_MAJOR 220
#define GM_MAJOR_OLD 41


/* variables to tune */
#define GM_ARCH_MAX_MINOR 256
#define GM_ARCH_MAX_INSTANCE 16

#ifdef __alpha__
#define GM_BC_PAGE_LEN 8192
#elif __intel__
#define GM_BC_PAGE_LEN 4096
#endif


/* Values of flags passed to
   dma_region_alloc() to indicate the
   type of receive buffer to allocate */

#define GM_ARCH_DMA_READ	0x10
#define GM_ARCH_DMA_WRITE	0x20
#define GM_ARCH_DMA_RDWR	(GM_ARCH_DMA_READ | GM_ARCH_DMA_WRITE)
#define GM_ARCH_DMA_CONSISTENT  0x40
#define GM_ARCH_DMA_STREAMING   0x80
#define GM_ARCH_DMA_CONTIGUOUS  0x100

#define GM_ARCH_INTR_CLAIMED         1
#define GM_ARCH_INTR_UNCLAIMED       2
#define GM_ARCH_SYNC_FOR_DEVICE      3
#define GM_ARCH_SYNC_FOR_CPU         4

#define GM_WOKE                      8
#define GM_SLEEP_TIMED_OUT           16



/************
 * Error notification values, used as return values for most driver functions.
 ************/

#define GM_EACCES		EACCES
#define GM_EBUSY		EBUSY
#define GM_EFAULT       	EFAULT
#define GM_EINVAL		EINVAL
#define GM_ENOMEM		ENOMEM
#define GM_ENOTTY		ENOTTY
#define GM_EPERM		EPERM
#define GM_EPROTO		EPROTO
#define GM_EUNATCH		EUNATCH

/************
 * Debugging output macros
 ************/

/* make these all KERN_INFO to simplify finding debugging printout */
#define GM_ARCH_KERN_INFO   KERN_WARNING
#define GM_ARCH_KERN_NOTICE KERN_WARNING
#define GM_ARCH_KERN_CRIT   KERN_WARNING
#define GM_ARCH_KERN_DEBUG  KERN_WARNING
#define GM_ARCH_KERN_WARNING KERN_WARNING
#define GM_ARCH_KERN_ERR    KERN_WARNING



#ifndef __GNUC__
#error gcc is required for this driver
#endif


#define GM_ARCH_INFO(args) gm_linux_info args
#define GM_ARCH_NOTE(args) gm_linux_note args
#define GM_ARCH_PANIC(args) gm_linux_panic args
#define GM_ARCH_PRINT(args) gm_linux_print args
#define GM_ARCH_WARN(args) gm_linux_warn args

#ifdef CONFIG_X86_PAE
typedef gm_u64_t gm_phys_t;
#if GM_SIZEOF_DP_T != 8
#error gm_dp_t should be 8 for X86/PAE kernels
#endif
#else
typedef unsigned long gm_phys_t;
#endif
#define gm_linux_pfn(phys) ((unsigned long)((phys)>> PAGE_SHIFT))

/* PAGE_OFFSET should be the first physical page of memory, that
   should works even if memory does not begin at physical address 0 */
#define GM_LINUX_PFN_ZERO (__pa((void*)PAGE_OFFSET) / PAGE_SIZE)
#if LINUX_XX >= 26
#define GM_LINUX_PFN_MAX (GM_LINUX_PFN_ZERO + num_physpages)
#else
#define GM_LINUX_PFN_MAX (GM_LINUX_PFN_ZERO + max_mapnr)
#endif
#define GM_LINUX_KERNEL_PFN_MAX (__pa((void*)high_memory) / PAGE_SIZE)

#if GM_CPU_alpha
/* alpha architecture */

/*
  - for 21264: the bit 43 is necessary, it  marks non-cacheable space.
  - for 21164: the non-cacheable bit (bit 39) is included in the base
  address of the PCI memory region.
  - for 21264: this unfortunately cannot be the case because Linux use
  a 41 bit KSEG (it would need a 48KSEG segment)
  - we always set the bit in linux >= 2.2, it will be ignored by 21164 
  harware.
*/
#define GM_LINUX_PAGE_CACHE 0
#define GM_LINUX_PAGE_NOCACHE (1L<<(43+19))


#define GM_LINUX_IOMEM2PHYS(a) (a)

#define GM_LINUX_PFN_MASK (PAGE_MASK & ((1UL<<41) - 1))
/* KSEG currently limited to 41 bit, altough there seems to
   be some support for 48 bit forthcoming
   on EV6 bit 1<<40 is sign extended when using
   other bit are matched again KSEG selector, 
   and should be removed from phys
*/
#define GM_LINUX_PHYS_FROM_PTE(a) \
((pte_val(a)>>(32-PAGE_SHIFT)) & GM_LINUX_PFN_MASK)
#endif

#if GM_CPU_x86 || GM_CPU_x86_64
/* i386 architecture */

#ifndef _PAGE_PWT
#define _PAGE_PWT   0x008
#endif

/* add in no write-combining too */
#define GM_LINUX_PAGE_NOCACHE (_PAGE_PCD | _PAGE_PWT)
#define GM_LINUX_PAGE_CACHE 0
#define GM_LINUX_IOMEM2PHYS(a) (a)
#if GM_CPU_x86_64
#define GM_LINUX_PFN_MASK (PHYSICAL_PAGE_MASK)
#elif defined CONFIG_X86_PAE
#define GM_LINUX_PFN_MASK (0x0ffffffffffff000ULL)
#else
#define GM_LINUX_PFN_MASK (~(gm_phys_t)(PAGE_SIZE-1))
#endif
#define GM_LINUX_PHYS_FROM_PTE(a) (pte_val(a) & GM_LINUX_PFN_MASK)
#endif


#if GM_CPU_ia64
/* ia64 architecture */
#define GM_LINUX_PAGE_NOCACHE _PAGE_MA_UC
#define GM_LINUX_PAGE_CACHE 0
#define GM_LINUX_IOMEM2PHYS(a) (a)
/* somehow walking the page table seems to result in an unexpected 1
   in bit 52, so lets clean out all the bits not concern
   above 51. --nelson
   no longer a terrible ia64 2.3.99 horrible hack in this form -- loic */
#define GM_LINUX_PFN_MASK _PFN_MASK
#define GM_LINUX_PHYS_FROM_PTE(a) (pte_val(a) & GM_LINUX_PFN_MASK)
#endif


#if GM_CPU_sparc && ! defined __sparc_v9__
/* sparc32 architecture */
/* ... */
#endif


#if GM_CPU_sparc64
/* sparc64 architecture */
#define GM_LINUX_PAGE_NOCACHE _PAGE_E
#define GM_LINUX_PAGE_CACHE _PAGE_CACHE
#define GM_LINUX_IOMEM2PHYS(a) (a)
/* Physical Address bits [40:13]      */
#define GM_LINUX_PFN_MASK  0x000001FFFFFFE000
#define GM_LINUX_PHYS_FROM_PTE(a) (pte_val(a) & GM_LINUX_PFN_MASK)
#endif


#if GM_CPU_powerpc
#define GM_LINUX_PAGE_NOCACHE (_PAGE_NO_CACHE | _PAGE_GUARDED)
#define GM_LINUX_PAGE_CACHE 0

/* ioremap expect phys address for all ppc subarchs */
#define GM_LINUX_IOMEM2PHYS(a) (a)
#define GM_LINUX_PFN_MASK PAGE_MASK
#define GM_LINUX_PHYS_FROM_PTE(a) (pte_val(a) & GM_LINUX_PFN_MASK)
#endif

#if GM_CPU_powerpc64
#define GM_LINUX_PAGE_NOCACHE (_PAGE_NO_CACHE | _PAGE_GUARDED)
#define GM_LINUX_PAGE_CACHE 0
#define GM_LINUX_IOMEM2PHYS(a) (a)
#define GM_LINUX_PFN_MASK PAGE_MASK
#define GM_LINUX_PHYS_FROM_PTE(a) \
   ((pte_val(a) >> (PTE_SHIFT - PAGE_SHIFT)) & GM_LINUX_PFN_MASK)
#endif



#define gm_linux_current current
#define gm_linux_up up
#define gm_linux_down down

#define HAS_GET_MODULE_SYMBOL

#define GM_SEND_RING GM_NUM_SEND_TOKENS
#define GM_RECV_RING (GM_NUM_ETHERNET_RECV_TOKENS / 2)
#define GM_SEND_RING_MAX_INDEX (GM_SEND_RING -1)
#define GM_RECV_RING_MAX_INDEX (GM_RECV_RING -1)


#define GM_ARCH_GFP_REGION 1
#define GM_ARCH_VMALLOC_REGION 2

#define GM_SLEEP_WOKE 0
#define GM_SLEEP_INTERRUPTED 1

#define USER_LOCK_DEAD 0x6598731
#define USER_LOCK_ALIVE 0x635290

#define GM_HAS_ATOMIC_T 1



#define GM_SLEEP_PRINT 0

#define GM_USE_INTR_LOCK 0
/* to disable interrupts while sending a ethernet packet,
   and serialize all GM code: normally not needed */

extern spinlock_t gm_linux_intr_lock;
extern struct gm_hash *gm_linux_pfn_hash;
extern int (*gm_linux_get_user_pages)(struct task_struct *tsk, 
				      struct mm_struct *mm,
				      unsigned long start, int len,
				      int write, int force,
				      struct page **pages,
				      struct vm_area_struct **vmas);

#endif /* _gm_arch_def_h_ */
