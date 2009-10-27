/*****************************************************************-*-c-*-
 * Myricom GM networking software and documentation		 	*
 * Copyright (c) 1999 by Myricom, Inc.				 	*
 * All rights reserved.	 See the file `COPYING' for copyright notice.	*
 ************************************************************************/


/************************************************************************
 * This file includes the OS-specific driver code for Linux 2.[024].
 *
 * written by: John Regehr <regehr@myri.com>
 *           Loic Prylli <loic@myri.com>:
 *           Nelson Escobar <nelson@myri.com>
 *           Bob Felderman <feldy@myri.com>
 *           Patrick Geoffray <patrick@myri.com>
 *
 * Additional credits:
 *           Pete Wyckoff <pw@osc.edu>: 2.4 highmem fixes
 *
 * send questions to: help@myri.com
 *
 * This code was derived from Bob Felderman's Myrinet API driver for
 * Linux, and Glenn Brown's GM driver for Solaris.
 *
 * Also see:
 *   gm/drivers/linux/gm/README, for supported platforms
 *   gm/drivers/linux/gm/TODO, for future work 
 *   _Linux Device Drivers_, Alessandro Rubini, O'Reilly, 1998
 *   _Linux Kernel Internals_, Michael Beck et. al, Addison-Wesley, 1997
 *
 ************************************************************************/

#include <linux/config.h>
#include <linux/module.h>

/* first for proper asm/io.h inclusion on Alpha */
#include "gm_internal.h"
#include "gm_call_trace.h"
#include "gm_page_hash.h"
#include "gm_lanai.h"
#include "gm_instance.h"
#include "gm_malloc_debug.h"
#include "gm_klog_debug.h"
#include "gm_debug_lanai_dma.h"
#include "gm_debug_malloc.h"
#include "gm_impl.h"
#include "gm_debug_board_init.h"
#include "gm_enable_fork_system.h"
#include "gm_enable_security.h"

#include <linux/smp.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <asm/page.h>
#include <asm/pgtable.h>
#include <asm/unistd.h>
#include <linux/delay.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/module.h>
#include <linux/file.h>
#include <linux/mman.h>
#include <linux/swap.h>
#include <linux/ioport.h>

#if GM_SUPPORT_PCI
#include <linux/pci.h>
#if (GM_CPU_powerpc || GM_CPU_powerpc64) && LINUX_XX >= 24
#include <asm/pci-bridge.h>
#endif /* GM_CPU_powerpc || GM_CPU_powerpc64) && LINUX_XX >= 24*/

#if GM_CPU_powerpc64
#if LINUX_XX >= 26
#include <asm/iommu.h>
typedef struct iommu_table * gm_ppc64_iommu_t;
#define GM_PPC64_IOMMU(dev) (PCI_GET_DN(dev)->iommu_table)
#define GM_PPC64_IOMMU_SIZE(iommu) ((iommu)->it_mapsize)
#else
typedef struct TceTable * gm_ppc64_iommu_t;
#define GM_PPC64_IOMMU(dev) (((struct device_node *)((dev)->sysdata))->tce_table)
#define GM_PPC64_IOMMU_SIZE(tbl) (tbl->size*(PAGE_SIZE/sizeof(union Tce)))
#include <asm/prom.h>
#include <asm/pci_dma.h>
#endif
#endif

#endif /*GM_SUPPORT_PCI*/

#include <linux/ptrace.h>
#include <linux/string.h>
#include <linux/utsname.h>

#include <linux/vmalloc.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#ifdef HAVE_LINUX_COMPILE_H
#include <linux/compile.h>
#endif

#if LINUX_24
#include <linux/init.h>
#endif

#include <linux/version.h>
#include <asm/io.h>
#include <asm/segment.h>
#include <linux/string.h>
#include <asm/page.h>

#if LINUX_XX >= 26
#include <linux/cdev.h>
static struct class_simple *gm_class;
#endif

#include "gm_arch_def.h"
#include "gm_arch_pci_map.h"

#ifdef USE_ZLIB
#include "zlib.h"
#endif

/* convert CSPI_MAP26xx to use 0/1 instead of def/undef */
#ifdef CSPI_MAP26xx
#undef CSPI_MAP26xx
#define CSPI_MAP26xx 1
#else
#define CSPI_MAP26xx 0
#endif

#if CSPI_MAP26xx == GM_SUPPORT_PCI
#error one and only one of CSPI_MAP26xx and GM_SUPPORT_PCI must be defined
#endif


#ifdef HAVE_GET_UNMAPPED_AREA_6ARGS
#define GM_LINUX_EXEC_SHIELD_ARG ,0
#else
#define GM_LINUX_EXEC_SHIELD_ARG
#endif

#if defined HAVE_SET_PAGE_DIRTY || LINUX_VERSION_CODE >= KERNEL_VERSION (2,4,18)
/* HAVE_SET_PAGE_DIRTY requires kernel source (not just headers), so
   avoid most false negatives by also testing kernel version */
#define GM_LINUX_HAVE_SET_PAGE_DIRTY 1
#else
#define GM_LINUX_HAVE_SET_PAGE_DIRTY 0
#endif

extern int (*smp_call_function_symbol)(void);
extern int (*kmap_high_symbol)(void);
/****************
 * Storage
 ****************/

struct semaphore gm_linux_open_mutex;
spinlock_t gm_linux_intr_lock;
gm_arch_sync_t gm_linux_pfn_sync;
unsigned gm_linux_udev;


/************************************************************************
 * Forward declarations (used to build tables)
 ************************************************************************/

static int gm_linux_ioctl (struct inode *inodeP, struct file *fileP,
			   unsigned int cmd, unsigned long arg);
static int gm_linux_mmap (struct file *file, struct vm_area_struct *vma);
static int gm_linux_open (struct inode *inodeP, struct file *fileP);
static unsigned long gm_linux_get_area (struct file *, unsigned long, unsigned long,
			      unsigned long, unsigned long);

/* this one called when file descriptor release */
static int gm_linux_close (struct inode *inodeP, struct file *fileP);

#if GM_SUPPORT_PCI
static int gm_linux_init_one (struct pci_dev *pdev,
			      const struct pci_device_id *ent);
static void gm_linux_remove_one (struct pci_dev *pdev);
#endif
static gm_status_t gm_linux_create_instance (enum gm_bus_type,
					     void *bus_data);

void gm_linux_cleanup_module (void);
static void gm_linux_destroy_instance (gm_instance_state_t *is);

static unsigned long gm_linux_pci_dev_base (struct pci_dev *dev);

static int gm_linux_localize_status (gm_status_t status);

/***********************************************************************
 * Loadable module required data structures
 ***********************************************************************/

static struct file_operations gm_linux_file_ops = {
  ioctl:gm_linux_ioctl,
  mmap:gm_linux_mmap,
  open:gm_linux_open,
  release:gm_linux_close,
  get_unmapped_area:gm_linux_get_area,
  owner: THIS_MODULE
};

#if GM_SUPPORT_PCI
static struct pci_device_id gm_pci_tbl[] = {
  {GM_PCI_VENDOR_MYRICOM, GM_PCI_DEVICE_MYRINET,
   PCI_ANY_ID, PCI_ANY_ID, 0, 0, 0},
  {GM_PCI_VENDOR_MYRICOM2, GM_PCI_DEVICE_MYRINET,
   PCI_ANY_ID, PCI_ANY_ID, 0, 0, 0},
  {0,},
};

static struct pci_driver gm_driver = {
  name:"Myricom GM driver",
  probe:gm_linux_init_one,
  remove:gm_linux_remove_one,
  id_table:gm_pci_tbl,
};
#endif


/***********************************************************************
 * Globals
 ***********************************************************************/

static gm_instance_state_t *gm_linux_instances[GM_ARCH_MAX_INSTANCE];
static int gm_linux_num_instances = 0;
static int gm_linux_pci_driver_registered = 0;
static atomic_t gm_linux_max_user_locked_pages;
static int gm_linux_max_user_locked_pages_start;
static int gm_linux_skip_init;
struct gm_hash *gm_linux_pfn_hash = (struct gm_hash *) 0;
static unsigned long gm_activate_page_symbol = 0;
static unsigned long gm_sprintf_symbol = 0;

/* with little mem, reserve half memory */
/* Found that 3/4 was too much on some boxes with linux-2.4 */
#define GM_MAX_USER_LOCKED_SMALLMEM(x) (((x)*4)/8)
/* with average amount of memory, preserve a fix amount */
#define GM_MAX_SAVE_FROM_LOCKED ((unsigned long)(64*1024*1024)/PAGE_SIZE)
/* with a lot of mem, divide first to avoid overflow,
   and reserve a part proportional to memsize */
#define GM_MAX_USER_LOCKED_BIGMEM(x) (((x)/8)*4)


#if GM_DEBUG
int gm_linux_print_level = 0;
static gm_instance_state_t *gm_linux_debug_is;
#endif

#if GM_PERR_POLLING
static struct completion gm_linux_perr_thread_exited;
static pid_t gm_linux_perr_pid;
static int gm_linux_perr_thread_should_exit = 0;
#endif

#if LOIC_LANAI_DBG
/* loic: use under kernel source debugging to take a clone
   of LANAI variable in main memory translating it endianess/pointers */
#include "gm_lanai_dump.h"

static gm_dump_lanai_globals_t *gm_linux_dbg_clone;

static void
gm_linux_dbg (void)
{
  if (gm_linux_dbg_clone)
    {
      gm_dump_lanai_globals_t *g = gm_linux_dbg_clone;
      gm_linux_dbg_clone = 0;
      gm_pseudo_free (g);
    }
  gm_linux_dbg_clone = gm_pseudo_global_clone (gm_linux_instances[0]);
}
#endif


/*
 * poor man's memory leak detection
 */
#if GM_DEBUG
static int kmalloc_cnt = 0, kfree_cnt = 0;
static int vmalloc_cnt = 0, vfree_cnt = 0;
static int ioremap_cnt = 0, iounmap_cnt = 0;
static int dma_alloc_cnt = 0, dma_free_cnt = 0;
static int kernel_alloc_cnt = 0, kernel_free_cnt = 0;
static int user_lock_cnt = 0, user_unlock_cnt = 0;
#endif


/****************************************************************
 ****************************************************************
 * Low level architecture dependent functions
 ****************************************************************
 ****************************************************************/

unsigned long
gm_arch_max_locked_pages (void)
{
  return (gm_linux_max_user_locked_pages_start);
}

/************
 * gm_port_state initialization 
 ************/

/* This is called just after the port state is created (in gm_minor.c)
   to perform architecture-specific initialization. */

gm_status_t
gm_arch_port_state_init (gm_port_state_t * ps)
{
  GM_CALLED ();

  gm_arch_sync_init (&ps->arch.sync, ps->instance);
  ps->arch.ref_count = 1;
  GM_RETURN_STATUS (GM_SUCCESS);
}


/* This is called just before the port state is destroyed (in
   gm_minor.c) to perform architecture-specific finalization. */

void
gm_arch_port_state_fini (gm_port_state_t * ps)
{
  GM_CALLED ();
  
  GM_PARAMETER_MAY_BE_UNUSED (ps);
  gm_arch_sync_destroy (&ps->arch.sync);
  GM_RETURN_NOTHING ();
}


/************
 * gm_port_state initialization 
 ************/

/* This is called at the end of gm_port_state_open() to perform architecture-
   specific initialization. */

gm_status_t
gm_arch_port_state_open (gm_port_state_t * ps)
{
  GM_CALLED ();
  GM_PARAMETER_MAY_BE_UNUSED (ps);
  GM_RETURN_STATUS (GM_SUCCESS);
}


/* This is called at the start of gm_port_state_close to perform
   architecture-specific finalization. */

void
gm_arch_port_state_close (gm_port_state_t * ps)
{
  GM_CALLED ();
  GM_PARAMETER_MAY_BE_UNUSED (ps);
  GM_RETURN_NOTHING ();
}



/***********************************************************************
 * Utility functions
 ***********************************************************************/

/* These are required so the gm_ioctl( ) can copy its arguments
   and results. */

gm_status_t
gm_arch_copyin (const void *what, void *where, gm_size_t amount)
{
  GM_CALLED_WITH_ARGS (("0x%p,0x%p,%ld", what, where, amount));
  copy_from_user (where, what, amount);
  GM_RETURN_STATUS (GM_SUCCESS);
}

gm_status_t
gm_arch_copyout (const void *what, void *where, gm_size_t amount)
{
  GM_CALLED_WITH_ARGS (("0x%p,0x%p,%ld", what, where, amount));
  copy_to_user (where, what, amount);
  GM_RETURN_STATUS (GM_SUCCESS);
}


/* atomic manipulation of an integer */
void
gm_arch_atomic_set (gm_atomic_t * v, gm_u32_t val)
{
  atomic_set (v, val);
}

gm_u32_t
gm_arch_atomic_read (gm_atomic_t * v)
{
  return atomic_read (v);
}


/* Utility functions for generic error printing */

/* The MCP printf functionality might print up to 1024 bytes via a
   GM_WRITE_INTERRUPT, and the code below adds slop to the
   beginning. --Glenn */

struct gm_print_foobar_struct
{
  char buf[10000];
  gm_u32_t after_buf;
};

static struct gm_print_foobar_struct gm_linux_print_struct;


static void
gm_linux_puts (const char *buf)
{
  /* Linux printk will corrupt memory if the buffer length is much
     longer than this. */
  const gm_size_t max_print_len = 256;
  gm_size_t buf_len;
  char minibuf[256];

  /* verify that printf HACKs did not corrupt kernel memory. */
  gm_assert (gm_linux_print_struct.after_buf == 0xcafebabe);

  /* Perform oversize prints by printing in small parts that
     have been copied to a safe buffer. */
  for (buf_len = strlen (buf);
       buf_len > max_print_len;
       buf_len -= sizeof (minibuf) - 1, buf += sizeof (minibuf) - 1)
    {
      gm_assert (gm_linux_print_struct.after_buf == 0xcafebabe);
      gm_assert (buf_len == strlen (buf));

      gm_strncpy (minibuf, buf, sizeof (minibuf) - 1);
      minibuf[sizeof (minibuf) - 1] = 0;
      printk ("%s", minibuf);
    }
  gm_assert (buf_len == strlen (buf));

  /* print the remainder of the packet */
  gm_assert (strlen (buf) <= max_print_len);
  printk ("%s", buf);
}

void
gm_linux_vprintk (const char *format, va_list ap)
{
  static spinlock_t print_lock = SPIN_LOCK_UNLOCKED;
  unsigned long flags;

  /* mark the end of the buff */
  (gm_linux_print_struct.buf + sizeof (gm_linux_print_struct.buf))[-1] = 0;

  spin_lock_irqsave (&print_lock, flags);
  /* print to the buffer */
  vsprintf (gm_linux_print_struct.buf, format, ap);

  /* Check for sprintf overflow */
  if ((gm_linux_print_struct.buf
       + sizeof (gm_linux_print_struct.buf))[-1] != 0)
    {
      /* warn of sprintf overflow. */
      gm_linux_puts (KERN_CRIT
		     "******** gm_linux_print_struct.buf[] vsprintf() "
		     "overflow ********\n");
      /* minimize the damage of the overflow by terminating the buffer */
      (gm_linux_print_struct.buf
       + sizeof (gm_linux_print_struct.buf))[-1] = 0;
    }

  /* print the formatted message */
  gm_linux_puts (gm_linux_print_struct.buf);
  spin_unlock_irqrestore (&print_lock, flags);
}


/****************************************************************
 * Replacements for linux printk(), which is broken for prints longer
 * than about 256 bytes.  This one should work for messages as large
 * as gm_linux_print_struct.buf. 
 ****************************************************************/

void
gm_linux_printk (const char *format, ...)
{
  va_list ap;

  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

void
gm_linux_info (const char *format, ...)
{
  va_list ap;

  gm_linux_printk (GM_ARCH_KERN_INFO);
  gm_linux_printk ("GM: ");
  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

void
gm_linux_note (const char *format, ...)
{
  va_list ap;

  gm_linux_printk (GM_ARCH_KERN_NOTICE);
  gm_linux_printk ("GM: ");
  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

void
gm_linux_panic (const char *format, ...)
{
  va_list ap;

  gm_linux_printk (GM_ARCH_KERN_CRIT);
  gm_linux_printk ("GM: ");
  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

void
gm_linux_print (const char *format, ...)
{
  va_list ap;

  gm_linux_printk (GM_ARCH_KERN_DEBUG);
  gm_linux_printk ("GM: ");
  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

void
gm_linux_warn (const char *format, ...)
{
  va_list ap;

  gm_linux_printk (GM_ARCH_KERN_WARNING);
  gm_linux_printk ("GM: ");
  va_start (ap, format);
  gm_linux_vprintk (format, ap);
  va_end (ap);
}

static int gm_linux_in_interrupt;
typedef long gm_jmp_buf[256];

#if GM_DEBUG
static gm_jmp_buf intr_jmp_buf;

#if GM_DEBUG_SETJMP
/* I do not want to reboot at each little assertion failed that happen 
   inside interrupt handler, so let's setjmp/longjmp to return instead 
   not that this isn't safe when using several boards: global variable... */
/* 256 is a conservative number, 1k should be enough to save state */

extern int __setjmp (gm_jmp_buf);
extern int __longjmp (gm_jmp_buf, int);

int
__sigjmp_save (gm_jmp_buf __env, int __savemask)
{
  GM_PARAMETER_MAY_BE_UNUSED (__env);
  GM_PARAMETER_MAY_BE_UNUSED (__savemask);

  return 0;
}
#endif /* !GM_DEBUG_SETJMP */

extern long *sys_call_table_symbol;
typedef int (*sys_exit_func) (int);

static void
gm_linux_hack_sys_exit (int val)
{
  if (sys_call_table_symbol)
    (*(sys_exit_func)(sys_call_table_symbol[__NR_exit])) (val);
  else
    panic ("gm_linux_hack_sys_exit() called with no syscall table");
}
#endif


/****************
 * Setjmp/longjump support (for debugging only)
 ****************/

/* GM_LINUX_DEBUG_SIGSETJMP: if possible, setup state for a later
   gm_linux_debug_longjmp.  This is a macro because it cannot be
   safely put in a function.

   Return 0 on the first return, and the nonzero value specified by the
   user on subsequent returns. */

#if GM_DEBUG_SETJMP
#define GM_LINUX_DEBUG_SIGSETJMP(buf, val) __sigsetjmp (buf, val)
#else
/* The longjmp will never happen, so just return 0. */
#define GM_LINUX_DEBUG_SIGSETJMP(buf, val) 0
#endif

/* If possible, perform the __longjmp set up by an earlier call to
   GM_DEBUG_SETJMP.  val must be nonzero and will be returned by
   GM_DEBUG_SETJMP. */

GM_FUNCTION_MAY_BE_UNUSED
static void
gm_linux_debug_longjmp (gm_jmp_buf jmp_buf, int retval)
{
#if GM_DEBUG_SETJMP
  printk (KERN_EMERG "Aie, GM-PANIC inside interrupt, "
	  "	let go out of this\n");
  __longjmp (jmp_buf, retval);
#else
  /* do nothing */
#endif
}

/****************
 * Arch abort routine
 ****************/

void
gm_arch_abort (void)
{
#if !GM_DEBUG
  panic ("gm_arch_abort() called");
#else  /* GM_DEBUG */
  if (gm_linux_debug_is && gm_linux_debug_is->lanai.running)
    {
      gm_disable_lanai (gm_linux_debug_is);
    }

  if (gm_linux_debug_is
      && atomic_read (&gm_linux_debug_is->page_hash.sync.mutex.count) == 0)
    {
      gm_arch_mutex_exit (&gm_linux_debug_is->page_hash.sync);
    }

  if (gm_linux_debug_is
      && atomic_read (&gm_linux_debug_is->command_sync.mutex.count) == 0)
    {
      gm_arch_mutex_exit (&gm_linux_debug_is->command_sync);
    }

  if (gm_linux_in_interrupt)
    {
      gm_linux_debug_longjmp (intr_jmp_buf, 1);
      /* If we get here, we could not recover using a longjmp. */
      panic ("GM-PANIC in interrupt handler: cannot recover");
    }

  if (current->state & PF_EXITING)
    {
      /* we probably failed in the close procedure, so we will
         never execute the dec count in linux_close */
      GM_MOD_DEC_USE_COUNT;
    }
  gm_linux_hack_sys_exit (12);
#endif /* GM_DEBUG */
}

int
gm_linux_is_regular_mapping (struct vm_area_struct *vma)
{
  static struct vm_operations_struct *pf_ops;
  struct vm_operations_struct *ops = vma->vm_ops;
  struct file *file = vma->vm_file;

  if (!ops || ops == pf_ops)
    {
      return 1;
    }
  if (1
#if LINUX_XX <= 22
      && !(ops->open || ops->close
	|| ops->unmap || ops->protect || ops->sync
	|| ops->wppage || ops->swapout)
#else
      && file && file->f_dentry && file->f_dentry->d_inode
      && S_ISREG(file->f_dentry->d_inode->i_mode)
#endif
      )
    {
      /* ops points to file_generic__mmap in linux/mm/filemap.c */
      pf_ops = ops;
      return 1;
    }
  else
    {
      GM_WARN (("register memory with vm_ops %p (flags=%lx, open = %p, close = %p)\n",
		ops, vma->vm_flags, ops->open, ops->close));
      return 0;
    }
}


/*
 * find the physical address of either a kernel page or a user pager
 * by walking the page tables (see Rubini p.287)
 *
 * NOTE: cannot be called from interrupt handler since we use the
 * current MM context
 */

#define GM_DEBUG_KVIRT_TO_PHYS 0

static pgd_t *gm_linux_pgd_offset (unsigned long addr, int kernel)
{
  if (kernel)
    {
      /* get kernel page table */
      return  pgd_offset_k (addr);
    }
  else
    {
      /* use current->mm to access process page table */
      return pgd_offset (current->mm, addr);
    }
}


gm_phys_t
gm_linux_kvirt_to_phys (gm_instance_state_t * is,
			unsigned long addr, int kernel)
{
  pgd_t *pgd;
  pmd_t *pmd;
#ifdef PUD_SHIFT
  pud_t *pud;
#else
  pgd_t *pud;
#endif


  pte_t *pte;
  gm_phys_t phys;
  void *ptr;

  GM_CALLED ();
  
  GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("(0x%lx, %d)\n", addr, kernel));

  if (kernel)
    {
      _GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("kernel\n"));
      /* if kernel:
         if vaddr in low range, conversion is done by translation (most cases).
         if vaddr after high_memory (vmalloc), we deal with the segment offset
         via VMALLOC_VMADDR */

      ptr = (void *) addr;
      if ((addr >= PAGE_OFFSET) && (addr < (unsigned long) high_memory))
	{
	  _GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("low\n"));
	  GM_RETURN (__pa (ptr));
	}

#if !CSPI_MAP26xx
      _GM_PRINT
	(GM_DEBUG_KVIRT_TO_PHYS,
	 ("base=%p limit=0x%p ptr=%p\n",
	  is->board_base, is->board_base + is->board_span, ptr));
      gm_assert (is->board_base);
      if ((ptr >= is->board_base)
	  && ((unsigned long) ptr < (unsigned long) is->board_base
	      + is->board_span))
	{
	  _GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("board\n"));
	  GM_RETURN ((is->arch.phys_base_addr + (unsigned long) ptr
		      - (unsigned long) is->board_base));
	}
#endif

#if !GM_CPU_powerpc
      /* beware, some variables are not exported on ppc 2.2.x */
      gm_assert (addr >= VMALLOC_START && addr < VMALLOC_END);
#endif
    }
  
  _GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("out of kernel\n"));
  
  pgd = gm_linux_pgd_offset(addr,kernel);
  if (!pgd)
    {
      GM_WARN (("gm_linux_kvirt_to_phys: unable to get PGD\n"));
      GM_RETURN (0);
    }

  /* first level */
  if (pgd_none (*pgd))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: no PGD\n"));
      GM_RETURN (0);
    }
  if (pgd_bad (*pgd))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: bad PGD\n"));
      GM_RETURN (0);
    }

#ifdef PUD_SHIFT
  pud = pud_offset (pgd, addr);
  
  if (pud_none (*pud))
    {
      return (0);
    }
  if (!pud_present (*pud))
    {
      return (0);
    }
  if (pud_bad (*pud))
    {
      GM_WARN (("mx_kvirt_to_page: bad PUD\n"));
      return (0);
    }
#else
  pud = pgd;
#endif

  pmd = pmd_offset (pud, addr);

  if (pmd_none (*pmd))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: no PMD\n"));
      GM_RETURN (0);
    }
  if (!pmd_present (*pmd))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: PMD not present\n"));
      GM_RETURN (0);
    }
  if (pmd_bad (*pmd))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: bad PMD\n"));
      GM_RETURN (0);
    }

  /* last level */
  pte = pte_offset_map (pmd, addr);
  if (pte_none (*pte))
    {
      GM_WARN (("gm_linux_kvirt_to_phys: no PTE\n"));
      goto pte_error;
    }
  if (!pte_present (*pte))
    {
      GM_INFO (("gm_linux_kvirt_to_phys: PTE not present\n"));
      goto pte_error;
    }
  if (!pte_write (*pte))
    {
      GM_INFO (("gm_linux_kvirt_to_phys: PTE not writable\n"));
      goto pte_error;
    }

  /* do not use virt_to_phys, that would not work for IO */
  phys = GM_LINUX_PHYS_FROM_PTE (*pte);
  GM_PRINT (GM_DEBUG_KVIRT_TO_PHYS, ("returning page 0x%lx\n", 
				     gm_linux_pfn (phys)));
  pte_unmap (pte);
  GM_RETURN (phys);

 pte_error:
  pte_unmap(pte);
  GM_RETURN (0);
}


/****************************************************************
 * memory allocation
 ****************************************************************/

/****************
 * bookkeeping
 ****************/

#define GM_DEBUG_MEM_ALLOCATION 0
#if !GM_DEBUG_MEM_ALLOCATION
#define record_memory_allocation(x,y,z) GM_SUCCESS
#define check_mem
#else
static struct allocation_record *first_allocation_record;

struct allocation_record
{
  struct allocation_record *next;
  void *ptr;
  unsigned long len;
  void *free;
};

static gm_status_t
record_memory_allocation (void *ptr, unsigned long len, void *free)
{
  struct allocation_record *record;

  record = kmalloc (sizeof (*record), GFP_KERNEL);
  if (!record)
    {
      goto abort_with_nothing;
    }
  record->next = first_allocation_record;
  record->ptr = ptr;
  record->len = len;
  record->free = free;
  first_allocation_record = record;
  return GM_SUCCESS;

 abort_with_nothing:
  return GM_FAILURE;
}


static gm_status_t
unrecord_memory_allocation (void *ptr, unsigned long len, void *free)
{
  struct allocation_record **where, *match;

  for (where = &first_allocation_record; *where; where = &(*where)->next)
    {
      if ((*where)->ptr == ptr
	  && (*where)->len == len && (*where)->free == free)
	{
	  match = *where;
	  *where = match->next;
	  kfree (match);
	  return GM_SUCCESS;
	}
    }
  return GM_FAILURE;
}


static void
print_memory_allocation_leaks (void)
{
}
#endif


/****************
 * kmalloc
 ****************/

static void *
gm_linux_kmalloc (unsigned int size, int priority)
{
#if GM_DEBUG
  kmalloc_cnt++;
#endif
  return kmalloc (size, priority);
}

static void
gm_linux_kfree (void *obj)
{
#if GM_DEBUG
  kfree_cnt++;
#endif
  kfree (obj);
}


/****************
 * vmalloc
 ****************/

static void *
gm_linux_vmalloc (unsigned long size)
{
#if GM_DEBUG
  vmalloc_cnt++;
#endif
  return vmalloc (size);
}

static void
gm_linux_vfree (void *obj)
{
#if GM_DEBUG
  vfree_cnt++;
#endif
  vfree (obj);
}

static void *
gm_linux_ioremap (unsigned long phys, unsigned long size)
{
  void *ptr;

  GM_CALLED ();
  
#if GM_DEBUG
  ioremap_cnt++;
#endif
  GM_PRINT (GM_PRINT_LEVEL >= 2, ("Mapping IO at 0x%llx\n", (gm_u64_t)phys));
#if GM_CPU_x86
  ptr = (void *) __ioremap (phys, size, _PAGE_PCD | _PAGE_PWT);
#else
  ptr = (void *) ioremap (phys, size);
#if GM_CPU_powerpc64 && defined IO_TOKEN_TO_ADDR
  ptr = (void *) IO_TOKEN_TO_ADDR(ptr);
#endif
#endif
  GM_PRINT (GM_PRINT_LEVEL >= 2, ("IO is mapped at 0x%p\n", ptr));
  GM_RETURN_PTR (ptr);
}

static void
gm_linux_iounmap (void *obj)
{
  GM_CALLED ();
  
#if GM_DEBUG
  iounmap_cnt++;
#endif
  iounmap (obj);

  GM_RETURN_NOTHING ();
}

#if GM_SUPPORT_PCI
/****************************************************************
 * PCI config space functions  
 ****************************************************************/

#define pcibios_to_gm_arch(rw, size, linuxname, c_type, star)		\
gm_status_t								\
gm_arch_##rw##_pci_config_##size (gm_instance_state_t *is,		\
				  gm_offset_t offset,			\
				  gm_u##size##_t star value)		\
{									\
  gm_assert (is);							\
  gm_assert (is->arch.pci_dev);						\
  return ((pci_##rw##_config_##linuxname (is->arch.pci_dev,		\
					  (unsigned char) offset,	\
					  (c_type star) value)		\
	   == PCIBIOS_SUCCESSFUL)					\
	  ? GM_SUCCESS							\
	  : GM_FAILURE);						\
}

pcibios_to_gm_arch (read, 32, dword, unsigned int, *);
pcibios_to_gm_arch (write, 32, dword, unsigned int,);
pcibios_to_gm_arch (read, 16, word, unsigned short, *);
pcibios_to_gm_arch (write, 16, word, unsigned short,);
pcibios_to_gm_arch (read, 8, byte, unsigned char, *);
pcibios_to_gm_arch (write, 8, byte, unsigned char,);

#endif /* GM_SUPPORT_PCI */

/****************************************************************
 * Synchronization functions  
 ****************************************************************/

void
gm_arch_sync_init (gm_arch_sync_t * s, gm_instance_state_t * is)
{
  GM_PARAMETER_MAY_BE_UNUSED (is);

  GM_PRINT (0, ("gm_arch_sync_init() called\n"));
  init_MUTEX (&s->mutex);
  init_MUTEX (&s->wake_sem);
  atomic_set (&s->wake_cnt, 0);
  init_waitqueue_head (&s->sleep_queue);
}

void
gm_arch_sync_reset (gm_arch_sync_t * s)
{
  atomic_set (&s->wake_cnt, 0);
}

void
gm_arch_sync_destroy (gm_arch_sync_t * s)
{
  GM_PARAMETER_MAY_BE_UNUSED (s);
}

void
gm_arch_mutex_enter (gm_arch_sync_t * s)
{
  GM_PARAMETER_MAY_BE_UNUSED (s);

  down (&(s->mutex));
}

void
gm_arch_mutex_exit (gm_arch_sync_t * s)
{
  GM_PARAMETER_MAY_BE_UNUSED (s);

  up (&(s->mutex));
}


/*****************************************************************
 * Sleep functions
 *****************************************************************/

/* The interrupt handler increments WAKE_CNT each time a wake interrupt
   is received and the user threads decrementing WAKE_CNT each time they
   claim a wake interrupt.  User threads accessing WAKE_CNT must disable
   interrupts and hold the WAKE_CNT_LOCK during the access to ensure the
   accesses are atomic. */

/****************
 * waking
 ****************/

/* Wake the thread sleeping on the synchronization variable. */

void
gm_arch_wake (gm_arch_sync_t * s)
{
  GM_PRINT (GM_SLEEP_PRINT, ("gm_arch_wake called on s = %p\n", s));

  /* record the wake interrupt by incrementing the wake count.  This
     need to be atomic because disabling interrupt globally on SMP 
     is very costly. */

  atomic_inc (&s->wake_cnt);
  wake_up (&s->sleep_queue);
}


/****************
 * sleeping
 ****************
 
 The following code claims a wake interrupt by atomically testing for a
 positive WAKE_CNT and decrementing WAKE_CNT.  We can assume we are the
 only one trying to consume wake_cnt, the caller is responsible to get a
 mutex to ensure this, so wake_cnt can only increase while we are here.
 A basic Linux rule: if you need to disable interrupts globally, your
 code is not written the right way :-) */

/* sleep until awakend or timeout */

gm_arch_sleep_status_t
gm_arch_timed_sleep (gm_arch_sync_t * s, int seconds)
{
  long timeout;
  gm_arch_sleep_status_t ret = GM_SLEEP_WOKE;
  DECLARE_WAITQUEUE (wait, current);

  GM_CALLED ();
  
  GM_PRINT (GM_SLEEP_PRINT, ("gm_arch_timed_sleep  s = %p  sec=%d\n",
			     s, seconds));
  gm_log_call_trace ();
  
  /* take the mutex */
  down (&s->wake_sem);
  timeout = seconds * HZ;

  /* put the process in the queue before testing the event */
  add_wait_queue (&s->sleep_queue, &wait);
  while (timeout > 0 && atomic_read (&s->wake_cnt) <= 0)
    {
      /* use UN(INTERRUPTIBLE) variant to prevent signals */
      set_current_state (TASK_UNINTERRUPTIBLE);
      /* test again wake_cnt after setting current->state to avoid
         race condition */
      if (atomic_read (&s->wake_cnt) <= 0)
	timeout = schedule_timeout (timeout);
      /* reset state to RUNNING in case the if was not taken */
      set_current_state (TASK_RUNNING);
    }

  remove_wait_queue (&s->sleep_queue, &wait);

  if (atomic_read (&s->wake_cnt) <= 0)
    {
      /* no interrupt, timed out */
      gm_always_assert (timeout <= 0);
      ret = GM_SLEEP_TIMED_OUT;
    }
  else
    {
      /* claims the interrupt */
      atomic_dec (&s->wake_cnt);
    }

  /* release the mutex */
  up (&s->wake_sem);
  GM_RETURN (ret);
}


/* sleep until awakened or get a signal (protected against multiple 
   usage, it should not be necessary, but it seems there is no 
   protection for sleep_sync in gm.c) */

gm_arch_sleep_status_t
gm_arch_signal_sleep (gm_arch_sync_t * s)
{
  gm_arch_sleep_status_t ret = GM_SLEEP_WOKE;

  GM_CALLED ();
  
  GM_PRINT (GM_SLEEP_PRINT, ("gm_arch_signal_sleep  s = %p\n", s));

  /* take the lock */
  if (down_interruptible (&s->wake_sem))
    GM_RETURN (GM_SLEEP_INTERRUPTED);

  /* sleep until explicitly awakened or interrupted. */

  if (wait_event_interruptible(s->sleep_queue, atomic_read (&s->wake_cnt) > 0))
    {
      ret = GM_SLEEP_INTERRUPTED;
    }
  else
    {
      gm_always_assert(atomic_read (&s->wake_cnt) > 0);
      atomic_dec (&s->wake_cnt);
    }

  /* release the mutex */
  up (&s->wake_sem);
  GM_RETURN (ret);
}


/*********************************************************************
 * kernel memory reservation 
 *********************************************************************/

static void
gm_linux_reserve_page (gm_instance_state_t * is, gm_phys_t phys)
{
  gm_assert (gm_linux_phys_is_valid_page (is, phys));
  set_bit (PG_reserved, &(GM_LINUX_PHYS_TO_PAGE (is, phys))->flags);
}


static void
gm_linux_unreserve_page (gm_instance_state_t * is, gm_phys_t phys)
{
  gm_assert (gm_linux_phys_is_valid_page (is, phys));
  clear_bit (PG_reserved, &(GM_LINUX_PHYS_TO_PAGE (is, phys))->flags);
}


/***************************************************************
 * User memory page locking/unlocking
 ***************************************************************/
static void gm_linux_activate_page(struct page * page)
{
  /* locked page on some linux versions might jam the inactive_dirty
     list, if we ensure they are active after being locked they should
     never end-up there */
#if defined HAVE_ASM_RMAP_H && LINUX_XX == 24
  if (gm_activate_page_symbol && LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,10))
    {
      typedef void FASTCALL(page_func(struct page*));
      page_func *func;
      func = (page_func *)gm_activate_page_symbol;
      (*func)(page);
    }
#elif LINUX_XX >= 26
  SetPageReferenced(page);
  mark_page_accessed(page);
#endif
}

gm_status_t
gm_arch_lock_user_buffer_page (gm_port_state_t *ps,
			       gm_up_t in,
			       gm_dp_t * dma_addr, gm_arch_page_lock_t * lock)
{
  gm_phys_t phys;
  unsigned long pfn;
  int *user_data = (int *)(gm_size_t) in;
  int tmp;
  struct page *page;
  struct vm_area_struct *vma;
  int *counter_ptr = (int *) 0;
  gm_instance_state_t *is;
  gm_status_t status;
  gm_status_t map_stat;
  int retry = 0;
  int was_locked, regular_inode;
  struct inode *inode;

  GM_CALLED ();
  
  GM_PRINT (GM_PRINT_LEVEL >= 4,
	    ("gm_arch_LOCK_user_buffer_page(%p addr=0x%lx)\n",
	     lock, (unsigned long) in));

  gm_assert (ps);
  is = ps->instance;
  gm_assert (is);
  
  if (verify_area (VERIFY_WRITE, (void *)(gm_size_t) in, GM_PAGE_LEN))
    {
      GM_NOTE (("verify_write failed for page %p\n", user_data));
      GM_RETURN (GM_MEMORY_FAULT);
    }

 retry_swapin:
  /* should bring the page into physical memory and solve 
     copy-on-write problems */
  if (get_user (tmp, user_data) || put_user (tmp, user_data))
    {
      GM_NOTE (("EFAULT while trying to lock-register page 0x%p\n",
		user_data));
      GM_RETURN (GM_MEMORY_FAULT);
    }

  /* 2.2.x is SMP compliant but there is still a big kernel lock that 
     protect everythin, so no risk of race condition. 2.4.x is fully 
     multithreaded, we need to be carefull */

  gm_mmap_down_write (current->mm);
  vma = find_vma (current->mm, (unsigned long) in);
  gm_always_assert (vma && vma->vm_start <= (unsigned long) in);
  inode = gm_linux_vma_inode (vma);
  regular_inode = inode && S_ISREG(inode->i_mode);

  lock->inode = 0;
  if (!regular_inode)
    {
      spin_lock (&current->mm->page_table_lock);
      phys = gm_linux_kvirt_to_phys (is, (unsigned long) in, 0);
      if (!phys)
	{
	  if (!retry++)
	    {
	      GM_INFO (("kvirt_to_phys(0x%lx) failed (will retry):OK\n",
			(unsigned long) in));
	      spin_unlock (&current->mm->page_table_lock);
	      gm_mmap_up_write (current->mm);
	      goto retry_swapin;
	    }
	  GM_WARN (("kvirt_to_phys(0x%lx) failed\n", (unsigned long) in));
	  status = GM_FAILURE;
	  goto out_with_mm_lock;
	}
      
      page = gm_linux_phys_is_valid_page (is, phys) ?
	GM_LINUX_PHYS_TO_PAGE (is, phys) : 0;
      
      if (gm_linux_phys_on_board (is, phys))
	{
	  GM_WARN (("trying to register myrinet board space\n"));
	  status = GM_INVALID_PARAMETER;
	  goto out_with_mm_lock;
	}

      if (!page || PageReserved (page))
	{
#if GM_CPU_x86 || GM_CPU_ia64
	  *dma_addr = phys;
	  lock->page = 0;
	  GM_NOTE_ONCE (("GM program registering non-memory area: OK"));
	  status = GM_SUCCESS;
#else
	  GM_WARN (("trying to register non-memory area\n"));
	  status = GM_INVALID_PARAMETER;
#endif 
	  goto out_with_mm_lock;
	}
      
#if 0
      if (PageReserved (page))
	{
	  GM_WARN (("trying to register a mapping of a "
		    "reserved memory area (virt = 0x%lx  phys = 0x%lx)\n",
		    (unsigned long) in, (unsigned long) phys));
	  status = GM_INVALID_PARAMETER;
	  goto out_with_mm_lock;
	}
      
      if (page->buffers)
	{
	  GM_WARN (("page contains IO buffers!!!! - NOT registering it "
		    "(virt = 0x%lx  phys = 0x%lx\n",
		    (unsigned long) in, (unsigned long) phys));
	  status = GM_FAILURE;
	  goto out_with_mm_lock;
	}
#endif
      
      if (gm_linux_page_count (page) < 1)
	{
	  GM_WARN (("trying to register a page with count %d\n",
		    gm_linux_page_count (page)));
	  status = GM_INTERNAL_ERROR;
	  goto out_with_mm_lock;
	}
      
      if (atomic_read (&gm_linux_max_user_locked_pages) <= 0)
	{
	  GM_PRINT (GM_PRINT_LEVEL > 5,
		    ("cannot register memory for lack of physical resources\n"
		     "total pages = %ld, locked pages = %ld  (%ld MBytes)\n",
		     gm_linux_mem_nbpages (),
		     gm_arch_max_locked_pages (),
		     gm_arch_max_locked_pages () >> (20 - PAGE_SHIFT)));
	  status = GM_OUT_OF_MEMORY;
	  goto out_with_mm_lock;
	}
      /* ok now let's do it: increment page count and lock the page */

      get_page (page);
      was_locked = regular_inode || TestSetPageLocked (page);
      /* need to unlock mm before using gm_hash 
	 or anything that can allocate memory */
      spin_unlock (&current->mm->page_table_lock);

      if (vma->vm_ops && !(vma->vm_flags & VM_SHM)
	  && !gm_linux_is_regular_mapping (vma))
	{
	  GM_WARN (("cannot register specially mapped memory ops=%p "
		    "flags=0x%lx regular_mapping=%d, %s\n", vma->vm_ops, vma->vm_flags,
		    gm_linux_is_regular_mapping (vma),
		    ps->arch.mm == current->mm ? "" : "mm mismatch"));
	  status = GM_INVALID_PARAMETER;
	  gm_mmap_up_write (current->mm);
	  goto out_with_page_locked;
	}
    }
  else if (GM_LINUX_HAVE_SET_PAGE_DIRTY && gm_linux_get_user_pages)
    {
      if (gm_linux_get_user_pages(current, current->mm, (unsigned long)in, 1, 1,
			 0, &page, NULL) != 1)
	{
	  GM_WARN (("Unable to get_user_pages for 0x%lx\n",(long)in));
	  status = GM_FAILURE;
	  goto out_with_mmap_lock;
	}
      phys = page_to_phys(page);
      was_locked = 1;
      atomic_inc(&inode->i_count);
      lock->inode = inode;
    }
  else
    {
      GM_WARN (("Cannot register shared mapping\n"));
      status = GM_INVALID_PARAMETER;
      goto out_with_mmap_lock;
    }
  lock->vm_flags = vma->vm_flags;
  
  
  gm_mmap_up_write (current->mm);

  if (!was_locked)
    {
      gm_linux_activate_page (page);
    }
  pfn = gm_linux_pfn (phys);
  gm_arch_mutex_enter (&gm_linux_pfn_sync);
  counter_ptr = gm_hash_find (gm_linux_pfn_hash, &pfn);
  gm_arch_mutex_exit (&gm_linux_pfn_sync);

  if (counter_ptr)
    {
      gm_always_assert (was_locked);
      GM_PRINT (GM_PRINT_LEVEL >= 4,
               ("Page already locked by GM ref_cnt now = %d\n",
                counter_ptr[0]));
    }
  else if (!regular_inode && was_locked)
    {
      GM_INFO (("trying to register an already locked page:OK\n"));
      wait_on_page_locked (page);
      put_page (page);
      goto retry_swapin;
    }

  /* safe because phys is main memory, pci-mem mappings are handled above */
  map_stat = gm_linux_pci_map (is, phys, &lock->dma_handle);
  *dma_addr = lock->dma_handle;

  if (map_stat != GM_SUCCESS)
    {
      static int seen;
      if (!seen++)
	{
	  GM_INFO (("gm_linux_pci_map failed:OK\n"));
	}
      status = map_stat;
      goto out_with_page_locked;
    }

  if (counter_ptr)
    {
      counter_ptr[0]++;
    }
  else
    {
      int counter = 1;
      gm_arch_mutex_enter (&gm_linux_pfn_sync);
      status = gm_hash_insert (gm_linux_pfn_hash, &pfn, &counter);
      gm_arch_mutex_exit (&gm_linux_pfn_sync);
      if (status != GM_SUCCESS)
       {
         GM_NOTE (("gm_hash_insert(%p,%p,%p) for pfn failed\n",
                   gm_linux_pfn_hash, &pfn, &counter));
         goto out_with_dma_handle;
       }
      atomic_dec (&gm_linux_max_user_locked_pages);
    }


#if GM_DEBUG
  user_lock_cnt++;
#endif

#if 0 /* GM_ENABLE_FORK_SYSTEM*/
  /* loic: this is a clever hack from bob, which transforms fork into
     a sort of vfork so that registered memory continue to work see
     linux/mm/{memory.c,filemap.c}, it would definitely be very very
     dangerous and nasty to change the shared status of a file mapping
     (note the data area and the beginning of bss are usually a
     private file mapping, so that does restrict the use of fork). We
     also do not want the stack to be shared.  This should hopefully
     allow simple usual "fork followed by exec" to work, but no
     guarantee whatsoever, on 2.4 we can ensure fork does not mess the
     page table of the original process by the use of the VM_DONTCOPY
     flags for areas that are not shared, but with registered pages
     (which means if something in the stack has been registered the
     fork will probably segfaults, same thing if the new program tries
     to access a part in a private file mapping partly registered */
  if (!vma->vm_ops && !(vma->vm_flags & VM_GROWSDOWN))
    vma->vm_flags |= VM_SHARED;
#if LINUX_XX >= 24
  else if (!(vma->vm_flags & VM_SHARED))
    vma->vm_flags |= VM_DONTCOPY;
#endif
#endif

  lock->page = page;
  lock->phys = phys;
  lock->virt_pagenum = (unsigned long) in >> PAGE_SHIFT;
  lock->ps = ps;
  gm_assert (lock->magic == 0);
  lock->magic = USER_LOCK_ALIVE;
  if (!ps->arch.send_queue_addr || in < ps->arch.send_queue_addr)
    {
      /* print it only once per port */
      GM_INFO (("pid %d:fork() support limited: send_queue is not first vma\n",
		current->pid));
      ps->arch.send_queue_addr = 1;
    }
  GM_RETURN_STATUS (GM_SUCCESS);


 out_with_mm_lock:
  spin_unlock (&current->mm->page_table_lock);
 out_with_mmap_lock:
  gm_mmap_up_write (current->mm);
  GM_RETURN_STATUS (status);


 out_with_dma_handle:
  gm_linux_pci_unmap (is, lock->dma_handle);
 out_with_page_locked:
  if (!was_locked && !regular_inode)
    {
      UnlockPage (page);
    }
  if (lock->inode)
    {
      iput(lock->inode);
    }
    
  put_page (page);
  GM_RETURN_STATUS (status);
}

void
gm_arch_unlock_user_buffer_page (gm_arch_page_lock_t * lock)
{
  struct page *page = lock->page;
  int *counter_ptr = (int *) 0;
  int c, page_inuse = 0;
  unsigned long pfn;

  GM_CALLED ();
  
  GM_PRINT (GM_PRINT_LEVEL >= 4,
	    ("gm_arch_UNlock_user_buffer_page(%p)\n", lock));

  if (lock)
    {
      page = lock->page;
    }
  else
    {
      GM_WARN (("gm_arch_unlock_user_buffer_page called with "
		"NULL lock pointer!\n"));
      GM_RETURN_NOTHING ();
    }
  if (!page)
    {
      /* unlocking special memory (PCI range for instance) */
      GM_RETURN_NOTHING ();
    }

  if (lock->magic != USER_LOCK_ALIVE)
    {
      GM_PANIC (("internal error, releasing a user page "
		 "with cookie= 0x%x\n", lock->magic));
      GM_RETURN_NOTHING ();
    }

  pfn = gm_linux_pfn (lock->phys);
  gm_arch_mutex_enter (&gm_linux_pfn_sync);
  counter_ptr = gm_hash_find (gm_linux_pfn_hash, &pfn);
  if (!counter_ptr)
    {
      GM_NOTE (("unlock_user_buffer failed to find entry for pfn = 0x%lx\n",
               gm_linux_pfn (lock->phys)));
      gm_arch_mutex_exit (&gm_linux_pfn_sync);

      return;
    }

  if (counter_ptr[0] > 1)
    {
      counter_ptr[0]--;
      GM_PRINT (GM_PRINT_LEVEL >= 4,
               ("Page still locked by GM ref_cnt now = %d\n",
                counter_ptr[0]));
      page_inuse = 1;
    }
  else
    {
      counter_ptr[0]--;
      atomic_inc (&gm_linux_max_user_locked_pages);
      counter_ptr = gm_hash_remove (gm_linux_pfn_hash, &pfn);
      if (!counter_ptr)
       {
         GM_NOTE (("UNlocking user_buffer_page remove ptr is NULL?\n"));
       }
      else
       {
         GM_PRINT (GM_PRINT_LEVEL >= 4,
                   ("UNlocking user_buffer_page counter = %d\n",
                    counter_ptr[0]));
       }
    }
  gm_arch_mutex_exit (&gm_linux_pfn_sync);
  c = gm_linux_page_count (page);
  if (c != 2)
    {
      GM_PRINT (GM_PRINT_LEVEL >= 4,
		("unlock_user_buffer_page: try to dereg a page "
		 "with count(%d) != 2\n", gm_linux_page_count (page)));
    }
  gm_assert (c <= 1000);	/* sanity check */
  gm_always_assert (c >= 1);
  gm_always_assert (!PageReserved (page));

  lock->magic = USER_LOCK_DEAD;

  if (!page_inuse && !lock->inode)
    {
      gm_always_assert (PageLocked (page));
#if 0
      /* I am pretty sure this is not the problem */
      /* who cares if the page is in the swap_cache, swap is cheap, as long as it is in memory */
      gm_always_assert (!PageSwapCache (page));
#endif
      UnlockPage (page);
    }

#if GM_CPU_powerpc64
  gm_always_assert (lock->dma_handle != -1);
#else
  gm_always_assert (lock->dma_handle);
#endif
  gm_linux_pci_unmap (lock->ps->instance, lock->dma_handle);

#if GM_DEBUG
  /* when doing an explicit register, issue a warning if the address
     space of the process has changed between the register and the
     deregister */
  {
    gm_port_state_t *ps = lock->ps;
    if (ps && ps->arch.ref_count > 0
	&& !(current->flags & PF_EXITING) && current->mm == ps->arch.mm)
      {
	gm_phys_t phys;
	int tmp;
	unsigned long virt = lock->virt_pagenum * PAGE_SIZE;
	if (get_user (tmp, (int*)virt) == 0)
	  {
	    phys = gm_linux_kvirt_to_phys (lock->ps->instance, 
					   (unsigned long) lock->virt_pagenum
					   * PAGE_SIZE, 0);
	    if (!phys)
	      {
		GM_WARN (("gm_arch_unlock_user_buffer_page: "
			  "gm_linux_kvirt_to_phys failed\n"));
	      }
	    if (phys != lock->phys)
	      {
		GM_WARN (("gm_arch_unlock_user_page: "
			  "vma was 0x%lx->0x%lx, now ->0x%lx\n",
			  lock->virt_pagenum * PAGE_SIZE, 
			  gm_linux_pfn (lock->phys), 
			  gm_linux_pfn ((unsigned long)phys)));
	      }
	  }
	else
	  {
	    GM_WARN (("gm_arch_unlock_user_page: "
		      "vma was 0x%lx->0x%lx, has been unmapped\n",
		      lock->virt_pagenum * PAGE_SIZE,
		      gm_linux_pfn (lock->phys)));
	  }
      }
  }
  user_unlock_cnt++;
#endif

  if (lock->inode)
    {
      if (!gm_linux_hugetlb(lock->vm_flags))
	{
#if GM_LINUX_HAVE_SET_PAGE_DIRTY
	  set_page_dirty (page);
#else
	  GM_PANIC (("lock->inode non-zero but set_page_dirty unavailable\n"));
#endif
	}
      put_page (page);
      iput (lock->inode);
    }
  else
    {
      put_page (page);
    }

  GM_RETURN_NOTHING ();
}

static int
gm_linux_get_order (unsigned long size)
{
  GM_CALLED ();
  
#if LINUX_VERSION_CODE < KERNEL_VERSION (2, 2, 18)
  /* get_order was introduced into <asm/page.h> after 2.2.14, but is
     in 2.2.18.  It should be harmless to use this replicated version
     in this interval. */
  {
    int order;
    
    size = (size-1) >> (PAGE_SHIFT-1);
    order = -1;
    do
      {
	size >>= 1;
	order++;
      }
    while (size);
    GM_RETURN_INT (order);
  }
#elif LINUX_22 || LINUX_24
  {
    GM_RETURN_INT (get_order (size));
  }
#else
#error
#endif
}


/* Allocate LEN bytes of PCI consistent DMA memory that is contiguous 
   in kernel space but possibly segmented in DMA space.
   
   If r->register_function is non-null, call r->register_page_function
   (r, dma_addr) for each page. */

gm_status_t
gm_arch_dma_region_alloc (gm_instance_state_t * is,
			  gm_arch_dma_region_t * r,
			  gm_size_t len, gm_u32_t flags,
			  gm_register_page_function_t register_page_func,
			  void *arg)
{
  gm_dp_t bus_addr;
  gm_phys_t phys;
  unsigned int num_pages, page_num;

  GM_CALLED ();

#if GM_DEBUG
  dma_alloc_cnt++;
#endif

  GM_PRINT (GM_PRINT_LEVEL >= 4,
	    ("gm_arch_dma_region_alloc: len = %ld, flags=%d\n", len, flags));

  num_pages = GM_PAGE_ROUNDUP (u32, len) / GM_PAGE_LEN;
  gm_always_assert (num_pages >= 1);

  r->is = is;
  r->addr = NULL;
  r->len = len;
  r->flags = flags;
  r->num_pages = num_pages;
  r->index = 0;

  r->dma_list = gm_calloc (num_pages, sizeof (r->dma_list[0]));
  if (!r->dma_list)
    {
      GM_WARN (
	       ("gm_arch_dma_region_alloc: memory alloc failed for d-list\n"));
      GM_RETURN_STATUS (GM_FAILURE);
    }
  if ((flags & GM_ARCH_DMA_CONTIGUOUS) || (num_pages == 1))
    {
      r->region = GM_ARCH_GFP_REGION;
      r->addr
	= (void *) __get_free_pages (GFP_KERNEL, gm_linux_get_order (len));
      GM_PRINT (GM_PRINT_LEVEL >= 4,
		("__get_free_pages %ld bytes, addr= 0x%p\n", len, r->addr));
    }
  else
    {
      /* try with vmalloc, it will be non-contiguous at the physical 
         layer, but GM needs only contiguous virtual addresses */
      r->region = GM_ARCH_VMALLOC_REGION;
      r->addr = gm_linux_vmalloc (num_pages * PAGE_SIZE);
      GM_PRINT (GM_PRINT_LEVEL >= 4,
		("gm_linux_vmalloc %ld bytes, addr= 0x%p\n", len, r->addr));
    }

  if (r->addr == NULL)
    {
      /* not enough memory */
      gm_free (r->dma_list);
      GM_WARN (("gm_arch_dma_region_alloc: memory alloc failed (len=%ld)\n",
		len));
      GM_RETURN_STATUS (GM_FAILURE);
    }

  gm_always_assert (GM_PAGE_ALIGNED (r->addr));


  /* Go through each page in the allocated  region to get the DMA pointer. */
  for (page_num = 0; page_num < num_pages; page_num++)
    {
      unsigned long addr = (unsigned long) r->addr + page_num * PAGE_SIZE;
      gm_status_t map_stat;

      phys = gm_linux_kvirt_to_phys (is, addr, 1);
      if (phys == 0)
        {
	  GM_WARN (("gm_arch_dma_region_alloc: phys=0(kvirt=0x%lx)\n",addr));
	  goto error_with_vmalloc;
        }
      map_stat = gm_linux_pci_map (is, phys, &bus_addr);
      if (map_stat != GM_SUCCESS)
	{
	  GM_WARN (("gm_arch_dma_region_alloc: "
		    "gm_linux_pci_map(is,0x%lx) failed\n", gm_linux_pfn (phys)));
	  goto error_with_vmalloc;
	}
      r->dma_list[page_num] = bus_addr;
    }
  if (register_page_func)
    {
      for (page_num = 0; page_num < num_pages; page_num++)
	{
	  gm_status_t status;
	  status = register_page_func (arg, r->dma_list[page_num], page_num);
	  gm_always_assert (status == GM_SUCCESS);
	}
    }
  GM_RETURN_STATUS (GM_SUCCESS);

 error_with_vmalloc:
  for (page_num = 0; page_num < num_pages; page_num++)
    if (r->dma_list[page_num])
      gm_linux_pci_unmap (is, r->dma_list[page_num]);
  gm_free (r->dma_list);
  r->dma_list = NULL;
  if (r->region == GM_ARCH_GFP_REGION)
    {
      free_pages ((unsigned long) r->addr, gm_linux_get_order (len));
    }
  else
    {
      gm_linux_vfree (r->addr);
    }
  r->addr = NULL;
  GM_RETURN_STATUS (GM_FAILURE);
}

void
gm_arch_dma_region_free (gm_arch_dma_region_t * r)
{
  int page_num;

  GM_CALLED ();
  
#if GM_DEBUG
  dma_free_cnt++;
#endif

  for (page_num = 0; page_num < r->num_pages; page_num++)
    {
      gm_phys_t phys;
      gm_assert (r->dma_list[page_num]);
      gm_linux_pci_unmap (r->is, r->dma_list[page_num]);
      phys =
	gm_linux_kvirt_to_phys (r->is,
				(unsigned long) r->addr +
				page_num * PAGE_SIZE, 1);
      if (!phys)
	{
	  GM_WARN (("gm_arch_dma_region_free: "
		    "gm_linux_kvirt_to_phys failed\n"));
	}
      gm_assert (gm_linux_phys_is_valid_page (r->is, phys));
      gm_linux_unreserve_page (r->is, phys);
    }


  gm_free (r->dma_list);
  r->dma_list = NULL;
  if (r->region == GM_ARCH_GFP_REGION)
    {
      free_pages ((unsigned long) r->addr, gm_linux_get_order (r->len));
    }
  else
    {
      gm_linux_vfree (r->addr);
    }

  /* just in case */
  r->addr = NULL;

  GM_RETURN_NOTHING ();
}

void *
gm_arch_dma_region_kernel_addr (gm_arch_dma_region_t * r)
{
  GM_CALLED ();
  GM_RETURN_PTR (r->addr);
}

gm_s32_t
gm_arch_dma_region_status (gm_arch_dma_region_t * r)
{
  GM_CALLED ();
  GM_PARAMETER_MAY_BE_UNUSED (r);
  GM_RETURN (0xf);
}


gm_dp_t
gm_arch_dma_region_dma_addr (gm_arch_dma_region_t * r)
{
  GM_CALLED ();
  gm_always_assert (r->index < r->num_pages);
  GM_RETURN (r->dma_list[r->index]);
}

gm_dp_t
gm_arch_dma_region_dma_addr_advance (gm_arch_dma_region_t * r)
{
  int previous;

  GM_CALLED ();
  
  previous = r->index;
  r->index += 1;
  gm_always_assert (previous < r->num_pages);
  GM_RETURN (r->dma_list[previous]);
}

gm_status_t
gm_arch_dma_region_sync (gm_arch_dma_region_t * r, int command)
{
  GM_CALLED ();

  GM_PARAMETER_MAY_BE_UNUSED (r);
  GM_PARAMETER_MAY_BE_UNUSED (command);

  GM_RETURN_STATUS (GM_SUCCESS);
}



/*********************************************************************
 * kernel memory allocation functions 
 *********************************************************************/

void *
__gm_arch_kernel_malloc (unsigned long len, int flags)
{
  GM_PARAMETER_MAY_BE_UNUSED (flags);

#if GM_DEBUG
  kernel_alloc_cnt++;
#endif

  /* 64 is a safe value, anyway it will work even if the threshold
     does not exactly correspond to kmalloc internals */
  if (len <= PAGE_SIZE - 64)
    {
      return gm_linux_kmalloc (len, GFP_KERNEL);
    }
  else
    {
      return gm_linux_vmalloc (len);
    }
}

void
__gm_arch_kernel_free (void *ptr)
{
#if GM_DEBUG
  kernel_free_cnt++;
#endif

  if ((ptr > (void *) PAGE_OFFSET) && (ptr < (void *) high_memory))
    {
      gm_linux_kfree (ptr);
    }
  else
    {
      gm_linux_vfree (ptr);
    }
}

/*********************************************************************
 * memory mapping (into kernel space)
 *********************************************************************/

gm_status_t
gm_arch_map_io_space (gm_instance_state_t * is, gm_u32_t offset, gm_u32_t len,
		      void **kaddr)
{
  GM_CALLED ();
  
  GM_PRINT (GM_PRINT_LEVEL >= 3,
	    ("gm_arch_map_io_space(%p, 0x%x, %d)\n", is, offset, len));

  *kaddr = gm_linux_ioremap ((unsigned long) is->arch.iomem_base
		       + (unsigned long) offset, len);
  GM_PRINT (GM_PRINT_LEVEL >= 2,
	    ("ioremapped 0x%p (offset 0x%x, len 0x%x)\n",
	     *kaddr, offset, len));

  GM_RETURN_STATUS ((*kaddr) ? GM_SUCCESS : GM_FAILURE);
}

void
gm_arch_unmap_io_space (gm_instance_state_t * is, gm_u32_t offset,
			gm_u32_t len, void **kaddr)
{
  GM_PARAMETER_MAY_BE_UNUSED (is);
  GM_PARAMETER_MAY_BE_UNUSED (offset);
  GM_PARAMETER_MAY_BE_UNUSED (len);

  GM_CALLED ();
  
  GM_PRINT (GM_PRINT_LEVEL >= 6,
	    ("iounmapping %p (offset 0x%x, len 0x%x)\n",
	     *kaddr, offset, len));

  gm_linux_iounmap (*kaddr);
  *kaddr = 0;

  GM_RETURN_NOTHING ();
}

/* needed for memory registration? */
gm_status_t
gm_arch_mmap_contiguous_segment (gm_port_state_t * ps, void *kaddr,
				 unsigned long blockSize, gm_up_t * vaddr)
{
  GM_CALLED ();
  
  GM_PARAMETER_MAY_BE_UNUSED (ps);
  GM_PARAMETER_MAY_BE_UNUSED (kaddr);
  GM_PARAMETER_MAY_BE_UNUSED (blockSize);
  GM_PARAMETER_MAY_BE_UNUSED (vaddr);

  GM_NOT_IMP ();
  GM_RETURN_STATUS (GM_FAILURE);
}

void
gm_arch_munmap_contiguous_segments (gm_port_state_t * ps)
{
  GM_PARAMETER_MAY_BE_UNUSED (ps);

  GM_NOT_IMP ();
}

/*********************************************************************
 * Miscellaneous functions
 *********************************************************************/

gm_status_t
gm_arch_get_page_len (unsigned long *result)
{
  /* this is okay because in Linux, we always know page size at
   * compile time (on x86 and alphas, at least) */
  *result = PAGE_SIZE;

  return GM_SUCCESS;
}

gm_status_t
gm_arch_gethostname (char *ptr, int len)
{
  gm_bzero (ptr, len);
  strncpy (ptr, system_utsname.nodename, len - 1);
  ptr[len - 1] = 0;

#if GM_SHORT_HOSTNAME
  {
    char *s;

    /* drop everything from first '.' on to end */
    if (s = strchr (ptr, '.'))
      {
	*s = 0;
      }
  }
#endif

  return GM_SUCCESS;
}

void
gm_arch_spin (gm_instance_state_t * is, gm_u32_t usecs)
{
  GM_PARAMETER_MAY_BE_UNUSED (is);

  GM_CALLED ();
  
  if (usecs == 0 && !in_interrupt ())
    {
      /* if usecs == 0, just do the sched_yield() equivalent */
      schedule();
    }
  else if (usecs < 100 || in_interrupt ())
    {
      udelay (usecs);
    }
  else
    {
      /* do not want to call udelay for long time. */
      /* let be uninterruptible to be sure the delay is respected */
      set_current_state (TASK_UNINTERRUPTIBLE);
      schedule_timeout (((usecs * HZ) / 1000000) + 1);
    }

  GM_RETURN_NOTHING ();
}

static void
gm_linux_test_lanai (void *data)
{
  gm_instance_state_t * is = data;

  GM_CALLED ();
  
  gm_pause_lanai (is);
  gm_unpause_lanai (is);
  if (!is->lanai.running)
    GM_WARN (("gm_linux_test_lanai: lanai is stopped\n"));

  GM_RETURN_NOTHING ();
}

/*********************************************************************
 * Directcopy
 *********************************************************************/
#if 0 && GM_ENABLE_DIRECTCOPY
#if LINUX_22
#error directcopy and linux-2.2 is unstable
#error please --disable-directcopy in your configure line
#endif

gm_status_t
gm_arch_directcopy_get (void *source_addr,
			void *target_addr, ulong length, uint pid_source)
{

  /* Only x86 and alpha has been tested and validated */
#if GM_CPU_x86 || GM_CPU_sparc64 || GM_CPU_ia64 || GM_CPU_alpha || GM_CPU_powerpc

  struct task_struct *task;
  struct vm_area_struct *vma;
  unsigned long source, target, len, phys_source;
  unsigned int offset_source, pack;
  char *map_addr;
  pgd_t *page_dir;
  pmd_t *page_middle;
  pte_t *page_table;
  struct page *page;
  char log_msg[128];

  /* pid 0 ? */
  if (pid_source == 0)
    {
      GM_NOTE (("Directcopy : bad pid (0)\n"));
      return GM_INVALID_PARAMETER;
    }

  /* nothing to do */
  if (length == 0)
    {
      return GM_SUCCESS;
    }
  else if ((ulong) (source_addr) == 0)
    {
      GM_NOTE (("Directcopy : buffer NULL but length %ld", length));
      return GM_INVALID_PARAMETER;
    }



#if LINUX_VERSION_CODE < VERSION_CODE(2,2,17)

  {
    struct task_struct *task1, *task2;

    task1 = current->prev_task;
    task2 = current->next_task;
    while (1)
      {
	if (task1 != NULL)
	  if (task1->pid == pid_source)
	    {
	      task = task1;
	      break;
	    }
	  else
	    task1 = task1->prev_task;
	if (task2 != NULL)
	  if (task2->pid == pid_source)
	    {
	      task = task2;
	      break;
	    }
	  else
	    task2 = task2->next_task;
	if ((task1 == NULL) && (task2 == NULL))
	  {
	    break;
	  }
      }
  }
#elif LINUX_24

  read_lock (&tasklist_lock);
  task = find_task_by_pid (pid_source);
  if (task)
    get_task_struct (task);
  read_unlock (&tasklist_lock);

#else

  task = find_task_by_pid (pid_source);

#endif

  if (!task)
    {
      GM_NOTE (("Directcopy : bad pid (%d)\n", pid_source));
      return GM_INVALID_PARAMETER;
    }


#if LINUX_24
  /* Worry about races with exit() */
  task_lock (task);
  if (task->mm)
    atomic_inc (&task->mm->mm_users);
  task_unlock (task);
  if (!task->mm)
    {
      GM_NOTE (("%s:%d  Directcopy invalid MM task = %p\n", __FILE__,
		__LINE__, task));
      strcpy (log_msg, "Directcopy : invalid MM\n");
      goto directcopy_panic;
    }
#endif

  source = (ulong) source_addr;
  target = (ulong) target_addr;
  len = length;
  offset_source = source & ~PAGE_MASK;

  down (&task->mm->mmap_sem);

  /* search for the VMA */
  vma = find_vma (task->mm, source);
  if (!vma)
    {
      strcpy (log_msg, "Directcopy : invalid VMA\n");
      goto directcopy_panic;
    }
  if ((vma->vm_start > source)
      && ((!(vma->vm_flags & VM_GROWSDOWN))
	  || (vma->vm_end - source > task->rlim[RLIMIT_STACK].rlim_cur)))
    {
      GM_NOTE (("Directcopy : invalid VMA\n"));
      strcpy (log_msg, "Directcopy : invalid VMA\n");
      goto directcopy_panic;
    }

  while (1)
    {
      /* physical memory page boundaries */
      pack = GM_PAGE_LEN - offset_source;
      if (pack > len)
	pack = len;

      /* in the following, we convert the virtual adresses in physical
         adresses by walking in the pages tables. */
      page_dir = pgd_offset (vma->vm_mm, source);
      if (pgd_none (*page_dir))
	{
	  GM_NOTE (("%s:%d  Directcopy no page directory = %p\n", __FILE__,
		    __LINE__, page_dir));
	  strcpy (log_msg, "Directcopy : no page directory\n");
	  goto directcopy_panic;
	}
      if (pgd_bad (*page_dir))
	{
#if LINUX_24
	  pgd_ERROR (*page_dir);
#else
	  pgd_clear (page_dir);
#endif
	  GM_NOTE (("%s:%d  Directcopy bad page directory = %p\n", __FILE__,
		    __LINE__, page_dir));
	  strcpy (log_msg, "Directcopy : bad page directory\n");
	  goto directcopy_panic;
	}

      page_middle = pmd_offset (page_dir, source);
      if (pmd_none (*page_middle))
	{
	  GM_NOTE (("%s:%d  Directcopy no middle page table = %p\n", __FILE__,
		    __LINE__, page_middle));
	  strcpy (log_msg, "Directcopy : no middle page table\n");
	  goto directcopy_panic;
	}
      if (pmd_bad (*page_middle))
	{
#if LINUX_24
	  pmd_ERROR (*page_middle);
#else
	  pmd_clear (page_middle);
#endif
	  GM_NOTE (("%s:%d  Directcopy bad middle page table = %p\n",
		    __FILE__, __LINE__, page_middle));
	  strcpy (log_msg, "Directcopy : bad middle page table\n");
	  goto directcopy_panic;
	}

      page_table = pte_offset (page_middle, source);
      if (!pte_present (*page_table))
	{
	  GM_NOTE (("%s:%d  Directcopy pte not present = %p\n", __FILE__,
		    __LINE__, page_table));
	  strcpy (log_msg, "Directcopy : pte not present\n");
	  goto directcopy_panic;
	}

#if LINUX_24
      page = pte_page (*page_table);
      flush_cache_page (vma, source);
      map_addr = kmap (page);
      copy_to_user ((void *) target, (void *) (map_addr + offset_source),
		    pack);
      flush_page_to_ram (page);
      kunmap (page);
#else
      phys_source = pte_page (*page_table);
      if (MAP_NR (phys_source) >= max_mapnr)
	{
	  /* for high addresses, map the page into kernel space */
	  unsigned long result, vpage;
	  phys_source =
	    (unsigned long) gm_linux_ioremap (__pa (phys_source), PAGE_SIZE);
	  if (!phys_source)
	    {
	      GM_NOTE (("%s:%d  Directcopy out of memory \n", __FILE__,
			__LINE__));
	      strcpy (log_msg, "Directcopy : out of memory\n");
	      goto directcopy_panic;
	    }
	  /* we don't need to translate the receiver side adresses,
	     the MMU is here for that */
	  copy_to_user ((void *) target,
			(void *) (phys_source + offset_source), pack);
	  gm_linux_iounmap ((void *) phys_source);
	}
      else
	copy_to_user ((void *) target, (void *) (phys_source + offset_source),
		      pack);
#endif

      source += pack;
      target += pack;
      len -= pack;
      offset_source = source & ~PAGE_MASK;

      if (len == 0)
	{
	  up (&task->mm->mmap_sem);
#if LINUX_24
	  atomic_dec (&task->mm->mm_users);
	  free_task_struct (task);
#endif
	  return GM_SUCCESS;
	}

      if (source >= vma->vm_end)
	{
	  if (!vma->vm_next)
	    {
	      GM_NOTE (("%s:%d  Directcopy VMA too short vma=%p \n", __FILE__,
			__LINE__, vma));
	      strcpy (log_msg, "Directcopy : VMA too short\n");
	      goto directcopy_panic;
	    }
	  if (vma->vm_next->vm_start != vma->vm_end)
	    {
	      GM_NOTE (("%s:%d  Directcopy non-contiguous VMAs vma=%p \n",
			__FILE__, __LINE__, vma));
	      strcpy (log_msg, "Directcopy : non-contiguous VMAs\n");
	      goto directcopy_panic;
	    }
	  vma = vma->vm_next;
	}
    }

  GM_NOTE (("Directcopy : Error\n"));
  up (&task->mm->mmap_sem);
#if LINUX_24
  atomic_dec (&task->mm->mm_users);
  free_task_struct (task);
#endif
  return GM_FAILURE;

 directcopy_panic:
  up (&task->mm->mmap_sem);
#if LINUX_24
  atomic_dec (&task->mm->mm_users);
  free_task_struct (task);
#endif
#if 0
  GM_PANIC ((log_msg));
#else
  GM_NOTE (("Should have panic'd the machine here, but want debug info\n"));
  GM_NOTE ((log_msg));
  return GM_FAILURE;
#endif

#else
#error DIRECTCOPY turned on for untested architecture??
  return GM_FAILURE;
#endif
}
#endif

/*********************************************************************
 * other stuff
 *********************************************************************/

/*
 * interrupt handler
 */

#define GM_DEBUG_LINUX_INTR 0
static irqreturn_t
gm_linux_intr (int irq, void *instance_id, struct pt_regs *regs)
{
  unsigned long flags;
  gm_instance_state_t *is = instance_id;
  int claimed = 1;
  GM_PARAMETER_MAY_BE_UNUSED (irq);
  GM_PARAMETER_MAY_BE_UNUSED (regs);

  GM_PRINT (GM_DEBUG_LINUX_INTR, ("got an interrupt to IRQ%d\n", irq));

  gm_assert (is);
  /* careful here, if several handlers are registered on different
     IRQs (host has several Myrinet boards), the function might be
     preempted by itself while holding the spin_lock, unless we
     disable interrupts. So if using spin_lock use the irqsave version
     which disable interrupts */
  if (GM_USE_INTR_LOCK)
    {
      spin_lock_irqsave (&gm_linux_intr_lock, flags);
    }

  /* if case we are sharing interrupts, return as soon as possible */
  if (gm_interrupting (is) == GM_ARCH_INTR_UNCLAIMED)
    {
      GM_PRINT (GM_DEBUG_LINUX_INTR, ("the interrupt was not claimed\n"));
      claimed = 0;
      goto end_intr;
    }
  else
    {
      GM_PRINT (GM_DEBUG_LINUX_INTR, ("the interrupt was claimed\n"));
    }
  if (!is->lanai.running)
    {
      /* do not print a message each time becuse if the interrupt is
         shared we may fill up some log quickly */
      static unsigned long last_jiffies;
      static int count;

      if (jiffies - last_jiffies > HZ)
	{
	  count = 0;
	  last_jiffies = jiffies;
	}
      if (count < 5)
	{
	  count++;
	  GM_PRINT (1, ("LANai is not running in interrupt handler.\n"));
	}
      gm_set_EIMR (is, 0);	/* disable interrupts */
      goto end_intr;
    }

#if 0
  if (test_and_set_bit (0, (void *) &is->arch.interrupt) != 0)
    {
      /* this was to workaround a linux 2.0.35 SMP bug, this should
	 never happen anymore moreover on SMP the spinlock will not
	 let us here anyway */
      GM_PANIC(("board %d: recursive interruption detected", is->id));
    }
#endif

  gm_linux_in_interrupt = 1;
  /* Call the generic GM interrupt handler, but allow for a longjmp()
     out of it in some cases if it panics. */
  
  if (GM_LINUX_DEBUG_SIGSETJMP (intr_jmp_buf, 0) == 0)
    {
      gm_handle_claimed_interrupt (is);
    }
  else
    {
      GM_INFO (("longjmp out of interrupt handler\n"));
      gm_disable_interrupts (is);
    }
  gm_linux_in_interrupt = 0;
#if 0
  clear_bit (0, (void *) &is->arch.interrupt);
#endif

 end_intr:
  if (GM_USE_INTR_LOCK)
    {
      spin_unlock_irqrestore (&gm_linux_intr_lock, flags);
    }
  return IRQ_RETVAL(claimed);
}

/* Set the DMA mask on archs where it is settable. */

gm_status_t
gm_linux_set_dma_mask (gm_instance_state_t *is)
{
#if GM_SUPPORT_PCI
  {
    struct pci_dev *dev = is->arch.pci_dev;
    gm_dp_t mask;
    mask = ((is->flags & GM_INSTANCE_64BIT_DMA_ADDR_OK)
	    ? ~0ULL
	    : 0xffffffffULL);
    /* note: 32bit PCI devices should always succeed here, normally only
       PCI devices with less than 32bit addressing can fail */
    if (pci_set_dma_mask (dev, mask) == 0)
      return GM_SUCCESS;
    else if (GM_CPU_powerpc64 && (is->flags & GM_INSTANCE_64BIT_DMA_ADDR_OK))
      {
	/* ppc64 with IOMMU disabled can only handle 32-bit DMA addrs */
	mask =  0xffffffffULL;
	if (pci_set_dma_mask (dev, mask) == 0) 
	  {
	    is->flags &=  ~GM_INSTANCE_64BIT_DMA_ADDR_OK;
	    return GM_SUCCESS;
	  }
      }
    return GM_FAILURE;
  }
#elif CSPI_MAP26xx
  {
    return GM_SUCCESS;
  }
#else
#error bad bus_type
#endif
}

void
gm_linux_board_debug_info (gm_instance_state_t *is)
{
  GM_CALLED ();
  
#if (GM_PRINT_LEVEL >= 7)
  {
    unsigned char *cptr;
    int i;

    printk ("GM: Board found config regs follow\n");
    cptr = (unsigned char *) &is->ifc.pci.config;
    for (i = 0; i < 64; i++)
      {
	if ((i % 4) == 0)
	  {
	    printk ("\nGM: %02x: ", i);
	  }
	if (cptr[i])
	  {
	    printk ("%02x ", cptr[i]);
	  }
	else
	  {
	    printk ("   ");
	  }
      }
    printk ("\n\n");
    printk ("GM:  iomem_base = 0x%lx   phys_base_addr = %p   irq = 0x%x\n",
	    is->arch.iomem_base, is->arch.phys_base_addr, is->arch.irq);
  }
#endif
  
  GM_RETURN_NOTHING ();
}

static void
gm_linux_devfs_register (gm_instance_state_t *is, int priv)
{
  char name[10];
  umode_t mode = priv ? (S_IRUSR | S_IWUSR) : (S_IRUGO | S_IWUGO);

  GM_CALLED ();
  
  sprintf (name, "%s%d", priv ? "gmp" : "gm", is->id);
#if LINUX_XX >= 26
  if (gm_linux_udev)
    {
      devfs_mk_cdev(MKDEV(GM_MAJOR, is->id * 2 + priv), S_IFCHR | mode, name);
      class_simple_device_add(gm_class, MKDEV(GM_MAJOR, is->id * 2 + priv),
			      NULL, name);
    }
#else
  is->arch.devfs_handle[priv] = 
    devfs_register (NULL, name, DEVFS_FL_DEFAULT, GM_MAJOR, is->id *2 + priv, 
                    S_IFCHR | mode, &gm_linux_file_ops, 0);
#endif
  GM_RETURN_NOTHING ();
}

static void
gm_linux_devfs_unregister (gm_instance_state_t *is, int priv)
{
  GM_CALLED ();
  
#if LINUX_XX >= 26
  if (gm_linux_udev)
    {
      devfs_remove("%s%d", priv ? "gmp" : "gm", is->id);
      class_simple_device_remove(MKDEV(GM_MAJOR, is->id * 2 + priv));
    }
#else
  devfs_unregister (is->arch.devfs_handle[priv]);
#endif
  GM_RETURN_NOTHING ();
}

#if GM_CPU_powerpc64 || GM_CPU_x86_64
#include <linux/smp_lock.h>    /* For (un)lock_kernel */
/* These should also work for sparc64 on Linux */

extern int register_ioctl32_conversion(unsigned int cmd,
				       int (*handler)(unsigned int,
						      unsigned int,
                                                      unsigned long,
                                                      struct file *));
int unregister_ioctl32_conversion(unsigned int cmd);

/* there seems to be some confusion on exactly what should be done if your 
   your ioctls are 64 bit clean.  Text documentation on the web says that you
   should pass sys_ioctl as your handler to register_ioctl32_conversion,
   but sys_ioctl takes a different number of arguments than the handler.  
   Comments in header files say that you should pass null, but in the x86_64 case
   passing a null function pointer causes an oops since the code tries to
   call the handler without any checks.  So we will just declare our own 
   handler that just calls sys_ioctl and solve all the problems. */

int 
gm_linux_ioctl32 (unsigned int fd, unsigned int cmd, 
		  unsigned long arg, struct file *filp)
{
  int ret;
  
  lock_kernel();
  ret = gm_linux_ioctl(filp->f_dentry->d_inode, filp, cmd, arg);
  unlock_kernel();
  return ret;
}

static int
gm_linux_register_ioctl32_conversions (void)
{
  int i, err;

  for (i = 0; i < GM_NUM_IO; i++)
    {
      if ((err =  register_ioctl32_conversion (GM_IO(i), gm_linux_ioctl32)) != 0)
        return err;
    }
  return 0;
}

static int
gm_linux_unregister_ioctl32_conversions (void)
{
  int i, err;

  for (i = 0; i < GM_NUM_IO; i++)
    {
      if ((err =  unregister_ioctl32_conversion (GM_IO(i))) != 0)
	return err;
    }
  return 0;
}

#endif /* GM_CPU_powerpc64 || GM_CPU_x86_64 */

/****************************************************************
 * gm_linux_create_instance
 *
 * Initializes the myrinet card specified.  If the card is
 *   initialized correctly, it increments gm_linux_num_instances
 *   and adds it into the device array.
 * Arguments:
 *   dev - a pointer to the pci structure for the myrinet card
 * Returns:
 *   GM_SUCCESS if card was initialized correctly
 *   GM_FAILURE otherwise
 ****************************************************************/

/* create a new device. Only at end and if no error occurs, we link it
   and increment gm_linux_num_instance. */

static gm_status_t
gm_linux_create_instance (enum gm_bus_type bus_type, void *bus_data)
{
  int i;
  gm_instance_state_t *is;
  gm_status_t status;
  struct pci_dev * dev = bus_data;
  static gm_pci_config_t pci_config;
  unsigned int class;
  unsigned short vendor, device;
  unsigned char byte_value;
  unsigned char *cptr;

  GM_CALLED ();
  
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("Using gm_linux_create_instance for instance %d (bus type %d)\n",
	     gm_linux_num_instances, bus_type));

#if GM_SUPPORT_PCI
  if (pci_enable_device(dev)) {
    GM_WARN(("The linux pci_enable_device call failed (bus %d, devfn %d)\n",
	     dev->bus->number, dev->devfn));
    status = GM_FAILURE;
    goto abort_with_nothing; 
  }
  pci_set_master(dev);
  /*
   * lots of this could be moved to arch-independent code
   * there are just sanity checks + filling of gm_pci_config
   */
  pci_read_config_dword (dev, PCI_CLASS_REVISION, &class);
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("Myrinet PCI probe device = %d  revision = 0x%x\n",
	     dev->devfn, class));
  if (class == 0xffffffff)
    {
      status = GM_FAILURE;
      goto abort_with_nothing;
    }

  /* make a copy of entire PCI config space for this device */
  cptr = (unsigned char *) &pci_config;
  memset (cptr, 0, sizeof (gm_pci_config_t));
  for (i = 0; i < 64 /*sizeof(gm_pci_config_t) */ ; i++)
    {
      if (pci_read_config_byte (dev, i, &byte_value) != PCIBIOS_SUCCESSFUL)
	{
	  GM_WARN (("myri_pci_probe, unit %d (bus %d, dev %d): "
		    "error reading PCI configuration\n",
		    gm_linux_num_instances, dev->bus->number, dev->devfn));
	  status = GM_FAILURE;
	  goto abort_with_nothing;
	}
      else
	{
	  cptr[i] = byte_value;
	}
    }

  pci_read_config_word (dev, PCI_VENDOR_ID, &vendor);
  pci_read_config_word (dev, PCI_DEVICE_ID, &device);

  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("myri_pci_probe testing vendor=0x%x  device=0x%x\n",
	     vendor, device));

  if (((vendor == GM_PCI_VENDOR_MYRICOM)
       && (device == GM_PCI_DEVICE_MYRINET))
      || ((vendor == GM_PCI_VENDOR_MYRICOM2)
	  && (device == GM_PCI_DEVICE_MYRINET)))
    {
      /* OK */
    }
  else
    {
      status = GM_FAILURE;
      goto abort_with_nothing;
    }
#endif

  is = (void *) gm_linux_kmalloc (sizeof (*is), GFP_KERNEL);
  if (!is)
    {
      GM_WARN (("couldn't get memory for instance_state\n"));
      status = GM_OUT_OF_MEMORY;
      goto abort_with_nothing;
    }
  memset (is, 0, sizeof (*is));
#if GM_DEBUG
  gm_linux_debug_is = is;
#endif

  is->arch.ethernet_send_lock = SPIN_LOCK_UNLOCKED;
  
  /* from now, do not return anymore without cleaning up everything */

#if GM_SUPPORT_PCI
  /* busbase is the value that can be passed to ioremap
     it might be different from both the PCI address,
     or the phys address or the kernel address 
  */
  is->arch.iomem_base = gm_linux_pci_dev_base (dev);
  if (!is->arch.iomem_base)
    {
      GM_WARN (("Bad PCI Info:base_address[0] = 0!!! (PCI iobase=0x%lx)\n",
		is->arch.iomem_base));
      status = GM_FAILURE;
      goto abort_with_instance_state;
    }

  is->arch.phys_base_addr = GM_LINUX_IOMEM2PHYS (is->arch.iomem_base);
  GM_PRINT (GM_TRACE_LANAI_DMA || GM_DEBUG_BOARD_INIT,
	    ("phys address is 0x%lx\n", is->arch.phys_base_addr));

  is->arch.irq = dev->irq;
  is->arch.pci_dev = dev;
  /* FIXME does it really belong in arch specific code */
  is->ifc.pci.config = pci_config;
  atomic_set(&is->arch.free_iommu_pages, 1024);
#elif CSPI_MAP26xx
  is->arch.irq = 0x4;
#else
#error
#endif

  gm_linux_board_debug_info (is);

  if (gm_linux_skip_init)
    {
      goto init_ok;
    }

  /* generic board initialization; load MCP and stuff */
  status = gm_instance_init (is, gm_linux_num_instances, bus_type);
  if (status != GM_SUCCESS)
    {
      GM_NOTE (("gm_instance_init failed\n"));
      goto abort_with_instance_state;
    }

  status = gm_linux_set_dma_mask (is);
  if (status != GM_SUCCESS)
    {
      GM_NOTE (("Error setting DMA mask\n"));
      goto abort_with_instance_init;
    }

  if (request_irq (is->arch.irq, (void *) gm_linux_intr,
		   SA_SHIRQ, "myri/gm", is) == 0)
    {
      GM_INFO (("Allocated IRQ%d\n", is->arch.irq));
    }
  else
    {
      GM_NOTE (("Could not allocate IRQ%d\n", is->arch.irq));
      status = GM_FAILURE;
      goto abort_with_instance_init;
    }

  /*
   * enable interrupts
   */
  gm_enable_interrupts (is);
  GM_PRINT (GM_DEBUG_BOARD_INIT, ("Interrupts enabled.\n"));

#ifndef GM_NOIP
  /* add the IP device */
  if (gmip_init (is) != 0)
    {
      status = GM_FAILURE;
      goto abort_with_enabled_interrupts;
    }
#endif
  GM_LINUX_INIT_WORK (&is->arch.test_lanai_task, gm_linux_test_lanai, is);

  /*
   * prepend to list of GM devices
   */
 init_ok:
  gm_assert (is->id == gm_linux_num_instances);
  gm_linux_devfs_register (is, 0);
  gm_linux_devfs_register (is, 1);
  GM_PRINT
    (GM_DEBUG_BOARD_INIT,
     ("gm_instance_init succeeded for unit %d\n", gm_linux_num_instances));
  gm_always_assert (gm_linux_num_instances < GM_ARCH_MAX_INSTANCE);
  gm_assert (gm_linux_instances[gm_linux_num_instances] == 0);
  gm_linux_instances[gm_linux_num_instances] = is;
  gm_linux_num_instances += 1;
#if GM_SUPPORT_PCI
  pci_set_drvdata (is->arch.pci_dev, is);
#endif
  GM_RETURN_STATUS (GM_SUCCESS);

  /* ERROR Handling */
 abort_with_enabled_interrupts:
  gm_disable_interrupts (is);
  GM_PRINT (GM_DEBUG_BOARD_INIT, ("freeing irq %d\n", is->arch.irq));
  free_irq (is->arch.irq, is);
 abort_with_instance_init:
  gm_instance_finalize (is);
 abort_with_instance_state:
  gm_linux_kfree (is);
 abort_with_nothing:
  GM_RETURN_STATUS (status);
}



/***********************************************************************
 * Module entry points.
 ***********************************************************************/

void gm_linux_cleanup_module (void);

MODULE_AUTHOR ("Myricom <help@myri.com>");
MODULE_DESCRIPTION ("Myrinet GM driver");
MODULE_LICENSE ("Myricom");

static unsigned long gm_max_locked_mbytes;
MODULE_PARM (gm_max_locked_mbytes, "l");

MODULE_PARM (gm_activate_page_symbol, "l");
MODULE_PARM (gm_sprintf_symbol, "l");
MODULE_PARM (gm_linux_udev, "i");

#if GM_DEBUG
MODULE_PARM (gm_linux_print_level, "i");
#endif

#if GM_SUPPORT_PCI
#if LINUX_XX >= 26
MODULE_DEVICE_TABLE(pci, gm_pci_tbl);
#endif
/****************************************************************
 * gm_linux_init_one
 *
 * Initializes one Myrinet card.  Called by the kernel pci
 *   scanning routines when the module is loaded.
 ****************************************************************/

static int
gm_linux_init_one (struct pci_dev *pdev, const struct pci_device_id *ent)
{
  GM_CALLED ();
  
  if (gm_linux_create_instance (GM_MYRINET_BUS_PCI, pdev) == GM_SUCCESS)
    {
      GM_RETURN_INT (0);
    }
  else
    {
      GM_NOTE (("Failed to initialize Myrinet Card\n"));
      GM_RETURN_INT (-ENODEV);
    }
}

/****************************************************************
 * gm_linux_remove_one
 *
 * Does what is necessary to shutdown one Myrinet device. Called
 *   once for each Myrinet card by the kernel when a module is
 *   unloaded.
 ****************************************************************/

static void
gm_linux_remove_one (struct pci_dev *pdev)
{
  gm_instance_state_t *is;

  GM_CALLED ();
  
  is = (gm_instance_state_t *) pci_get_drvdata (pdev);
  gm_linux_destroy_instance (is);
  pci_set_drvdata (pdev, 0);

  GM_RETURN_NOTHING ();
}
#endif /* GM_SUPPORT_PCI */

#define GM_DEBUG_LINUX_DESTROY_INSTANCE 0

static void
gm_linux_destroy_instance (gm_instance_state_t *is)
{
  GM_CALLED ();
  
  gm_assert (is != NULL);
  gm_assert (gm_linux_instances[is->id] == is);

  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("flushing scheduled tasks\n"));
  gm_linux_flush_scheduled_work ();

  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("unregistering for devfs\n"));
  gm_linux_devfs_unregister (is, 0);
  gm_linux_devfs_unregister (is, 1);

  if (gm_linux_skip_init)
    {
      goto unlink;
    }

#ifndef GM_NOIP
  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("finalizing IP\n"));
  gmip_finalize (is);
#endif
  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("disabling interrupts\n"));
  gm_disable_interrupts (is);

#if 0
  /* remove debugging timer (not used currently but...) */
  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("deleting timer\n"));
  del_timer (&is->arch.timer);
#endif
  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("freeing IRQ\n"));
  free_irq (is->arch.irq, is);
  GM_PRINT (GM_DEBUG_LINUX_DESTROY_INSTANCE,
	    ("finalizing instance\n"));
  gm_instance_finalize (is);
 unlink:
  gm_linux_instances[is->id] = 0;
  gm_linux_kfree (is);
  gm_linux_num_instances -= 1;

  GM_RETURN_NOTHING ();
}

#if defined UTS_VERSION
static char gm_linux_uts_version[] = UTS_VERSION;
#else
static char gm_linux_uts_version[] = "";
#endif
static char gm_linux_uts_release[] = UTS_RELEASE;

/* Build a string in the driver to indicate the kernel version for
   which it was built.  The string should be in the form of a /bin/sh
   variable setting.  We will extract this string later using the
   "strings" program and install the driver at
   /lib/modules/<UTS_RELEASE>/kernel/drivers/net. */
char *GM_UTS_RELEASE = "GM_UTS_RELEASE=\"" UTS_RELEASE "\"";

unsigned long
gm_linux_mem_nbpages (void)
{
  static unsigned long nbpages;
  if (!nbpages)
    {
      struct sysinfo mem_info;
      si_meminfo (&mem_info);
#if LINUX_XX >= 24
      /* From Linux 2.4 totalram is expressed as a number of pages */
      nbpages = mem_info.totalram;
#else
      /* before 2.4 totalram is expressed in bytes */
      nbpages = mem_info.totalram / PAGE_SIZE;
#endif
    }
  return nbpages;
}

#if GM_PERR_POLLING

static void
gm_linux_perr_thread_body (void)
{
  int unit;
  gm_status_t gm_status;
  gm_u16_t pci_status;
  gm_instance_state_t *is;

  gm_linux_down (&gm_linux_open_mutex);
  for (unit = 0; unit < gm_linux_num_instances; unit++)
    {
      is = gm_linux_instances[unit];
      if (is == NULL)
	continue;

      if (gm_is_lX (is) && is->lanai.running)
	{
	  gm_status = gm_arch_read_pci_config_16 
	    (is, GM_OFFSETOF (gm_pci_config_t, Status), &pci_status);
	  if (gm_status != GM_SUCCESS)
	    {
	      GM_WARN (("Unable to read PCI config space on board %d\n", unit));
	    } 
	  else if (pci_status & GM_PCI_STATUS_PERR)
	    {
	      GM_WARN (("PCI Parity error detected on board %d\n", unit));
	      gm_disable_lanai (is);
	      _GM_WARN (("Board has been disabled\n"));
	    }
	}
    }
  gm_linux_up (&gm_linux_open_mutex);
}

static int
gm_linux_perr_thread (void *ignored)
{
  unsigned long timeout;
  wait_queue_head_t wait;

  daemonize ();
  reparent_to_init();
  spin_lock_irq(&current->sigmask_lock);
  sigemptyset(&current->blocked);
  recalc_sigpending(current);
  spin_unlock_irq(&current->sigmask_lock);
  strncpy (current->comm, "GM PERR Polling thread", sizeof(current->comm) - 1);
  current->comm[sizeof(current->comm) - 1] = '\0';
  init_waitqueue_head (&wait);

  while (1) {
    timeout = HZ;
    do 
      {
	timeout = interruptible_sleep_on_timeout (&wait, timeout);
      } while (!signal_pending (current) && (timeout > 0));
    
    if (signal_pending (current)) 
      {
	spin_lock_irq(&current->sigmask_lock);
	flush_signals(current);
	spin_unlock_irq(&current->sigmask_lock);
      }

    if (gm_linux_perr_thread_should_exit)
      break;
    gm_linux_perr_thread_body ();
  }
  
  complete_and_exit (&gm_linux_perr_thread_exited, 0);
  return 0;
}
#endif

#if LINUX_XX >= 26

static struct cdev gm_cdev = {
  .kobj   = {.name = "gm", },
  .owner  = THIS_MODULE,
};


static int
gm_linux_class_init(void)
{
  if (gm_linux_udev)
    {
      gm_class = class_simple_create(THIS_MODULE, "gm");
      if (gm_class == NULL)
	{
	  GM_WARN(("class_simple_create returned %p\n", gm_class));
	  return ENXIO;
	}
    }
  return 0;
}

static void
gm_linux_class_fini(void)
{
  if (gm_linux_udev) 
    {
      class_simple_destroy(gm_class);
    }
}

static int
gm_linux_cdev_init(void)
{
  int err;
  dev_t dev = MKDEV(GM_MAJOR, 0);

  err = register_chrdev_region(dev, 2*GM_ARCH_MAX_INSTANCE, "gm");
  if (err != 0)
    {
      GM_WARN(("register_chrdev_region failed with status %d\n", err));
      return err;
    }
  cdev_init(&gm_cdev, &gm_linux_file_ops);
  err = cdev_add(&gm_cdev, dev, 2*GM_ARCH_MAX_INSTANCE);
  if (err != 0)
    {
      GM_WARN(("cdev_add() failed with status %d\n", err));
      kobject_put(&gm_cdev.kobj);
      unregister_chrdev_region(dev, 2*GM_ARCH_MAX_INSTANCE);
      return err;
    }
  return 0;
}

static void
gm_linux_cdev_fini(void)
{
  cdev_del(&gm_cdev);
  unregister_chrdev_region(MKDEV(GM_MAJOR, 0), 2*GM_ARCH_MAX_INSTANCE);
}

#endif /* LINUX_XX >= 26 */


static void
gm_linux_pci_map_init(void)
{
#if GM_CPU_powerpc64
  int i,j;
  for (i = 0;i < gm_linux_num_instances;i++)
    {
      int nb_users = 0;
      gm_instance_state_t *is = gm_linux_instances[i];
      gm_ppc64_iommu_t tbl = GM_PPC64_IOMMU(is->arch.pci_dev);;
      for (j = 0;j < gm_linux_num_instances; j++)
	{
	  gm_ppc64_iommu_t tbl2;
	  tbl2 = GM_PPC64_IOMMU(gm_linux_instances[j]->arch.pci_dev);
	  if (tbl2 == tbl)
	    {
	      nb_users += 1;
	    }
	}
      /* use 3/4 of a iommu for myrinet */
      atomic_set(&is->arch.free_iommu_pages, 
		 (GM_PPC64_IOMMU_SIZE(tbl) * 3)/4/nb_users);
      GM_INFO(("Board %d: using iommu span of %d Mbytes (Table is %d Mbytes)\n", 
	       is->id, pages_to_mb(atomic_read(&is->arch.free_iommu_pages)),
	       pages_to_mb(GM_PPC64_IOMMU_SIZE(tbl))));
    }
#elif GM_CPU_alpha
  int i;
  for (i = 0;i < gm_linux_num_instances; i++)
    {
      atomic_set(&gm_linux_instances[i]->arch.free_iommu_pages, 
		 GM_ALPHA_MAX_IOMMU_MAPS);
    }
#endif
}

int
gm_linux_init_module (void)
{
  unsigned long running_page_offset;

  GM_CALLED ();
  GM_VAR_MAY_BE_UNUSED (running_page_offset);
  GM_VAR_MAY_BE_UNUSED (gm_activate_page_symbol);
  GM_VAR_MAY_BE_UNUSED (gm_sprintf_symbol);
  
  gm_linux_print_struct.after_buf = 0xcafebabe;

  GM_INFO (("Version %s build %s\n", _gm_version, _gm_build_id));

#if GM_DEBUG
  if (sys_call_table_symbol == 0) 
    {
      GM_WARN (("Kernel does not export system call table\n"));
    }
#endif
  GM_INFO (("On %s, kernel version: %s %s\n",
	    system_utsname.machine, system_utsname.release, system_utsname.version));

  if (strcmp (system_utsname.release,gm_linux_uts_release) != 0 ||
      strcmp(system_utsname.version, gm_linux_uts_version) != 0) 
    {
      GM_INFO (("GM module compiled with kernel headers of %s %s\n",
		gm_linux_uts_release, gm_linux_uts_version));
    }

#if defined CONFIG_X86_PAE
  if (!cpu_has_pae || !(read_cr4() & X86_CR4_PAE))
    {
      GM_WARN (("Trying to load PAE-enabled module on a non-PAE kernel\n"));
      _GM_WARN (("please recompile appropriately\n"));
      GM_RETURN_INT (-ENODEV);
    }
#elif GM_CPU_x86 && LINUX_XX >= 24
  if (cpu_has_pae && (read_cr4() & X86_CR4_PAE))
    {
      GM_WARN (("Trying to load non-PAE module on a PAE-enabled kernel\n"));
      _GM_WARN (("please recompile appropriately\n"));
      GM_RETURN_INT (-ENODEV);
    }
#endif

#if defined(CONFIG_SMP)
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("GM driver compiled with CONFIG_SMP enabled\n"));
#else
  if (smp_call_function_symbol != 0)
    {
      GM_WARN (("GM driver has not been compiled for a SMP kernel\n"));
      _GM_WARN (("running kernel is SMP\n"));
      _GM_WARN (("please recompile appropriately\n"));
      GM_RETURN_INT (-ENODEV);
    }
#endif

#if defined(CONFIG_MODVERSIONS)
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("GM driver compiled with CONFIG_MODVERSIONS enabled\n"));
#endif

#if defined(CONFIG_BIGMEM)
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("GM driver compiled with CONFIG_BIGMEM enabled\n"));
#endif

#if defined(CONFIG_HIGHMEM)
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("GM driver compiled with CONFIG_HIGHMEM enabled\n"));
#endif

#if GM_ENABLE_DIRECTCOPY
  GM_PRINT (GM_DEBUG_BOARD_INIT,
	    ("GM driver compiled with GM DIRECTCOPY enabled\n"));
#endif

  /* Report the "highmem" memory configurations. */
  
  if (GM_LINUX_PFN_ZERO != 0
      || GM_LINUX_KERNEL_PFN_MAX != GM_LINUX_PFN_MAX)
    {
      GM_INFO (("Highmem memory configuration:\n"));
      _GM_INFO (("PFN_ZERO=0x%lx, PFN_MAX=0x%lx, KERNEL_PFN_MAX=0x%lx\n",
		 (unsigned long) GM_LINUX_PFN_ZERO,
		 (unsigned long) GM_LINUX_PFN_MAX,
		 (unsigned long) GM_LINUX_KERNEL_PFN_MAX));
    }
  
#if 0 /* GM_CPU_x86 */
  /* This check has been disabled since RedHat and others have started applying the
     4G/4G virtually memory patch to their kernels, which breaks this check. */
  /* try to get the running kernel PAGE_OFFSET to check module consistency
   * assume kernel code is in the first 128Mo after PAGE_OFFSET 
   */
  running_page_offset = (unsigned long)printk & ~(128*1024*1024 - 1);
  if (running_page_offset != PAGE_OFFSET)
    {
      GM_WARN (("GM : PAGE_OFFSET(kernel)=0x%lx COMPILED=0x%lx\n", 
		running_page_offset, PAGE_OFFSET));
      _GM_WARN (("\tmodule was not compiled with the right kernel source tree\n"));
      GM_RETURN_INT (-ENODEV);
    }
#endif /* GM_CPU_x86 */

  /* CONFIG_HIGHMEM should be defined if all RAM is not mapped into kernel */
#if (GM_CPU_x86 || GM_CPU_powerpc) && !defined CONFIG_BIGMEM && ! defined CONFIG_HIGHMEM
  if (GM_LINUX_PFN_MAX != GM_LINUX_KERNEL_PFN_MAX || kmap_high_symbol)
    {
      GM_WARN (("GM : PFN_MAX != KERNEL_PFN_MAX, kernel was compiled with CONFIG_HIGHMEM\n"));
      _GM_WARN (("\tmodule was not compiled with the right kernel source tree\n"));
      GM_RETURN_INT (-ENODEV);
    }
#endif

/****************** PCI CHIPSET TWEAKS: WARNING *************************
 *                                                                      *
 *  The patches below were supplied by customers who reported that      *
 *  their PCI performance was improved when using these patches         *
 *  on a particular chipset.                                            *
 *  These patches tweak certain bits in the chipset and have not been   *
 *  verified or reviewed by Myricom and may have other, possibly        *
 *  negative, side-effects. Before applying one of these patches,       *
 *  you may wish to check for a newer BIOS for your machine.            *
 *  Also, a newer linux kernel may provide better PCI performance,      *
 *  and might be a safer course of action than applying one of          *
 *  these patches.                                                      *
 *                                                                      *
 *  Use these patches at your own risk.                                 *
 *                                                                      *
 ***********************************************************************/

  /* Before enabling this code see the PCI CHIPSET TWEAKS: WARNING */

#define GM_21154 0  
#if GM_21154
  /* 21154 PCI Bridge stuff */
  {
    int i, myint;
    unsigned short myword = 0x1234;
    unsigned char mychar;
    struct pci_dev *pcidev = NULL;

    pcidev = pci_find_device (0x1011, 0x0026, pcidev);
    if (pcidev)
      {
	pci_read_config_byte (pcidev, 0x40, &mychar);
	printk ("GM: 21154 before pci_bios_read_config_byte(0x40) = 0x%02x\n",
		mychar);

	pci_write_config_byte (pcidev, 0x40, 0);

	pci_read_config_byte (pcidev, 0x40, &mychar);
	printk ("GM: 21154 after pci_bios_read_config_byte(0x40) = 0x%02x\n",
		mychar);

	printk ("GM: Reading 21154 \n");
	for (i = 0; i < 0x44; i += 4)
	  {
	    pci_read_config_dword (pcidev, i, &myint);
	    if (myint)
	      {
		printk ("GM: 21154: %08x: 0x%08x\n", i, myint);
	      }
	  }
      }
  }
#endif /* 21154 */


  /* Before enabling this code see the PCI CHIPSET TWEAKS: WARNING */

#define GM_INTEL_840 0
#if GM_INTEL_840
  /* Intel840 PCI stuff */
  {
    int i, myint;
    unsigned short myword = 0x1234;
    struct pci_dev *pcidev = NULL;

    pcidev = pci_find_device (0x8086, 0x1360, pcidev);
    if (pcidev)
      {
	pci_read_config_word (pcidev, 0x50, &myword);
	myword = myword & ~0x0018;
	pci_write_config_word (pcidev, 0x50, myword);
      }
  }
#endif /* INTEL840 */

  /* Before enabling this code see the PCI CHIPSET TWEAKS: WARNING */

#define GM_INTEL_860 0
#if GM_INTEL_860
  /* Intel860 PCI stuff */
  {
    int i, myint;
    unsigned short myword = 0x1234;
    struct pci_dev *pcidev = NULL;

    pcidev = pci_find_device (0x8086, 0x1360, pcidev);
    if (pcidev)
      {
	pci_read_config_word (pcidev, 0x50, &myword);
	myword = myword & ~0x0018;
	myword = myword | 0x0004;      /* try DT depth = 512 bytes */
	pci_write_config_word (pcidev, 0x50, myword);
      }
  }
#endif /* INTEL860 */

  /* Before enabling this code see the PCI CHIPSET TWEAKS: WARNING */

#define GM_INTEL_450NX 0
#if GM_INTEL_450NX
  /* 450NX stuff */
  {
    int i;
    unsigned short myword = 0x1234;
    unsigned char mychar = 0x12;
    unsigned int myint;
    struct pci_dev *pcidev = NULL;

    pcidev = pci_find_device (0x8086, 0x84cb, pcidev);

    if (pcidev)
      {
	pci_read_config_word (pcidev, 0x40, &myword);
	pci_read_config_byte (pcidev, 0xA0, &mychar);
	printk ("GM: i450NX pci_bios_read_config_word(0, 18, 0x40) "
		"= 0x%04x\n", myword);
	printk ("GM: i450NX pci_bios_read_config_byte(0, 18, 0xA0) "
		"= 0x%02x\n", mychar);

	pci_write_config_word (pcidev, 0x40, 0x2360);
	/* pci_write_config_byte(pcidev, 0xA0, 0xFF); */

	pci_read_config_word (pcidev, 0x40, &myword);
	pci_read_config_byte (pcidev, 0xA0, &mychar);
	printk ("GM: i450NX pci_bios_read_config_word(0, 18, 0x40) "
		"= 0x%04x\n", myword);
	printk ("GM: i450NX pci_bios_read_config_byte(0, 18, 0xA0) "
		"= 0x%02x\n", mychar);

	printk ("GM: Reading device 18\n");
	for (i = 0; i < 0x100; i += 4)
	  {
	    pci_read_config_dword (pcidev, i, &myint);
	    if (myint)
	      {
		printk ("%08x: 0x%08x\n", i, myint);
	      }
	  }

	pcidev = pci_find_device (0x8086, 0x84cb, pcidev);

	if (pcidev)
	  {
	    pci_read_config_word (pcidev, 0x40, &myword);
	    pci_read_config_byte (pcidev, 0xA0, &mychar);
	    printk ("GM: i450NX feldy pci_bios_read_config_word(0, 19, 0x40) "
		    "= 0x%04x\n", myword);
	    printk ("GM: i450NX pci_bios_read_config_byte(0, 19, 0xA0) "
		    "= 0x%02x\n", mychar);

	    pci_write_config_word (pcidev, 0x40, 0x2360);
	    /* pci_write_config_byte(pcidev, 0xA0, 0xFF); */

	    pci_read_config_word (pcidev, 0x40, &myword);
	    pci_read_config_byte (pcidev, 0xA0, &mychar);
	    printk ("GM: i450NX pci_bios_read_config_word(0, 19, 0x40) "
		    "= 0x%04x\n", myword);
	    printk ("GM: i450NX pci_bios_read_config_byte(0, 19, 0xA0) "
		    "= 0x%02x\n", mychar);

	    printk ("GM: Reading device 19\n");
	    for (i = 0; i < 0x100; i += 4)
	      {
		pci_read_config_dword (pcidev, i, &myint);
		if (myint)
		  {
		    printk ("GM: %08x: 0x%08x\n", i, myint);
		  }
	      }
	  }
      }
  }
#endif /* GM_INTEL_450NX */

  /* Before enabling this code see the PCI CHIPSET TWEAKS: WARNING */

#define GM_KT266A 0
#if GM_KT266A
  /* VIA KT266A Stuff */
  {
    struct pci_dev *dev = NULL;
    u8 result;
    int reg;

    if ((dev = pci_find_device(PCI_VENDOR_ID_VIA, 
			       PCI_DEVICE_ID_VIA_8367_0, NULL)) != NULL) {
      reg = 0x0d;
      result = 0x00;
      pci_write_config_byte(dev, reg, result);

      // Disable PCI Bus master time-out
      reg = 0x75;
      pci_read_config_byte(dev, reg, &result);
      result &= 0xf8;
      pci_write_config_byte(dev, reg, result);
    }
  }
#endif /* VIA KT266A */
  
  gm_always_assert (sizeof (gm_u64_t) == 8);
  gm_always_assert (sizeof (gm_u32_t) == 4);
  gm_always_assert (sizeof (gm_u16_t) == 2);
  gm_always_assert (sizeof (gm_u8_t) == 1);

  if (gm_arch_get_page_len (&GM_PAGE_LEN) != GM_SUCCESS)
    {
      /* can't fail in Linux */
      GM_WARN (("gm_arch_get_page_len failed??\n"));
      GM_RETURN_INT (-ENXIO);
    }

  /*  this is sanity limits on registered memory to prevent the user
      from bringing the system to a complete stop */
  if (gm_linux_mem_nbpages() >= (2 * GM_MAX_SAVE_FROM_LOCKED))
    {
      /* system with decent amount of memory, take the lower bound
       of BIGMEM and SMALLMEM computation */
      unsigned long bigmem_limit;
      gm_linux_max_user_locked_pages_start
	= gm_linux_mem_nbpages() - GM_MAX_SAVE_FROM_LOCKED;
      bigmem_limit =
	GM_MAX_USER_LOCKED_BIGMEM (gm_linux_mem_nbpages());
      if (gm_linux_max_user_locked_pages_start > bigmem_limit)
	gm_linux_max_user_locked_pages_start = bigmem_limit;
    }
  else
    {
      gm_linux_max_user_locked_pages_start
	= GM_MAX_USER_LOCKED_SMALLMEM (gm_linux_mem_nbpages());
    }
  if (gm_max_locked_mbytes)
    {
      GM_INFO (("Max amount available for registration overriden by user\n"));
      gm_linux_max_user_locked_pages_start = 
	gm_max_locked_mbytes << (20 - PAGE_SHIFT);
    }

  atomic_set (&gm_linux_max_user_locked_pages, gm_linux_max_user_locked_pages_start);

#if defined HAVE_ASM_RMAP_H && LINUX_XX == 24
#if GM_CPU_ia64
  {
    /* the symbol from system.map are code pointers,
       C function pointers  points to a descriptor (code_ptr,got_ptr)
       compare and reconstruct accordingly */
    static struct fptr {
      gm_u64_t code;
      gm_u64_t got;
    } func_desc;
    struct fptr *s = (void *)sprintf;
    if (gm_activate_page_symbol && s->code != gm_sprintf_symbol) {
      gm_activate_page_symbol = 0;
      GM_WARN (("gm_activate_page_symbol ignored: sprintf check failed\n"));
    } else {
      func_desc.code = gm_activate_page_symbol;
      func_desc.got = s->got;
      gm_activate_page_symbol = (gm_size_t)&func_desc;
    }
  }
#else
  if (gm_activate_page_symbol && gm_sprintf_symbol != (gm_size_t)sprintf)
    {
      gm_activate_page_symbol = 0;
      GM_WARN (("gm_activate_page_symbol ignored: sprintf check failed\n"));
    }
#endif
  if (gm_activate_page_symbol)
    {
      GM_INFO (("activate_page (0x%lx) used: good\n", gm_activate_page_symbol));
    }
  else
    {
      GM_INFO (("no activate_page, swapping might cause unresponsive state\n"));
    }
#endif

  GM_INFO (("Memory available for registration: %ld pages (%ld MBytes)\n",
	    gm_arch_max_locked_pages (),
	    gm_arch_max_locked_pages () >> (20 - PAGE_SHIFT)));


#if LINUX_XX >= 26
  if (gm_linux_class_init())
    {
      GM_RETURN_INT (-EBUSY);
    }
  if (gm_linux_cdev_init())
    {
      gm_linux_class_fini();
      GM_RETURN_INT (-EBUSY);
    }

#else  
  if (register_chrdev (GM_MAJOR, "gm", &gm_linux_file_ops))
    {
      GM_NOTE (("failed to register character device\n"));
      GM_RETURN_INT (-EBUSY);
    }
  register_chrdev (GM_MAJOR_OLD, "gm", &gm_linux_file_ops);
#endif

#if GM_CPU_powerpc64 || GM_CPU_x86_64
  if ( (gm_linux_register_ioctl32_conversions()) )
    {
       GM_WARN (("failed to register ioctl32 conversions\n"));
    }
#endif

  GM_PRINT (GM_PRINT_LEVEL >= 5, ("page size is %ld\n", GM_PAGE_LEN));

  /* from now we go through gm_linux_cleanup_module in case of error */
  {
    gm_status_t status;
    
    status = gm_init ();
    if (status != GM_SUCCESS)
      {
	GM_WARN (("Could not initialize GM internals.\n"));
	GM_RETURN_INT (-gm_linux_localize_status (status));
      }
  }

  /* initialize the hash table to track locking virtual memory */
  gm_linux_pfn_hash = gm_create_hash (gm_hash_compare_longs,
                                      gm_hash_hash_long,
                                      sizeof (unsigned long), sizeof (int),
                                      0, 0);

  if (!gm_linux_pfn_hash)
    {
      GM_WARN (("init_module: couldn't initialize physical memory "
               "gm hash table\n"));
      /* necessary to unregister the char device */
      gm_linux_cleanup_module ();
      return -ENODEV;
    }
  gm_arch_sync_init (&gm_linux_pfn_sync, 0);

  /* moved here in case is used during initialization */
  /* these initialization does not allocate resources,
     so even if module init failed no need to free resources */
  init_MUTEX (&gm_linux_open_mutex);
  spin_lock_init (&gm_linux_intr_lock);

  if (!gm_linux_get_user_pages)
    {
      GM_INFO (("kernel does not export get_user_pages\n"));
    }
#if GM_SUPPORT_PCI
  if (pci_module_init (&gm_driver) != 0)
    {
      gm_linux_cleanup_module ();
      GM_WARN (("No board initialized\n"));
      GM_RETURN_INT (-ENODEV);
    }
  gm_linux_pci_driver_registered = 1;
#elif CSPI_MAP26xx
  if (gm_linux_create_instance (GM_MYRINET_BUS_MAP26xx, 0) != GM_SUCCESS)
    {
      gm_linux_cleanup_module ();
      GM_INFO (("Failed to create cspi instance\n"));
      GM_RETURN_INT (-ENODEV);
    }
#else
#error bus_type undefined!!
#endif

#if GM_PERR_POLLING
  init_completion (&gm_linux_perr_thread_exited);
  gm_linux_perr_pid = kernel_thread (gm_linux_perr_thread, 0,
				     CLONE_FS | CLONE_FILES);
  if (gm_linux_perr_pid < 0)
    GM_WARN (("Unable to start perr polling thread\n"));
  else
    GM_INFO (("PERR Polling thread running with PID %ld\n",
	      (long)gm_linux_perr_pid));
#endif
  
  gm_linux_pci_map_init();
  GM_INFO (("%d Myrinet board(s) found and initialized\n", 
	    gm_linux_num_instances));
  GM_RETURN_INT (0);
}

#define GM_DEBUG_LINUX_CLEANUP_MODULE 0

void
gm_linux_cleanup_module (void)
{
  GM_CALLED ();

#if GM_PERR_POLLING
  if (gm_linux_perr_pid >= 0)
    {
      int ret;

      gm_linux_perr_thread_should_exit = 1;
      ret = kill_proc (gm_linux_perr_pid, SIGTERM, 1);
      if (ret) 
	{
	  GM_WARN (("failed to signal PERR thread\n"));
	}
      else
	{
	  wait_for_completion (&gm_linux_perr_thread_exited);
	}
    }
#endif
  
#if GM_SUPPORT_PCI
  if (gm_linux_pci_driver_registered)
    {
      pci_unregister_driver (&gm_driver);
    }
#elif CSPI_MAP26xx
  if (gm_linux_instances[0])
    gm_linux_destroy_instance (gm_linux_instances[0]);
#else
#error bus_type??
#endif
  GM_PRINT (GM_DEBUG_LINUX_CLEANUP_MODULE, ("cleaning up\n"));
  GM_PRINT
    (GM_DEBUG_LINUX_CLEANUP_MODULE, ("unregistering character device\n"));

#if LINUX_XX >= 26
  gm_linux_class_fini();
  gm_linux_cdev_fini();
#else
  unregister_chrdev (GM_MAJOR_OLD, "gm");
  unregister_chrdev (GM_MAJOR, "gm");
#endif
  GM_PRINT
    (GM_DEBUG_LINUX_CLEANUP_MODULE, ("unregistered character device\n"));

#if GM_CPU_powerpc64 || GM_CPU_x86_64
  if ( (gm_linux_unregister_ioctl32_conversions()) )
    {
       GM_WARN (("failed to unregister ioctl32 conversions\n"));
    }
#endif

  if (gm_linux_pfn_hash)
    {
      gm_destroy_hash (gm_linux_pfn_hash);
      gm_linux_pfn_hash = (struct gm_hash *) 0;
      gm_arch_sync_destroy (&gm_linux_pfn_sync);
    }

  gm_finalize ();

#if GM_DEBUG
  printk ("memory leak info:\n");
  printk ("  kmallocs      = %d\n", kmalloc_cnt);
  printk ("  kfrees        = %d\n", kfree_cnt);
  printk ("  vmallocs      = %d\n", vmalloc_cnt);
  printk ("  vfrees        = %d\n", vfree_cnt);
  printk ("  ioremaps      = %d\n", ioremap_cnt);
  printk ("  iounmaps      = %d\n", iounmap_cnt);
  printk ("  dma allocs    = %d\n", dma_alloc_cnt);
  printk ("  dma frees     = %d\n", dma_free_cnt);
  printk ("  kernel allocs = %d\n", kernel_alloc_cnt);
  printk ("  kernel frees  = %d\n", kernel_free_cnt);
  printk ("  user buffer lock = %d\n", user_lock_cnt);
  printk ("  user buffer unlock  = %d\n", user_unlock_cnt);
#endif
  
  GM_RETURN_NOTHING ();
}

module_init (gm_linux_init_module);
module_exit (gm_linux_cleanup_module);

/***********************************************************************
 * Character device entry points.
 ***********************************************************************/

static int
gm_linux_localize_status (gm_status_t status)
{
  GM_CALLED ();
  
#define CASE(from,to) case from : GM_RETURN_INT (to)
  switch (status)
    {
      CASE (GM_SUCCESS, 0);
      CASE (GM_INPUT_BUFFER_TOO_SMALL, EFAULT);
      CASE (GM_OUTPUT_BUFFER_TOO_SMALL, EFAULT);
      CASE (GM_TRY_AGAIN, EAGAIN);
      CASE (GM_BUSY, EBUSY);
      CASE (GM_MEMORY_FAULT, EFAULT);
      CASE (GM_INTERRUPTED, EINTR);
      CASE (GM_INVALID_PARAMETER, EINVAL);
      CASE (GM_OUT_OF_MEMORY, ENOMEM);
      CASE (GM_INVALID_COMMAND, EINVAL);
      CASE (GM_PERMISSION_DENIED, EPERM);
      CASE (GM_INTERNAL_ERROR, EPROTO);
      CASE (GM_UNATTACHED, EUNATCH);
      CASE (GM_UNSUPPORTED_DEVICE, ENXIO);
    default:
      GM_RETURN_INT (gm_linux_localize_status (GM_INTERNAL_ERROR));
    }
}

#define GM_DEBUG_IOCTL 0

/* misc GM functionality - pass control to arch-independent code */
static int
gm_linux_ioctl (struct inode *inodeP, struct file *fileP,
		unsigned int cmd, unsigned long arg)
{
  gm_port_state_t *ps;
  gm_status_t status = GM_SUCCESS;
  GM_PARAMETER_MAY_BE_UNUSED (inodeP);

  GM_CALLED ();

  if (cmd == GM_SET_PORT_NUM)
    {
      gm_linux_down (&gm_linux_open_mutex);
    }

  GM_PRINT (GM_DEBUG_IOCTL, ("gm_linux_ioctl called with cmd 0x%x %s.\n",
			     cmd, _gm_ioctl_cmd_name (cmd)));

  ps = fileP->private_data;
  gm_arch_mutex_enter (&ps->sync);

  GM_PRINT (GM_PRINT_LEVEL >= 5,
	    ("gm_linux_ioctl: cmd = %s\n", _gm_ioctl_cmd_name (cmd)));

  status = gm_ioctl (ps, cmd, (void *) arg, INT_MAX, (void *) arg, INT_MAX, 0);

  gm_arch_mutex_exit (&ps->sync);
  if (cmd == GM_SET_PORT_NUM)
    {
      gm_linux_up (&gm_linux_open_mutex);
    }
  GM_RETURN_INT (gm_linux_localize_status (status));
}

#define MAPPING_OF_TYPE(type, off)					\
  (ps->mappings.type.offset <= off					\
   && off < (ps->mappings.type.offset					\
             + (gm_offset_t)ps->mappings.type.len))



static void
linux_vm_open (struct vm_area_struct *vma)
{
  gm_port_state_t *ps;

  GM_CALLED ();
  
  gm_linux_down (&gm_linux_open_mutex);
  gm_assert (vma->vm_file);
  ps = (gm_port_state_t *) vma->vm_file->private_data;
  gm_arch_mutex_enter (&ps->arch.sync);
  gm_always_assert (ps->arch.ref_count > 0);
  ps->arch.ref_count += 1;
  gm_arch_mutex_exit (&ps->arch.sync);
  gm_linux_up (&gm_linux_open_mutex);
  
  if (vma->vm_mm != current->mm && current->mm == ps->arch.mm &&
      vma->vm_pgoff == ps->mappings.send_queue.offset / PAGE_SIZE) 
    {
      static int done;
      if (!done)
	{
	  done = 1;
	  GM_INFO ((" A gm program is using fork():OK\n"));
	}
      gm_linux_fork_wrap (vma);
      
    }
  GM_RETURN_NOTHING ();
}

static void
linux_vm_close (struct vm_area_struct *vma)
{
  gm_port_state_t *ps;
  int ready_to_close;

  GM_CALLED ();
  
  gm_assert (vma->vm_file);

  ps = (gm_port_state_t *) vma->vm_file->private_data;
  gm_arch_mutex_enter (&ps->arch.sync);
  gm_always_assert (ps->arch.ref_count > 0);
  ready_to_close = (--ps->arch.ref_count == 0);
  gm_arch_mutex_exit (&ps->arch.sync);
  if (ready_to_close)
    {
      GM_INFO (("myri/gm: closing port after last mapping finished\n"));
      gm_linux_port_close (ps);
    }

  GM_RETURN_NOTHING ();
}


static struct vm_operations_struct gm_linux_vm_ops = {
  open:linux_vm_open,		/* open */
  close:linux_vm_close,		/* close */
};

/* map some pages into user space */

#define GM_DEBUG_LINUX_MMAP 0

static unsigned long
gm_linux_get_area (struct file *file, unsigned long addr, unsigned long len,
		   unsigned long pgoff, unsigned long flags)
{
  gm_port_state_t *ps;
  ps = file->private_data;
  if (pgoff == ps->mappings.send_queue.offset / PAGE_SIZE)
    {
      /* map the send_queue at the beginning of the address space for
	 fork() support, find a hole at the beginning (ignoring other
	 lanai mappings)
      */
      struct vm_area_struct *vma;
      /* Trying to put the send_queue at 2Mb that ensures we are first
	 in the address space on all supported arches, and still
	 ensuring *(int*)0 still segfaults */
      unsigned long low_addr = 2 * 1024 * 1024;
      vma = find_vma (current->mm, low_addr);
      while (vma && vma->vm_ops == &gm_linux_vm_ops)
	{
	  low_addr = vma->vm_end;
	  vma = find_vma (current->mm, low_addr);
	}
      if (vma && low_addr + len  < vma->vm_start 
	  && low_addr + len < current->mm->start_brk
	  && low_addr + len < current->mm->start_data)
	{
	  /* found the right hole !! */
	  addr = low_addr;
	  flags |= MAP_FIXED;
	}
      else
	{
	  /* this notice is partly redundant with the checking in
	     gm_arch_lock_user_buffer, but we'd like to know if the
	     send_queue cannot be put right at the beginning, even if
	     it still before any registered page */
	  GM_NOTE_ONCE (("Limited fork() support with GM ports opened\n"));
	}
    }
  addr = get_unmapped_area (0, addr, len, pgoff, flags GM_LINUX_EXEC_SHIELD_ARG);
  return addr;
}


static int
gm_linux_mmap (struct file *fileP, struct vm_area_struct *vma)
{
  gm_port_state_t *ps;
  gm_status_t status;
  gm_offset_t off;
  gm_size_t len;
  unsigned long start, end, pos;
  pgprot_t prot;
  void *kptr;
  gm_phys_t phys;
  unsigned int requested_permissions;

  GM_CALLED ();

  ps = fileP->private_data;
  gm_arch_mutex_enter (&ps->sync);
  gm_assert (ps->arch.ref_count > 0);
  gm_assert (vma->vm_ops == NULL);
  vma->vm_ops = &gm_linux_vm_ops;
  vma->vm_flags |= VM_IO;
#if 0 /* we need to copy it to detect fork */
  vma->vm_flags |= VM_DONTCOPY;
#endif
  start = vma->vm_start;
  end = vma->vm_end;
  off = vma_get_pgoff (vma) * GM_PAGE_LEN;
  len = end - start;

  requested_permissions = 0;
  if ((vma->vm_flags) & VM_READ)
    {
      requested_permissions |= GM_MAP_READ;
    }
  if ((vma->vm_flags) & VM_WRITE)
    {
      requested_permissions |= GM_MAP_WRITE;
    }

  if (vma->vm_pgoff  == ps->mappings.send_queue.offset / PAGE_SIZE)
    {
      ps->arch.send_queue_addr = vma->vm_start;
    }
  if (GM_DEBUG_LINUX_MMAP)
    {
      GM_PRINT (GM_PRINT_LEVEL >= 0,
		("mmap: offset= %lx  start= %lx  end= %lx  len= %lx\n",
		 off, start, end, len));
      if (MAPPING_OF_TYPE (copy_block, off))
	{
	  GM_PRINT (GM_PRINT_LEVEL >= 0,
		    ("mmap: mapping a copy block segment\n"));
	}
      else if (MAPPING_OF_TYPE (sram, off))
	{
	  GM_PRINT (GM_PRINT_LEVEL >= 0, ("mmap: mapping an sram segment\n"));
	}
      else if (MAPPING_OF_TYPE (control_regs, off))
	{
	  GM_PRINT (GM_PRINT_LEVEL >= 0,
		    ("mmap: mapping a control_regs segment\n"));
	}
      else if (MAPPING_OF_TYPE (special_regs, off))
	{
	  GM_PRINT (GM_PRINT_LEVEL >= 0,
		    ("mmap: mapping a special_regs segment\n"));
	}
      else
	{
	  GM_PRINT (GM_PRINT_LEVEL >= 0,
		    ("mmap: mapping unknown type of segment\n"));
	}
    }

  status = gm_prepare_to_mmap (ps, off, len, requested_permissions);
  if (status != GM_SUCCESS)
    {
      GM_PRINT (GM_DEBUG_LINUX_MMAP,
		("whoops: gm_prepare_to_mmap returned an error\n"));
      goto abort_with_mutex;
    }

  /*
   * do the mapping a page at a time
   */
  for (pos = 0; pos < len; pos += PAGE_SIZE)
    {
      status = gm_mmap (ps, off + pos, &kptr);
      if (status != GM_SUCCESS)
	{
	  GM_PRINT (GM_DEBUG_LINUX_MMAP, ("whoops: gm_generic_mmap failed 0\n"));
	  goto abort_with_mutex;
	}

      /*
       * remap_page_range wants a physical address; only the recv_queue
       * was allocated with kmalloc (we do not care, kvirt_to_phys is 
       * working in any case) */

      phys = gm_linux_kvirt_to_phys (ps->instance, (unsigned long) kptr, 1);
      if (!phys)
	{
	  GM_WARN (("gm_linux_mmap: gm_linux_kvirt_to_phys failed\n"));
	  goto abort_with_mutex;
	}

      GM_PRINT (GM_DEBUG_LINUX_MMAP,
		("off = 0x%lx  kptr = %p  phys = 0x%lx  user = 0x%lx\n",
		 off, kptr, gm_linux_pfn (phys), start + pos));

      /*
       * pages allocated with the kernel allocators need to be reserved
       * before being mapped into user space
       *
       * FIXME: should unreserve pages upon deallocation, or better yet,
       * use the nopage technique from Rubin ip.283 so we don't have to
       * reserve at all
       */


      prot = ((vma)->vm_page_prot);

      if (!gm_linux_phys_on_board (ps->instance, phys))
	{
	  gm_linux_reserve_page (ps->instance, phys);
	  gm_always_assert (gm_linux_pfn (phys) < GM_LINUX_KERNEL_PFN_MAX);
	  GM_PRINT (GM_DEBUG_LINUX_MMAP,
		    ("pfn = 0x%lx  remaparg = 0x%lx\n", gm_linux_pfn (phys), 
		     gm_linux_pfn (phys)));
	  
	  if (gm_linux_remap_pfn_range (vma, start + pos, gm_linux_pfn(phys), PAGE_SIZE, prot))
	    {
	      GM_PRINT (GM_DEBUG_LINUX_MMAP, ("oops: remap page range failed\n"));
	      goto abort_with_mutex;
	    }
	}
      else
	{
#if !CSPI_MAP26xx
	  /* FIXME... should really check to see if 'phys' falls in
	     the different resource of the CSPI board instead of just
	     ignoring this check --nelson */

	  if ((phys < ps->instance->arch.phys_base_addr) ||
	      (phys >= (ps->instance->arch.phys_base_addr
			+ ps->instance->board_span)))
	    {
	      if (GM_DEBUG_LINUX_MMAP)
		{
		  if (MAPPING_OF_TYPE (copy_block, off))
		    {
		      GM_PRINT (GM_PRINT_LEVEL >= 0,
				("mmap: mapping a copy block segment\n"));
		    }
		  else if (MAPPING_OF_TYPE (sram, off))
		    {
		      GM_PRINT (GM_PRINT_LEVEL >= 0,
				("mmap: mapping an sram segment\n"));
		    }
		  else if (MAPPING_OF_TYPE (control_regs, off))
		    {
		      GM_PRINT (GM_PRINT_LEVEL >= 0,
				("mmap: mapping a control_regs segment\n"));
		    }
		  else if (MAPPING_OF_TYPE (special_regs, off))
		    {
		      GM_PRINT (GM_PRINT_LEVEL >= 0,
				("mmap: mapping a special_regs segment\n"));
		    }
		  else
		    {
		      GM_PRINT (GM_PRINT_LEVEL >= 0,
				("mmap: mapping unknown type of segment\n"));
		    }
		}

	      GM_WARN (("Bad physical address in gm_linux_mmap\n"
			"    pfn=0x%lx  base_pfn=0x%lx  span(nbpages)=0x%lx\n"
			"    KERNEL_PFN_MAX = 0x%lx  PFN_MAX = 0x%lx "
			"PFN_ZERO = 0x%lx\n",
			gm_linux_pfn (phys), 
			gm_linux_pfn (ps->instance->arch.phys_base_addr),
			gm_linux_pfn (ps->instance->board_span),
			GM_LINUX_KERNEL_PFN_MAX,
			GM_LINUX_PFN_MAX,
			GM_LINUX_PFN_ZERO));


	      goto abort_with_mutex;
	    }
#endif /* !CSPI_MAP26xx */

	  /*
	   * disable caching - is this necessary?
	   * loic: yes from lanai access as done here
	   */
	  pgprot_val (prot) &= ~GM_LINUX_PAGE_CACHE;
	  pgprot_val (prot) |= GM_LINUX_PAGE_NOCACHE;
	  gm_assert (!MAPPING_OF_TYPE (copy_block, off));
	  GM_PRINT (GM_DEBUG_LINUX_MMAP,
		    ("pfn = 0x%lx  ioremaparg = 0x%lx\n", gm_linux_pfn (phys), 
		     gm_linux_pfn (phys)));
	  
	  {
	    GM_PRINT (GM_PRINT_LEVEL >= 2,("calling io_remap_page_range\n"));
	    if (gm_linux_io_remap_page_range (vma, start + pos, phys, PAGE_SIZE, prot))
	      {
		GM_PRINT (GM_DEBUG_LINUX_MMAP, ("oops: ioremap page range failed\n"));
		goto abort_with_mutex;
	      }
	  }
	}
    }

  status = gm_finish_mmap (ps, off, len, start);
  if (status != GM_SUCCESS)
    {
      GM_PRINT (GM_DEBUG_LINUX_MMAP,
		("whoops: gm_finish_mmap returned an error\n"));
      goto abort_with_mutex;
    }

  GM_PRINT (GM_DEBUG_LINUX_MMAP, ("mmap was successful\n"));
  gm_arch_mutex_enter (&ps->arch.sync);
  ps->arch.ref_count += 1;
  gm_arch_mutex_exit (&ps->arch.sync);
  gm_arch_mutex_exit (&ps->sync);
  GM_RETURN_INT (0);

 abort_with_mutex:
  GM_PRINT (GM_DEBUG_LINUX_MMAP, ("gm_linux_mmap() failed\n"));
  gm_arch_mutex_exit (&ps->sync);
  GM_RETURN_INT (-1);
}

/* open a device. */
static int
gm_linux_open (struct inode *inodeP, struct file *fileP)
{
  unsigned int unit;
  gm_instance_state_t *is;
  gm_port_state_t *ps;
  gm_status_t status;

  GM_CALLED ();

  gm_linux_down (&gm_linux_open_mutex);

  unit = minor (inodeP->i_rdev) / 2;
  if (unit >= GM_ARCH_MAX_INSTANCE)
    {
      gm_linux_up (&gm_linux_open_mutex);
      GM_RETURN_INT (-ENODEV);
    }
  is = gm_linux_instances[unit];
  if (!is)
    {
      gm_linux_up (&gm_linux_open_mutex);
      GM_RETURN_INT (-ENODEV);
    }
  gm_assert (is->id == unit);

  /* Alloc and initialize a port state structure for the new open.
     This code looks funny since Linux does not use GM minor numbers.

     NOTE: The linux-specific initialization of the port state is in
     gm_arch_port_state_init(). */

  {
    int fake_minor;

    status = gm_minor_alloc (is, &fake_minor);
    if (status != GM_SUCCESS)
      {
	gm_linux_up (&gm_linux_open_mutex);
	GM_RETURN_INT (-gm_linux_localize_status (status));
      }
    ps = gm_minor_get_port_state (fake_minor);

    /* Give the new device the same privileges as the factory device. */
    
    ps->privileged = minor (inodeP->i_rdev) & 1;
    GM_PRINT (GM_DEBUG_SECURITY,
	      ("new device got privileged=%d\n", ps->privileged));
    ps->arch.mm = current->mm;
  }

  /* Cache the port state so that we can avoid the hash table
     lookup overhead of gm_minor_get_port_state() from now on. */

  fileP->private_data = ps;

  /* don't let the kernel unload the module while a device is open */

  GM_MOD_INC_USE_COUNT;

  gm_linux_up (&gm_linux_open_mutex);

  GM_RETURN_INT (0);
}

/* close a device */
static int
gm_linux_close (struct inode *inodeP, struct file *fileP)
{
  gm_port_state_t *ps;
  int ready_to_close;
  GM_PARAMETER_MAY_BE_UNUSED (inodeP);

  GM_CALLED ();


  ps = fileP->private_data;
  gm_assert (ps);
  gm_arch_mutex_enter (&ps->arch.sync);
  gm_always_assert (ps->arch.ref_count > 0);
  /* only close the port if there is no more mapping */
  ready_to_close = (--ps->arch.ref_count == 0);
  gm_arch_mutex_exit (&ps->arch.sync);
  if (ready_to_close)
    {
      gm_linux_port_close (ps);
    }
  else
    {
      GM_INFO (("Application closed file descriptor while "
		"mappings still alive: port destruct delayed\n"));
    }
  fileP->private_data = 0;
  GM_RETURN_INT (0);
}

void
gm_linux_port_close (gm_port_state_t * ps)
{
  gm_always_assert (ps->arch.ref_count == 0);

  GM_CALLED ();
  
  GM_PRINT (GM_PRINT_LEVEL >= 3, ("gm_linux_port_close\n"));

  /* Finalize and free the port state in the usual GM way.  This
     code looks funny since Linux does not use GM minor numbers. */

  gm_linux_down (&gm_linux_open_mutex);
  gm_minor_free (ps->minor);
  gm_linux_up (&gm_linux_open_mutex);

  GM_MOD_DEC_USE_COUNT;

  GM_RETURN_NOTHING ();
}


#if !GM_CPU_powerpc && !GM_CPU_powerpc64 && !GM_CPU_alpha && !GM_CPU_ia64
/* KLUDGE: defeat evil Linux macros */
/* loic: I put this at end, only zlib need it??? */
#undef memcpy
#undef memset

/* Needed to allow gcc structure copy to work. */

void *
memcpy (void *dest, const void *from, __kernel_size_t bytes)
{
  __memcpy (dest, from, bytes);
  return dest;
}

void *
memset (void *s, int c, __kernel_size_t n)
{
  while (n--)
    {
      ((char *) s)[n] = c;
    }
  return s;
}

#endif



/****************************************************************
 ****************************************************************
 * Low level architecture dependent functions
 ****************************************************************
 ****************************************************************/

#if LINUX_22
int
pci_module_init (struct pci_driver *drv)
{
  int i = 0;
  struct pci_dev *pcidev = 0;
  const struct pci_device_id *tbl;

  GM_CALLED ();
  
  for (tbl = drv->id_table; tbl->vendor; tbl++)
    {
      printk ("GM: searching for pcidev %x %x\n", tbl->vendor, tbl->device);
      pcidev = NULL;
      do
	{
	  pcidev = pci_find_device (tbl->vendor, tbl->device, pcidev);
	  if (pcidev && drv->probe (pcidev, drv->id_table) == 0)
	    {
	      drv->devs[i++] = pcidev;
	    }
	  if (i == VXX_MAX_DEV_PER_DRIVER)
	    {
	      GM_RETURN_INT (0);
	    }
	}
      while (pcidev);
    }
  GM_RETURN_INT (i ? 0 : -ENODEV);
}

void
pci_unregister_driver (struct pci_driver *drv)
{
  int i;

  GM_CALLED ();
  
  for (i = 0; i < VXX_MAX_DEV_PER_DRIVER && drv->devs[i]; i++)
    {
      if (drv->remove)
	{
	  drv->remove (drv->devs[i]);
	}
      drv->devs[i] = NULL;
    }

  GM_RETURN_NOTHING ();
}
#endif

#if LINUX_22
struct net_device *
gm_linux_dev_alloc (char *s, int *err, int num)
{
  struct device *dev;

  GM_CALLED ();
  
  dev = kmalloc (sizeof (*dev) + strlen (s) + 2, GFP_KERNEL);
  if (!dev)
    {
      *err = -ENOMEM;
    }
  else
    {
      memset (dev, 0, sizeof (*dev));
      dev->name = (char *) (dev + 1);
      sprintf (dev->name, s, num);
    }
  GM_RETURN (dev);
}
#endif

#if  LINUX_XX >= 26
struct net_device *
gm_linux_dev_alloc (char *s, int *err, int num)
{
  char name[IFNAMSIZ];
  struct net_device *dev;

  GM_CALLED ();
  sprintf(name, s, num);
  dev = alloc_netdev (0, name, ether_setup);
  if (!dev)
    {
      *err = -ENOMEM;
    }
  else
    {
      *err = 0;
    }
  GM_RETURN (dev);
}
#endif

#if GM_SUPPORT_PCI
static unsigned long 
gm_linux_pci_dev_base(struct pci_dev *dev)
{
  unsigned long base;

  GM_CALLED ();
  
  /* we do not know exactly what is in pdev->base, but it is supposed
     to be passed to ioremap without change, well except on some archs
     which were not doing the right thing until 2.4 (probably nobody
     accessed memory mapped devices on them before) */
  base = pci_resource_start (dev, 0);

  /* here are the specific fix for various arch/version combinations */
#if GM_CPU_powerpc && LINUX_XX <= 22
  /* ioremap expect the physical address.
     on prep architecture, phys addr = PCI_BUS_ADDR + _ISA_MEM_BASE,
     linux 2.2 prep leave a unusable pdev->base address
     I do not find any clean way to know if we are on a prep
     architecture from linux-2.2/include/asm-ppc/io.h,
     but checking is ISA_MEM_BASE is 0x80000000 should be OK and
     adding it if necessary */
  if (_ISA_MEM_BASE == 0x80000000 && base < 0x80000000)
    base += _ISA_MEM_BASE;
#elif GM_CPU_alpha && LINUX_XX <= 22
  /* on alpha 2.2 ioremap is identity and takes a kernel virtual
     address. Linux 2.2 does not automatically use the byte adressable
     space, so we fix it */
  if (strcmp (alpha_mv.vector_name,"LX164") == 0 ||
      strcmp (alpha_mv.vector_name,"Miata") == 0)
    {
      base = PYXIS_BW_MEM + (base & 0xffffffffUL);
    }
  else
    {
      base = dense_mem (base) + (base & 0xffffffffUL);
    }
#elif GM_CPU_powerpc && LINUX_XX == 24
  /* - typically on powermac, there is a linux bug related to PCI-PCI
     bridge handling in early 2.4 kernels, cause Linux to be confused 
     and not to initialize the base field of the device correctly,
  */
  if (base == 0)
    {
      unsigned int bus_base;
      struct pci_controller *hose = (struct pci_controller *)dev->sysdata;
      pci_read_config_dword(dev, PCI_BASE_ADDRESS_0, &bus_base);
      bus_base &= PCI_BASE_ADDRESS_MEM_MASK;
      GM_WARN (("Buggy Linux:pci_resource_start=0x%lx,bus_base=0x%x,"
		"hose=%p,hose->pci_mem_offset=%lx\n",
		pci_resource_start (dev,0), bus_base,
		hose, hose ? hose->pci_mem_offset : 0));
      if (bus_base && hose)
	base = bus_base + hose->pci_mem_offset;
    }
#elif GM_CPU_powerpc64 && LINUX_XX >= 24
  /* - on ppc64 with eeh on, the kernel hides the real address without
     providing any way to map its fake handle to use space. In
     both cases get the information from the PCI register who never lies. */
  if ((base >> 60UL) > 0)
    {
      /* we got either a fake (token), or a already mapped address */
      unsigned int bus_base;
      struct pci_controller *hose = ((struct device_node *)dev->sysdata)->phb;
      pci_read_config_dword(dev, PCI_BASE_ADDRESS_0, &bus_base);
      bus_base &= PCI_BASE_ADDRESS_MEM_MASK;
      GM_WARN (("Linux faking pci_resource_start:pci_resource_start=0x%lx,"
		"bus_base=0x%x,hose->pci_mem_offset=%lx\n",
		pci_resource_start (dev,0), bus_base,
		hose ? hose->pci_mem_offset : 0));
      if (bus_base && hose)
      	base = bus_base + hose->pci_mem_offset;
    }
#endif
  GM_RETURN (base);
}
#endif


/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
