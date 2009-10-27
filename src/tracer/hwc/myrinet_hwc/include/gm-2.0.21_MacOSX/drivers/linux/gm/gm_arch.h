
/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef _gm_arch_h_
#define _gm_arch_h_

/**********************************************************************
 * This file architecture-specific marcros and inline functions
 * required by gm.c and gm_arch.c. See gm_arch_types.h for typedefs.
 **********************************************************************/

/************************************************************************
 * Items in this section are not required by gm.c, but needed by the
 * implementations of the functions required by gm.c.
 ************************************************************************/

#include "gm_arch_def.h"
#include "gm_arch_types.h"
#include "gm_compiler.h"
#include "gm_debug.h"
#include "gm_impl.h"
#include "gm_internal.h"
#include "gm_lanai.h"
#include "gm_arch_linux.h"

extern void *gm_instancep;
extern void *gm_minorp;

extern unsigned gm_control_program_4k_length;
extern unsigned gm_control_program_8k_length;
extern char gm_control_program_4k[];
extern char gm_control_program_8k[];

#define GM_LINUX_PRINTF_FUNC __gm_gcc_attribute__ ((format (printf, 1, 2)))

void gm_linux_info (const char *format, ...) GM_LINUX_PRINTF_FUNC;
void gm_linux_note (const char *format, ...) GM_LINUX_PRINTF_FUNC;
void gm_linux_panic (const char *format, ...) GM_LINUX_PRINTF_FUNC;
void gm_linux_print (const char *format, ...) GM_LINUX_PRINTF_FUNC;
void gm_linux_printk (const char *format, ...) GM_LINUX_PRINTF_FUNC;
void gm_linux_warn (const char *format, ...) GM_LINUX_PRINTF_FUNC;

#if GM_DEBUG
extern int gm_linux_print_level;
#include "gm_debug.h"

#undef GM_PRINT_LEVEL
#define GM_PRINT_LEVEL gm_linux_print_level
#endif

extern struct semaphore gm_linux_open_mutex;


static inline int
gm_linux_phys_on_board (gm_instance_state_t * is, gm_phys_t phys)
{
#if CSPI_MAP26xx
  if ((phys >= 0x40000000 && phys < 0x40800000) ||	/* sram */
      (phys >= 0x48000000 && phys < 0x48004000) ||	/* specials */
      (phys >= 0xf0000000 && phys < 0xf0004000) ||	/* control */
      (phys >= 0xffffff00 && phys <= 0xffffffff))	/* eeprom */
    {
      return 1;
    }
  else
    {
      return 0;
    }
#else /* !CSPI_MAP26xx */
  unsigned long min_addr, max_addr;

  min_addr = (gm_phys_t) is->arch.phys_base_addr;
  max_addr = (gm_phys_t) is->arch.phys_base_addr + is->board_span;

  if ((phys >= min_addr) && (phys < max_addr))
    {
      return 1;
    }
  else
    {
      return 0;
    }
#endif /* !CSPI_MAP26xx */
}

static inline int
gm_linux_phys_is_valid_page (gm_instance_state_t * is, gm_phys_t phys)
{
  unsigned long pfn = gm_linux_pfn (phys);
#if LINUX_XX >= 26
  return pfn_valid(pfn);
#elif LINUX_XX == 24 && defined CONFIG_X86_PAE
  return pfn >= GM_LINUX_PFN_ZERO && pfn < GM_LINUX_PFN_MAX;
#else
 {
   void * kaddr = phys_to_virt (phys);
   struct page * page = virt_to_page (kaddr);
   return VALID_PAGE (page);
 }
#endif
}

static inline struct page *
GM_LINUX_PHYS_TO_PAGE (gm_instance_state_t * is, gm_phys_t phys)
{
  gm_assert (gm_linux_phys_is_valid_page (is, phys));
#if LINUX_XX >= 26
  return pfn_to_page (gm_linux_pfn (phys));
#elif LINUX_XX >= 24 && !defined CONFIG_X86_PAE
  {
    void * kaddr = phys_to_virt(phys);
    return virt_to_page (kaddr);
  }
#else
  return mem_map + gm_linux_pfn (phys) - GM_LINUX_PFN_ZERO;
#endif
}

void gm_linux_fork(struct vm_area_struct *vma);
void gm_linux_fork_wrap(struct vm_area_struct *vma);
gm_phys_t gm_linux_kvirt_to_phys (gm_instance_state_t * is,
				  unsigned long addr, int kernel);

unsigned long gm_linux_mem_nbpages (void);

void gm_linux_port_close (gm_port_state_t * ps);



#endif /* _gm_arch_h_ */
