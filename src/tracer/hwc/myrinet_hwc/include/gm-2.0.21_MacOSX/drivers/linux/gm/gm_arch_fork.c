/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.	      				 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#include "gm_internal.h"
#include "gm_arch_def.h"
#include "gm_impl.h"

#include <asm/page.h>
#include <linux/mm.h>
#include <asm/pgtable.h>
#include <linux/pagemap.h>
#include <asm/pgalloc.h>
#include <linux/highmem.h>

void gm_linux_make_writable(pte_t *ptep)
{
#if GM_CPU_powerpc || (GM_CPU_powerpc64 && LINUX_XX <= 24)
  pte_update(ptep, 0, _PAGE_RW);
#elif GM_CPU_powerpc64 && LINUX_XX >= 26 /* powerpc64 >= 2.6 */
  /* set_pte not available on ppc64 2.6 */
  unsigned long oldval;
  do {
    oldval = pte_val (*(volatile pte_t *)ptep);
  } while ((oldval & _PAGE_BUSY)
	   || cmpxchg(ptep, oldval, oldval | _PAGE_RW) != oldval);
#else
  set_pte (ptep, pte_mkwrite (*ptep));
#endif
}

extern gm_arch_sync_t gm_linux_pfn_sync;

static int
gm_linux_cow (struct vm_area_struct *vma, pmd_t * src_pmd, pmd_t * dst_pmd,
             unsigned long addr_base)
{
  void *new, *old;
  struct page *new_page;
  pte_t *src_pte, *dst_pte;
  struct page *old_page;
  unsigned long pfn;

  GM_VAR_MAY_BE_UNUSED (old);
  GM_VAR_MAY_BE_UNUSED (new);
  /* we need to get the mutex before the pte_offset_map related spinlocks */
  gm_arch_mutex_enter (&gm_linux_pfn_sync);
  src_pte = pte_offset_map (src_pmd, addr_base);
  dst_pte = pte_offset_map_nested (dst_pmd, addr_base);
  if (!pte_present(*src_pte))
    {
      gm_arch_mutex_exit (&gm_linux_pfn_sync);
      goto cow_end;
    }
  pfn = gm_linux_pfn (GM_LINUX_PHYS_FROM_PTE (*src_pte));
  if (!gm_hash_find (gm_linux_pfn_hash, &pfn))
    {
      gm_arch_mutex_exit (&gm_linux_pfn_sync);
      goto cow_end;
    }
  gm_arch_mutex_exit (&gm_linux_pfn_sync);
  old_page = pte_page (*src_pte);
  gm_always_assert (pte_present (*src_pte) && pte_present (*dst_pte) &&
		    !pte_write (*src_pte) && !pte_write (*dst_pte) &&
		    pte_page (*dst_pte) == old_page);
  /* page count should be: 1 in parent, 1 in child, 1 in page_hash */
  gm_always_assert (gm_linux_page_count (old_page) >= 2);
  if (gm_linux_page_count (old_page) < 3)
    {
      GM_PRINT (GM_PRINT_LEVEL >= 2,
		("after_fork:old_page->count(%d) != 3 \n",
		 gm_linux_page_count (old_page)));
      goto cow_end;
    }
  /* we need to release the pte mapping before forcing the cow that
     can sleep */
  pte_unmap_nested (dst_pte);
  pte_unmap (src_pte);
  if (!gm_linux_get_user_pages)
    {
#if LINUX_XX >= 26 || defined HAVE_ASM_RMAP_H
      GM_WARN (("fork() not supported on this kernel when GM ports are open"));
      return -1;
#else
      new_page = alloc_page (GFP_HIGHUSER);
      if (!new_page)
	{
	  GM_WARN (("copy-on-write error: not enough memory!\n"));
	  return -1;
	}
      /* copy new_page */
      new = kmap (new_page);
      old = kmap (old_page);
      memcpy(new,old,PAGE_SIZE);
      kunmap (old_page);
      kunmap (new_page);
      flush_cache_page (vma, addr_base);
      src_pte = pte_offset_map (src_pmd, addr_base);
      dst_pte = pte_offset_map_nested (dst_pmd, addr_base);
#if GM_CPU_powerpc
      *dst_pte = pte_mkwrite (pte_mkdirty (mk_pte (new_page, vma->vm_page_prot)));
#else
      set_pte (dst_pte,
	       pte_mkwrite (pte_mkdirty (mk_pte (new_page, vma->vm_page_prot))));
#endif
      page_cache_release (old_page);	/* remove child reference */
      /* put again original page in rw mode */
      gm_linux_make_writable (src_pte);
      /* we do not need flush_tlb_{page/mm} as we are in fork, we only modified the
	 child tlb which has never been active and there will be a
	 flush_tlb_page before that happens, we are in dup_mmap in the
	 parent and there is a flush_tlb_mm at the end of this
	 function */
#endif /* HAVE_ASM_RMAP_H */
    }
  else /* fonction get_user_pages available */
    {
      int res;

#ifdef CONFIG_STACK_GROWSUP
      GM_WARN (("fork() not supported on this kernel when GM ports are open"));
      return -1;
#endif
      /* We need to find another way to do this, here is how it works now:
         the mm_struct of the child process is not completely setup yet,
         and get_user_pages (which does the cow) relies on find_vma, 
         find_vma will work with known kernel versions if we put the vma in the cache.
         (exception if CONFIG_STACK_GROWSUP is defined). */
      vma->vm_mm->mmap_cache = vma;
      res = (*gm_linux_get_user_pages) (current,vma->vm_mm, addr_base, 1, 
					1, 0, &new_page, NULL);
      if (res != 1)
	{
	  GM_WARN (("copy-on-write error(%d): not enough memory?\n",res));
	  return -1;
	}
      gm_always_assert (new_page != old_page);
      src_pte = pte_offset_map (src_pmd, addr_base);
      dst_pte = pte_offset_map_nested (dst_pmd, addr_base);
      gm_always_assert (pte_present (*src_pte) && pte_present (*dst_pte) &&
			!pte_write (*src_pte) && pte_write (*dst_pte) &&
			pte_page (*dst_pte) == new_page &&
			pte_page (*src_pte) == old_page);
      /* page count should be: 1 in parent, 1 in child, 1 in page_hash */
      gm_always_assert (gm_linux_page_count (old_page) >= 2);
      if ((LINUX_XX <= 24 && gm_linux_page_count (new_page) != 2) ||
	  (LINUX_XX >= 26 && gm_linux_page_count (new_page) < 2))
	{
	  GM_WARN(("warning,after fork:new_page->count=%d\n",
		   gm_linux_page_count (new_page)));
	}
      page_cache_release (new_page);
      gm_linux_make_writable (src_pte);
    }
 cow_end:
  pte_unmap_nested (dst_pte);
  pte_unmap (src_pte);
  return 0;
}

void
gm_linux_fork (struct vm_area_struct *vma)
{
  unsigned long addr;
  pgd_t *src_pgd, *dst_pgd;
#ifdef PUD_SHIFT
  pud_t *src_pud, *dst_pud;
#else
  pgd_t *src_pud, *dst_pud;
#endif
  pmd_t *src_pmd, *dst_pmd;

  gm_assert (vma->vm_mm != current->mm);
  /* OK we must be probably in a fork call */

  for (addr = vma->vm_start; addr < vma->vm_end; addr += PAGE_SIZE) {
    src_pgd = pgd_offset (current->mm, addr);
    dst_pgd = pgd_offset (vma->vm_mm, addr);
    /* check  valid pgd entries */
    if (pgd_none (*src_pgd) || pgd_bad (*src_pgd) ||
	pgd_none (*dst_pgd) || pgd_bad (*dst_pgd))
      {
	continue;
      }
#ifdef PUD_SHIFT
    src_pud = pud_offset (src_pgd, addr);
    dst_pud = pud_offset (dst_pgd, addr);
    /* check  valid pud entries */
    if (pud_none (*src_pud) || pud_bad (*src_pud) ||
	pud_none (*dst_pud) || pud_bad (*dst_pud))
      {
	continue;
      }
#else
    src_pud = src_pgd;
    dst_pud = dst_pgd;
#endif

    src_pmd = pmd_offset (src_pud, addr);
    dst_pmd = pmd_offset (dst_pud, addr);

    /* check valid pmd entries */
    if (pmd_none (*src_pmd) || pmd_bad (*src_pmd) ||
	pmd_none (*dst_pmd) || pmd_bad (*dst_pmd))
      {
	continue;
      }
    gm_linux_cow (vma, src_pmd, dst_pmd, addr);
  }
}



/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  c-backslash-column:72
  End:
*/
