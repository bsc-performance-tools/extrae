/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2000 by Myricom, Inc.	      				 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#include "gm_internal.h"
#include "gm_call_trace.h"

#include <linux/mm.h>
#include <linux/file.h>

#define GM_LINUX_VM_METHOD(vma,method) do {	\
  if ((vma)->vm_ops && (vma)->vm_ops->method)	\
    {						\
      (vma)->vm_ops->method(vma);		\
    } } while (0)

 
/* gm_linux_wrap: allows to "overload" the vm_ops methods of a region
   to get notified when it is "opened" or "closed", because there it
   no refcounting, the overloading open procedure should ensure a
   child region points to the original methods. That way at all time,
   one gm_linux_vm_ops_t is associated with exactly one region */
static gm_linux_vm_ops_t * gm_linux_wrap (struct vm_area_struct *vma, 
					  void (*open)(struct vm_area_struct *),
					  void (*close)(struct vm_area_struct *))
{
  gm_linux_vm_ops_t *gm_ops = gm_malloc (sizeof(*gm_ops));
  if (!gm_ops)
    {
      /* FIXME: we should try to disable the port here */
      GM_WARN (("ERROR:Out of memory when forking in gm app!!\n"));
      _GM_WARN (("GM Port will be confused!!\n"));
    }
  gm_bzero (gm_ops,sizeof(*gm_ops));
  
  gm_ops->gm.ori = vma->vm_ops;
  if (vma->vm_ops)
      gm_ops->wrap = *vma->vm_ops;
  
  gm_ops->wrap.open = open;
  gm_ops->wrap.close = close;
  gm_ops->gm.magic = 0x72666d67;
  gm_ops->gm.vma = vma;
  vma->vm_ops = &gm_ops->wrap;
  gm_linux_module_get(THIS_MODULE);
  return gm_ops;
}

/* gm_linux_dewrap: restores a region to its original set of methods,
   remove the overload of open/close, and free the associated
   memory */
static void gm_linux_dewrap (struct vm_area_struct *vma)
{
  gm_linux_vm_ops_t *gm_ops = (gm_linux_vm_ops_t *)vma->vm_ops;

  gm_always_assert (gm_ops->gm.magic == 0x72666d67); /* gmfrk */
  vma->vm_ops = gm_ops->gm.ori;
  gm_bzero (gm_ops,sizeof(*gm_ops));
  gm_free (gm_ops);
  gm_linux_module_put(THIS_MODULE);
  
}


static void
gm_linux_fork_open (struct vm_area_struct *vma)
{
  gm_linux_vm_ops_t *gm_ops = (gm_linux_vm_ops_t *)vma->vm_ops;

  /* ensure the wrapping never extend to the newly created vma */
  vma->vm_ops = gm_ops->gm.ori;
  GM_LINUX_VM_METHOD (vma,open);
  
  gm_always_assert (gm_ops->gm.vma->vm_start == vma->vm_start);
  gm_always_assert (vma->vm_mm != current->mm);
  if (gm_ops->gm.vma_to_dewrap)
    {
      gm_linux_dewrap (gm_ops->gm.vma_to_dewrap);
    }
  gm_linux_dewrap (gm_ops->gm.vma);

  gm_linux_fork (vma);
}



/* dead code: the dewrap of any vma (in parent) is done before going
   out of fork(), so before closing any parent vma */
void
gm_linux_fork_close (struct vm_area_struct *vma)
{
  gm_linux_dewrap (vma);
  GM_WARN(("Internal Error in the fork code(): GM port might be confused\n"));
  GM_LINUX_VM_METHOD (vma,close);
}

/* dead code: the first vma in child is dewrapped before
   going out of fork() (so before any possibility of duplication) */
static void
gm_linux_ref_vma_open (struct vm_area_struct *vma)
{
  gm_linux_vm_ops_t *gm_ops = (gm_linux_vm_ops_t *)vma->vm_ops;
  vma->vm_ops = gm_ops->gm.ori;
  GM_LINUX_VM_METHOD (vma,open);

  gm_linux_dewrap (gm_ops->gm.vma);
  GM_WARN (("Internal Error in the fork code(): GM port might be confused\n"));
}

static void
gm_linux_ref_vma_close (struct vm_area_struct *vma)
{
  /* fork failed before reaching last vma */
  gm_linux_dewrap (vma);
  GM_LINUX_VM_METHOD (vma,close);
  
  GM_WARN(("fork() has failed for pid %d:OK\n", current->pid));
  /* dewrap any vma that wasn't reach before failure */
  for (vma = current->mm->mmap; vma; vma = vma->vm_next)
    {
      if (vma->vm_ops && vma->vm_ops->open == gm_linux_fork_open)
	{
	  gm_linux_dewrap (vma);
	}
    }
}

void
gm_linux_fork_wrap(struct vm_area_struct *vma)
{
  struct vm_area_struct *tmp;
  gm_linux_vm_ops_t *last = 0;
  
  for (tmp = current->mm->mmap; tmp; tmp = tmp->vm_next)
    {
      if (!(tmp->vm_flags & (VM_DONTCOPY | VM_SHARED | VM_IO | VM_RESERVED))
	  && tmp->vm_start > vma->vm_start)
	{
	  if (tmp->vm_ops && tmp->vm_ops->open == gm_linux_fork_open)
	    {
	      /* it looks like we have already wrapped the last vma !! */
	      GM_INFO (("Last vma already wrapped: assuming several gm ports "
			" opened in same process:OK\n"));
	      return;
	    }
	  last = gm_linux_wrap (tmp, gm_linux_fork_open, gm_linux_fork_close);
	  if (!last)
	    {
	      return;
	    }
	}
    }
  if (!last)
    {
      GM_WARN (("GM+fork support:no vma after send queue: please report!!!\n"));
      return;
    }
  /* call gm_linux_fork_open on regions forked() before the
     send_queue. There should be none, if send_queue mapping is first
     as it should be */
  for (tmp = vma->vm_mm->mmap; tmp; tmp = tmp->vm_next)
    {
      if (!(tmp->vm_flags & (VM_DONTCOPY | VM_SHARED | VM_IO | VM_RESERVED))
	  && tmp->vm_start < vma->vm_start)
	{
	  static int info_done;
	  if (!info_done)
	    {
	      GM_INFO (("GM+fork support: send_queue is not first in address space: OK\n"));
	      info_done = 1;
	    }
	  gm_linux_fork (tmp);
	}
    }
  if (last && gm_linux_wrap (vma, gm_linux_ref_vma_open, gm_linux_ref_vma_close))
    {
      last->gm.vma_to_dewrap = vma;
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
