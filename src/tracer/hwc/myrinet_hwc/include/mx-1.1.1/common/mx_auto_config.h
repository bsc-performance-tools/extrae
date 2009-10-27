/* common/mx_auto_config.h.in.  Generated from configure.ac by autoheader.  */

#ifndef MX_AUTO_CONFIG_H

#define MX_AUTO_CONFIG_H

#ifdef MX_AUTODETECT_H

#error mx_autodetect.h should not be before mx_auto_config.h

#endif

/* Define to 1 if you have the <asm/rmap.h> kernel header file */
#undef HAVE_ASM_RMAP_H

/* Define this on Linux if sysfs class_simple exists, earlier 2.6 kernels */
#undef HAVE_CLASS_SIMPLE

/* Define this on Linux if do_mmap_pgoff takes a mm arg first, 2.4 UML kernels
   */
#undef HAVE_DO_MMAP_PGOFF_7ARGS

/* Define this on Linux if do_munmap takes a acct arg, 2.4 or later RedHat
   kernels */
#undef HAVE_DO_MUNMAP_4ARGS

/* if iommu_table->it_mapsize exists */
#undef HAVE_IT_MAPSIZE

/* Define to 1 if you have the <linux/compile.h> header file */
#undef HAVE_LINUX_COMPILE_H

/* Define this on Linux if mmap_up_write/down_write exist (RedHat) */
#undef HAVE_MMAP_UP_WRITE

/* whether struct page has count or _count */
#undef HAVE_OLD_PAGE_COUNT

/* Define this on Linux if pte_kunmap exists (Suse 8.0 highpte support) */
#undef HAVE_PTE_KUNMAP

/* Define this on Linux if the function or macro pte_offset_map exists
   (highpte support) */
#undef HAVE_PTE_OFFSET_MAP

/* Define this on Linux if the function or macro pte_offset_map_nested exists
   (highpte support) */
#undef HAVE_PTE_OFFSET_MAP_NESTED

/* Define this on Linux if remap_page_range takes a vma arg first, 2.5 or
   later RedHat kernels */
#undef HAVE_REMAP_PAGE_RANGE_5ARGS

/* Define this on Linux if remap_pfn_range exists */
#undef HAVE_REMAP_PFN_RANGE

/* Is 10g support enabled */
#undef MX_10G_ENABLED

/* Is debugging enabled */
#undef MX_DEBUG

/* Is development code enabled */
#undef MX_DEVEL

/* Are SSE2 instructions permitted? */
#undef MX_ENABLE_SSE2

/* Is kernel library enabled */
#undef MX_KERNEL_LIB

/* Is logging enabled */
#undef MX_MCP_LOGGING

/* Enable One sided operations */
#undef MX_ONE_SIDED

/* Use the userspace-driver */
#undef MX_OS_UDRV

/* Is RDMA Win cache enabled */
#undef MX_RDMAWIN_CACHE

/* Is LXGDB support enabled */
#undef MX_UBA

/* Is Lanai emulated in unix process */
#undef MX_ULANAI

/* Define to the address where bug reports for this package should be sent. */
#undef PACKAGE_BUGREPORT

/* Define to the full name of this package. */
#undef PACKAGE_NAME

/* Define to the full name and version of this package. */
#undef PACKAGE_STRING

/* Define to the one symbol short name of this package. */
#undef PACKAGE_TARNAME

/* Define to the version of this package. */
#undef PACKAGE_VERSION

#include "mx_autodetect.h"

#endif
