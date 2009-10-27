#include <asm/atomic.h>

/* Powerpc64 returns a -1 for error when using pci_map_single (0 might
   be a valid dma address) Every other architecture either uses 0 or
   never fails
*/

#if GM_CPU_powerpc64
#define INVALID_DMA_ADDR ((dma_addr_t)-1)
#else
#define INVALID_DMA_ADDR ((dma_addr_t)0)
#endif

#if GM_CPU_alpha

#define GM_ALPHA_PCI_IOMMU 1
#define GM_ALPHA_MAX_IOMMU_MAPS (256*1024*1024/PAGE_SIZE)

#if LINUX_VERSION_CODE < VERSION_CODE(2,4,13)

#include <asm/machvec.h>
#include <asm/hwrpb.h>
#include <asm/core_tsunami.h>
#include <asm/core_cia.h>
#ifdef ST_DEC_MARVEL
#include <asm/core_marvel.h>
#endif /* ST_DEC_MARVEL */

static  gm_u64_t alpha_dac_offset = (1UL << 40);

static inline void
gm_linux_set_csr_bit (volatile gm_u64_t * reg, int mask)
{
  *reg |= mask;
  mb ();
  *reg;
}

static inline int
gm_linux_mwin_enabled (void)
{
  static int done, answer;
  /* compute the result only once */
  if (done)
    return answer;

  if (hwrpb->sys_type == ST_DEC_TSUNAMI)
    {
      if (!(TSUNAMI_pchip0->pctl.csr & pctl_m_mwin) ||
	  !(TSUNAMI_pchip1->pctl.csr & pctl_m_mwin))
	{
	  GM_WARN (
		   ("gm_linux_mwin_enabled: Turning monster window on (was not set)\n"));
	  gm_linux_set_csr_bit (&TSUNAMI_pchip0->pctl.csr, pctl_m_mwin);
	  gm_linux_set_csr_bit (&TSUNAMI_pchip0->pctl.csr, pctl_m_mwin);
	  gm_always_assert (TSUNAMI_pchip0->pctl.csr & pctl_m_mwin);
	  gm_always_assert (TSUNAMI_pchip1->pctl.csr & pctl_m_mwin);
	}
      answer = 1;
    }
  else if (hwrpb->sys_type == ST_DEC_EB164 || hwrpb->sys_type == ST_DEC_MIATA)
    {
      volatile gm_u64_t *csr = (volatile gm_u64_t *) CIA_IOC_CIA_CNFG;
      GM_WARN (
	       ("gm_linux_mwin_enabled: pyxis chipset with more than 2Gb RAM, wow!!\n"));
      if (!(*csr & CIA_CNFG_PCI_MWEN))
	{
	  GM_WARN (
		   ("gm_linux_mwin_enabled: Turning monster window on (was not set)\n"));
	  gm_linux_set_csr_bit (csr, CIA_CNFG_PCI_MWEN);
	}
      gm_always_assert (*csr & CIA_CNFG_PCI_MWEN);
      answer = 1;
    }
  else
#ifdef ST_DEC_MARVEL
    /* marvel's monster windows are always enabled, but the DAC offset
       is different than normal */
    if (hwrpb->sys_type == ST_DEC_MARVEL)
      {
	alpha_dac_offset = IO7_DAC_OFFSET;
	answer = 1;
      }
  else
#endif
    {
      GM_WARN (
	       ("gm_linux_mwin_enabled: do not know how to check for Monster Window on %s (%d,%d)\n",
		alpha_mv.vector_name, hwrpb->sys_type, hwrpb->sys_variation));
      answer = 0;
    }
  if (!answer)
    GM_WARN (
	     ("gm_linux_mwin_enabled: Cannot use PCIA64, please contact help@myri.com\n"));
  GM_PRINT (GM_PRINT_LEVEL >= 1,
	    ("gm_linux_mwin_enabled:Monster Window %s\n",
	     answer ? "ON" : "OFF"));
  done = 1;
  return answer;
}

#endif /* < 2413: gm_linux_mwin_enabled() */
#endif /* CPU_Alpha */

static inline gm_status_t
gm_linux_pci_map (gm_instance_state_t * is, gm_phys_t phys,
		  gm_dp_t *mapped_addr)
{
  gm_dp_t bus;

  gm_assert (gm_linux_phys_is_valid_page (is, phys));
  gm_assert ((phys & (PAGE_SIZE -1)) == 0);
#if GM_CPU_x86  || GM_CPU_ia64 || GM_CPU_x86_64
  /* x86 & ia64 phys to bus mapping is always identity.  No need to check
     GM_INSTANCE_64BIT_DMA_ADDR_OK, since configurations without
     this flag are unsupported */
  bus = phys;
#elif GM_CPU_powerpc
  bus = phys + (gm_phys_t) PCI_DRAM_OFFSET;
#elif LINUX_XX <= 22 
  bus = virt_to_bus (__va (phys));
#elif GM_CPU_alpha
  /* try direct mapping, then try monster window if we have 64bit 
     Myrinet board, then try allocating an IOMMU entry */
  bus = virt_to_bus (__va (phys));
  /* refuse to use the last Mbyte of memory, there are some buggy
   kernels in the wild who do not pay attention to the commonly used
   cypress chip */
  if (!bus || bus >= 0xfff00000)
    {
#if LINUX_VERSION_CODE >= VERSION_CODE(2,4,13)
      if (pci_dac_dma_supported (is->arch.pci_dev, 
				 is->arch.pci_dev->dma_mask))
       {
         struct page *page = GM_LINUX_PHYS_TO_PAGE (is, phys);
         bus = pci_dac_page_to_dma (is->arch.pci_dev, page, 0,
                                    PCI_DMA_BIDIRECTIONAL);
       }
#else
      if (gm_linux_mwin_enabled ())
	{
	  gm_always_assert (phys <= alpha_dac_offset);
	  bus = phys | alpha_dac_offset;
	}
#endif
      else if (GM_ALPHA_PCI_IOMMU
	       && atomic_read (&is->arch.free_iommu_pages) > 0)
	{
	  bus =
	    pci_map_single (is->arch.pci_dev, __va (phys), PAGE_SIZE,
			    PCI_DMA_BIDIRECTIONAL);
	  if (bus != INVALID_DMA_ADDR)
	    {
	      atomic_dec (&is->arch.free_iommu_pages);
	    }
	}
      else
	{
	  return GM_FAILURE;
	}
    }
#elif GM_CPU_powerpc64
  if ( atomic_read (&is->arch.free_iommu_pages) > 0)
    {
      bus = pci_map_single (is->arch.pci_dev, __va (phys), PAGE_SIZE,
 			    PCI_DMA_BIDIRECTIONAL);
      if (bus != INVALID_DMA_ADDR)
 	{
 	  atomic_dec (&is->arch.free_iommu_pages);
	}
    }
  else
    {
      return GM_FAILURE;
    }
#elif GM_CPU_sparc64
#error we need more complex support here
  bus =
    pci_map_single (is->arch.pci_dev, __va (phys), PAGE_SIZE,
		    PCI_DMA_BIDIRECTIONAL);
#else
#error Wrong architecture?
#endif
 
  if (bus == INVALID_DMA_ADDR)
      {
	return GM_FAILURE;
      }
  *mapped_addr = bus;
  return GM_SUCCESS;
}

static inline void
gm_linux_pci_unmap (gm_instance_state_t * is, gm_dp_t bus)
{

  gm_always_assert (bus != INVALID_DMA_ADDR);

#if LINUX_22 || GM_CPU_x86 || GM_CPU_powerpc || GM_CPU_ia64 || GM_CPU_x86_64
  /* nothing */ ;
#elif GM_CPU_alpha
  if (bus > 0xffffffffUL ||
      (bus >= __direct_map_base
       && bus < __direct_map_base + __direct_map_size))
    /* nothing */ ;
  else if (GM_ALPHA_PCI_IOMMU)
    {
      atomic_inc (&is->arch.free_iommu_pages);
      pci_unmap_single (is->arch.pci_dev, (dma_addr_t) bus, PAGE_SIZE,
			PCI_DMA_BIDIRECTIONAL);
    }
  else
    GM_PANIC (("gm_linux_pci_unmap: bad address\n"));
#elif GM_CPU_powerpc64
  atomic_inc (&is->arch.free_iommu_pages);
  pci_unmap_single (is->arch.pci_dev, (dma_addr_t) bus, PAGE_SIZE,
 		    PCI_DMA_BIDIRECTIONAL);

#elif GM_CPU_sparc64
  pci_unmap_single (is->arch.pci_dev, (dma_addr_t) bus, PAGE_SIZE,
		    PCI_DMA_BIDIRECTIONAL);
#else
#error Wrong architecture?
#endif
}




