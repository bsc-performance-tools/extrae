/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_board_mapping_h_
#define _gm_board_mapping_h_

#include "gm_simple_types.h"

GM_ENTRY_POINT gm_status_t gm_contiguous_dma_free (struct gm_port *port);
GM_ENTRY_POINT gm_status_t gm_contiguous_dma_malloc (struct gm_port *port,
						     gm_size_t len,
						     gm_dp_t *ret);
GM_ENTRY_POINT gm_status_t gm_map_board (struct gm_port *port,
					 void **control_regs,
					 gm_size_t *control_regs_mapped_len,
					 void **special_regs,
					 gm_size_t *special_regs_mapped_len,
					 void **sram,
					 gm_size_t *sram_mapped_len);
GM_ENTRY_POINT void gm_unmap_board (struct gm_port *port,
				    void *control_regs,
				    void *special_regs,
				    void *sram);
     
#endif /* _gm_board_mapping_h_ */
