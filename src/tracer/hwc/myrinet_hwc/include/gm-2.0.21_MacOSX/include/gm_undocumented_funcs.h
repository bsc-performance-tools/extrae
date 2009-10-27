/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file defines the undocumented API calls exported by libgm. */

#ifndef _gm_undocumented_funcs_h_
#define _gm_undocumented_funcs_h_

#include "gm.h"
#include "gm_simple_types.h"
#include "gm_mapper_state.h"

struct gm_mapping_specs;
struct gm_myrinet_eeprom;
struct gm_port;

/****************
 * unsupported exported functions
 ****************/

GM_ENTRY_POINT gm_status_t _gm_clear_all_routes (struct gm_port *p);
GM_ENTRY_POINT gm_status_t _gm_clear_counters (struct gm_port *p);
GM_ENTRY_POINT gm_status_t _gm_cop_wakeup (struct gm_port *);
GM_ENTRY_POINT gm_status_t _gm_cop_end (struct gm_port *);
GM_ENTRY_POINT gm_status_t _gm_cop_send (struct gm_port *, int d);
GM_ENTRY_POINT gm_status_t _gm_cop_receive (struct gm_port *, int *d);
GM_ENTRY_POINT gm_status_t _gm_user_write_lanai_register
(struct gm_port *, gm_u32_t offset, gm_u32_t value);
GM_ENTRY_POINT void	   _gm_dump_receive_queue (struct gm_port *);
GM_ENTRY_POINT gm_status_t _gm_enable_raw_receives (struct gm_port *p);

#if GM_BUILDING_INTERNALS /* Allow only if gm_up_t layouts are compatible. */
GM_ENTRY_POINT gm_status_t _gm_finish_mmap (struct gm_port *,
					    gm_offset_t off,
					    gm_size_t len, gm_up_t uvma);
#endif
GM_ENTRY_POINT gm_status_t _gm_get_board_config_strings (struct gm_port *port,
							 char (*buf)[256]);
GM_ENTRY_POINT gm_status_t _gm_get_eeprom (struct gm_port *p,
					   struct gm_myrinet_eeprom *eeprom);
GM_ENTRY_POINT gm_status_t _gm_get_firmware_string (struct gm_port *,
						    gm_lp_t lanai_ptr,
						    char *str,
						    gm_size_t *len);
#if GM_BUILDING_INTERNALS /* Allow only if struct layouts are compatible. */
GM_ENTRY_POINT gm_status_t _gm_get_globals (struct gm_port *,
					    gm_u8_t *ptr,
					    gm_size_t length);
GM_ENTRY_POINT gm_status_t _gm_get_globals_by_request (struct gm_port *,
						       gm_u8_t *ptr,
						       unsigned int off,
						       gm_size_t length);
#endif
GM_ENTRY_POINT gm_status_t _gm_get_page_hash_cache_size (struct gm_port
							 *,
							 unsigned int
							 *cache_size);
GM_ENTRY_POINT const char *_gm_get_kernel_build_id (struct gm_port *p);
GM_ENTRY_POINT gm_status_t _gm_get_mapping_specs (struct gm_port *p,
						  struct gm_mapping_specs *ms);
GM_ENTRY_POINT gm_status_t _gm_get_opener_pids (struct gm_port *p,
						gm_pid_t *pids);
GM_ENTRY_POINT const char *_gm_get_build_id (void);
GM_ENTRY_POINT gm_status_t _gm_get_page_len (unsigned long *result);
GM_ENTRY_POINT gm_status_t _gm_get_mapper_state (struct gm_port *the_port,
						 struct gm_mapper_state *ms);
GM_ENTRY_POINT gm_status_t _gm_get_node_type (struct gm_port *p,
					      const gm_u8_t unique_id[6],
					      gm_u32_t *node_type);
GM_ENTRY_POINT const char *_gm_get_version (void);
GM_ENTRY_POINT gm_status_t _gm_handle_alarm (struct gm_port *);
GM_ENTRY_POINT gm_status_t _gm_handle_flushed_alarm (struct gm_port *p);
GM_ENTRY_POINT gm_status_t _gm_mapper_open (struct gm_port **p,
					    unsigned int unit,
					    enum gm_api_version gm_version);
GM_ENTRY_POINT unsigned int _gm_max_used_node_id (struct gm_port *);
GM_ENTRY_POINT void        _gm_nt_foo (void);
GM_ENTRY_POINT void        _gm_preserve_lanai_globals(struct gm_port * p);
GM_ENTRY_POINT void        _gm_provide_raw_receive_buffer (struct gm_port *p,
							   void *ptr);
GM_ENTRY_POINT void        _gm_raw_send (struct gm_port *p, void *ptr,
					 unsigned int len,
					 unsigned int route_len);
GM_ENTRY_POINT void        _gm_raw_send_with_callback
(struct gm_port *p, void *ptr, unsigned int len, unsigned int route_len,
 gm_send_completion_callback_t cb, void *context);
GM_ENTRY_POINT gm_status_t _gm_set_mapper_level (struct gm_port *p, int level);
GM_ENTRY_POINT gm_status_t _gm_set_mapper_state (struct gm_port *the_port,
						 const struct gm_mapper_state
						 *ms);
GM_ENTRY_POINT gm_status_t _gm_set_host_name (struct gm_port *p,
					      char (*host_name)
					      [GM_MAX_HOST_NAME_LEN+1]);
GM_ENTRY_POINT gm_status_t _gm_set_node_id (struct gm_port *p,
					    unsigned int node_id);
GM_ENTRY_POINT gm_status_t _gm_set_opener_pid (struct gm_port *p,
					       gm_pid_t pid);
GM_ENTRY_POINT gm_status_t _gm_set_physical_pages (struct gm_port *);
GM_ENTRY_POINT gm_status_t _gm_set_port_num (struct gm_port *p,
					     unsigned long _id);
GM_ENTRY_POINT gm_status_t _gm_set_route (struct gm_port *p,
					  const gm_u8_t unique_id[6],
					  unsigned int len, char *route);
GM_ENTRY_POINT gm_status_t _gm_set_node_type (struct gm_port *p,
					      const gm_u8_t unique_id[6],
					      gm_u32_t node_type);
GM_ENTRY_POINT gm_status_t _gm_set_unique_id (struct gm_port *p,
					      unsigned int node_id,
					      char id[6]);
GM_ENTRY_POINT void        _gm_unknown_debug_buffers (struct gm_port * p,
						      gm_recv_event_t * e);
GM_ENTRY_POINT gm_status_t _gm_user_ioctl (struct gm_port *p, unsigned int cmd,
					   void *buf, gm_size_t bufsize);

#endif /* _gm_undocumented_funcs_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
