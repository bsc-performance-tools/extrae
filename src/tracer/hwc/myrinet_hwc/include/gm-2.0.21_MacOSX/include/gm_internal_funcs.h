/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_internal_funcs_h_
#define _gm_internal_funcs_h_

#include "gm.h"

/****************
 * Internal functions that you should not assume you know how to use
 * unless you wrote them yourself and have reviews the implications of
 * using the function.
 ****************/

gm_u16_t __gm_ip_checksum (void *_message, unsigned long _len);
GM_ENTRY_POINT void __gm_stbar (void);

/****************
 * Simple internal functions
 ****************/

void _gm_bcopy_down (const void *from, void *to, gm_size_t len);
void _gm_bcopy_up (const void *from, void *to, gm_size_t len);
gm_status_t _gm_call_trace_init (void);
void _gm_call_trace_fini (void);
gm_status_t _gm_check_types (void);
void _gm_dma_finalize (struct gm_port *p);
void _gm_finalize_all (void);
long _gm_hash_compare_ups (void *key1, void *key2);
unsigned long _gm_hash_hash_up (void *key);
void _gm_hex_dump (const void *ptr, gm_size_t len);
char *_gm_ioctl_cmd_name (unsigned int cmd);
gm_u16_t _gm_ip_checksum (void *message, unsigned long len);
gm_status_t _gm_ip_checksum_verify (void *message, unsigned long len,
				    gm_u16_t cks);
void _gm_mark_set_iterate (struct gm_mark_set *set,
			   void (*callback) (void *context, gm_mark_t *mark),
			   void *context);
gm_status_t _gm_mmap (struct gm_port *p, gm_offset_t offset,
		      gm_size_t len, int flags, void **result);
gm_status_t _gm_munmap (struct gm_port *p, void *ptr, unsigned long len);
struct gm_port *_gm_open_ports_remove (struct gm_port *my_gm_port);
void _gm_open_ports_close_all (void);
void _gm_open_ports_insert (struct gm_port *gm_my_port);
int _gm_open_ports_test(void);
void _gm_overlapping_bcopy (void *from, void *to, gm_size_t len);
void _gm_perform_on_exit_callbacks (gm_status_t);
const char *_gm_recv_event_name (enum gm_recv_event_type t);
void _gm_register_memory_mutex_enter (void);
void _gm_register_memory_mutex_exit (void);
gm_status_t gm_register_recv_queue (struct gm_port *port, gm_up_t user_vma);
void _gm_request_sleep (struct gm_port *p);
GM_ENTRY_POINT void _gm_sent (struct gm_port *p, void *context, gm_status_t status);
gm_status_t _gm_sleep (struct gm_port *p);
gm_status_t _gm_unknown (struct gm_port *gm_my_port, union gm_recv_event *gm_e);
gm_status_t _gm_update_global_id_caches (struct gm_port *port);

void gm_crc32_test (void);
int gm_fork(void);
char *gm_get_buf_status_name (enum gm_buf_status s);
void gm_safer_bzero (void *ptr, int len);
gm_status_t __gm_init_local_mutex (void);
void __gm_finalize_local_mutex (void);
char *gm_strcpy (char *dest, const char *src);

#endif /* _gm_internal_funcs_h_ */

/*
  This file uses GM standard indentation.

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
