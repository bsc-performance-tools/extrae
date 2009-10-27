/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/****************************************************************
 * prologue
 ****************************************************************/

#ifndef _gm_internal_h_
#define _gm_internal_h_

#ifdef	__cplusplus
extern "C"
{
#if 0
}				/* indent hack */
#endif
#endif

/****************************************************************
 * Headers
 ****************************************************************/

/* warn if the compiler is no good */
#include "gm_compiler.h"

/************
 * Myricom debugging HACKS
 ************/

/* gm_config.h set the configuration options for this architecture.
   This must be included before gm.h so it can override the
   auto-detect definitions there when this file is included by the
   MCP. */

#include "gm_config.h"

/* defines only the GM types used in the GM API. */

#include "gm.h"

/* gm_types.h defines gm_dp_t &c used in GM internals.  Depends on
   the types in gm.h */

#include "gm_types.h"

/* Include the driver types so that kernel code can simply include
   this file to get the entire GM internal driver software interface. */

#define GM_OPENER_STRING_MAX_LEN 64
#if GM_KERNEL
#  include "gm_arch.h"		/* defines internal driver SPI */
#  include "gm_impl.h"          /* defines port_state types    */
#endif

/* Define the details of user<->kernel I/O for the user-level and
   kernel-level code that performs the I/O. */

#if !GM_BUILDING_FIRMWARE
#include "gm_io.h"
#endif

#include "gm_enable_ethernet.h"
#include "gm_internal_funcs.h"
#include "gm_stbar.h"
#include "gm_undocumented_funcs.h"

/* read back the last int in the structure pointed to by ptr to flush
   the write to the LANai. */

#if GM_NEED_READBACK_TO_FLUSH_WRITES
#define GM_READBACK_TO_FLUSH_WRITES(port,ptr) do {	       		\
  (port)->trash = ((int *)(ptr+1))[-1];					\
} while (0)
#else
#define GM_READBACK_TO_FLUSH_WRITES(port,ptr)
#endif

#define GM_FLUSH_SEND_EVENT(port,hst) do {				\
  GM_STBAR ();								\
  GM_READBACK_TO_FLUSH_WRITES (port, hst);				\
} while (0)
  
#define GM_SIMULATE_SIMULATE()

/* Default implementations of GM_ARCH_LOCK_ETHER_SEND_QUEUE, which
   serialize access to the ethernet send queue on drivers where the
   send queue may be accessed from both kernel threads and the
   interrupt handler. */

#ifndef GM_ARCH_LOCK_ETHER_SEND_QUEUE
#define GM_ARCH_LOCK_ETHER_SEND_QUEUE(port) /* do nothing */
#define GM_ARCH_UNLOCK_ETHER_SEND_QUEUE(port) /* do nothing */
#endif

/****************************************************************
 * Constants
 ****************************************************************/

enum {
  /* max length of a string returned by _gm_get_kernel_build_id().
     Actual strings may be longer. */
  GM_MAX_KERNEL_BUILD_ID_LEN = 256
};

#define GM_SIZEOF_SIZE_T GM_SIZEOF_VOID_P
GM_TOP_LEVEL_ASSERT (sizeof (gm_size_t) == GM_SIZEOF_SIZE_T);

/**********************************************************************
 * Typdefs
 **********************************************************************/

/****
 * gm_should_bcopy() defs
 ****/

#define GM_SHOULD_BCOPY__MAX_THRESHOLD (1<<15)
typedef GM_BITMAP_DECL (gm_should_bcopy_bitmap_t,
			GM_SHOULD_BCOPY__MAX_THRESHOLD);

/****
 * Ports
 ****/

struct gm_send_queue_slot_token
{
  struct gm_send_queue_slot_token *next;
  gm_send_completion_callback_t sent_handler;
  void *sent_handler_context;
  void *debug_in_use;
};

/* Allow the use of the symbol "lanai" when the lanai compiler is being
   used. */

#ifdef lanai
#undef lanai
#define lanai lanai
#endif

typedef struct gm_port
{
  /* The version of the GM API supported by this port. */
  enum gm_api_version api_version;
  /* a field the user may use for any purpose.  Not touched by GM. */
  void *context;

#if GM_KERNEL
  struct gm_port_state *kernel_port_state;
  /* state for gm_dma_malloc/gm_dma_free */
  struct gm_hash *kernel_ptr_to_region_hash;
  unsigned long kernel_ptr_to_region_hash_population;
#endif /* GM_KERNEL */
 
  unsigned this_node_id;
  unsigned max_node_id;
  int wake_scheduled;

  /* Pointer to mapped lanai SRAM, including lanai-resident queues */

  struct gm_port_unprotected_lanai_side *lanai;

  /********************
  ** Host->LANai queues (in LANai SRAM)
  ********************/

  /* send queue slot tokens */

  struct gm_send_queue_slot_token send_queue_slot_token[GM_NUM_SEND_QUEUE_SLOTS];
  struct gm_send_queue_slot_token *first_free_send_queue_slot_token;
  struct gm_send_queue_slot_token *last_free_send_queue_slot_token;
				/* for forging GM_SENT_EVENTS */
#if GM_SIZEOF_UP_T == 8 && GM_SIZEOF_VOID_P==4
  /* we are running 32 bit apps on a 64 bit kernel, so make these 
     32 bit since the sent_list gets passed to the user as 
     part of the GM_SENT_EVENT */
  gm_u32_n_t sent_list[GM_NUM_SEND_QUEUE_SLOTS+1];
  gm_u32_n_t *sent_list_slot;
#else
  gm_up_n_t sent_list[GM_NUM_SEND_QUEUE_SLOTS+1];
  gm_up_n_t *sent_list_slot;
#endif
  /* Send token queue. */
  volatile gm_u32_t send_token_cnt[GM_NUM_PRIORITIES];
  volatile struct gm_send_queue_slot *send_queue_start;
  volatile struct gm_send_queue_slot *send_queue_limit;
  
  /* Recv token queue */
  volatile struct gm_host_recv_token *recv_token_queue_start;
  volatile struct gm_host_recv_token *recv_token_queue_slot;
  volatile struct gm_host_recv_token *recv_token_queue_limit;

  /********************
  ** LANai->host queues (in host memory)
  ********************/

  /* Recv message queue. */
  void *recv_queue_allocation;
  volatile struct gm_recv_queue_slot *recv_queue_start;
  volatile struct gm_recv_queue_slot *recv_queue_slot;
  volatile struct gm_recv_queue_slot *recv_queue_limit;
  
  /********************
   * Alarm stuff
   *******************/
  
  gm_u64_t lanai_alarm_time;
  enum lanai_alarm_state {LANAI_ALARM_IDLE,
			  LANAI_ALARM_SET,
			  LANAI_ALARM_FLUSHING} lanai_alarm_state;

  /* DMAable memory allocation state */
  
  struct 
  {
    unsigned num_zones;
    struct gm_zone *zone[32];
    void *zone_base[32];
    gm_size_t zone_len[32];
    gm_size_t alloced_mem_len;
  } dma;

  /********************
   * Other state
   ********************/

  unsigned unit;
  unsigned id;
  unsigned max_used_id;
  unsigned remote_memory_access_enabled;

  gm_recv_token_t _recv_tokens[GM_NUM_RECV_TOKENS + 1];
  struct gm_lanai_globals *lanai_globals;
  union gm_lanai_special_registers *lanai_special_regs;
#ifdef WIN32
  HANDLE fd;
#else
  int fd;
#endif
  struct gm_mapping_specs *mappings;
  volatile int trash;
  char unique_board_id[6];

  /* Alarm state */

  gm_alarm_t *first_alarm;

  /* pointer to the lanai-side real-time clock
     (the first page of LANai SRAM) */

  struct gm_first_sram_page *first_sram_page;

  /* _gm_get_kernel_build_id() state */

  char *kernel_build_id;

  struct gm_port *next_open_port;

  /* State for gm_ticks() */
  
  gm_u32_t gm_ticks_old_high;
  
  unsigned send_count;

  unsigned int alarm_token_cnt;

  /* Fields used for ports using the ethernet API. */
  struct
  {
    struct
    {
      struct
      {
	gm_ethernet_gather_list_n_t *ptr;
	gm_dp_t dma_addr;
      } list[GM_NUM_SEND_TOKENS];
      int index;
    } gather;
    
    struct
    {
      struct
      {
	gm_ethernet_scatter_list_n_t *ptr;
	gm_dp_t dma_addr;
      } list[GM_NUM_RECV_TOKENS];
      int index;
    } scatter;
  } *ethernet;

  struct
  {
    unsigned int first_uncached_node_id;
    /* Array to translate from node_id to global_id.  Space is
       preallocated for gm_max_node_id() entries. */
    unsigned int *array;
    /* Fast hash table to translate from global_id to node_id.
       Space is preallocated for gm_max_node_id() entries. */
    struct gm_ptr_hash *hash;
  } global_id_cache;
}
gm_port_t;

/****************************************************************
 * Enumerations
 ****************************************************************/

enum gm_mapping_mode
{
  GM_MAP_READ = 1,
  GM_MAP_WRITE = 2,
  GM_MAP_RDWR = 3
};

/****************************************************************
 * Globals
 ****************************************************************/

GM_ENTRY_POINT extern struct gm_mutex *_gm_global_mutex;
GM_ENTRY_POINT extern int _gm_initialized;
GM_ENTRY_POINT extern const char _gm_build_id[GM_MAX_KERNEL_BUILD_ID_LEN];
GM_ENTRY_POINT extern const char *_gm_version;

/* Flag to indicate if we are currently executing in the interrupt
   handler routine. */

#if GM_KERNEL
extern int gm_in_intr;
#else
#define gm_in_intr 0
#endif

/****************************************************************
 * Cast to up_t
 ****************************************************************/

/* cast from pointer to up_t needs to go through an intermediate
   integer type. Otherwise they might be sign-extended :-( */
#define GM_PTR_TO_UP(ptr) ((gm_up_t)(gm_size_t)(void *)(ptr))

/****************************************************************
 * epilogue
 ****************************************************************/

#ifdef __cplusplus
#if 0
{				/* indent hack */
#endif
}
#endif


#endif /* ifndef _gm_internal_h_ */
