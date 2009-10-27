#error This file is no longer maintained.  --Glenn

#ifndef gm_lanai_dump_h
#define gm_lanai_dump_h


#include "gm_debug_lanai_struct.h"
#include <asm/page.h>

struct gm_dump_lanai_globals;

void gm_pseudo_free (struct gm_dump_lanai_globals *g);
#if GM_KERNEL
struct gm_dump_lanai_globals *gm_pseudo_global_clone (gm_instance_state_t *
						      is);
#else
struct gm_dump_lanai_globals *gm_pseudo_global_clone (struct gm_port *);
#endif


/* A plain copy+byte-swapping would be sufficient if pointers 
   were the same size, but I want to use it for the Alpha 
   so I have to create new pseudo type to mimic the lanai ones.

   this file is constructed as follows (change in gm_types need 
   to be copied here):
   
   structure where 32 bt field are replace by host pointers:
   gm_lanai_globals  -> gm_dump_lanai_globals
   gm_port_unprotected_lanai_side _PORT par gm_dump 
   (replace the #ifdef lanai3 + GM_PAGE_LEN by PAGE_SIZE)
   copy gm_connection -> dump
   copy gm_send_record -> dump
   copy gm_send_token -> dump
   copy gm_subport -> dump
   copy gm_port_protected -> dump
  */

struct gm_dump_subport;
struct gm_dump_connection;
union gm_dump_send_token;
struct gm_dump_port_protected_lanai_side;

typedef struct gm_dump_port_unprotected_lanai_side
{
  /* Host<->LANai token queues */
  struct gm_send_queue_slot send_token_queue[GM_NUM_SEND_QUEUE_SLOTS];
  /* 8 */
  /* Extra slot for wraparound. */
  gm_host_recv_token_t recv_token_queue[GM_NUM_RECV_TOKEN_QUEUE_SLOTS + 1];
  gm_u8_t padding[PAGE_SIZE
		 -
		 (GM_NUM_SEND_QUEUE_SLOTS *
		  sizeof (struct gm_send_queue_slot) +
		  ((GM_NUM_RECV_TOKEN_QUEUE_SLOTS + 1) *
		   sizeof (gm_host_recv_token_t)))];
}
gm_dump_port_unprotected_lanai_side_t;


typedef struct gm_dump_subport
{
  struct gm_dump_subport *next;
  struct gm_dump_subport *prev;
  struct gm_dump_connection *connection;

  union gm_dump_send_token *first_send_token;
  union gm_dump_send_token *last_send_token;

  gm_s32_t delay_until;
  gm_u8_t id;
  gm_u8_t disabled;
  gm_u8_t _reserved[2];
}
gm_dump_subport_t;


typedef struct gm_dump_send_record
{
  struct gm_dump_send_record *next;
  union gm_dump_send_token *send_token;
  /* 8 */
  gm_up_t before_ptr;
  gm_sexno_t sexno;
#if GM_SIZEOF_HP_T == 8
  gm_u32_t reserved_after_sexno;
#endif
  /* 8 */
  gm_u32_t before_len;
  gm_s32_t orig_send_time;
  /* 8 */
  gm_s32_t resend_time;
  gm_u32_t _reserved_after_resend_time;
}
gm_dump_send_record_t;

typedef union gm_dump_send_token
{
  struct gm_dump_st_common
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t _reserved_after_type[3];
    /* 8 */
    gm_u32_t sendable;		/* Must be nonzero for all types that
				   may be sent */
    gm_dump_subport_t *subport;
  }
  common;
  struct gm_dump_st_ackable
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t _reserved_after_type;
    gm_u16_t target_subport_id;
    /* 8 */
    gm_u32_t send_len;
    gm_dump_subport_t *subport;
    /* 8 */
    gm_up_t orig_ptr;
    gm_up_t send_ptr;
  }
  ackable;
  struct gm_dump_st_reliable
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t size;
    gm_u16_t target_subport_id;
    /* 8 */
    gm_u32_t send_len;
    gm_dump_subport_t *subport;
    /* 8 */
    gm_up_t orig_ptr;
    gm_up_t send_ptr;
#if GM_FAST_SMALL_SEND
    gm_lp_t data;
#endif
  }
  reliable;
  struct gm_dump_st_directed
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t size;		/* HACK */
    gm_u16_t target_subport_id;
    /* 8 */
    gm_u32_t send_len;
    gm_dump_subport_t *subport;
    /* 8 */
    gm_up_t orig_ptr;
    gm_up_t send_ptr;
    /* 8 */
    gm_up_t remote_ptr;
#if GM_SIZEOF_HP_T == 4
    gm_u8_t reserved_after_remote_ptr[4];
#endif
  }
  directed;
  struct gm_dump_st_raw
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t size;		/* HACK */
    gm_u16_t target_subport_id;	/* HACK */
    /* 8 */
    gm_u32_t send_len;
    gm_dump_subport_t *subport;
    /* 8 */
    gm_up_t orig_ptr;
    gm_up_t send_ptr;		/* HACK */
  }
  raw;
  struct gm_dump_st_probe
  {
    union gm_dump_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_s8_t _reserved_after_type;
    gm_u16_t target_subport_id;
    /* 8 */
    gm_u32_t send_len;
    gm_dump_subport_t *subport;
  }
  probe;
  struct gm_dump_st_ethernet
  {
    union gm_send_token *next;
    GM_SEND_TOKEN_TYPE_8 (type);
    gm_u8_t _reserved_after_type[3];
    /* 8 */
    gm_u32_t sendable;		/* must be nonzero */
    gm_dump_subport_t *subport;
    /* NOTE: There is lots of associated state in gm.ethernet.send.state.
       This is possible because there is only one ethernet send token. */
  }
  ethernet;
}
gm_dump_send_token_t;


typedef struct gm_dump_connection
{
  /* Fields used to form a doubly-linked list of connections with
     outstanding sends. */
  struct gm_dump_connection *next_active;
  struct gm_dump_connection *prev_active;
  /* 8 */
  struct gm_dump_connection *next_to_ack;
  gm_u8_t probable_crc_error_cnt;
  gm_u8_t misrouted_packet_error_cnt;
  gm_u8_t flags;
  gm_u8_t route_len;
  /* 8 */
  /* Ack prestaging area: ROUTE must preceed ACK_PACKET */
  gm_u8_t route[GM_MAX_NETWORK_DIAMETER];	/* 4 words */
  /* 8 */
  gm_ack_packet_t ack_packet;	/* 4 words */
  /* 8 */
  /* end of ack prestaging area */
  gm_sexno_t send_sexno;
  gm_u16_t active_subport_bitmask;	/* 0 if inactive */
  gm_u16_t _reserved_after_active_subport_bitmask;
  /* 8 */
  struct gm_dump_send_record *first_send_record;
  struct gm_dump_send_record *last_send_record;
  /* 8 */
  /* may be nonzero even if inactive */
  struct gm_dump_subport *first_active_send_port;
  gm_sexno_t close_sexno;
  /* 8 */
  gm_s32_t open_time;
  gm_s32_t close_time;

}
gm_dump_connection_t;


typedef struct gm_dump_port_protected_lanai_side
{
  struct gm_send_queue_slot *send_token_queue_slot;	/*lanai addr */
  gm_host_recv_token_t *recv_token_queue_slot;	/*lanai addr */
  /* 8 */
  gm_u8_t alarm_set;
  gm_u8_t privileged;
  volatile gm_u8_t wake_host;
  gm_u8_t open;
  gm_u8_t _reserved_after_wake_host[4];
  /* 8 */
  /* lanai internal queues */
  gm_dump_send_token_t *free_send_tokens;	/*lanai addr */
  gm_recv_token_t *free_recv_tokens;	/*lanai addr */
  /* 8 */
  gm_dump_send_token_t _send_tokens[GM_NUM_SEND_TOKENS];
  /* 8 */
  gm_recv_token_t _recv_tokens[GM_NUM_RECV_TOKENS];
  /* 8 */
  /* Extra storage for "raw" tokens----v */
  gm_recv_token_t *free_recv_token[GM_NUM_PRIORITIES][34];
  /* 8 */
  /* The LANai->host recv queue */
#if GM_SIZEOF_HP_T == 4
  gm_u32_t reserved_before_sent_slot;
#elif GM_SIZEOF_HP_T != 8
#  error there is an alignment problem
#endif
  gm_hp_lp_t sent_slot;
  /* 8 */
  struct gm_dump_port_protected_lanai_side *next_with_alarm;
  gm_u32_t recv_queue_slot_num;
  /* 8 */
  /* Assume page size is the worst-case here */
  gm_dp_t recv_queue_slot_dma_addr[GM_NUM_RECV_QUEUE_SLOTS
				  + GM_NUM_RECV_QUEUE_SLOTS % 2];
  /* 8 */
  gm_up_t recv_queue_slot_host_addr[GM_NUM_RECV_QUEUE_SLOTS
				   + GM_NUM_RECV_QUEUE_SLOTS % 2];
  /* 8 */
  struct gm_dump_port_protected_lanai_side *next_with_sent_packets;
  gm_u32_t active_subport_cnt;
  /* 8 */
  /* Staging area for GM_SENT_EVENT events.  Extra slots for
     null-termination and alignment. */
  gm_up_t sent[GM_NUM_SEND_TOKENS + 2];
  /* 8 */
  gm_sent_t sent_event;
  /* 8 */
  gm_s32_t alarm_time;
  gm_u32_t id;
  /* 8 */
  gm_u32_t unacceptable_recv_sizes[GM_NUM_PRIORITIES];
  /* 8 */
  struct gm_dump_port_unprotected_lanai_side *PORT;
  gm_u32_t reserved_after_PORT;
  /* 8 */
}
gm_dump_port_protected_lanai_side_t;

typedef struct gm_dump_lanai_globals
{
  gm_u32_t magic;
  struct gm_dump_port_unprotected_lanai_side *_PORT;
  /* 8 */
  volatile union gm_dma_descriptor dma_descriptor;
  /* 8 */
  gm_u32_t reserved_after_dma_descriptor;
  /* Arrays that will never be indexed
     directly except during
     initialization. */

  /* HACK: These arrays are stored near
     the bottom of memory by mapping
     them into the code segment. They
     must each start in the bottom 32K
     of memory. */
  gm_lp_lp_t handler;
  /* 8 */
  gm_s8_lp_t event_index_table;
  gm_u16_t this_global_id;
  gm_u16_t port_to_wake;
  /* 8 */
  struct
  {
    /* Description of host-resident page
       hash table, which is broken into
       pieces. */
    gm_dp_t bogus_sdma_ptr;
    gm_dp_t bogus_rdma_ptr;
    /* 8 */
    /* A cache of the most recently used
       hash table entries. */
    gm_page_hash_cache_t cache;
    /* 8 */
  }
  page_hash;
  
  /* 8 */
  /* offset for converting lanai
     addresses to host addresses. */
  gm_dump_port_protected_lanai_side_t *first_port_with_sent_packets;
  gm_dump_subport_t *free_subports;
  /* 8 */
  gm_dump_subport_t _subport[GM_NUM_SUBPORTS];
  /* 8 */
  /* Timeout info */
  gm_s32_t timeout_time;
  gm_s32_t sram_length;
  /* 8 */
  /********************************************
   * State machine state
   ********************************************/
  gm_u32_t _state;
  /* RECV state */
  gm_s32_t free_recv_chunk_cnt;
  /* 8 */
  gm_dump_port_protected_lanai_side_t *current_rdma_port;
  gm_dump_port_protected_lanai_side_t *registered_raw_recv_port;
  /* 8 */
  /* SEND state */
  gm_u32_t free_send_chunk_cnt;
  gm_dump_connection_t *first_connection_to_ack;
  /* 8 */
  /* Send/Recv staging areas */
  gm_send_chunk_t send_chunk[2];
  /* 8 */
  gm_recv_chunk_t recv_chunk[2];
  /* 8 */
  gm_recv_token_lp_t recv_token_bin[GM_RECV_TOKEN_HASH_BINS];
  /* 8 */
  gm_dump_connection_t *first_active_connection;
  gm_dump_send_record_t *free_send_records;
  /* 8 */
  gm_dump_send_record_t _send_record[GM_NUM_SEND_RECORDS];
  /* 8 */
  /* Page-crossing dma continuation state */
  gm_u32_t remaining_sdma_ctr;	/* may be in register; use macro to access */
  gm_u32_t remaining_rdma_ctr;	/* may be in register; use macro to access */
  /* 8 */
  gm_lp_t remaining_sdma_lar;
  gm_lp_t remaining_rdma_lar;
  /* 8 */
  gm_up_t remaining_sdma_hp;
  gm_up_t remaining_rdma_hp;
  /* 8 */
  /* Buffer for staging recv token DMAs */
  struct _gm_recv_event recv_token_dma_stage;
  /* 8 */
  struct _gm_ethernet_recv_event ethernet_recv_event_dma_stage;
  /* 8 */
  gm_failed_send_event_t failed_send_event_dma_stage;
  /* 8 */
  struct
  {
    gm_u8_t _reserved[GM_RDMA_GRANULARITY - 1];
    GM_RECV_EVENT_TYPE_8 (type);
  }
  report_dma_stage;
  /* 8 */
  gm_u32_t nack_delay;
  gm_s32_t rand_seed;
  /* 8 */
  gm_u32_t backlog_delay;
  /* the three following variable are made 32 bits for the Alpha,
     ensure best performace, no waste of memory since we would need
     padding if they were made smaller */
  volatile gm_u32_t pause_rqst;
  volatile gm_u32_t pause_ack;
  gm_s32_t port_to_close;
  /* 8 */
  gm_s8_lp_t failed_file;
  gm_s32_t failed_line;
  /* 8 */
  gm_u32_t nack_cnt;
  gm_u32_t drop_cnt;
  /* 8 */
  gm_u32_t resend_cnt;
  gm_u32_t _reserved_after_resent_cnt;
  /* 8 */
  gm_u32_t bogus_header_cnt;
  gm_u32_t out_of_sequence_cnt;
  /* 8 */
  gm_s32_t led;
  gm_u32_t too_small_cnt;
  /* 8 */

  gm_dump_port_protected_lanai_side_t port[GM_NUM_PORTS];
  /* 8 */
  gm_dump_port_protected_lanai_side_t *finishing_rdma_for_port;
  gm_dump_port_protected_lanai_side_t *first_port_with_alarm;
  /* 8 */
#if GM_DEBUG_LANAI_STRUCT
  /* debugging info */
  struct
  {
    gm_u32_t host_dma_pages[GM_MAX_HOST_PAGES / 32];
    gm_u32_t first_dma_page;
    gm_u32_t reserved;
    /* 8 */
    gm_u32_t info[4];
    /* 8 */
    gm_up_t hp[2];
  }
  debug;
#endif
  /* 8 */
  gm_s32_t record_log;
  gm_s8_lp_t log_slot;
  /* 8 */
  gm_s8_lp_t log_end;
  gm_s8_lp_t lzero;
  /* 8 */
  gm_s8_t log[GM_LOG_LEN];
  /* 8 */
  volatile gm_s8_lp_t last_handler;
  gm_lp_t print_string;
  /* 8 */
  volatile gm_u32_t resume_after_halt;
  volatile gm_s32_t volatile_zero;	/* BAD */
  /* 8 */
  gm_s8_t dispatch_seen[128][2];
  /* 8 */
  gm_u32_t dispatch_cnt[128][2];
  /* 8 */
  gm_u32_t hit_cnt;
  gm_u32_t miss_cnt;
  /* 8 */
  gm_u32_t pause_cnt;
  gm_s32_t hashed_token_cnt;
  /* 8 */
  struct gm_mapper_state mapper_state;
  /* 8 */
  struct
  {
    struct
    {
      /* A token reserved for ethernet sends */
      gm_dump_send_token_t token;
      /* 8 */

      /* the staging areas for broadcasts */
      struct
      {
	/* 8 */
	gm_u8_t filled;
	gm_u8_t _reserved_after_filled[5];
	gm_u8_t padding[2];
	/* 8 */
	gm_u8_t route[GM_MAX_NETWORK_DIAMETER];
	/* 8 */
	union
	{
	  gm_u8_t as_bytes[GM_MTU];
	  struct
	  {
	    char target_ethernet_id[6];
	    char source_ethernet_id[6];
	  }
	  as_ethernet;
	  struct
	  {
	    gm_u16_t myrinet_packet_type;
	    char target_ethernet_id[6];
	    char source_ethernet_id[6];
	  }
	  as_marked_ethernet;
	}
	packet;
      }
      chunk[2];
      /* 8 */
      gm_u16_t target;
      gm_u8_t gather_cnt;
      gm_u8_t gather_pos;
      gm_lp_t next_lar;
      /* 8 */
      gm_ethernet_segment_descriptor_t
      gather_segment[GM_MAX_ETHERNET_GATHER_CNT];
      /* 8 */
      gm_u32_t total_len;
      gm_u8_t mark;
      gm_u8_t busy;
      gm_u8_t _reserved_after_mark[2];
    }
    send;
    struct
    {
      gm_ethernet_recv_token_t token[GM_NUM_ETHERNET_RECV_TOKENS];
      /* 8 */
      gm_ethernet_recv_token_lp_t token_slot;
      gm_lp_t next_lar;
      /* 8 */
      gm_s32_t remaining_len;
      gm_u32_t total_len;
      /* 8 */
      gm_u8_t scatter_pos;
      gm_u8_t reserved[7];
      /* 8 */
    }
    recv;
    /* 8 */
    gm_u8_t addr_stage[8];
    /* 8 */
    gm_dp_t addr_table_piece[GM_MAX_NUM_ADDR_TABLE_PIECES];
  }
  ethernet;
  /* 8 */
  gm_dp_t name_table_piece[GM_MAX_NUM_NAME_TABLE_PIECES];
  /* 8 */
  gm_u8_t trash[8];
  /* 8 */
  gm_u16_t max_global_id;		/* Max number of supported connections */
  gm_u8_t reserved_after_max_global_id[2];
  gm_u32_t end_magic;
  /* 8 */
  /* Connections grow off the end of the globals, consuming as much
     memory as is available. */
  gm_dump_connection_t *connection;
}
gm_dump_lanai_globals_t;


#endif /*  gm_lanai_dump_h */
