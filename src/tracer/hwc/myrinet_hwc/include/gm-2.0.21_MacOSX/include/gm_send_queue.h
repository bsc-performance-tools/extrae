/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#include "gm_call_trace.h"
#include "gm_compiler.h"
#include "gm_debug.h"
#include "gm_debug_send_tokens.h"
#include "gm_internal.h"
#include "gm_list.h"

#if GM_STRUCT_WRITE_COMBINING

#define GM_COPY_TO_IO_SPACE(lval, rval) do {				\
  gm_copy_to_io_space (&(lval), &(rval), sizeof(lval));			\
} while (0)

static gm_inline void
gm_copy_to_io_space (volatile void *to, void *from, unsigned size)
{
  register int i;

  gm_assert ((size > 0)
	     && !(size & 3) && !((gm_size_t) to & 3) && !((gm_size_t) from & 3));

#if 0				/* put 0 to disable 64 bit write */
  if (!(size & 7) && !((long) to & 7) && !((long) from & 7))
    {
      volatile register gm_u64_t *to64;
      gm_u64_t *from64, tmp64;

      to64 = (volatile gm_u64_t *) to;
      from64 = (gm_u64_t *) from;
      for (i = size / 8 - 1; i; i--)
	{
	  *to64++ = *from64++;
	}
      tmp64 = *from64;
      GM_STBAR ();
      *to64 = tmp64;
      GM_STBAR ();
    }
  else
#endif
    {
      register volatile gm_u32_t *to32;
      gm_u32_t *from32, tmp;

      to32 = (volatile gm_u32_t *) to;
      from32 = (gm_u32_t *) from;
      for (i = size / 4 - 1; i; i--)
	{
	  *to32++ = *from32++;
	}
      tmp = *from32;
      GM_STBAR ();
      *to32++ = tmp;
      GM_STBAR ();
    }
}
#endif /* STRUCT_WRITE_COMBINING */

static gm_inline struct gm_send_queue_slot *
__gm_send_queue_slot (gm_port_t * p, gm_send_completion_callback_t callback,
		      void *context)
{
  struct gm_send_queue_slot_token *sqst, *next;
  unsigned int token;

  GM_CALLED_WITH_ARGS (("%p,%p,%p", p, callback, context));
  GM_PRINT (GM_DEBUG_SEND_TOKENS,
	    ("%d outstanding sends\n", ++p->send_count));
  
  GM_ASSERT_NO_CYCLE (struct gm_send_queue_slot_token,
		      p->first_free_send_queue_slot_token, next);
  sqst = p->first_free_send_queue_slot_token;
  ;
  ;
  gm_assert (sqst);
  next = sqst->next;
  sqst->next = 0;
  sqst->sent_handler = callback;
  sqst->sent_handler_context = context;
  p->first_free_send_queue_slot_token = next;
  token = (unsigned int) (sqst - &p->send_queue_slot_token[0]);
  GM_PRINT (GM_DEBUG_SEND_TOKENS, ("using send slot 0x%x\n", token));
  if (GM_DEBUG)
    {
      if (next)
	{
	  GM_PRINT
	    (GM_DEBUG_SEND_TOKENS,
	     ("next is 0x%lx\n",
	      (unsigned long) (next - &p->send_queue_slot_token[0])));
	}
      else
	{
	  GM_PRINT (GM_DEBUG_SEND_TOKENS,
		    ("*** no more free send slots ***\n"));
	}
      
      /* Mark the token as in use. */
      
      gm_assert (sqst->debug_in_use == 0);
      sqst->debug_in_use = (void *) -1;
    }

  /* Make sure either callback or context is set for debugging purposes. */

  if (GM_DEBUG && !callback)
    sqst->sent_handler_context = (void *) 1;

  GM_RETURN ((struct gm_send_queue_slot *) (&p->send_queue_start[0] + token));
}

#define GM_SEND_QUEUE_SLOT(port, callback, context, type)		\
/**/ GM_SEND_QUEUE_SLOT_EVENT (__gm_send_queue_slot (port,		\
						     callback,		\
						     context),		\
			       type)

static gm_inline void
__gm_post_simple_send_event (gm_port_t * p, enum gm_send_event_type type,
			     gm_send_completion_callback_t callback,
			     void *context)
{
  struct gm_simple_send_event volatile *s;
#if GM_STRUCT_WRITE_COMBINING
  struct gm_simple_send_event batch_write;
#endif

  s = GM_SEND_QUEUE_SLOT (p, callback, context, simple);
  gm_assert (gm_ntoh_u8 (s->type) == GM_NO_SEND_EVENT);
#if GM_STRUCT_WRITE_COMBINING
  batch_write.type = gm_hton_u8 ((gm_u8_t) type);
  GM_COPY_TO_IO_SPACE (*s, batch_write);
#else
  s->type = gm_htonc (type);
#endif
  GM_FLUSH_SEND_EVENT (p, s);
}

static gm_inline void
__gm_free_send_tokens (gm_port_t * p, unsigned int priority,
		       unsigned int count)
{
  gm_assert (priority <= GM_MAX_PRIORITY);
  p->send_token_cnt[priority] += count;
}

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
