/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#if defined(HAVE_ONLINE)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif
#ifdef HAVE_SYS_UIO_H
# include <sys/uio.h>
#endif

#include "buffers.h"
#include "utils.h"

#define EVENT_INDEX(buffer, event) (event - Buffer_GetFirst(buffer))
#define ALL_BITS_SET 0xFFFFFFFF

/* Forward declarations */
static void Mask_ChangeRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id, int set);
static int Mask_Get (Buffer_t *buffer, event_t *event, int mask_id);

static void dump_buffer (int fd, int n_blocks, struct iovec *blocks);

static DataBlocks_t * new_DataBlocks (Buffer_t *buffer);
#if !defined(ARCH_SPARC64)
static void DataBlocks_AddSorted (DataBlocks_t *blocks, void *ini_address, void *end_address);
static void DataBlocks_Add (DataBlocks_t *blocks, void *ini_address, void *end_address);
#endif
static void DataBlocks_Free (DataBlocks_t *blocks);

Buffer_t * new_Buffer (int n_events, char *file, int enable_cache)
{
	Buffer_t *buffer = NULL;
#if defined(HAVE_ONLINE)
#if 0
	pthread_mutexattr_t attr;
#endif
	int rc;
#endif

	xmalloc(buffer, sizeof(Buffer_t));
	buffer->FillCount = 0;
	buffer->MaxEvents = n_events;

	xmalloc(buffer->FirstEvt, n_events * sizeof(event_t));
	buffer->LastEvt = buffer->FirstEvt + n_events;
	buffer->HeadEvt = buffer->FirstEvt;
	buffer->CurEvt = buffer->FirstEvt;

	if (file == NULL)
	{
		buffer->fd = -1;
	}
	else 
	{
          /*
           * We found a system where the mpirun seems to close the stdin, 
           * then this open assigns the fd 0, and later writes trigger an 
           * error of invalid fd. If the fd assigned is 0, repeat the open. 
           */
          do
          {
            buffer->fd = open (file, O_CREAT | O_TRUNC | O_RDWR, 0644);
          }
          while (buffer->fd == 0);

          if (buffer->fd == -1)
          {
                fprintf(stderr, "new_Buffer: Error opening file '%s'.\n", file);
                perror("open");
                exit(1);
          }
        } 

#if defined(HAVE_ONLINE) 
#if 0
	pthread_mutexattr_init( &attr );
	pthread_mutexattr_settype( &attr, PTHREAD_MUTEX_RECURSIVE_NP );

	rc = pthread_mutex_init( &(buffer->Lock), &attr );
	if ( rc != 0 )
	{
		perror("pthread_mutex_init");
		fprintf(stderr, "new_Buffer: Failed to initialize mutex.\n");
		pthread_mutexattr_destroy( &attr );
		exit(1);
	}
	pthread_mutexattr_destroy( &attr );
#else
	rc = pthread_mutex_init( &(buffer->Lock), NULL );
	if ( rc != 0 )
	{
		perror("pthread_mutex_init");
		fprintf(stderr, "new_Buffer: Failed to initialize mutex.\n");
		exit(1);
	}
#endif
#endif

	xmalloc(buffer->Masks, n_events * sizeof(Mask_t));
	Mask_Wipe(buffer);

	buffer->FlushCallback = CALLBACK_FLUSH;


	buffer->NumberOfCachedEvents = 0;
	buffer->CachedEvents         = NULL;
	buffer->VictimCache          = NULL;
	if (enable_cache)
	{
		buffer->VictimCache = new_Buffer(BUFFER_CACHE_SIZE, file, 0);
	}

	return buffer;
}

void Buffer_Free (Buffer_t *buffer)
{
	if (buffer != NULL)
	{
		xfree (buffer->FirstEvt);
#if defined(HAVE_ONLINE)
		pthread_mutex_destroy(&(buffer->Lock));
#endif
		xfree (buffer->Masks);

                xfree (buffer->CachedEvents);
                if (buffer->VictimCache != NULL)
		{
			Buffer_Free(buffer->VictimCache);
		}
		xfree (buffer);
	}
}

void Buffer_AddCachedEvent(Buffer_t *buffer, INT32 event_type)
{
  if ((buffer != NULL) && (buffer->VictimCache != NULL))
  {
    buffer->NumberOfCachedEvents ++;
    xrealloc(buffer->CachedEvents, buffer->CachedEvents, sizeof(INT32) * buffer->NumberOfCachedEvents);
    buffer->CachedEvents[ buffer->NumberOfCachedEvents - 1 ] = event_type;
  }
}

int Buffer_IsEventCached(Buffer_t *buffer, INT32 event_type)
{
  int i = 0;
  if ((buffer != NULL) && (buffer->VictimCache != NULL))
  {
    for (i=0; i<buffer->NumberOfCachedEvents; i++)
    {
      if (buffer->CachedEvents[i] == event_type) return 1;
    }
  }
  return 0;
}

void Buffer_CacheEvent(Buffer_t *buffer, event_t *event)
{
  if (buffer != NULL)
  {
    if (Buffer_IsEventCached(buffer, Get_EvEvent(event)))
    {
      Buffer_InsertSingle(buffer->VictimCache, event);
    }
  }
}

/**
 * Returns the number of bytes written in the fd of the given buffer
 * \return The number of bytes written to disk
 */
unsigned long long Buffer_GetFileSize (Buffer_t *buffer)
{
#if defined(DEAD_CODE)
	struct stat buf;
#endif
	off_t size = 0;

	if ((buffer != NULL) && (buffer->fd != -1))
	{
		off_t current_position;

		current_position = lseek (buffer->fd, 0, SEEK_CUR);
		size = lseek (buffer->fd, 0, SEEK_END);
		lseek (buffer->fd, current_position, SEEK_SET);

#if defined(DEAD_CODE)
		fstat(buffer->fd, &buf);
		size = buf.st_size;
#endif
	}
	return (unsigned long long)size;
}

/**
 * Assigns a callback to be called when the buffer gets filled up
 * \param buffer A valid buffer
 * \param callback Pointer to function
 */
void Buffer_SetFlushCallback (Buffer_t *buffer, int (*callback)(struct Buffer *))
{
   buffer->FlushCallback = callback;
}

/**
 * Invokes the flush callback assigned to buffer
 * \param buffer The buffer to be flushed
 * \return The value returned by the callback, 0 if buffer had no callback assigned
 */
int Buffer_ExecuteFlushCallback (Buffer_t *buffer)
{
	int rc = 0;

#if defined(LOCK_AT_FLUSH)
	Buffer_Lock (buffer);
#endif
	if (buffer->FlushCallback != NULL)
	{
		rc = ((buffer->FlushCallback) (buffer));
	}
#if defined(LOCK_AT_FLUSH)
	Buffer_Unlock (buffer);
#endif
	return rc;
}

void Buffer_Close (Buffer_t *buffer)
{
	if (buffer->fd != -1)
	{
		Buffer_FlushCache(buffer);
		close(buffer->fd);
	}
	buffer->fd = -1;
}

int Buffer_IsClosed (Buffer_t *buffer)
{
	return (buffer->fd == -1);
}

int Buffer_IsEmpty (Buffer_t *buffer)
{
    return (buffer->FillCount == 0);
}

int Buffer_IsFull (Buffer_t *buffer)
{
    return (buffer->FillCount == buffer->MaxEvents);
}

static event_t * Buffer_GetFirst (Buffer_t *buffer)
{
    return (buffer->FirstEvt); 
}

static event_t * Buffer_GetLast (Buffer_t *buffer)
{
    return (buffer->LastEvt);
}

event_t * Buffer_GetFirstEvent (Buffer_t *buffer)
{
    if (Buffer_GetFillCount(buffer) > 0)
      return Buffer_GetHead(buffer);
    else 
      return NULL;
}

event_t * Buffer_GetLastEvent (Buffer_t *buffer)
{
    if (Buffer_GetFillCount(buffer) > 0)
      return (Buffer_GetTail(buffer) - 1);
    else 
      return NULL;
}

event_t * Buffer_GetHead (Buffer_t *buffer)
{
    return (buffer->HeadEvt);
}

event_t * Buffer_GetTail (Buffer_t *buffer)
{
    return (buffer->CurEvt);
}

int Buffer_GetFillCount (Buffer_t *buffer)
{
    return (buffer->FillCount);
}

int Buffer_RemainingEvents (Buffer_t *buffer)
{
    return (buffer->MaxEvents - buffer->FillCount);
}

int Buffer_EnoughSpace (Buffer_t *buffer, int num_events)
{
	return (Buffer_RemainingEvents(buffer) >= num_events);
}

event_t * Buffer_GetNext (Buffer_t *buffer, event_t *current)
{
    event_t *next = current;
    if (++next == Buffer_GetLast(buffer)) next = Buffer_GetFirst(buffer);
    return next;
}

void Buffer_Lock (Buffer_t *buffer)
{
#if defined(HAVE_ONLINE)
	pthread_mutex_lock(&(buffer->Lock));
#else
	UNREFERENCED_PARAMETER(buffer);
#endif
}

void Buffer_Unlock (Buffer_t *buffer)
{
#if defined(HAVE_ONLINE)
	pthread_mutex_unlock(&(buffer->Lock));
#else
	UNREFERENCED_PARAMETER(buffer);
#endif
}

/*
	writev_wrapper
	writev can be interrupted in BG systems. We build this wrapper to emulate the
	writev operation based on regular writes.
*/

static ssize_t writev_wrapper (int fd, const struct iovec *iov, int iovcnt) 
{
	size_t tmp = 0;
	ssize_t written = 0,total = 0;
	int i;

	for (i = 0; i < iovcnt; i++, iov++)
	{
		tmp = 0;
		while (tmp < iov->iov_len)
		{
			written = write (fd,
				(const void *)((int *)(iov->iov_base) + tmp), iov->iov_len - tmp);

			if (written < 0)  
				return written;
			tmp += written;
		}

		total += tmp;
	}
	return total;
}

static void dump_buffer (int fd, int n_blocks, struct iovec *blocks)
{
   int     idx = 0;
   int     remaining_blocks = 0;
   int     written_blocks = 0;
   ssize_t nbytes = 0;

#if defined(DEBUG)
   fprintf(stderr, "[dump_buffer] Start writing %d blocks (%p) to fd=%d\n", n_blocks, blocks, fd);
#endif
   if (blocks != NULL)
   {
      remaining_blocks = n_blocks;
      while (remaining_blocks > 0)
      {
         written_blocks = MIN(IOV_MAX, remaining_blocks);

         nbytes = writev_wrapper (fd, (const struct iovec *)blocks + idx, written_blocks);
         if (nbytes == -1)
         {
            fprintf(stderr, "dump_buffer: Error writing to disk.\n");
            perror("writev");
            exit(1);
         }

         remaining_blocks -= written_blocks;
         idx += written_blocks;
      }
   }
#if defined(DEBUG)
   fprintf(stderr, "[dump_buffer] Done\n");
#endif
}

void Buffer_InsertMultiple(Buffer_t *buffer, event_t *events_list, int num_events)
{
	int i, retry = num_events;

	while ((retry > 0) && (!Buffer_EnoughSpace(buffer, num_events)))
	{
		int rc;
		rc = Buffer_ExecuteFlushCallback(buffer);
		if (rc == 0) return;
		retry --;
	}
	if (!Buffer_EnoughSpace(buffer, num_events))
	{
		fprintf (stderr, "Buffer_InsertMultiple: No room for %d events.\n", num_events);
		exit(1);
	}
	/* XXX: We should just memcpy the whole events_list */
	for (i=0; i<num_events; i++)
	{
		Buffer_InsertSingle(buffer, &(events_list[i]));
	}
}

void Buffer_InsertSingle(Buffer_t *buffer, event_t *new_event)
{
#if defined(LOCK_AT_INSERT)
	Buffer_Lock (buffer);
#endif

	if (Buffer_IsFull (buffer))
	{
		int rc;
		rc = Buffer_ExecuteFlushCallback (buffer);
		if (rc == 0) return;
	}

	/* Insert new event */
	memcpy(buffer->CurEvt, new_event, sizeof(event_t));
	Mask_UnsetAll (buffer, buffer->CurEvt);

	/* Move tail forwards */
	buffer->CurEvt = Buffer_GetNext(buffer, buffer->CurEvt);
	buffer->FillCount ++;

#if defined(LOCK_AT_INSERT)
	Buffer_Unlock (buffer);
#endif
}

int Buffer_Flush(Buffer_t *buffer)
{
	DataBlocks_t *db = new_DataBlocks (buffer);
	event_t *head = NULL, *tail = NULL;
	int num_flushed, overflow;
#if defined(ARCH_SPARC64)
	ssize_t r;
#endif

	if ((Buffer_IsEmpty(buffer)) || (Buffer_IsClosed(buffer))) 
	{
		return 0;
	}

	head = Buffer_GetHead(buffer);
	tail = head;
	num_flushed = Buffer_GetFillCount(buffer);
	CIRCULAR_STEP (tail, num_flushed, buffer->FirstEvt, buffer->LastEvt, &overflow);

#if !defined(ARCH_SPARC64)

# if defined(HAVE_ONLINE)
	/* Select events depending on the mask */
	Filter_Buffer(buffer, head, tail, db);
# else
	/* Select all events from head to tail */
	DataBlocks_Add (db, head, tail);
# endif

	/* Forward to the end of the file. This is necessary because of the interactions between the normal tracing buffer and its cache. When 
         * one of these buffers flush, the other buffer's pointer does not get updated to the end of the file, and so if it flushes next, data may get 
         * overwritten */
        lseek(buffer->fd, 0, SEEK_END);

	/* Write to disk */
	dump_buffer (buffer->fd, db->NumBlocks, db->BlocksList);

	/* Free resources */
	DataBlocks_Free(db);

#else /* ARCH_SPARC64 */

	r = write (buffer->fd, head, buffer->FillCount*sizeof(event_t));
	if (r != buffer->FillCount*sizeof(event_t))
		fprintf (stderr, "ERROR! Wrote %ld bytes instead of %ld bytes\n", r, buffer->FillCount*sizeof(event_t));

#endif

	//Do not call DiscardAll. This allows one thread to flush another thread's buffer that is not locked.
	//Buffer_DiscardAll(buffer);
    buffer->FillCount -= num_flushed;
    buffer->HeadEvt = tail;

	return 1;
}

int Buffer_FlushCache(Buffer_t *buffer)
{
  if ((buffer != NULL) && (buffer->VictimCache != NULL))
  {
    return Buffer_Flush(buffer->VictimCache);
  }
  return 0;
}

#if !defined(ARCH_SPARC64)
void Filter_Buffer(Buffer_t *buffer, event_t *first_event, event_t *last_event, DataBlocks_t *io_db)
{
    void *ini_addr = NULL;
    event_t *current = NULL;

    current = first_event;
    do
    {
	if (Mask_IsSet(buffer, current, MASK_NOFLUSH) && !Buffer_IsEventCached(buffer, Get_EvEvent(current)))
        {
            if (ini_addr != NULL)
            {
                DataBlocks_Add(io_db, ini_addr, (void *)current);
                ini_addr = NULL;
            }
        }
        else
        {
            if (ini_addr == NULL)
            {
                ini_addr = (void *)current;
            }
        }

        /* Next */
	current = Buffer_GetNext(buffer, current);
    } while (current != last_event);

    if (ini_addr != NULL)
    {
        DataBlocks_Add(io_db, ini_addr, (void *)current);
    }
}
#endif

int Buffer_DiscardOldest (Buffer_t *buffer)
{
    event_t *old_head = NULL, *new_head = NULL;

    old_head = buffer->HeadEvt;
    Buffer_CacheEvent(buffer, old_head);

    new_head = Buffer_GetNext(buffer, buffer->HeadEvt);
    buffer->FillCount --;
    buffer->HeadEvt = new_head;

    return 1;
}

int Buffer_Discard10Pct (Buffer_t *buffer)
{
    event_t *head = NULL, *last = NULL;
    int pct10 = buffer->MaxEvents * 0.1;

    head = buffer->HeadEvt;
    last = buffer->LastEvt;

    head += pct10;
    if (head >= last)
    {
        head = buffer->FirstEvt + (head - last);
    }

    buffer->FillCount -= pct10;
    buffer->HeadEvt = head;

	return 1;
}

int Buffer_DiscardAll (Buffer_t *buffer)
{
	buffer->FillCount = 0;
	buffer->HeadEvt = buffer->CurEvt;

	return 1;
}

/***************************************************************************/
/***************************************************************************/
/************************        B L O C K S        ************************/
/***************************************************************************/
/***************************************************************************/

static DataBlocks_t * new_DataBlocks (Buffer_t *buffer)
{  
    DataBlocks_t *blocks = NULL;
   
    xmalloc (blocks, sizeof(DataBlocks_t));

    if (blocks != NULL)
    {
        blocks->FirstAddr = buffer->FirstEvt;
        blocks->LastAddr = buffer->LastEvt;
    
        blocks->MaxBlocks = BLOCKS_CHUNK;
        blocks->NumBlocks = 0;
        xmalloc (blocks->BlocksList, sizeof(struct iovec) * blocks->MaxBlocks);
    }
    return blocks;
}

#if !defined(ARCH_SPARC64)
static void DataBlocks_AddSorted (DataBlocks_t *blocks, void *ini_address, void *end_address)
{
    blocks->NumBlocks ++;
    if (blocks->NumBlocks >= blocks->MaxBlocks)
    {
        blocks->MaxBlocks += BLOCKS_CHUNK;

        xrealloc (blocks->BlocksList, blocks->BlocksList, sizeof(struct iovec) * blocks->MaxBlocks);
    }
    blocks->BlocksList[ blocks->NumBlocks - 1 ].iov_base = (void *)ini_address;
	blocks->BlocksList[ blocks->NumBlocks - 1 ].iov_len  = end_address - ini_address;
}
#endif

#if !defined(ARCH_SPARC64)
static void DataBlocks_Add (DataBlocks_t *blocks, void *ini_address, void *end_address)
{
    if (blocks != NULL)
    {
        if (ini_address < end_address)
        {
            DataBlocks_AddSorted (blocks, ini_address, end_address);
        }
        else
        {
            DataBlocks_AddSorted (blocks, ini_address, blocks->LastAddr);
            DataBlocks_AddSorted (blocks, blocks->FirstAddr, end_address);
        }
    }
}
#endif

static void DataBlocks_Free (DataBlocks_t *blocks)
{
   xfree (blocks->BlocksList);
   xfree (blocks);
}

/***************************************************************************/
/***************************************************************************/
/************************     I T E R A T O R S     ************************/
/***************************************************************************/
/***************************************************************************/

/**
 * Creates a new buffer iterator
 * \param buffer The buffer to create an iterator for
 * \return The new iterator
 */
static BufferIterator_t * new_Iterator(Buffer_t *buffer)
{
	BufferIterator_t *it = NULL;

	ASSERT_VALID_BUFFER(buffer);

	xmalloc(it, sizeof(BufferIterator_t));

	it->Buffer         = buffer;
	it->OutOfBounds    = Buffer_IsEmpty(buffer);
	it->CurrentElement = NULL;
	it->StartBound     = Buffer_GetHead(buffer);
	it->EndBound       = Buffer_GetTail(buffer);
#if defined(DEBUG)
	fprintf(stderr, "[DBG_BUFFERS] new_Iterator: Buffer=%p, OutOfBounds=%d\n", it->Buffer, it->OutOfBounds);
#endif
	return it;
}

/**
 * Clones an iterator
 * \param orig The iterator to copy
 * \return The new iterator
 */
BufferIterator_t * BufferIterator_Copy (BufferIterator_t *orig)
{
  BufferIterator_t *copy = NULL; 
  if (orig != NULL)
  {
    xmalloc(copy, sizeof(BufferIterator_t));
    copy->Buffer         = orig->Buffer;
    copy->OutOfBounds    = orig->OutOfBounds;
    copy->CurrentElement = orig->CurrentElement;
    copy->StartBound     = orig->StartBound;
    copy->EndBound       = orig->EndBound;
  }
  return copy;
}

/**
 * Creates a new iterator pointing to the first event in buffer
 * \param buffer The buffer to create an iterator for
 * \return The new iterator
 */
BufferIterator_t * BufferIterator_NewForward (Buffer_t *buffer)
{
	BufferIterator_t *it = new_Iterator (buffer);

	ASSERT_VALID_ITERATOR(it);

	it->CurrentElement = Buffer_GetHead(buffer);

	return it;
}

/**
 * Creates a new iterator pointing to the last event in buffer
 * \param buffer The buffer to create an iterator for
 * \return The new iterator
 */
BufferIterator_t * BufferIterator_NewBackward (Buffer_t *buffer)
{
	int overflow;
	BufferIterator_t *it = new_Iterator (buffer);

	ASSERT_VALID_ITERATOR(it);

	it->CurrentElement = buffer->CurEvt;
	/* CurEvt points to the next event going to be written, so we have to rewind 1 position for the last valid event */

	CIRCULAR_STEP (it->CurrentElement, -1, it->Buffer->FirstEvt, it->Buffer->LastEvt, &overflow);

	return it;
}

BufferIterator_t * BufferIterator_NewRange (Buffer_t *buffer, unsigned long long start_time, unsigned long long end_time)
{
	BufferIterator_t *itf, *itb, *itrange;
	int found_start_bound = FALSE, found_end_bound = FALSE;
	int count1 = 0, count2 = 0;
	
	itrange = new_Iterator(buffer);
	ASSERT_VALID_ITERATOR(itrange);
	itf = BIT_NewForward(buffer);
	itb = BIT_NewBackward(buffer);

	/* Search for the start boundary */
	while ((!BIT_OutOfBounds(itf)) && (!found_start_bound))
	{
		event_t *cur = BIT_GetEvent(itf);
		if (Get_EvTime(cur) >= start_time)
		{
			found_start_bound = TRUE;
			itrange->StartBound = cur;
		}
		count1++;
		BIT_Next(itf);
	}

	/* Search for the end boundary */
	while ((!BIT_OutOfBounds(itb)) && (!found_end_bound))
	{
		event_t *cur = BIT_GetEvent(itb);
		if (Get_EvTime(cur) <= end_time)
		{	
			found_end_bound = TRUE;
			itrange->EndBound = cur;
		}
		count2++;
		BIT_Prev(itb);
	}

	itrange->CurrentElement = itrange->StartBound;
	itrange->OutOfBounds = !(found_start_bound && found_end_bound);
	return itrange;
}


/**
 * Moves a buffer iterator to the next event
 * \param it The iterator to be moved
 */
void BufferIterator_Next (BufferIterator_t *it)
{
	ASSERT_VALID_BOUNDS(it);

	it->CurrentElement = Buffer_GetNext(it->Buffer, it->CurrentElement);
	/* it->OutOfBounds = (it->CurrentElement == Buffer_GetTail(it->Buffer)); */
	it->OutOfBounds = (it->CurrentElement == it->EndBound);
}

/**
 * Moves a buffer iterator to the previous event
 * \param it The iterator to be moved
 */
void BufferIterator_Previous (BufferIterator_t *it)
{
	int overflow;
	ASSERT_VALID_BOUNDS(it);

	/* it->OutOfBounds = (it->CurrentElement == Buffer_GetHead(it->Buffer)); */
	it->OutOfBounds = (it->CurrentElement == it->StartBound);
	if (!it->OutOfBounds)
	{
		CIRCULAR_STEP (it->CurrentElement, -1, it->Buffer->FirstEvt, it->Buffer->LastEvt, &overflow);
	}
}

/**
 * Check whether buffer iterator is out of bounds
 * \param it The iterator to be checked
 * \return 1 if the iterator is out of bounds, 0 otherwise
 */
int BufferIterator_OutOfBounds (BufferIterator_t *it)
{
	return ((it != NULL) ? it->OutOfBounds : TRUE);
}

/**
 * Returns a pointer to the event being pointed by the iterator 'it'
 * \param it A valid buffer iterator
 * \return Pointer to the event
 */
event_t * BufferIterator_GetEvent (BufferIterator_t *it)
{
	ASSERT_VALID_BOUNDS(it);

	return (it->CurrentElement);
}

/**
 * Frees all memory used by an iterator
 * \param it The iterator to be freed
 */
void BufferIterator_Free (BufferIterator_t *it)
{
	xfree(it);
}

void Mask_Wipe (Buffer_t *buffer)
{
    memset (buffer->Masks, 0, buffer->MaxEvents * sizeof(Mask_t));
}

void Mask_Set (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->Masks[index] |= mask_id;
}

void Mask_SetAll (Buffer_t *buffer, event_t *event)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->Masks[index] = ALL_BITS_SET;	
}

void Mask_SetRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id)
{
    Mask_ChangeRegion (buffer, start, end, mask_id, 1);
}

void Mask_Unset (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->Masks[index] &= ~mask_id;
}

void Mask_UnsetAll (Buffer_t *buffer, event_t *event)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->Masks[index] = 0;	
}

void Mask_UnsetRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id)
{
	Mask_ChangeRegion (buffer, start, end, mask_id, 0);
}

static void Mask_ChangeRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id, int set)
{
	event_t *current = start;

    do
    {
		if (set)
			Mask_Set (buffer, current, mask_id);
		else
			Mask_Unset (buffer, current, mask_id);

		current = Buffer_GetNext (buffer, current);
    } while (current != end);

    /* Both range limits included */
	if (set)
		Mask_Set (buffer, current, mask_id);
	else
		Mask_Unset (buffer, current, mask_id);
}

void Mask_Flip (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->Masks[index] ^= mask_id;
}

static int Mask_Get (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	return ((buffer->Masks[index] & (mask_id)) == mask_id);
}

int Mask_IsSet (Buffer_t *buffer, event_t *event, int mask_id)
{
	return (Mask_Get (buffer, event, mask_id) == 1);
}

int Mask_IsUnset (Buffer_t *buffer, event_t *event, int mask_id)
{
	return (Mask_Get (buffer, event, mask_id) == 0);
}

void BufferIterator_MaskSet (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    Mask_Set (it->Buffer, it->CurrentElement, mask_id);
}

void BufferIterator_MaskSetAll (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);
    Mask_SetAll (it->Buffer, it->CurrentElement);
}

void BufferIterator_MaskUnset (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    Mask_Unset (it->Buffer, it->CurrentElement, mask_id);
}

void BufferIterator_MaskUnsetAll (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);
    Mask_UnsetAll (it->Buffer, it->CurrentElement);
}

int BufferIterator_IsMaskSet (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    return Mask_IsSet (it->Buffer, it->CurrentElement, mask_id);
}

int BufferIterator_IsMaskUnset (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    return Mask_IsUnset (it->Buffer, it->CurrentElement, mask_id);
}

