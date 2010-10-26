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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <config.h>
#if defined(HAVE_MRNET)
# include <pthread.h>
#endif
#include "buffers.h"
#include "utils.h"

#define EVENT_INDEX(buffer, event) (event - Buffer_GetFirst(buffer))
#define ALL_BITS_SET 0xFFFFFFFF

/* Forward declarations */
static void NewMask_ChangeRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id, int set);
static int NewMask_Get (Buffer_t *buffer, event_t *event, int mask_id);

static event_t * Buffer_GetFirst (Buffer_t *buffer);
static event_t * Buffer_GetLast (Buffer_t *buffer);
#if defined(MASKS_DEAD_CODE)
static void Buffer_SetMask (Buffer_t *buffer, int index, int value);
static int Buffer_CheckMask (Buffer_t *buffer, int index);
static void Buffer_SetMaskRegion (Buffer_t *buffer, event_t *first_evt, event_t *last_evt, int value);
#endif /* MASKS_DEAD_CODE */
static void dump_buffer (int fd, int n_blocks, struct iovec *blocks);


static DataBlocks_t * new_DataBlocks (Buffer_t *buffer);
static void DataBlocks_AddSorted (DataBlocks_t *blocks, void *ini_address, void *end_address);
static void DataBlocks_Add (DataBlocks_t *blocks, void *ini_address, void *end_address);
static void DataBlocks_Free (DataBlocks_t *blocks);

Buffer_t * new_Buffer (int n_events, char *file)
{
	Buffer_t *buffer = NULL;
#if defined(HAVE_MRNET)
	pthread_mutexattr_t attr;
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
	else if ((buffer->fd = open (file, O_CREAT | O_TRUNC | O_RDWR, 0644)) == -1)
	{
		fprintf(stderr, "new_Buffer: Error opening file '%s'.\n", file);
		perror("open");
		exit(1);
	}

#if defined(HAVE_MRNET) 
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
#endif

#if defined(MASKS_DEAD_CODE)
	xmalloc(buffer->Mask, n_events * sizeof(int));
#endif
	xmalloc(buffer->NewMasks, n_events * sizeof(Mask_t));
	NewMask_Wipe(buffer);

	buffer->FlushCallback = CALLBACK_FLUSH;

	return buffer;
}

void Buffer_Free (Buffer_t *buffer)
{
	if (buffer != NULL)
	{
		xfree (buffer->FirstEvt);
#if defined(HAVE_MRNET)
		pthread_mutex_destroy(&(buffer->Lock));
#endif
#if defined(MASKS_DEAD_CODE)
		xfree (buffer->Mask);
#endif
		xfree (buffer->NewMasks);
		xfree (buffer);
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

#if defined(MASKS_DEAD_CODE)
void Buffer_ClearMask (Buffer_t *buffer)
{
    memset (buffer->Mask, 0, buffer->MaxEvents * sizeof(int));
}

static void Buffer_SetMask (Buffer_t *buffer, int index, int value)
{
    buffer->Mask[index] = value;
}

void Buffer_MaskIn (Buffer_t *buffer, event_t *evt)
{
    Buffer_SetMask(buffer, evt - Buffer_GetFirst(buffer), 1);
}

void Buffer_MaskOut (Buffer_t *buffer, event_t *evt)
{
    Buffer_SetMask(buffer, evt - Buffer_GetFirst(buffer), 0);
}

static int Buffer_CheckMask (Buffer_t *buffer, int index)
{
    return buffer->Mask[index];
}

int Buffer_IsMaskedOut (Buffer_t *buffer, event_t *evt)
{
    return ( Buffer_CheckMask(buffer, evt - Buffer_GetFirst(buffer)) == 0 );
}

int Buffer_IsMaskedIn (Buffer_t *buffer, event_t *evt)
{
    return ( Buffer_CheckMask(buffer, evt - Buffer_GetFirst(buffer)) == 1 );
}

static void Buffer_SetMaskRegion (Buffer_t *buffer, event_t *first_evt, event_t *last_evt, int value)
{
    event_t *current = first_evt;

    do
    {
        if (value)
            Buffer_MaskIn (buffer, current);
        else
            Buffer_MaskOut (buffer, current);

		current = Buffer_GetNext (buffer, current);
    } while (current != last_evt);

    /* Both range limits included */
    Buffer_MaskIn (buffer, last_evt);
}

void Buffer_MaskRegionIn (Buffer_t *buffer, event_t *first_evt, event_t *last_evt)
{
    Buffer_SetMaskRegion (buffer, first_evt, last_evt, 1);
}

void Buffer_MaskRegionOut (Buffer_t *buffer, event_t *first_evt, event_t *last_evt)
{
    Buffer_SetMaskRegion (buffer, first_evt, last_evt, 0);
}
#endif /* MASKS_DEAD_CODE */

void Buffer_Lock (Buffer_t *buffer)
{
#if defined(HAVE_MRNET)
	pthread_mutex_lock(&(buffer->Lock));
#else
	UNREFERENCED_PARAMETER(buffer);
#endif
}

void Buffer_Unlock (Buffer_t *buffer)
{
#if defined(HAVE_MRNET)
	pthread_mutex_unlock(&(buffer->Lock));
#else
	UNREFERENCED_PARAMETER(buffer);
#endif
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

         nbytes = writev (fd, (const struct iovec *)blocks + idx, written_blocks);
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
#if defined(MASKS_DEAD_CODE)
	Buffer_MaskIn (buffer, buffer->CurEvt);
#else
	NewMask_UnsetAll (buffer, buffer->CurEvt);
#endif /* MASKS_DEAD_CODE */

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

	if ((Buffer_IsEmpty(buffer)) || (Buffer_IsClosed(buffer))) return 0;


	head = Buffer_GetHead(buffer);
	tail = head;
	num_flushed = Buffer_GetFillCount(buffer);
	CIRCULAR_STEP (tail, num_flushed, buffer->FirstEvt, buffer->LastEvt, &overflow);

#if defined(HAVE_MRNET)
	/* Select events depending on the mask */
	Filter_Buffer(buffer, head, tail, db);
#else
	/* Select all events from head to tail */
	DataBlocks_Add (db, head, tail);
#endif

	/* Write to disk */
	dump_buffer (buffer->fd, db->NumBlocks, db->BlocksList);

	/* Free resources */
	DataBlocks_Free(db);

	//Do not call DiscardAll. This allows one thread to flush another thread's buffer that is not locked.
	//Buffer_DiscardAll(buffer);
    buffer->FillCount -= num_flushed;
    buffer->HeadEvt = tail;

	return 1;
}

void Filter_Buffer(Buffer_t *buffer, event_t *first_event, event_t *last_event, DataBlocks_t *io_db)
{
    void *ini_addr = NULL;
    event_t *current = NULL;

    current = first_event;
    do
    {
#if defined(MASKS_DEAD_CODE)
        if (Buffer_IsMaskedOut(buffer, current))
#else
		if (NewMask_IsSet(buffer, current, MASK_NOFLUSH))
#endif
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

int Buffer_DiscardOldest (Buffer_t *buffer)
{
    event_t *head = NULL;

	head = Buffer_GetNext(buffer, buffer->HeadEvt);

    buffer->FillCount --;
    buffer->HeadEvt = head;

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
#if 0
/* Una funcion como esta deberia estar en la banda del traceo ? */
void Action_When_Buffer_Is_Full (Buffer_t *buffer)
{
#if defined(HAVE_MRNET)
    Buffer_Lock(buffer);
#endif

    switch(DefaultActionWhenFull)
    {
        case OVERWRITE:
            Buffer_DiscardOldest (buffer);
            break;
        case DISCARD:
            Buffer_DiscardAll (buffer);
            break;
        case FLUSH:
            Buffer_Flush(buffer, FALSE);
            break;
        case SYNC_FLUSH:
#if defined(HAVE_MRNET)
//          Buffer_Unlock(thread);
            MRNet_Sync_Flush ();
#endif
            break;
        default:
            fprintf(stderr, PACKAGE_NAME": Buffer is full and no action specified!\n");
            exit(1);
    }
#if defined(HAVE_MRNET)
    Buffer_Unlock(buffer);
#endif
}
#endif

#if 0
void advance_current(Buffer_t *buffer)
{
#if defined(HAVE_MRNET)
    Buffer_MaskIn(buffer, buffer->CurEvt);
#endif
	buffer->CurEvt = Buffer_GetNext (buffer, buffer->CurEvt);
	buffer->FillCount ++;

	if (Buffer_IsFull (buffer))
	{
		Action_When_Buffer_Is_Full (buffer);
	}

#if USE_HARDWARE_COUNTERS
    CheckForHWCSetChange ( CHANGE_TIME, thread );
#endif
}

int linearize(Buffer_t *buffer, event_t **buffer)
{
    event_t *linear_buffer, *head, *current;
    int num_events, num_ev1, num_ev2;
    int size, size1, size2;

    head = buffer->HeadEvt;
    current = buffer->CurEvt;

    if (current > head)
    {
        num_events = current - head;
        size = num_events * sizeof(event_t);
        xmalloc(linear_buffer, size);
        memcpy(linear_buffer, head, size);
    }
    else
    {
        num_ev1 = buffer->LastEvt - head;
        num_ev2 = current - buffer->FirstEvt; 
        num_events = num_ev1 + num_ev2;
        size1 = num_ev1 * sizeof(event_t);
        size2 = num_ev2 * sizeof(event_t);
        size = size1 + size2;
        xmalloc(linear_buffer, size);
        memcpy (linear_buffer, head, size1);
        memcpy (linear_buffer+num_ev1, buffer->FirstEvt, size2);
    }
    *buffer = linear_buffer;
    return num_events;
}
#endif

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

	it->Buffer = buffer;
	it->OutOfBounds = Buffer_IsEmpty(buffer);
#if defined(DEBUG)
	fprintf(stderr, "[DBG_BUFFERS] new_Iterator: Buffer=%p, OutOfBounds=%d\n", it->Buffer, it->OutOfBounds);
#endif
	it->StartBound = Buffer_GetHead(buffer);
	it->EndBound = Buffer_GetTail(buffer);

	return it;
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

#if defined(MASKS_DEAD_CODE)
void BufferIterator_MaskIn (BufferIterator_t *it)
{
	ASSERT_VALID_BOUNDS(it);

	Buffer_MaskIn (it->Buffer, it->CurrentElement);
}

void BufferIterator_MaskOut (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);

    Buffer_MaskOut (it->Buffer, it->CurrentElement);
}

int BufferIterator_IsMaskedIn (BufferIterator_t *it)
{
	ASSERT_VALID_BOUNDS(it);

	return Buffer_IsMaskedIn (it->Buffer, it->CurrentElement);
}

int BufferIterator_IsMaskedOut (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);

    return Buffer_IsMaskedOut (it->Buffer, it->CurrentElement);
}
#endif /* MASKS_DEAD_CODE */

/********************************************************/
#if 1

void NewMask_Wipe (Buffer_t *buffer)
{
    memset (buffer->NewMasks, 0, buffer->MaxEvents * sizeof(Mask_t));
}

void NewMask_Set (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->NewMasks[index] |= mask_id;
}

void NewMask_SetAll (Buffer_t *buffer, event_t *event)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->NewMasks[index] = ALL_BITS_SET;	
}

void NewMask_SetRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id)
{
    NewMask_ChangeRegion (buffer, start, end, mask_id, 1);
}

void NewMask_Unset (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->NewMasks[index] &= ~mask_id;
}

void NewMask_UnsetAll (Buffer_t *buffer, event_t *event)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->NewMasks[index] = 0;	
}

void NewMask_UnsetRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id)
{
	NewMask_ChangeRegion (buffer, start, end, mask_id, 0);
}

static void NewMask_ChangeRegion (Buffer_t *buffer, event_t *start, event_t *end, int mask_id, int set)
{
	event_t *current = start;

    do
    {
		if (set)
			NewMask_Set (buffer, current, mask_id);
		else
			NewMask_Unset (buffer, current, mask_id);

		current = Buffer_GetNext (buffer, current);
    } while (current != end);

    /* Both range limits included */
	if (set)
		NewMask_Set (buffer, current, mask_id);
	else
		NewMask_Unset (buffer, current, mask_id);
}

void NewMask_Flip (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	buffer->NewMasks[index] ^= mask_id;
}

static int NewMask_Get (Buffer_t *buffer, event_t *event, int mask_id)
{
	int index = EVENT_INDEX(buffer, event);
	return ((buffer->NewMasks[index] & (mask_id)) == mask_id);
}

int NewMask_IsSet (Buffer_t *buffer, event_t *event, int mask_id)
{
	return (NewMask_Get (buffer, event, mask_id) == 1);
}

int NewMask_IsUnset (Buffer_t *buffer, event_t *event, int mask_id)
{
	return (NewMask_Get (buffer, event, mask_id) == 0);
}

void BufferIterator_MaskSet (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    NewMask_Set (it->Buffer, it->CurrentElement, mask_id);
}

void BufferIterator_MaskSetAll (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);
    NewMask_SetAll (it->Buffer, it->CurrentElement);
}

void BufferIterator_MaskUnset (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    NewMask_Unset (it->Buffer, it->CurrentElement, mask_id);
}

void BufferIterator_MaskUnsetAll (BufferIterator_t *it)
{
    ASSERT_VALID_BOUNDS(it);
    NewMask_UnsetAll (it->Buffer, it->CurrentElement);
}

int BufferIterator_IsMaskSet (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    return NewMask_IsSet (it->Buffer, it->CurrentElement, mask_id);
}

int BufferIterator_IsMaskUnset (BufferIterator_t *it, int mask_id)
{
    ASSERT_VALID_BOUNDS(it);
    return NewMask_IsUnset (it->Buffer, it->CurrentElement, mask_id);
}


#endif
