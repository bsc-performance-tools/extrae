/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2002 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

/* This file implements CRC-32 support optimized for hashing.  That is,
   the support attempts to minimize cache misses that would be very
   costly for computing CRC's over small amounts of data. */

#ifndef _gm_crc32_h_
#define _gm_crc32_h_

#include "gm.h"
#include "gm_cache_line.h"
#if GM_BUILDING_FIRMWARE
#include "gm_lanai_def.h"
#endif

/****************************************************************
 * Simple macros
 ****************************************************************/

/* The CRC32 generator polynomial. */
#define GM_CRC32_POLY 0x04C11DB7UL

/****************************************************************
 * Typedefs
 ****************************************************************/

/****************
 * Structure containing CRC-32 tables
 ****************/

/* Space reserved for software crc-32 tables.  If the tables need to be
   cache line alined, reserve space so we can round down the address of
   this structure to a cache line boundary. */

extern struct __gm_crc32_tables 
{
#if GM_NO_CACHE
  /* the LANais don't have caches so they don't worry about caching
     issues and always use 8-bit CRC table. */
  gm_u32_t table_8[1<<8];
#elif GM_CACHE_LINE_LEN == 128
  /* Hosts with 128 byte cache lines always compute the CRC 4 bits at
     a time with a 16-entry table that easily fits a single cache
     line.  A 32-entry table would also fit, but then we would be
     computing the CRC 5 bits at a time, and this doesn't map well to
     the user interface, which computes CRC-32 8, 16, or 32-bits at a
     time. */
  gm_u8_t pad [GM_CACHE_LINE_LEN];
  gm_u32_t table_4[1<<4];
#elif GM_CACHE_LINE_LEN == 32
  /* Hosts with 32 byte cache lines always compute the CRC 3 bits at a
     time, with an 8-entry table that just fits in a single cache
     line. */
  gm_u8_t pad [GM_CACHE_LINE_LEN];
  gm_u32_t table_3[1<<3];
#else
#error unsupported GM_CACHE_LINE_LEN
#endif /* GM_NO_CACHE */
} __gm_crc32_tables;

/****************
 * Macro for accessing CRC-32 tables, which may use hacks to ensure
 * cache alignment of the tables.
 ****************/

/* Access the crc32 table for N-bit-at-a-time
   computation.  The compiler will optimize the alignment operation to
   happen only once per function that uses this macro. */

#if GM_NO_CACHE
/* Simply access the table, since there is no cache, and therefore no
   cache alignment hack. */
#define GM_CRC32_TABLE_ENTRY(N,pos) (__gm_crc32_tables.table_ ## N[(pos)])
#else  /* !GM_NO_CACHE */
/* Access the table, with the position of the structure having been
   rounded down to a cache line boundary to ensure that none of the
   tables cross a cache line boundary unnecessarily. */
#define GM_CRC32_TABLE_ENTRY(N,pos)					\
(((struct __gm_crc32_tables *)						\
  ((gm_size_t) &__gm_crc32_tables & ~(GM_CACHE_LINE_LEN-1)))	\
 ->table_ ## N[(pos)])
#endif /* !GM_NO_CACHE */

/****************************************************************
 * Prototypes
 ****************************************************************/

extern void gm_crc32_test (void);
extern void gm_init_crc32 (void);

/****************************************************************
 * Generic accumulation of CRC32 in software.
 *
 * Imperically, we know that an in-cache table lookup has about the
 * same cost as computing a single bit of CRC without a table.  We
 * choose the implementations below to minimize the CRC-32 computation
 * time assuming the CRC-32 table is in L1 cache.  Since each table
 * fits within a single cache line, this seems reasonable.  At worst,
 * we will suffer a single cache miss.
 ****************************************************************/

/* Compute 1 bit of the CRC-32 without tables. */

#define GM_CRC32_COMPUTE_1_BIT(in, accum) do {				\
  (accum)								\
    = (((accum) << 1)							\
       ^ (((((in) >> (8*sizeof(in)-1)) & 1)				\
	  ^ (((accum) >> 31) & 1))					\
	  ? GM_CRC32_POLY : 0));					\
  (in) <<= 1;								\
} while (0)

/* Table-based CRC32 accumulation macro, computing N bits at a time,
   using the high N bits of "in." N must match the CRC table size for
   this compile, and "in" is shifted up N bytes, consuming the bits
   used in the computation. */

#define GM_CRC32_ACCUM(in, accum, N) do {				\
  (accum)								\
    = (((accum) << (N))							\
       ^ (GM_CRC32_TABLE_ENTRY						\
	  (N,								\
	   ((((accum)>>(32-(N))) ^ ((in)>>(8*sizeof(in)-(N))))		\
	    & ((1<<(N))-1)))));						\
  (in) <<= (N);								\
} while (0)

/****************************************************************
 * Inlines and static functions
 ****************************************************************/

/****************
 * 8, 16, and 32-bit CRC functions.
 *
 * Loops are deliberately NOT unrolled for the host to minimize cache
 * misses.  Loops are deliberately unrolled in the GM_NO_CACHE case,
 * making this faster, since there is no cache.
 ****************/

static inline
void
gm_crc32_u8 (gm_u8_t in, gm_u32_t *accum)
{
#if GM_NO_CACHE
  GM_CRC32_ACCUM (in, *accum, 8);
#elif GM_CACHE_LINE_LEN == 128
  {
    int i;
  
    for (i=2; i; i--)
      {
	GM_CRC32_ACCUM (in, *accum, 4);
      }
  }
#elif GM_CACHE_LINE_LEN == 32
  {
    int i;
  
    for (i=2; i; i--)
      {
	GM_CRC32_ACCUM (in, *accum, 3);
	GM_CRC32_COMPUTE_1_BIT (in, *accum);
      }
  }
#else
#error
#endif
}

static inline
void
gm_crc32_u16 (gm_u16_t in, gm_u32_t *accum)
{
#if GM_NO_CACHE
  {
    GM_CRC32_ACCUM (in, *accum, 8);
    GM_CRC32_ACCUM (in, *accum, 8);
  }
#elif GM_CACHE_LINE_LEN == 128
  {
    int i;

    for (i=0; i<4; i++)
      {
	GM_CRC32_ACCUM (in, *accum, 4);
      }
  }
#elif GM_CACHE_LINE_LEN == 32
  {
    int i;

    for (i=5; i; i--)
      {
	GM_CRC32_ACCUM (in, *accum, 3);
      }
    GM_CRC32_COMPUTE_1_BIT (in, *accum);
  }
#else
#error
#endif
}

static inline
void
gm_crc32_u32 (gm_u32_t in, gm_u32_t *accum)
{
#if GM_NO_CACHE
  {
    GM_CRC32_ACCUM (in, *accum, 8);
    GM_CRC32_ACCUM (in, *accum, 8);
    GM_CRC32_ACCUM (in, *accum, 8);
    GM_CRC32_ACCUM (in, *accum, 8);
  }
#elif GM_CACHE_LINE_LEN == 128
  {
    int i;

    for (i=0; i<8; i++)
      {
	GM_CRC32_ACCUM (in, *accum, 4);
      }
  }
#elif GM_CACHE_LINE_LEN == 32
  {
    int i;

    GM_CRC32_COMPUTE_1_BIT (in, *accum);
    GM_CRC32_COMPUTE_1_BIT (in, *accum);
    for (i=0; i<10; i++)
      {
	GM_CRC32_ACCUM (in, *accum, 3);
      }
  }
#else
#error
#endif
}

static inline
void
gm_crc32_size (gm_size_t in, gm_u32_t *accum)
{
  if (GM_SIZEOF_VOID_P == 8)
    {
      gm_assert (sizeof (void *) == sizeof (gm_size_t));
      gm_crc32_u32 ((gm_u32_t) (in >> (8 * sizeof (gm_size_t) / 2)), accum);
    }
  gm_crc32_u32 ((gm_u32_t) in, accum);
}

/****************************************************************
 * LANaiX CRC32 abstaction
 *
 * This interface abstracts the LANaiX CRC32 interface, so that it may
 * be used on earlier LANais, albeit with less performance than on the
 * LX.  In all cases, however, this achieves the best performance we
 * know how to achieve for the small CRC inputs that we use in
 * hashing.
 ****************************************************************/

/* Do we have a CRC-32 engine in hardware ? */
#define GM_CRC32_IN_HARDWARE (GM_BUILDING_FIRMWARE && LX)

#if GM_CRC32_IN_HARDWARE
#define GM_CRC32 CRC32		/* Use hardware CRC32 register. */
#else
extern gm_u32_t GM_CRC32;	/* Use a global as an accumulator. */
#endif

static inline
void
GM_CRC32_SET (gm_u32_t in)
{
#if GM_CRC32_IN_HARDWARE
  CRC32 = in;
#else
  GM_CRC32 = in;
#endif
}

static inline
gm_u32_t
GM_CRC32_GET (void)
{
#if GM_CRC32_IN_HARDWARE
  asm volatile ("nop\n\tnop");
  return CRC32;
#else
  return GM_CRC32;
#endif
}

static inline
void
GM_CRC32_BYTE (gm_u8_t in)
{
#if GM_CRC32_IN_HARDWARE
  CRC32_BYTE = in;
#else
  gm_crc32_u8 (in, &GM_CRC32);
#endif
}

static inline
void
GM_CRC32_HALF (gm_u16_t in)
{
#if GM_CRC32_IN_HARDWARE
  CRC32_HALF = in;
#else
  gm_crc32_u16 (in, &GM_CRC32);
#endif
}

static inline
void
GM_CRC32_WORD (gm_u32_t in)
{
#if GM_CRC32_IN_HARDWARE
  CRC32_WORD = in;
#else
  gm_crc32_u32 (in, &GM_CRC32);
#endif
}

#endif /* _gm_crc32_h_ */
