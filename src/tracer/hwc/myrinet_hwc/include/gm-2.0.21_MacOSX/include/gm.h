/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/**
   @file gm.h
   The official GM API include file.
 
   author: glenn@myri.com
*/

#ifndef _gm_h_
#define _gm_h_

/*# This file includes the minimum amount needed to support the GM API.
   It should not include any other header file or depend on any macro
   definitions that are not automatically set by the compiler. */

/************************************************************************
 * Begin Prologue
 ************************************************************************/

#ifdef	__cplusplus
extern "C"
{
#if 0
}				/* indent hack */
#endif
#endif

/****************
 * Compiler command line switch defaults.
 ****************/

#ifndef GM_BUILDING_GM
#define GM_BUILDING_GM 0
#define GM_BUILDING_FIRMWARE 0
#define GM_BUILDING_INTERNALS 0
#endif

#ifndef GM_KERNEL
#define GM_KERNEL 0
#endif

/****************
 * Includes
 ****************/

#if GM_BUILDING_FIRMWARE
#include "gm_config.h"
#else
/* There must be no inludes for user (a.k.a.: "GM client") compilations. */
#endif

/************************************************************************
 * End Prologue
 ************************************************************************/

/** Define API version numbers that may be used by the preprocessor. */

/** Hex equivalent of GM_API_VERSION_1_0 */
#define	GM_API_VERSION_1_0 0x100
/** Hex equivalent of GM_API_VERSION_1_1 */
#define	GM_API_VERSION_1_1 0x101
/** Hex equivalent of GM_API_VERSION_1_2 */
#define	GM_API_VERSION_1_2 0x102
/** Hex equivalent of GM_API_VERSION_1_3 */
#define	GM_API_VERSION_1_3 0x103
/** Hex equivalent of GM_API_VERSION_1_4 */
#define	GM_API_VERSION_1_4 0x104
/** Hex equivalent of GM_API_VERSION_1_5 */
#define	GM_API_VERSION_1_5 0x105
/** Hex equivalent of GM_API_VERSION_1_6 */
#define	GM_API_VERSION_1_6 0x106
/** Hex equivalent of GM_API_VERSION_2_0 */
#define	GM_API_VERSION_2_0 0x200
/** Hex equivalent of GM_API_VERSION_2_0_6 */
#define	GM_API_VERSION_2_0_6 0x20006
/** Hex equivalent of GM_API_VERSION_2_0_12 */
#define	GM_API_VERSION_2_0_12 0x2000c
/** Hex equivalent of GM_API_VERSION_2_0_16 */
#define	GM_API_VERSION_2_0_16 0x20010

/** Set the default API version used in this file. */

#ifndef GM_API_VERSION
#define GM_API_VERSION GM_API_VERSION_2_0_16
#endif

/* Extract the major, minor, and subminor parts of GM_API_VERSION. */

#define GM_API_MAJOR ((GM_API_VERSION & 0xffff) == GM_API_VERSION	\
		      ? GM_API_VERSION >> 8				\
		      : GM_API_VERSION >> 16)
#define GM_API_MINOR ((GM_API_VERSION & 0xffff) == GM_API_VERSION	\
		      ? GM_API_VERSION & 0xff				\
		      : (GM_API_VERSION >> 8) & 0xff)
#define GM_API_SUBMINOR ((GM_API_VERSION & 0xffff) == GM_API_VERSION	\
			 ? 0						\
			 : GM_API_VERSION & 0xff)

/* Check that the user has selected an API version compatible with this
   header file. */

#if GM_API_VERSION < GM_API_VERSION_1_0
#error GM_API_VERSION is too small.
#elif GM_API_VERSION > GM_API_VERSION_2_0_16
#error GM_API_VERSION is too large.
#endif

/****************************************************************
 * Prevent direct access to these functions in GM intself, so that we
 * can check for memory leaks.
 ****************************************************************/

#if GM_BUILDING_GM
#include "gm_debug_malloc.h"
#endif

/** For htonl, etc. */

/****
 * Constants
 ****/

/** Maximum length of GM host name */
#define GM_MAX_HOST_NAME_LEN 128
/** Maximum length of GM port name */
#define GM_MAX_PORT_NAME_LEN 32
/** No such GM node id */
#define GM_NO_SUCH_NODE_ID 0

/************
 * Determine CPU based on what compiler is being used.
 ************/

#if defined GM_CPU_alpha
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_lanai
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_mips
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_powerpc
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_powerpc64
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_sparc
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_sparc64
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_x86
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_x86_64
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_hppa
#  define GM_CPU_DEFINED 1
#elif defined GM_CPU_ia64
#  define GM_CPU_DEFINED 1
#else
#  define GM_CPU_DEFINED 0
#endif

#if !GM_CPU_DEFINED
#  if defined _MSC_VER		/* Microsoft compiler */
#    if defined _M_IX86
#      define GM_CPU_x86 1
#    elif defined _M_IA64
#      define GM_CPU_ia64 1
#    elif defined _M_AMD64
#      define GM_CPU_x86_64 1
#    elif defined _M_ALPHA
#      define GM_CPU_alpha 1
#    else
#      error Could not determine CPU type.  You need to modify gm.h.
#    endif
#  elif defined __APPLE_CC__	/* Apple OSX compiler defines __GNUC__ */
#    if defined __ppc__ 	/* but doesn't support #cpu syntax     */
#      define GM_CPU_powerpc 1
#    elif define __i386__
#      define GM_CPU_x86 1
#    else
#      error Could not determine CPU type.  You need to modify gm.h.
#    endif
#  elif defined mips
#    define GM_CPU_mips 1
#  elif defined(__GNUC__)
#    if #cpu(alpha)
#      define GM_CPU_alpha 1
#    elif #cpu(hppa)
#      define GM_CPU_hppa 1
#    elif defined lanai
#      define GM_CPU_lanai 1
#    elif defined lanai3
#      define GM_CPU_lanai 1
#    elif defined lanai7
#      define GM_CPU_lanai 1
#    elif defined(powerpc64)
#      define GM_CPU_powerpc64 1
#    elif defined(__powerpc64__)
#      define GM_CPU_powerpc64 1
#    elif defined(__ppc__)
#      define GM_CPU_powerpc 1
#    elif defined(__powerpc__)
#      define GM_CPU_powerpc 1
#    elif defined(powerpc)
#      define GM_CPU_powerpc 1
#    elif defined(_POWER)
#      define GM_CPU_powerpc 1
#    elif defined(_IBMR2)
#      define GM_CPU_powerpc 1
#    elif #cpu(ia64)
#      define GM_CPU_ia64 1
#    elif #cpu(sparc64)
#      define GM_CPU_sparc64 1
#    elif defined sparc
#      define GM_CPU_sparc 1
#    elif defined __i386
#      define GM_CPU_x86 1
#    elif defined i386
#      define GM_CPU_x86 1
#    elif #cpu(x86_64)
#      define GM_CPU_x86_64 1
#    elif defined(CPU)   /* This is how vxWorks defines their CPUs */
#      if (CPU==PPC603)
#	 define GM_CPU_powerpc 1
#      elif (CPU==PPC604)
#	 define GM_CPU_powerpc 1
#      elif (CPU==PPC405)
#        define GM_CPU_powerpc 1
#      else
#        error Could not determine CPU type.  If this is VxWorks, you will need to modify gm.h to add your cpu type.
#      endif
#    else
#      error Could not determine CPU type.  You need to modify gm.h.
#    endif
#  elif (defined __powerpc64__)
#      define GM_CPU_powerpc64 1
#  elif (defined (_POWER) && defined(_AIX))
#      define GM_CPU_powerpc 1
#  elif (defined __powerpc__)
#      define GM_CPU_powerpc 1
#  elif (defined (__DECC) || defined (__DECCXX)) && defined(__alpha)
#      define GM_CPU_alpha 1
#  elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#    if defined(sparc64) || defined(__sparcv9)
#      define GM_CPU_sparc64 1
#    elif defined(sparc)
#      define GM_CPU_sparc 1
#    elif defined(__x86_64__)
#      define GM_CPU_x86_64 1
#    elif defined i386
#      define GM_CPU_x86 1
#    else
#      error Could not determine CPU type for SunPro C compiler.
#    endif
#  elif defined __PGI 
#    if defined  __x86_64__
#      define GM_CPU_x86_64 1
#    else	
#      define GM_CPU_x86 1
#    endif
#  elif defined __linux__	/* Portland Group Compiler */
#    if defined __i386__
#      define GM_CPU_x86 1
#    elif defined __ia64__
#      define GM_CPU_ia64 1
#    elif defined  __x86_64__
#      define GM_CPU_x86_64 1
#    else
#      error Could not determine CPU type.  You need to modify gm.h.
#    endif
#  elif defined(__hppa) || defined(_PA_RISC1_1)
#      define GM_CPU_hppa 1
#  else
#    error Could not determine CPU type.  You need to modify gm.h.
#  endif
#  undef GM_CPU_DEFINED
#  define GM_CPU_DEFINED 1
#endif

/** Define all undefined GM_CPU switches to 0 to prevent problems
   with "gcc -Wundef" */

#ifndef GM_CPU_alpha
#define GM_CPU_alpha 0
#endif
#ifndef GM_CPU_ia64
#define GM_CPU_ia64 0
#endif
#ifndef GM_CPU_hppa
#define GM_CPU_hppa 0
#endif
#ifndef GM_CPU_lanai
#define GM_CPU_lanai 0
#endif
#ifndef GM_CPU_mips
#define GM_CPU_mips 0
#endif
#ifndef GM_CPU_powerpc
#define GM_CPU_powerpc 0
#endif
#ifndef GM_CPU_powerpc64
#define GM_CPU_powerpc64 0
#endif
#ifndef GM_CPU_sparc
#define GM_CPU_sparc 0
#endif
#ifndef GM_CPU_sparc64
#define GM_CPU_sparc64 0
#endif
#ifndef GM_CPU_x86
#define GM_CPU_x86 0
#endif
#ifndef GM_CPU_x86_64
#define GM_CPU_x86_64 0
#endif

/************
 * Enable inlining if and only if we know it won't crash the compiler.
 ************/


#ifndef gm_inline
#  if defined _MSC_VER
#    define gm_inline __inline
#  elif defined __GNUC__
#    define gm_inline __inline__
#  elif GM_CPU_mips
#    define gm_inline __inline
#    define inline __inline
#  elif defined __DECC
#    define gm_inline __inline
#  elif defined __DECCXX
#    define gm_inline
#  else
#    define gm_inline
#  endif
#endif

/************
 * Define sized types
 ************/

/* gm_s64_t   64-bit signed   integer
   gm_s32_t   32-bit signed   integer
   gm_s16_t   16-bit signed   integer
   gm_s8_t     8-bit signed   integer
   gm_u64_t   64-bit unsigned integer
   gm_u32_t   32-bit unsigned integer
   gm_u16_t   16-bit unsigned integer
   gm_u8_t     8-bit unsigned integer */

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__DECC) || defined(__DECCXX) || defined (__IBMC__) || defined (__IBMCPP__)
#if ((GM_CPU_sparc64 || GM_CPU_powerpc64) && defined(__linux__)) && !GM_BUILDING_FIRMWARE
#if GM_KERNEL
typedef signed long gm_s64_t;
typedef unsigned long gm_u64_t;
#else
typedef signed long long gm_s64_t;
typedef unsigned long long gm_u64_t;
#endif /* GM_KERNEL */
#else
#if (GM_CPU_alpha || GM_CPU_sparc64) && !GM_BUILDING_FIRMWARE
typedef signed long gm_s64_t;
typedef unsigned long gm_u64_t;
#else
typedef signed long long gm_s64_t;
typedef unsigned long long gm_u64_t;
#endif
#endif

/** gm_s32_t is a 32-bit signed integer. */
typedef signed int gm_s32_t;
/** gm_s16_t is a 16-bit signed integer. */
typedef signed short gm_s16_t;
/** gm_s8_t is an 8-bit signed integer. */
typedef signed char gm_s8_t;
/** gm_u32_t is a 32-bit unsigned integer. */
typedef unsigned int gm_u32_t;
/** gm_u16_t is a 16-bit unsigned integer. */
typedef unsigned short gm_u16_t;
/** gm_u8_t is an 8-bit unsigned integer. */
typedef unsigned char gm_u8_t;
#elif defined _MSC_VER
typedef signed __int64 gm_s64_t;
typedef unsigned __int64 gm_u64_t;
typedef signed __int32 gm_s32_t;
typedef signed __int16 gm_s16_t;
typedef signed __int8 gm_s8_t;
typedef unsigned __int32 gm_u32_t;
typedef unsigned __int16 gm_u16_t;
typedef unsigned __int8 gm_u8_t;
#elif GM_CPU_mips		/* see /usr/include/sgidefs.h */
#if (_MIPS_SZLONG == 64)
typedef long gm_s64_t;
typedef unsigned long gm_u64_t;
#elif defined(_LONGLONG)
typedef long long gm_s64_t;
typedef unsigned long long gm_u64_t;
#else
/* __long_long is a hidden builtin, ansi-compliant 64-bit type.  It
   should be used only here; henceforward, the gm_ types should be used. */
typedef __long_long gm_s64_t;
typedef unsigned __long_long gm_u64_t;
#endif
typedef int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif defined __PGI || defined(i386) || defined(__i386)		/* Need cpu macros? */
typedef signed long long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC)) && (defined(sparc64) || defined(__sparcv9))
typedef signed long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC)) && defined(__sparc)
typedef signed long long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC)) && defined __x86_64__
typedef signed long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC)) && defined i386
typedef signed long long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif defined(__hpux)
typedef signed long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#elif defined(__AIX__) || defined(_AIX)
typedef signed long long gm_s64_t;
typedef signed int gm_s32_t;
typedef signed short gm_s16_t;
typedef signed char gm_s8_t;
typedef unsigned long long gm_u64_t;
typedef unsigned int gm_u32_t;
typedef unsigned short gm_u16_t;
typedef unsigned char gm_u8_t;
#else
#  error Could not define sized types.  You need to modify gm.h.
#endif

/************
 * LANai-compatible host-side pointer representation
 ************/

/* Define size of host pointers */

#if GM_BUILDING_FIRMWARE
#  define GM_SIZEOF_VOID_P 4
#elif defined(__AIX__) || defined(_AIX)
#    if defined(__64BIT__)
#      define GM_SIZEOF_VOID_P 8
#    else
#      define GM_SIZEOF_VOID_P 4
#    endif
#else
#  if GM_CPU_sparc64 && defined(__linux__) && !GM_KERNEL
#    define GM_SIZEOF_VOID_P 4
#  elif GM_CPU_sparc64
#    define GM_SIZEOF_VOID_P 8
#  elif GM_CPU_sparc
#    define GM_SIZEOF_VOID_P 4
#  elif GM_CPU_x86
#    define GM_SIZEOF_VOID_P 4
#  elif GM_CPU_x86_64
#    define GM_SIZEOF_VOID_P 8
#  elif GM_CPU_ia64
#    define GM_SIZEOF_VOID_P 8
#  elif GM_CPU_alpha
#    define GM_SIZEOF_VOID_P 8
#  elif GM_CPU_mips
#    ifdef _MIPS_SZPTR
#      define GM_SIZEOF_VOID_P (_MIPS_SZPTR / 8)
#    else
#      error Failed to define _MIPS_SZPTR
#    endif
#  elif GM_CPU_powerpc
#    define GM_SIZEOF_VOID_P 4
#  elif GM_CPU_powerpc64
#    define GM_SIZEOF_VOID_P 8
#  elif GM_CPU_hppa
#    define GM_SIZEOF_VOID_P 8
#  else
#    error Could not determine host pointer size.  You need to modify gm.h.
#  endif
#endif

/* GM_SIZEOF_UP_T */

/****************
 * GM_SIZEOF_UP_T: GM user pointers
 *
 * This is a tricky definition.  From the point of view of the GM
 * internals (the firmware, libgm, the driver, and certain debug
 * programs) all user pointers are the maximum size supported on the
 * system, and GM_SIZEOF_UP_T is defined by the build environment.
 *
 * From the point of view of the GM client application, gm_up_t's are
 * the same size as a "void *" in that application by default.
 *
 * Note that in structure in the GM API, there is a 32-bit pad before
 * any 32-bit user pointer to make GM's internal and external layouts
 * for these structures be compatible.
 ****************/

#if GM_BUILDING_INTERNALS

#ifndef GM_SIZEOF_UP_T
#error GM_SIZEOF_UP_T should be defined on the compiler command line
#endif

#else  /* ndef GM_BUILDING_INTERNALS */

#define GM_SIZEOF_UP_T GM_SIZEOF_VOID_P

#endif /* ndef GM_BUILDING_INTERNALS */

/** The "gm_up_t" is a LANai-compatible representation of a user
   virtual memory address.  If the host has 32-bit pointers, then a
   gm_up_t has 32 bits.  If the host has 64-bit pointers, but only
   32-bits of these pointers are used, then a gm_up_t still has 32
   bits (for performance reasons).  Finally, if the host has 64-bit
   pointers and more than 32 bits of the pointer is used, then a
   gm_up_t has 64-bits. */


/* Define host pointer types

   gm_up_t   unsigned int the size of a user pointer */

#if GM_SIZEOF_UP_T == 4
typedef gm_u32_t gm_up_t;
#elif GM_SIZEOF_UP_T == 8
typedef gm_u64_t gm_up_t;
#else
#  error Host pointer size is not supported.
#endif

typedef gm_u64_t gm_remote_ptr_t;	/* assume remote pointers are large */

/****************************************************************
 * Endianess
 ****************************************************************/

/* Determine endianness */

#if GM_CPU_alpha
#  define GM_CPU_BIGENDIAN 0
#elif GM_CPU_lanai
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_mips
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_powerpc
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_powerpc64
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_sparc
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_sparc64
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_hppa
#  define GM_CPU_BIGENDIAN 1
#elif GM_CPU_x86
#  define GM_CPU_BIGENDIAN 0
#elif GM_CPU_x86_64
#  define GM_CPU_BIGENDIAN 0
#elif GM_CPU_ia64
#  define GM_CPU_BIGENDIAN 0
#else
#  error Could not determine endianness.  You need to modify gm.h.
#endif

/****************
 * Network-order types, strongly typed on the host
 ****************/

/* The default behaviour is to turn on strong typing, except when
   using 64-bit Solaris compilers, which will pad the strong types,
   breaking this feature.
   loic: SUN compiler generates call to .stretx for functions returning
   structures, that leads to undefined symbols for kernel code,
   so I disabled strong typing even for 32bit with SUN compilers */

#ifndef GM_STRONG_TYPES
#if (defined __SUNPRO_C || defined __SUNPRO_CC) && (defined __sparc || defined __sparcv9)
#define GM_STRONG_TYPES 0
#else
#define GM_STRONG_TYPES 1
#endif
#endif /* ndef GM_STRONG_TYPES */

/* Prevent usage of strong types on 64-bit Solaris machines.  These are
   big-endian anyway, so no one should need it there. */

#if GM_STRONG_TYPES && (defined __SUNPRO_C || defined __SUNPRO_CC) && (defined __sparc || defined __sparcv9)
#error GM_STRONG_TYPES is incompatible with Solaris compilers
#endif

#if !GM_BUILDING_FIRMWARE && GM_STRONG_TYPES

/** A pointer to memory on a (potentially) remote machine.  Such pointers
    are always 64-bits, since we don't know if the remote pointer is 32-
    or 64-bits. */
typedef struct
{
  gm_remote_ptr_t n;
}
gm_remote_ptr_n_t;
/** A 16-bit signed value in network byte order. */
typedef struct
{
  gm_s16_t n;
}
gm_s16_n_t;
/** A 32-bit signed value in network byte order. */
typedef struct
{
  gm_s32_t n;
}
gm_s32_n_t;
/** A 64-bit signed value in network byte order. */
typedef struct
{
  gm_s64_t n;
}
gm_s64_n_t;
/** A 8-bit signed value in network byte order.  (Silly, I know.) */
typedef struct
{
  gm_s8_t n;
}
gm_s8_n_t;
/** A 16-bit unsigned value in network byte order. */
typedef struct
{
  gm_u16_t n;
}
gm_u16_n_t;
/** A 32-bit unsigned value in network byte order. */
typedef struct
{
  gm_u32_t n;
}
gm_u32_n_t;
/** A 64-bit unsigned value in network byte order. */
typedef struct
{
  gm_u64_t n;
}
gm_u64_n_t;
/** An byte value in network byte order.  (Silly, I know.) */
typedef struct
{
  gm_u8_t n;
}
gm_u8_n_t;
/** A user-space pointer in network byte order.  (On systems
    supporting multiple pointer sizes, this is large enough to store
    the largest pointer size, even if the process is using smaller
    pointers.) */
typedef struct
{
  gm_up_t n;
}
gm_up_n_t;
typedef gm_up_n_t gm_up_n_up_n_t;
#define GM_N(x) ((x).n)

#else /* GM_BUILDING_FIRMWARE || !GM_STRONG_TYPES */

typedef gm_remote_ptr_t gm_remote_ptr_n_t;
typedef gm_s16_t gm_s16_n_t;
typedef gm_s32_t gm_s32_n_t;
typedef gm_s64_t gm_s64_n_t;
typedef gm_s8_t gm_s8_n_t;
typedef gm_u16_t gm_u16_n_t;
typedef gm_u32_t gm_u32_n_t;
typedef gm_u64_t gm_u64_n_t;
typedef gm_u8_t gm_u8_n_t;
typedef gm_up_t gm_up_n_t;
typedef gm_up_t gm_up_n_up_n_t;
#define GM_N(x) (x)

#endif /* GM_BUILDING_FIRMWARE || !GM_STRONG_TYPES*/

/****************************************************************
 * Byte order conversion
 ****************************************************************/

/********************************
 * Unconditional conversion
 ********************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/* Define __gm_swap* macros, which always reverse the order of bytes. */

gm_inline static gm_u8_t
__gm_swap_u8 (gm_u8_t x)
{
  return x;
}

gm_inline static gm_u16_t
__gm_swap_u16 (gm_u16_t x)
{
  return (((x >> 8) & 0xff) | ((x & 0xff) << 8));
}

/* Architecture-independent C implementation. */

gm_inline static gm_u32_t
__gm_swap_u32_C (gm_u32_t x)
{
  return (((x >> 24) & 0xff)
	  | ((x >> 8) & 0xff00) | ((x & 0xff00) << 8) | ((x & 0xff) << 24));
}

/* Architecture-specific assembly implementations */

gm_inline static gm_u32_t
__gm_swap_u32_asm (gm_u32_t x)
{
#if defined __GNUC__ && GM_CPU_x86 && defined __OPTIMIZE__ && __OPTIMIZE__ && 0
__asm__ ("bswap %0": "=r" (x):"0" (x));
  return x;
#else
  return __gm_swap_u32_C (x);
#endif
}

gm_inline static gm_u32_t
__gm_swap_u32 (gm_u32_t x)
{
#ifdef __GNUC__
  /* Use C implementation for constants to allow compile-time swapping.
     Otherwise, use fast assembly implementation. */
  return __builtin_constant_p (x) ? __gm_swap_u32_C (x) :
    __gm_swap_u32_asm (x);
#else
  return __gm_swap_u32_C (x);
#endif
}

/* These unconditional conversion macros are used by the conditional
   conversion macros below */

gm_inline static gm_u64_t
__gm_swap_u64 (gm_u64_t x)
{
  volatile union
  {
    gm_u64_t u64;
    gm_u32_t u32[2];
  }
  ret, old;

  old.u64 = x;
  ret.u32[0] = __gm_swap_u32 (old.u32[1]);
  ret.u32[1] = __gm_swap_u32 (old.u32[0]);
  return ret.u64;
}

/****************************************************************
 * Strongly typed conversion
 ****************************************************************/

/********************************
 * Conditional conversion
 ********************************/

#if GM_CPU_BIGENDIAN || GM_BUILDING_FIRMWARE
#define GM_NET_SWAP 0
#else
#define GM_NET_SWAP 1
#endif

/* Swap using weak typing. */

#if GM_NET_SWAP
#define __GM_NET_SWAP(type,size,x) \
((gm##type##size##_t) __gm_swap_u##size ((gm_u##size##_t) (x)))
#else
#define __GM_NET_SWAP(type,size,x) ((gm##type##size##_t) (x))
#endif

#if GM_NET_SWAP
#define __GM_PCI_SWAP(type,size,x) ((gm##type##size##_t) (x))
#else
#define __GM_PCI_SWAP(type,size,x) \
((gm##type##size##_t) __gm_swap_u##size ((gm_u##size##_t) (x)))
#endif

/* These gm_ntoh* macros perform strong type checking, allowing the
   compiler to ensure that no network-ordered fields are used by host
   code without first being converted to host byte order by these
   functions.  The only way to circumvent this strong checking is
   using memory copies or complicated casts.  The compiler will catch
   simple casts. */

/****************
 * network to host
 ****************/

/* Macro to define gm_ntoh_u8(), etc. */

#define _GM_NTOH(type, size)						\
									\
/* swap using weak typing */						\
									\
gm_inline static							\
gm##type##size##_t							\
__gm_ntoh##type##size (gm##type##size##_t x)				\
{									\
  return __GM_NET_SWAP (type, size, x);					\
}									\
									\
/* swap using strong typing */						\
									\
gm_inline static							\
gm##type##size##_t							\
_gm_ntoh##type##size (gm##type##size##_n_t x)				\
{									\
  return __gm_ntoh##type##size (GM_N (x));				\
}									\
									\
struct gm_ignore_the_semicolon

_GM_NTOH (_u, 8);
_GM_NTOH (_u, 16);
_GM_NTOH (_u, 32);
_GM_NTOH (_u, 64);

_GM_NTOH (_s, 8);
_GM_NTOH (_s, 16);
_GM_NTOH (_s, 32);
_GM_NTOH (_s, 64);

/****************
 * host to network
 ****************/

/* Macro to define gm_hton_u8(), etc. */

#define _GM_HTON(type, size)						\
									\
/* swap using weak typing */						\
									\
gm_inline static							\
gm##type##size##_t							\
__gm_hton##type##size (gm##type##size##_t x)				\
{									\
  return __GM_NET_SWAP (type, size, x);					\
}									\
									\
/* swap using strong typing */						\
									\
gm_inline static							\
gm##type##size##_n_t							\
_gm_hton##type##size (gm##type##size##_t x)				\
{									\
  gm##type##size##_n_t ret;						\
									\
  GM_N (ret) = __gm_hton##type##size (x);				\
  return ret;								\
}

_GM_HTON (_u, 8)
_GM_HTON (_u, 16)
_GM_HTON (_u, 32)
_GM_HTON (_u, 64)

_GM_HTON (_s, 8)
_GM_HTON (_s, 16)
_GM_HTON (_s, 32)
_GM_HTON (_s, 64)

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/********************************
 * host/network conversion macros.
 ********************************/

/* These are just like the "^_gm" versions above, but exist to give
   etags something to jump to.  The "_gm" versions above are in a macro
   to avoid code replication. */

gm_inline static gm_u64_t
gm_ntoh_u64 (gm_u64_n_t x)
{
  return _gm_ntoh_u64 (x);
}
gm_inline static gm_s64_t
gm_ntoh_s64 (gm_s64_n_t x)
{
  return _gm_ntoh_s64 (x);
}
gm_inline static gm_u32_t
gm_ntoh_u32 (gm_u32_n_t x)
{
  return _gm_ntoh_u32 (x);
}
gm_inline static gm_s32_t
gm_ntoh_s32 (gm_s32_n_t x)
{
  return _gm_ntoh_s32 (x);
}
gm_inline static gm_u16_t
gm_ntoh_u16 (gm_u16_n_t x)
{
  return _gm_ntoh_u16 (x);
}
gm_inline static gm_s16_t
gm_ntoh_s16 (gm_s16_n_t x)
{
  return _gm_ntoh_s16 (x);
}
gm_inline static gm_u8_t
gm_ntoh_u8 (gm_u8_n_t x)
{
  return _gm_ntoh_u8 (x);
}
gm_inline static gm_s8_t
gm_ntoh_s8 (gm_s8_n_t x)
{
  return _gm_ntoh_s8 (x);
}

gm_inline static gm_u64_n_t
gm_hton_u64 (gm_u64_t s)
{
  return _gm_hton_u64 (s);
}
gm_inline static gm_s64_n_t
gm_hton_s64 (gm_s64_t s)
{
  return _gm_hton_s64 (s);
}
gm_inline static gm_u32_n_t
gm_hton_u32 (gm_u32_t s)
{
  return _gm_hton_u32 (s);
}
gm_inline static gm_s32_n_t
gm_hton_s32 (gm_s32_t s)
{
  return _gm_hton_s32 (s);
}
gm_inline static gm_u16_n_t
gm_hton_u16 (gm_u16_t s)
{
  return _gm_hton_u16 (s);
}
gm_inline static gm_s16_n_t
gm_hton_s16 (gm_s16_t s)
{
  return _gm_hton_s16 (s);
}
gm_inline static gm_u8_n_t
gm_hton_u8 (gm_u8_t c)
{
  return _gm_hton_u8 (c);
}
gm_inline static gm_s8_n_t
gm_hton_s8 (gm_s8_t c)
{
  return _gm_hton_s8 (c);
}

/* Macro for accessing words in the little endian PCI config space */

gm_inline static gm_u32_t
gm_htopci_u32 (gm_u32_t x)
{
  return __GM_PCI_SWAP (_u, 32, x);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS 

#if !GM_BUILDING_FIRMWARE
gm_inline static void *
gm_ntohp (gm_up_n_t x)
{
  /* Odd syntax here to placate "icc -Wall" (Intel cc) */
#if GM_SIZEOF_UP_T == 8
  return (void *) ((char *) 0 + __gm_ntoh_u64 (GM_N (x)));
#elif GM_SIZEOF_UP_T == 4
  return (void *) ((char *) 0 + __gm_ntoh_u32 (GM_N (x)));
#endif
}
#endif

/****************
 * backwards compatibility
 ****************/

/** byte order macros: gm_ntoh* and gm_hton*, where
 ** "c" a 8-bit int (char)
 ** "s" a 16-bit int (short)
 ** "l" a 32-bit int (long)
 ** "ll" a 64-bit int (long long)
 ** "_dp" a 32-bit DMA pointer.
 ** "_lp" a 32-bit LANai pointer.
 ** "_up" a lanai-compatible representation of a user virtual memory addr. */

#if !GM_BUILDING_GM

/* These aliases are for client use only */

gm_inline static gm_u8_t
gm_ntohc (gm_u8_n_t x)
{
  return gm_ntoh_u8 (x);
}
gm_inline static gm_u16_t
gm_ntohs (gm_u16_n_t x)
{
  return gm_ntoh_u16 (x);
}
gm_inline static gm_u32_t
gm_ntohl (gm_u32_n_t x)
{
  return gm_ntoh_u32 (x);
}
gm_inline static gm_u64_t
gm_ntohll (gm_u64_n_t x)
{
  return gm_ntoh_u64 (x);
}

gm_inline static gm_u8_n_t
gm_htonc (gm_u8_t x)
{
  return gm_hton_u8 (x);
}
gm_inline static gm_u16_n_t
gm_htons (gm_u16_t x)
{
  return gm_hton_u16 (x);
}
gm_inline static gm_u32_n_t
gm_htonl (gm_u32_t x)
{
  return gm_hton_u32 (x);
}
gm_inline static gm_u64_n_t
gm_htonll (gm_u64_t x)
{
  return gm_hton_u64 (x);
}

#define gm_assert_p(a) gm_assert (a)

#endif /* !GM_BUILDING_GM */

/************
 * GM_ENTRY_POINT definition
 ************/

#ifndef GM_ENTRY_POINT		/* need to override in NT4 make-os.in */

#if defined _MSC_VER		/* microsoft compiler */
#  if GM_KERNEL
#    define GM_ENTRY_POINT
#  elif GM_BUILDING_GM_LIB
#    define GM_ENTRY_POINT __declspec (dllexport)
#  else
#    define GM_ENTRY_POINT __declspec (dllimport)
#  endif
#else
#  define GM_ENTRY_POINT
#endif

#endif /* ndef GM_ENTRY_POINT */

/* Debugging */

#ifndef GM_DEBUG
#  if !GM_BUILDING_GM
#    define GM_DEBUG 0
#  else  /* GM_BUILDING_GM */
#    error GM_DEBUG must be defined for internal GM builds.
#  endif /* GM_BUILDING_GM */
#endif /* !GM_DEBUG */

/* A macro like "gm_assert()", only causing an error at compile-time
   rather than run time.  The expression A must be a constant
   expression.  */

#if !defined __GNUC__ || defined __STRICT_ANSI__
#define __GM_COMPILE_TIME_ASSERT(a)
#else
#define __GM_COMPILE_TIME_ASSERT(a) do {				\
  char (*__GM_COMPILE_TIME_ASSERT_var)[(a) ? 1 : -1] = 0;		\
  (void) __GM_COMPILE_TIME_ASSERT_var; /* prevent unused var warning */	\
} while (0)
#endif

/****************
 * gm_always_assert() and gm_assert()
 ****************/

/* Macro for determining if X is a constant that can be evaluated at
   compile time.  This macro is used by gm_always_assert*() to perform as many
   compile-time checks as possible. */

#if defined __GNUC__ && !defined __STRICT_ANSI__
#define __gm_builtin_constant_p(x) __builtin_constant_p (x)
#else
#define __gm_builtin_constant_p(x) 0
#endif

/* __gm_check_syntax(): (Undocumented, unsupported.) Allow the
   compiler to check syntax, on the expression, but don't actually
   execute it. */

#define __gm_check_syntax(x) do {if (0) (void) (x);} while (0)

/* gm_assert() Assert iff debugging turned on. */

#if GM_DEBUG
#define gm_assert(a) _gm_always_assert (a, #a)
#else
#define gm_assert(a) __gm_check_syntax (a)
#endif

#if GM_BUILDING_FIRMWARE && 0
#define __GM_ASSERTION_FAILED(txt, line, file) do {			\
  _gm_assertion_failed ("", line, file);				\
} while (0)
#else
#define __GM_ASSERTION_FAILED(txt, line, file) do {			\
  _gm_assertion_failed (txt, line, file);				\
} while (0)
#endif

/* Default file name for debugging prints.  GM builds' preprocessing
   scripts supply their own definition, overridding the default with
   a shorter string to save memory. */

#ifndef __GM_FILE__
#define __GM_FILE__ __FILE__
#endif

/* Special always assert macro with preexpanded assertion string.
   This prevents the preprocessor from expanding the expression before
   converting it to a string. */

#define _gm_always_assert(a,txt) do {					\
  __GM_COMPILE_TIME_ASSERT (!__gm_builtin_constant_p (a) || (a));	\
  if (!(a))								\
    {									\
      __GM_ASSERTION_FAILED (txt, __LINE__, __GM_FILE__);		\
    }									\
} while (0)

/* gm_always_assert(): Always assert, even if debugging is turned off,
   unless we are able to to the check at compile-time. */

#define gm_always_assert(a) _gm_always_assert (a, #a)

#define GM_ABORT() __GM_ASSERTION_FAILED ("0", __LINE__, __GM_FILE__)

/****************
 * Compiler warning supression macros
 ****************/

#define GM_PARAMETER_MAY_BE_UNUSED(p) ((void)(p))
#define GM_VAR_MAY_BE_UNUSED(v) ((void)(v))
#define GM_LOOPS_FOREVER() { if (0) break; }

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/****************************************************************
 * typedefs
 ****************************************************************/

/** GM Send Completion Status codes */ 
typedef enum gm_status
{
  /** The send succeeded.  This status code does not indicate an error. */
  GM_SUCCESS = 0,
  /** Operation Failed */
  GM_FAILURE = 1,
  /** Input buffer is too small */
  GM_INPUT_BUFFER_TOO_SMALL = 2,
  /** Output buffer is too small */
  GM_OUTPUT_BUFFER_TOO_SMALL = 3,
  /** Try Again */
  GM_TRY_AGAIN = 4,
  /** GM Port is Busy */
  GM_BUSY = 5,
  /** Memory Fault */
  GM_MEMORY_FAULT = 6,
  /** Interrupted */
  GM_INTERRUPTED = 7,
  /** Invalid input parameter */
  GM_INVALID_PARAMETER = 8,
  /** Out of Memory */
  GM_OUT_OF_MEMORY = 9,
  /** Invalid Command */
  GM_INVALID_COMMAND = 10,
  /** Permission Denied */
  GM_PERMISSION_DENIED = 11,
  /** Internal Error */
  GM_INTERNAL_ERROR = 12,
  /** Unattached */
  GM_UNATTACHED = 13,
  /** Unsupported Device */
  GM_UNSUPPORTED_DEVICE = 14,
  /** The target port is open and responsive and the message is of an
     acceptable size, but the receiver failed to provide a matching receive
     buffer within the timeout period. This error can be caused by the
     receive neglecting its responsibility to provide receive buffers in a
     timely fashion or crashing.  It can also be caused by severe congestion
     at the receiving node where many senders are contending for the same
     receive buffers on the target port for an extended period.  This error
     indicates a programming error in the client software. */
  GM_SEND_TIMED_OUT = 15,
  /** The receiver indicated (in a call to gm_set_acceptable_sizes()) the
     size of the message was unacceptable.  This error indicates a
     programming error in the client software. */
  GM_SEND_REJECTED = 16,
  /** The message cannot be delivered because the destination port has been
     closed. */
  GM_SEND_TARGET_PORT_CLOSED = 17,
  /** The target node could not be reached over the Myrinet.  This error can
     be caused by the network becoming disconnected for too long, the remote
     node being powered off, or by network links being rearranged when the
     Myrinet mapper is not running. */
  GM_SEND_TARGET_NODE_UNREACHABLE = 18,
  /**  The send was dropped at the client's request.  (The client called
     gm_drop_sends().)  This status code does not indicate an error. */
  GM_SEND_DROPPED = 19,
  /** Clients should never see this internal error code. */
  GM_SEND_PORT_CLOSED = 20,
  /** Node ID is not yet set */
  GM_NODE_ID_NOT_YET_SET = 21,
  /** GM Port is still shutting down */
  GM_STILL_SHUTTING_DOWN = 22,
  /** GM Clone Busy */
  GM_CLONE_BUSY = 23,
  /** No such device */
  GM_NO_SUCH_DEVICE = 24,
 /** Aborted. */
  GM_ABORTED = 25,
#if GM_API_VERSION >= GM_API_VERSION_1_5
  /** Incompatible GM library and driver */
  GM_INCOMPATIBLE_LIB_AND_DRIVER = 26,
  /** Untranslated System Error */
  GM_UNTRANSLATED_SYSTEM_ERROR = 27,
  /** Access Denied */
  GM_ACCESS_DENIED = 28,
#endif
#if GM_API_VERSION >= GM_API_VERSION_2_0
  /** No Driver Support */
  GM_NO_DRIVER_SUPPORT = 29,
  /** PTE Ref Cnt Overflow */
  GM_PTE_REF_CNT_OVERFLOW = 30,
  /** Not supported in the kernel */
  GM_NOT_SUPPORTED_IN_KERNEL = 31,
  /** Not supported for this architecture */
  GM_NOT_SUPPORTED_ON_ARCH = 32,
  /** No match */
  GM_NO_MATCH = 33,
  /** User error */
  GM_USER_ERROR = 34,
  /** Timed out */
  GM_TIMED_OUT = 35,
  /** Data has been corrupted */
  GM_DATA_CORRUPTED = 36,
  /** Hardware fault */
  GM_HARDWARE_FAULT = 37,
  /** Send orphaned */
  GM_SEND_ORPHANED = 38,
  /** Minor overflow */
  GM_MINOR_OVERFLOW = 39,
  /** Page Table is Full */
  GM_PAGE_TABLE_FULL = 40,
  /** UC Error */
  GM_UC_ERROR = 41,
  /** Invalid Port Number */
  GM_INVALID_PORT_NUMBER = 42,
  /** No device files found */
  GM_DEV_NOT_FOUND = 43,
  /** Lanai not running */
  GM_FIRMWARE_NOT_RUNNING = 44,
  /** No match for yellow pages query. */
  GM_YP_NO_MATCH = 45,
  /** Parity error detected in SRAM. */
  GM_SRAM_PARITY_ERROR = 46,
  /** Fatal situation detected in firmware. */
  GM_FIRMWARE_ABORT = 47,
#endif
  /* If you add error codes, also update libgm/gm_strerror.c. --Glenn */

  GM_NUM_STATUS_CODES		/* may change in value */
#if !GM_BUILDING_GM
    /* DEPRECATED */ , GM_NUM_ERROR_CODES = GM_NUM_STATUS_CODES
#endif
    /* Do not add new codes here. */
}
gm_status_t;

/** Priority Levels */
enum gm_priority
{
  /** Low priority message */ 
  GM_LOW_PRIORITY = 0,
  /** High priority message */ 
  GM_HIGH_PRIORITY = 1,
 /** Number of priority types */ 
  GM_NUM_PRIORITIES = 2
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS 

enum gm_buf_status
{
  gm_in_send = 0,
  gm_in_recv,
  gm_in_app,
  gm_invalid_status
};

struct gm_buf_handle
{
  void *addr;
  int size;
  enum gm_buf_status status;
  struct gm_buf_handle *next;
};


typedef struct gm_alarm
{
  struct gm_alarm *next;
  enum
  {
    GM_ALARM_FREE = 0,
    GM_ALARM_SET
  }
  state;
  struct gm_port *port;
  void (*callback) (void *context);
  void *context;
  gm_u64_t deadline;
}
gm_alarm_t;

/* This is a list of GM API versions supported by this version of GM. */

enum gm_api_version
{
  /* symbolic versions for C code to allow symbolic debugging. */
  _GM_API_VERSION_1_0 = GM_API_VERSION_1_0,
#if GM_API_VERSION >= GM_API_VERSION_1_1
  _GM_API_VERSION_1_1 = GM_API_VERSION_1_1,
#endif
#if GM_API_VERSION >= GM_API_VERSION_1_2
  _GM_API_VERSION_1_2 = GM_API_VERSION_1_2,
#endif
#if GM_API_VERSION >= GM_API_VERSION_1_3
  _GM_API_VERSION_1_3 = GM_API_VERSION_1_3,
#endif
#if GM_API_VERSION >= GM_API_VERSION_1_4
  _GM_API_VERSION_1_4 = GM_API_VERSION_1_4,
#endif
#if GM_API_VERSION >= GM_API_VERSION_1_5
  _GM_API_VERSION_1_5 = GM_API_VERSION_1_5,
#endif
#if GM_API_VERSION >= GM_API_VERSION_1_6
  _GM_API_VERSION_1_6 = GM_API_VERSION_1_6,
#endif
#if GM_API_VERSION >= GM_API_VERSION_2_0
  _GM_API_VERSION_2_0 = GM_API_VERSION_2_0,
#endif
#if GM_API_VERSION >= GM_API_VERSION_2_0_6
  _GM_API_VERSION_2_0_6 = GM_API_VERSION_2_0_6,
#endif
  __GM_API_VERSION_IGNORE_COMMA
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/************************************************************************
 * Macros
 ************************************************************************/

/* Determine if N is aligned on an M-byte boundary, where M is a power
   of 2 */

#define __GM_MISALIGNMENT(n,m) ((gm_size_t)(n) & ((gm_size_t)(m) - 1))
#define GM_ALIGNED(n,m)	(__GM_MISALIGNMENT(n,m) == 0)

/* NOTE: all GM alignment requirements will be lifted */

/** GM RDMA GRANULARITY */
#define GM_RDMA_GRANULARITY 64
/** GM MAX DMA GRANULARITY */
#define GM_MAX_DMA_GRANULARITY 8
#if GM_RDMA_GRANULARITY % GM_MAX_DMA_GRANULARITY
#  error GM_RDMA_GRANULARITY must be a multiple of GM_MAX_DMA_GRANULARITY
#endif

#if !GM_BUILDING_FIRMWARE
#define GM_DMA_GRANULARITY GM_MAX_DMA_GRANULARITY
#endif

#define GM_DMA_ALIGNED(p)	GM_ALIGNED (p, GM_DMA_GRANULARITY)

/************************************************************************
 * Recv event type
 ************************************************************************/

/** Receive Event Types */
enum gm_recv_event_type
{
#if !GM_BUILDING_FIRMWARE			/* don't use these in the MCP */
  /** No significant receive event is pending. */
  GM_NO_RECV_EVENT = 0,
  /** deprecated */
  GM_SENDS_FAILED_EVENT = 1,
  /** This event should be treated as an unknown event (passed to gm_unknown())
*/
  GM_ALARM_EVENT = 2,
  /** */
  GM_SENT_EVENT = 3,
 /** */
  _GM_SLEEP_EVENT = 4,
 /** */
  GM_RAW_RECV_EVENT = 5,
 /** */
  GM_BAD_SEND_DETECTED_EVENT = 6,
 /** */
  GM_SEND_TOKEN_VIOLATION_EVENT = 7,
 /** */
  GM_RECV_TOKEN_VIOLATION_EVENT = 8,
 /** */
  GM_BAD_RECV_TOKEN_EVENT = 9,
 /** */
  GM_ALARM_VIOLATION_EVENT = 10,
  /**  This event indicates that a normal receive (GM_LOW_PRIORITY) has
  occurred.  */
  GM_RECV_EVENT = 11,
  /**  This event indicates that a normal receive (GM_HIGH_PRIORITY) has
  occurred.  */
  GM_HIGH_RECV_EVENT = 12,
  /**  This event indicates that a normal receive (GM_LOW_PRIORITY) has
  occurred, and the PEER indicates that the sender/receiver ports are
  the same.  */
  GM_PEER_RECV_EVENT = 13,
  /**  This event indicates that a normal receive (GM_HIGH_PRIORITY) has
  occurred, and the PEER indicates that the sender/receiver ports are
  the same.  */
  GM_HIGH_PEER_RECV_EVENT = 14,
  /** A small-message receive occurred (GM_LOW_PRIORITY) with the
  small message stored in the receive queue for improved small-message
  performance. */
  GM_FAST_RECV_EVENT = 15,
  /** A small-message receive occurred (GM_HIGH_PRIORITY) with the
  small message stored in the receive queue for improved small-message
  performance. */
  GM_FAST_HIGH_RECV_EVENT = 16,
  /** A small-message receive occurred (GM_LOW_PRIORITY) with the
  small message stored in the receive queue for improved small-message
  performance.  The PEER indicates that the sender/receiver ports are
  the same. */
  GM_FAST_PEER_RECV_EVENT = 17,
  /** A small-message receive occurred (GM_HIGH_PRIORITY) with the
  small message stored in the receive queue for improved small-message
  performance.  The PEER indicates that the sender/receiver ports are
  the same. */
  GM_FAST_HIGH_PEER_RECV_EVENT = 18,
  /**  */
  GM_REJECTED_SEND_EVENT = 19,
  /**  */
  GM_ORPHANED_SEND_EVENT = 20,
  /** Types used to make a new event in
     the queue. */
  /** Directed send notification */
  /* _GM_PUT_NOTIFICATION_EVENT = 21, */
  /* GM_FREE_SEND_TOKEN_EVENT = 22, */
  /* GM_FREE_HIGH_SEND_TOKEN_EVENT = 23, */
  /** */
  GM_BAD_RESEND_DETECTED_EVENT = 24,
  /** */
  GM_DROPPED_SEND_EVENT = 25,
  /** */
  GM_BAD_SEND_VMA_EVENT = 26,
  /** */
  GM_BAD_RECV_VMA_EVENT = 27,
  /** */
  _GM_FLUSHED_ALARM_EVENT = 28,
  /** */
  GM_SENT_TOKENS_EVENT = 29,
  /** */
  GM_IGNORE_RECV_EVENT = 30,
  /** */
  GM_ETHERNET_RECV_EVENT = 31,
  /** */
  GM_FATAL_FIRMWARE_ERROR_EVENT = 32,
#endif				/* GM_BUILDING_FIRMWARE not defined */
  /****** Types used to make a new event in the queue. ******/
  GM_NEW_NO_RECV_EVENT = 128,
  /** deprecated */
  GM_NEW_SENDS_FAILED_EVENT = 129,
  /** */
  GM_NEW_ALARM_EVENT = 130,
  /** */
  GM_NEW_SENT_EVENT = 131,
  /** */
  _GM_NEW_SLEEP_EVENT = 132,
  /** */
  GM_NEW_RAW_RECV_EVENT = 133,
  /** */
  GM_NEW_BAD_SEND_DETECTED_EVENT = 134,
  /** */
  GM_NEW_SEND_TOKEN_VIOLATION_EVENT = 135,
  /** */
  GM_NEW_RECV_TOKEN_VIOLATION_EVENT = 136,
  /** */
  GM_NEW_BAD_RECV_TOKEN_EVENT = 137,
  /** */
  GM_NEW_ALARM_VIOLATION_EVENT = 138,
  /** normal receives */
  GM_NEW_RECV_EVENT = 139,
  /** */
  GM_NEW_HIGH_RECV_EVENT = 140,
  /** */
  GM_NEW_PEER_RECV_EVENT = 141,
  /** */
  GM_NEW_HIGH_PEER_RECV_EVENT = 142,
  /** streamlined small message receives */
  GM_NEW_FAST_RECV_EVENT = 143,
  /** */
  GM_NEW_FAST_HIGH_RECV_EVENT = 144,
  /** */
  GM_NEW_FAST_PEER_RECV_EVENT = 145,
  /** */
  GM_NEW_FAST_HIGH_PEER_RECV_EVENT = 146,
  /** */
  GM_NEW_REJECTED_SEND_EVENT = 147,
  /** */
  GM_NEW_ORPHANED_SEND_EVENT = 148,
  /** Directed send notification */
  _GM_NEW_PUT_NOTIFICATION_EVENT = 149,
  /** */
  GM_NEW_FREE_SEND_TOKEN_EVENT = 150,
  /** */
  GM_NEW_FREE_HIGH_SEND_TOKEN_EVENT = 151,
  /** */
  GM_NEW_BAD_RESEND_DETECTED_EVENT = 152,
  /** */
  GM_NEW_DROPPED_SEND_EVENT = 153,
  /** */
  GM_NEW_BAD_SEND_VMA_EVENT = 154,
  /** */
  GM_NEW_BAD_RECV_VMA_EVENT = 155,
  /** */
  _GM_NEW_FLUSHED_ALARM_EVENT = 156,
  /** */
  GM_NEW_SENT_TOKENS_EVENT = 157,
  /** */
  GM_NEW_IGNORE_RECV_EVENT = 158,
  /** */
  GM_NEW_ETHERNET_RECV_EVENT = 159,
  /** */
  GM_NEW_FATAL_FIRMWARE_ERROR_EVENT = 160,
  /** Add new types here. */

  GM_NUM_RECV_EVENT_TYPES
  /** DO NOT add new types here. */
};

/****************
 * recv
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef struct gm_recv
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* 8 */
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t message_pad;
#endif
  gm_up_n_t message;
  /* 8 */
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t buffer_pad;
#endif
  gm_up_n_t buffer;
  /* 8 */
  gm_u32_n_t reserved_after_buffer;
  gm_u32_n_t length;
  /* 8 */
  gm_u16_n_t sender_node_id;
  gm_u16_n_t reserved_after_sender_node_id;
  gm_u8_n_t tag;
  gm_u8_n_t size;
  gm_u8_n_t sender_port_id;
  gm_u8_n_t type;
}
gm_recv_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/****************
 * sent token report
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

struct _gm_sent_token_report
{
  gm_u8_n_t token;
  gm_u8_n_t status;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#define GM_MAX_SENT_TOKEN_REPORTS_PER_EVENT ((GM_RDMA_GRANULARITY/2) - 1)

/****************
 * tokens sent
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef struct gm_tokens_sent
{
  struct _gm_sent_token_report report[GM_MAX_SENT_TOKEN_REPORTS_PER_EVENT];
  gm_u8_n_t reserved_before_type;
  gm_u8_n_t type;
}
gm_tokens_sent_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/****************
 * sent
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define GM_TOKENS_SENT_EVENT__FIRST (ts) ((gm_u8_t *) (ts + 1) - 2)

typedef struct gm_sent
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* Pad to 32 bytes */
  gm_u8_n_t _reserved[16];
#if GM_SIZEOF_UP_T == 4
  gm_u8_n_t _reserved_before_message_list[4];
#endif
  gm_up_n_up_n_t message_list;
  gm_u8_n_t _reserved_after_message_list[8 - 1];
  gm_u8_n_t type;
}
gm_sent_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/****************
 * failed send
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

				/* Struct of GM_RDMA_GRANULARITY for
				   reporting rejected sends. */
typedef struct gm_failed_send_event
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* Pad to 32 bytes */
  gm_u8_n_t _reserved[16];
  /* 8 */
#if GM_SIZEOF_UP_T == 4
  gm_u32_n_t message_pad;
#endif
  gm_up_n_t message;
  /* 8 */
  gm_u32_n_t reserved_after_message;
  gm_u16_n_t target_node_id;
  gm_u8_n_t target_subport_id;
  gm_u8_n_t type;
}
gm_failed_send_event_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */ 

/****************
 * fatal firmware error
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef struct gm_fatal_firmware_error
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* 8 */
#if GM_MAX_DMA_GRANULARITY != 8
#error
#endif
  /* Pad to 32 bytes */
  gm_u8_n_t _reserved[32 - 8];
  /* 8 */
  gm_u32_n_t status;
  gm_u8_n_t reserved_after_status[3];
  gm_u8_n_t type;
}
gm_fatal_firmware_error_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */ 

/****************
 * flushed alarm
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef struct gm_flushed_alarm
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* 8 */
#if GM_MAX_DMA_GRANULARITY != 8
#error
#endif
  /* Pad to 32 bytes */
  gm_u8_n_t _reserved[32 - 8];
  /* 8 */
  gm_u32_n_t usecs_remaining;
  gm_u8_n_t reserved_after_usecs_remaining[3];
  gm_u8_n_t type;
}
gm_flushed_alarm_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */ 

/****************
 * Ethernet recv
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef struct gm_ethernet_recv
{
  /* Pad to GM_RDMA_GRANULARITY bytes */
#if (32 % GM_RDMA_GRANULARITY)
  gm_u8_n_t _rdma_padding[GM_RDMA_GRANULARITY - (32 % GM_RDMA_GRANULARITY)];
#endif
  /* Pad to 32 bytes */
#if (16 + 2 * GM_SIZEOF_UP_T) % 32
  gm_u8_n_t _reserved[32 - (16 + 2 * GM_SIZEOF_UP_T) % 32];
#endif
  gm_up_n_t message;
  gm_up_n_t buffer;
  /* 8 */
  gm_u16_n_t ip_checksum;	/* IPv4 partial checksum, or checksum
				   of entire packet; flags tell which */
  gm_u16_n_t ip_header_len;	/* future use */
  gm_u32_n_t length;
  /* 8 */
  gm_u16_n_t sender_node_id;
  gm_u8_n_t flags;
#define	GM_IPV4_PARTIAL_CHECKSUM	0x1	/* checksum is an IPv4
						   partial checksum */
#define GM_TCP_UDP_HEADER_SPLIT		0x2	/* future use */
  gm_u8_n_t reserved_after_flags;
  gm_u8_n_t tag;
  gm_u8_n_t size;
  gm_u8_n_t sender_port_id;
  gm_u8_n_t type;
}
gm_ethernet_recv_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */ 

/****************
 * recv event
 ****************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS

typedef union gm_recv_event
{
  /* Each of these must be the same size so the TYPE field at the
     end of each is aligned. */
  gm_tokens_sent_t tokens_sent;
  gm_recv_t recv;
  gm_sent_t sent;
  gm_failed_send_event_t failed_send;
  gm_fatal_firmware_error_t fatal_firmware_error;
  gm_flushed_alarm_t flushed_alarm;
  gm_ethernet_recv_t ethernet_recv;
}
gm_recv_event_t;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */ 

#define GM_RECV_EVENT_TYPE(e) (gm_ntoh_u8 ((e)->recv.type))

/* GM equivalent of ANSI "offsetof()" macro, since WindowsNT botches
   (or used to botch) it somewhere in the DDK header files. */

#define GM_OFFSETOF(type,field) ((gm_offset_t) &((type *)0)->field)

/** opaque handle to the gm_port struct. */
struct gm_port;
/** gm_send_completion_callback_t typedef function. */
typedef void (*gm_send_completion_callback_t) (struct gm_port * p,
					       void *context,
					       gm_status_t status);

/************************************************************************
 * Globals (undefined until gm_init() or gm_open() called).
 ************************************************************************/

#if !defined GM_PAGE_LEN
extern GM_ENTRY_POINT unsigned long GM_PAGE_LEN;
#endif

/****************************************************************
 * Data type abstraction typedefs
 ****************************************************************/

#if defined _M_IA64 || defined _M_AMD64 || (GM_CPU_hppa && !GM_BUILDING_FIRMWARE)
typedef gm_u64_t gm_size_t;
typedef gm_s64_t gm_offset_t;
#else
typedef unsigned long gm_size_t;
typedef signed long gm_offset_t;
#endif

/****************************************************************
 * Function prototypes
 ****************************************************************/

GM_ENTRY_POINT void _gm_assertion_failed (const char *, int, const char *);

/** Aborts the current process (@ref gm_abort "Details"). */
GM_ENTRY_POINT void gm_abort (void);

/** Allocates a send token (@ref gm_alloc_send_token "Details"). */
GM_ENTRY_POINT int gm_alloc_send_token (struct gm_port *p,
					unsigned int priority);
/** Allows any remote GM port to modify the contents of any GM DMAable
memory (@ref gm_allow_remote_memory_access "Details"). */
GM_ENTRY_POINT gm_status_t gm_allow_remote_memory_access (struct gm_port *p);

/** Copies a region of memory (@ref gm_bcopy "Details"). */
GM_ENTRY_POINT void gm_bcopy (const void *from, void *to, gm_size_t len);

/** Blocks until there is a receive event and then returns a pointer to the
event. (@ref gm_blocking_receive "Details"). */
GM_ENTRY_POINT union gm_recv_event *gm_blocking_receive (struct gm_port *p);

/** Like gm_blocking_receive(), except it sleeps the current thread
immediately if no receive is pending. (@ref gm_blocking_receive_no_spin
"Details"). */
GM_ENTRY_POINT union
  gm_recv_event *gm_blocking_receive_no_spin (struct gm_port *p);

/** Zeros a region in memory (@ref gm_bzero "Details"). */
GM_ENTRY_POINT void gm_bzero (void *ptr, gm_size_t len);

/** Callocs a region in memory (@ref gm_calloc "Details"). */
GM_ENTRY_POINT void *gm_calloc (gm_size_t len, gm_size_t cnt);

/** opaque handle the gm_alarm struct. */
struct gm_alarm;

/** Cancels alarm (@ref gm_cancel_alarm "Details"). */
GM_ENTRY_POINT void gm_cancel_alarm (struct gm_alarm *gm_alarm);

/** Closes a GM port (@ref gm_close "Details"). */
GM_ENTRY_POINT void gm_close (struct gm_port *p);

/** Unreliable send (@ref gm_datagram_send "Details"). */
GM_ENTRY_POINT void gm_datagram_send (struct gm_port *p, void *message,
				      unsigned int size, gm_size_t len,
				      unsigned int priority,
				      unsigned int target_node_id,
				      unsigned int target_port_id,
				      gm_send_completion_callback_t callback,
				      void *context);

/** Unreliable send of gm_u32_t messages (@ref gm_datagram_send_4 "Details"). */
GM_ENTRY_POINT void gm_datagram_send_4 (struct gm_port *p, gm_u32_t message,
					unsigned int size, gm_size_t len,
					unsigned int priority,
					unsigned int target_node_id,
					unsigned int target_port_id,
					gm_send_completion_callback_t
					callback, void *context);

/** Deregisters memory (@ref gm_deregister_memory "Details"). */ 
GM_ENTRY_POINT gm_status_t gm_deregister_memory (struct gm_port *p, void *ptr,
						 gm_size_t length);

/* Deprecated function. */ 
GM_ENTRY_POINT void gm_directed_send (struct gm_port *p, void *source_buffer,
				      gm_remote_ptr_t target_buffer,
				      gm_size_t len,
				      enum gm_priority priority,
				      unsigned int target_node_id,
				      unsigned int target_port_id);

/** Directed send (PUT) (@ref gm_put "Details"). */
GM_ENTRY_POINT void gm_directed_send_with_callback (struct gm_port *p,
						    void *source_buffer,
						    gm_remote_ptr_t
						    target_buffer,
						    gm_size_t len,
						    enum gm_priority priority,
						    unsigned int
						    target_node_id,
						    unsigned int
						    target_port_id,
						    gm_send_completion_callback_t
						    callback, void *context);

/** Callocs a region of DMAable memory (@ref gm_dma_calloc "Details"). */
GM_ENTRY_POINT void *gm_dma_calloc (struct gm_port *p, gm_size_t count,
				    gm_size_t length);

/** Frees a region of DMAable memory (@ref gm_dma_free "Details"). */ 
GM_ENTRY_POINT void gm_dma_free (struct gm_port *p, void *addr);

/** Mallocs a region of DMAable memory (@ref gm_dma_malloc "Details"). */
GM_ENTRY_POINT void *gm_dma_malloc (struct gm_port *p, gm_size_t length);

/** Flushes an alarm (@ref gm_flush_alarm "Details"). */
GM_ENTRY_POINT void gm_flush_alarm (struct gm_port *p);

/** Frees a region of memory (@ref gm_free "Details"). */
GM_ENTRY_POINT void gm_free (void *ptr);

/** Frees a send token (@ref gm_free_send_token "Details"). */
GM_ENTRY_POINT void gm_free_send_token (struct gm_port *p,
					unsigned int priority);

/** Frees multiple send tokens (@ref gm_free_send_tokens "Details"). */
GM_ENTRY_POINT void gm_free_send_tokens (struct gm_port *p,
					 unsigned int priority,
					 unsigned int count);

/** Copies the host name of the local node (@ref gm_get_host_name "Details"). */
GM_ENTRY_POINT gm_status_t gm_get_host_name (struct gm_port *port,
					     char name[GM_MAX_HOST_NAME_LEN]);

/** Returns GM_GET_NODE_TYPE (@ref gm_get_node_type "Details"). */
GM_ENTRY_POINT gm_status_t gm_get_node_type (struct gm_port *port,
					     int *node_type);

/** Copies the GM ID of the interface (@ref gm_get_node_id "Details"). */
GM_ENTRY_POINT gm_status_t gm_get_node_id (struct gm_port *port,
					   unsigned int *n);

/** Copies the board ID of the interface (@ref gm_get_unique_board_id
"Details"). */
GM_ENTRY_POINT gm_status_t gm_get_unique_board_id (struct gm_port *port,
						   char unique[6]);

/** Copies copies the 6-byte ethernet address of the interface
(@ref gm_get_mapper_unique_id "Details"). */
GM_ENTRY_POINT gm_status_t gm_get_mapper_unique_id (struct gm_port *port,
						    char unique[6]);

/** Prints the hex equivalent of data (@ref gm_hex_dump "Details"). */
GM_ENTRY_POINT void gm_hex_dump (const void *ptr, gm_size_t len);

/** This function is deprectated.  Use gm_host_name_to_node_id_ex() instead.
Returns the GM ID associated with a host_name
(@ref gm_host_name_to_node_id "Details"). */
GM_ENTRY_POINT unsigned int gm_host_name_to_node_id (struct gm_port *port,
						     char *_host_name);

/** Initializes user-allocated storage for an alarm.
(@ref gm_initialize_alarm "Details"). */
GM_ENTRY_POINT void gm_initialize_alarm (struct gm_alarm *my_alarm);

/** Like the ANSI isprint() except it works in the kernel and MCP.
(@ref gm_isprint "Details"). */
GM_ENTRY_POINT int gm_isprint (int c);

/** Mallocs a region in memory (@ref gm_malloc "Details"). */
GM_ENTRY_POINT void *gm_malloc (gm_size_t len);

/** Returns a pointer to a newly allocated aligned uninitialized page of
memory. (@ref gm_page_alloc "Details"). */
GM_ENTRY_POINT void *gm_page_alloc (void);

/** Frees a page of memory (@ref gm_page_free "Details"). */
GM_ENTRY_POINT void gm_page_free (void *addr);

/** Allocates a page-aligned buffer of memory (@ref gm_page_free "Details"). */
GM_ENTRY_POINT void *gm_alloc_pages (gm_size_t len);

/** Frees the pages of memory (@ref gm_free_pages "Details"). */
GM_ENTRY_POINT void gm_free_pages (void *addr, gm_size_t len);

/** Returns the maximum length of a message that will fit in a GM buffer
(@ref gm_max_length_for_size "Details"). */
GM_ENTRY_POINT gm_size_t gm_max_length_for_size (unsigned int size);

/** Stores the maximum GM node ID supported by the network interface card
(@ref gm_max_node_id "Details"). */
GM_ENTRY_POINT gm_status_t gm_max_node_id (struct gm_port *port,
					   unsigned int *n);

/* deprecated function. */ 
GM_ENTRY_POINT gm_status_t gm_max_node_id_inuse (struct gm_port *port,
						 unsigned int *n);

/** Emulates the ANSI memcmp() function (@ref gm_memcmp "Details"). */
GM_ENTRY_POINT int gm_memcmp (const void *a, const void *b, gm_size_t len);

/** Copies a message into a buffer if needed.
(@ref gm_memorize_message "Details"). */
GM_ENTRY_POINT void *gm_memorize_message (void *message, void *buffer,
					  unsigned int len);

/** Returns the minimum supported message size
(@ref gm_min_message_size "Details"). */
GM_ENTRY_POINT unsigned int gm_min_message_size (struct gm_port *port);

/** Returns the minimum GM message buffer size required to store a message.
(@ref gm_min_size_for_length "Details"). */
GM_ENTRY_POINT unsigned int gm_min_size_for_length (gm_size_t length);

/** Returns the value of GM_MTU. (@ref gm_mtu "Details"). */
GM_ENTRY_POINT unsigned int gm_mtu (struct gm_port *port);

/** This function is deprecated.  Use gm_node_id_to_host_name_ex()
instead.  Returns a pointer to the host name of the host containing
the network interface card with GM node id node_id. (@ref
gm_node_id_to_host_name_ex "Details"). */
GM_ENTRY_POINT char *gm_node_id_to_host_name (struct gm_port *port,
					      unsigned int node_id);

/** Stores the MAC address for the interface (@ref gm_node_id_to_unique_id
"Details"). */
GM_ENTRY_POINT gm_status_t gm_node_id_to_unique_id (struct gm_port *port,
						    unsigned int n,
						    char unique[6]);

/** Returns the number of ports supported by this build.
(@ref gm_num_ports "Details"). */
GM_ENTRY_POINT unsigned int gm_num_ports (struct gm_port *p);

/** Returns the number of send tokens for this port.
(@ref gm_num_send_tokens "Details"). */
GM_ENTRY_POINT unsigned int gm_num_send_tokens (struct gm_port *p);

/** Returns the number of receive tokens for this port.
(@ref gm_num_receive_tokens "Details"). */
GM_ENTRY_POINT unsigned int gm_num_receive_tokens (struct gm_port *p);

/** Returns the id of the GM port (@ref gm_get_port_id "Details"). */
GM_ENTRY_POINT unsigned int gm_get_port_id(struct gm_port *p);

/** Opens a GM port on an interface (@ref gm_open "Details"). */
GM_ENTRY_POINT gm_status_t gm_open (struct gm_port **p, unsigned int unit,
				    unsigned int port, const char *port_name,
				    enum gm_api_version version);

/* Deprecated function. */
GM_ENTRY_POINT void gm_provide_receive_buffer (struct gm_port *p, void *ptr,
					       unsigned int size,
					       unsigned int priority);

/** Provides GM with a buffer into which it can receive messages (@ref
gm_provide_receive_buffer_with_tag "Details"). */
GM_ENTRY_POINT void gm_provide_receive_buffer_with_tag (struct gm_port *p,
							void *ptr,
							unsigned int size,
							unsigned int priority,
							unsigned int tag);

/** Returns nonzero if a receive event is pending (@ref gm_receive_pending
"Details"). */
GM_ENTRY_POINT int gm_receive_pending (struct gm_port *p);

/** Returns the nonzero event type if an event is pending (@ref
gm_next_event_peek "Details"). */
GM_ENTRY_POINT int gm_next_event_peek (struct gm_port *p, gm_u16_t *sender);

/** Returns a receive event. (@ref gm_receive "Details"). */
GM_ENTRY_POINT union gm_recv_event *gm_receive (struct gm_port *p);

/** Registers virtual memory for DMA transfers. (@ref gm_register_memory
"Details"). */
GM_ENTRY_POINT gm_status_t gm_register_memory (struct gm_port *p,
					       void *ptr, gm_size_t length);

/** Tests for the availability of a send token without allocating the send
token. (@ref gm_send_token_available "Details"). */
GM_ENTRY_POINT int gm_send_token_available (struct gm_port *p,
					    unsigned int priority);

/* Deprecated function. */
GM_ENTRY_POINT void gm_send (struct gm_port *p, void *message,
			     unsigned int size, gm_size_t len,
			     unsigned int priority,
			     unsigned int target_node_id,
			     unsigned int target_port_id);

/** A fully asynchronous send. (@ref gm_send_with_callback "Details"). */
GM_ENTRY_POINT
void gm_send_with_callback (struct gm_port *p, void *message,
			    unsigned int size, gm_size_t len,
			    unsigned int priority,
			    unsigned int target_node_id,
			    unsigned int target_port_id,
			    gm_send_completion_callback_t callback,
			    void *context);

/* Deprecated function. */
GM_ENTRY_POINT void gm_send_to_peer (struct gm_port *p, void *message,
				     unsigned int size, gm_size_t len,
				     unsigned int priority,
				     unsigned int target_node_id);

/** A fully asychronous send from/to the same GM port on the sending
and receiving side.  (@ref gm_send_to_peer_with_callback "Details"). */
GM_ENTRY_POINT
void gm_send_to_peer_with_callback (struct gm_port *p, void *message,
				    unsigned int size, gm_size_t len,
				    unsigned int priority,
				    unsigned int target_node_id,
				    gm_send_completion_callback_t callback,
				    void *context);

/** Informs GM of the acceptable sizes of GM messages received on a port.
(@ref gm_set_acceptable_sizes "Details"). */
GM_ENTRY_POINT gm_status_t gm_set_acceptable_sizes (struct gm_port *p,
						    enum gm_priority
						    priority, gm_size_t mask);

/** Sets an alarm, which may already be pending. (@ref gm_set_alarm
"Details"). */
GM_ENTRY_POINT void gm_set_alarm (struct gm_port *p,
				  struct gm_alarm *my_alarm,
				  gm_u64_t usecs,
				  void (*callback) (void *), void *context);

/** Reimplements strlen. (@ref gm_strlen "Details"). */
GM_ENTRY_POINT gm_size_t gm_strlen (const char *cptr);

/** Reimplements strncpy. (@ref gm_strlen "Details"). */
GM_ENTRY_POINT char *gm_strncpy (char *to, const char *from, int len);

/** Reimplements strcmp. (@ref gm_strcmp "Details"). */ 
GM_ENTRY_POINT int gm_strcmp (const char *a, const char *b);

/** Reimplements strncmp. (@ref gm_strncmp "Details"). */
GM_ENTRY_POINT int gm_strncmp (const char *a, const char *b, int len);

/** Reimplements strncasecmp. (@ref gm_strncasecmp"Details"). */
GM_ENTRY_POINT int gm_strncasecmp (const char *a, const char *b, int len);

/** Returns a 64-bit extended version of the LANai real time clock (RTC).
(@ref gm_ticks "Details"). */
GM_ENTRY_POINT gm_u64_t gm_ticks (struct gm_port *port);

/** Returns the board id number for an interface. (@ref gm_unique_id
"Details"). */
GM_ENTRY_POINT gm_status_t gm_unique_id (struct gm_port *port,
					 char unique[6]);

/** Returns the GM node id for a specific interface.
(@ref gm_unique_id_to_node_id "Details"). */
GM_ENTRY_POINT gm_status_t gm_unique_id_to_node_id (struct gm_port *port,
						    char unique[6],
						    unsigned int *node_id);

/** GM Event Handler.  (@ref gm_unknown "Details"). */
GM_ENTRY_POINT void gm_unknown (struct gm_port *p, union gm_recv_event *e);

/** _gm_get_route function.  (@ref _gm_get_route "Details"). */
GM_ENTRY_POINT gm_status_t _gm_get_route (struct gm_port *p,
					  unsigned int node_id, char *route,
					  unsigned int *len);

/****************
 * buffer debugging
 ****************/

/** Dumps the contents of a buffer.  (@ref gm_dump_buffers "Details"). */
GM_ENTRY_POINT void gm_dump_buffers (void);

/** Registers a GM buffer. (@ref gm_register_buffer "Details"). */
GM_ENTRY_POINT void gm_register_buffer (void *addr, int size);

/** Deregisters a GM buffer. (@ref gm_unregister_buffer "Details"). */
GM_ENTRY_POINT int gm_unregister_buffer (void *addr, int size);

/************
 * Lookaside table entry points
 ************/

/** Creates a lookaside list. (@ref gm_create_lookaside "Details"). */
GM_ENTRY_POINT struct gm_lookaside *gm_create_lookaside (gm_size_t entry_len,
							 gm_size_t
							 min_entry_cnt);

/** Destroys a lookaside list. (@ref gm_destroy_lookaside "Details"). */  
GM_ENTRY_POINT void gm_destroy_lookaside (struct gm_lookaside *l);

/** Allocates an entry from the lookaside table, with debugging.
(@ref gm_lookaside_alloc "Details"). */
GM_ENTRY_POINT void *gm_lookaside_alloc (struct gm_lookaside *l);

/** allocates and clear an entry from the lookaside table.
(@ref gm_lookaside_zalloc "Details"). */
GM_ENTRY_POINT void *gm_lookaside_zalloc (struct gm_lookaside *l);

/** Frees an allocated entry in the lookaside list.
(@ref gm_lookaside_free "Details"). */
GM_ENTRY_POINT void gm_lookaside_free (void *ptr);

/************
 * Hash table entry points
 ************/

#define GM_HASH_SMOOTH 1	/* not yet supported */

/** Creates a hash table. (@ref gm_create_hash "Details"). */
GM_ENTRY_POINT struct gm_hash
  *gm_create_hash (long (*gm_user_compare) (void *key1, void *key2),
		   unsigned long (*gm_user_hash) (void *key1),
		   gm_size_t key_len, gm_size_t data_len,
		   gm_size_t gm_min_entries, int flags);

/** Destroys a hash table. (@ref gm_destroy_hash "Details"). */
GM_ENTRY_POINT void gm_destroy_hash (struct gm_hash *h);

/** Removes an entry from the hash table. (@ref gm_hash_remove "Details"). */
GM_ENTRY_POINT void *gm_hash_remove (struct gm_hash *hash, void *key);

/** Finds an entry in the hash table. (@ref gm_hash_find "Details"). */
GM_ENTRY_POINT void *gm_hash_find (struct gm_hash *hash, void *key);

/** Inserts an entry in the hash table. (@ref gm_hash_insert "Details"). */
GM_ENTRY_POINT gm_status_t gm_hash_insert (struct gm_hash *hash, void *key,
					   void *datum);

/** Replaces a key in the hash table. (@ref gm_hash_rekey "Details"). */
GM_ENTRY_POINT void gm_hash_rekey (struct gm_hash *hash, void *old_key,
				   void *new_key);

/** Compares strings in the hash table. (@ref gm_hash_compare_strings
"Details"). */
GM_ENTRY_POINT long gm_hash_compare_strings (void *key1, void *key2);

/** Hashes a string. (@ref gm_hash_hash_string "Details"). */ 
GM_ENTRY_POINT unsigned long gm_hash_hash_string (void *key);

/** Compares longs in the hash table. (@ref gm_hash_compare_longs
"Details"). */
GM_ENTRY_POINT long gm_hash_compare_longs (void *key1, void *key2);

/** Hashes a long. (@ref gm_hash_hash_long "Details"). */ 
GM_ENTRY_POINT unsigned long gm_hash_hash_long (void *key);

/** Compares ints in the hash table. (@ref gm_hash_compare_ints
"Details"). */
GM_ENTRY_POINT long gm_hash_compare_ints (void *key1, void *key2);

/** Hashes an int. (@ref gm_hash_hash_int "Details"). */
GM_ENTRY_POINT unsigned long gm_hash_hash_int (void *key);

/** Compares ptrs in the hash table. (@ref gm_hash_compare_ptrs
"Details"). */
GM_ENTRY_POINT long gm_hash_compare_ptrs (void *key1, void *key2);

/** Hashes a ptr. (@ref gm_hash_hash_ptr "Details"). */
GM_ENTRY_POINT unsigned long gm_hash_hash_ptr (void *key);

/************
 * crc
 ************/

/** Crc function.  (@ref gm_crc "Details"). */
GM_ENTRY_POINT unsigned long gm_crc (void *ptr, gm_size_t len);

/** Crc str function.  (@ref gm_crc_str "Details"). */
GM_ENTRY_POINT unsigned long gm_crc_str (const char *ptr);

/************
 * random number generation
 ************/

/** Rand function.  (@ref gm_rand "Details"). */
GM_ENTRY_POINT int gm_rand (void);

/** Srand function.  (@ref gm_srand "Details"). */
GM_ENTRY_POINT void gm_srand (int seed);

/** rand_mod function.  (@ref gm_rand_mod "Details"). */
GM_ENTRY_POINT unsigned int gm_rand_mod (unsigned int modulus);

/************
 * init/finalize
 ************/

/** Initializes GM. (@ref gm_init "Details"). */
GM_ENTRY_POINT gm_status_t gm_init (void);

/** Decrements the GM initialization counter and if it becomes zero, frees
 all resources associated with GM in the current process.
(@ref gm_finalize "Details"). */
GM_ENTRY_POINT void gm_finalize (void);

/************
 * base 2 logarithm computation
 ************/

/** Log 2 roundup table. (@ref gm_log2_roundup_table "Details"). */
GM_ENTRY_POINT extern const unsigned char gm_log2_roundup_table[257];

/** Log 2 computation. (@ref gm_log2_roundup "Details"). */
GM_ENTRY_POINT unsigned long gm_log2_roundup (unsigned long n);

/************
 * GM mutex's (currently open-coded for user-mode)
 ************/

/** Create mutex. (@ref gm_create_mutex "Details"). */
GM_ENTRY_POINT struct gm_mutex *gm_create_mutex (void);

/** Destroy mutex. (@ref gm_destroy_mutex "Details"). */
GM_ENTRY_POINT void gm_destroy_mutex (struct gm_mutex *mu);

/** Enter mutex. (@ref gm_mutex_enter "Details"). */
GM_ENTRY_POINT void gm_mutex_enter (struct gm_mutex *mu);

/** Exit mutex. (@ref gm_mutex_exit "Details"). */
GM_ENTRY_POINT void gm_mutex_exit (struct gm_mutex *mu);

/************
 * GM zone
 ************/

/** Create a zone. (@ref gm_zone_create_zone "Details"). */
GM_ENTRY_POINT struct gm_zone *gm_zone_create_zone (void *base,
						    gm_size_t len);

/** Destroy a zone. (@ref gm_zone_destroy_zone "Details"). */
GM_ENTRY_POINT void gm_zone_destroy_zone (struct gm_zone *zone);

/** Free a zone. (@ref gm_zone_free "Details"). */
GM_ENTRY_POINT void *gm_zone_free (struct gm_zone *zone, void *a);

/** Malloc a zone. (@ref gm_zone_malloc "Details"). */
GM_ENTRY_POINT void *gm_zone_malloc (struct gm_zone *zone, gm_size_t length);

/** Calloc a zone. (@ref gm_zone_calloc "Details"). */
GM_ENTRY_POINT void *gm_zone_calloc (struct gm_zone *zone, gm_size_t count,
				     gm_size_t length);

/** Address in a zone. (@ref gm_zone_addr_in_zone "Details"). */
GM_ENTRY_POINT int gm_zone_addr_in_zone (struct gm_zone *zone, void *p);

/****************
 * Default event handlers.
 *
 * gm_unknown() will call these for you, but you can call them directly
 * to enhance performance.
 ****************/

/* Deprecated function. */
GM_ENTRY_POINT void
gm_handle_put_notification (struct gm_port *p, gm_recv_event_t * e);
/* Deprecated function. */
GM_ENTRY_POINT void gm_handle_sent_tokens (struct gm_port *p,
					   gm_recv_event_t * e);

/****************
 * fault tolerance
 ****************/

/** Reenables packet transmission of messages. (@ref gm_resume_sending
"Details"). */
GM_ENTRY_POINT void gm_resume_sending (struct gm_port *p,
				       unsigned int priority,
				       unsigned int target_node_id,
				       unsigned int target_port_id,
				       gm_send_completion_callback_t callback,
				       void *context);

/** Tells the LANai to drop all enqueued sends and reenable packet
transmission on that connection. (@ref gm_drop_sends "Details"). */
GM_ENTRY_POINT void gm_drop_sends (struct gm_port *port,
				   unsigned int priority,
				   unsigned int target_node_id,
				   unsigned int target_port_id,
				   gm_send_completion_callback_t callback,
				   void *context);

/****************
 * PID support
 ****************/

/** typedef for gm_pid_t. */
typedef gm_u32_t gm_pid_t;

/** Wrapper around the UNIX getpid().  (@ref gm_getpid "Details"). */
GM_ENTRY_POINT gm_pid_t gm_getpid (void);

/****************
 * Direct copy support
 ****************/

/** Copies data from a local process to the memory area.
(@ref gm_directcopy_get "Details"). */
GM_ENTRY_POINT gm_status_t gm_directcopy_get (struct gm_port *p,
					      void *source_addr,
					      void *target_addr,
					      gm_size_t length,
					      unsigned int source_instance_id,
					      unsigned int source_port_id);

/****************
 * Misc
 ****************/

/** Similar to ANSI perror().  (@ref gm_perror "Details"). */
GM_ENTRY_POINT void gm_perror (const char *message, gm_status_t error);

/** emulates the ANSI standard sleep(). (@ref gm_sleep "Details"). */
GM_ENTRY_POINT int gm_sleep (unsigned seconds);

/** Causes the current process to exit with a status appropriate to the GM
status code. (@ref gm_exit "Details"). */
GM_ENTRY_POINT void gm_exit (gm_status_t status);

/** Emulates or invokes the ANSI standard printf(). (@ref gm_printf
"Details"). */
GM_ENTRY_POINT int
gm_printf (const char *format, ...)
#ifdef __GNUC__
  __attribute__ ((format (printf, 1, 2)))
#endif
  ;

/****************************************************************
 * GM-1.4 additions
 ****************************************************************/

#if GM_API_VERSION >= GM_API_VERSION_1_4

/** Error function for GM. (@ref gm_strerror "Details"). */
GM_ENTRY_POINT char *gm_strerror (gm_status_t error);

/** Enables nack down. (@ref gm_set_enable_nack_down "Details"). */
GM_ENTRY_POINT gm_status_t gm_set_enable_nack_down(struct gm_port *port,
						   int flag);

/** Returns the maximum GM node ID that is in use by the network attached
to the port. (@ref gm_max_node_id_in_use "Details"). */
GM_ENTRY_POINT gm_status_t gm_max_node_id_in_use (struct gm_port *port,
						  unsigned int *n);
#endif

/****************
 * GM-1.6 additions
 ****************/

#if GM_API_VERSION >= GM_API_VERSION_1_6
  
#define GM_STRUCT_CONTAINING(type,field,field_instance)			\
((type *)((char *)(field_instance) - GM_OFFSETOF (type, field)))
#define GM_NUM_ELEM(ar) (sizeof (ar) / sizeof (*ar))
#define GM_POWER_OF_TWO(n) (!((n)&((n)-1)))
#define GM_MISALIGNMENT(n,m) __GM_MISALIGNMENT(n,m)

/** Invokes the ANSI standard vprintf().  (@ref gm_vprintf "Details"). */
GM_ENTRY_POINT int gm_eprintf (const char *format, ...)
#ifdef __GNUC__
  __attribute__ ((format (printf, 1, 2)))
#endif
  ;

/** Reimplements the UNIX memset().  (@ref gm_memset "Details"). */
GM_ENTRY_POINT void *gm_memset (void *s, int c, gm_size_t n);

/** Reimplements the UNIX strdup().  (@ref gm_strdup "Details"). */
GM_ENTRY_POINT char *gm_strdup (const char *);

#endif

/****************
 * GM mark support
 ****************/

typedef gm_size_t gm_mark_t;
struct gm_mark_set;

/** Marks an area. (@ref gm_mark "Details"). */
GM_ENTRY_POINT gm_status_t gm_mark (struct gm_mark_set *set, gm_mark_t *m);

/** Evaluates if a mark is valid. (@ref gm_mark_is_valid "Details"). */
GM_ENTRY_POINT int gm_mark_is_valid (struct gm_mark_set *set, gm_mark_t *m);

/** Create a mark set. (@ref gm_create_mark_set "Details"). */
GM_ENTRY_POINT gm_status_t gm_create_mark_set (struct gm_mark_set **set,
					       unsigned long init_count);

/** Destroy the mark set. (@ref gm_destroy_mark_set "Details"). */
GM_ENTRY_POINT void gm_destroy_mark_set (struct gm_mark_set *set);

/** Unmarks a valid mark. (@ref gm_unmark "Details"). */
GM_ENTRY_POINT void gm_unmark (struct gm_mark_set *set, gm_mark_t *m);

/** Unmarks all valid marks . (@ref gm_unmark_all "Details"). */
GM_ENTRY_POINT void gm_unmark_all (struct gm_mark_set *set);

/****************
 * gm_on_exit
 ****************/

typedef void (*gm_on_exit_callback_t)(gm_status_t status, void *arg);

/** Call the callbacks in the reverse order registered inside gm_exit(),
passing GM exit status and registered argument to the callback.
(@ref gm_on_exit "Details"). */
GM_ENTRY_POINT gm_status_t gm_on_exit (gm_on_exit_callback_t, void *arg);

/****************************************************************
 * GM-2.0 new entry points
 ****************************************************************/

#if GM_API_VERSION >= GM_API_VERSION_2_0

/** RDMA Read Operation (GET) (@ref gm_get "Details"). */
GM_ENTRY_POINT void gm_get (struct gm_port *p,
			    gm_remote_ptr_t remote_buffer,
			    void *local_buffer,
			    gm_size_t len,
			    enum gm_priority priority,
			    unsigned int target_node_id,
			    unsigned int target_port_id,
			    gm_send_completion_callback_t callback,
			    void *context);

/** Directed send (PUT) (@ref gm_put "Details"). */
GM_ENTRY_POINT void gm_put (struct gm_port *p,
			    void *local_buffer,
			    gm_remote_ptr_t remote_buffer,
			    gm_size_t len,
			    enum gm_priority priority,
			    unsigned int target_node_id,
			    unsigned int target_port_id,
			    gm_send_completion_callback_t callback,
			    void *context);

/** Reimplements the UNIX strdup().  (@ref gm_strdup "Details"). */
GM_ENTRY_POINT char *gm_strdup (const char *);

/** Invokes the ANSI standard eprintf().  (@ref gm_eprintf "Details"). */
GM_ENTRY_POINT int gm_eprintf (const char *format, ...);

/** Given a pointer to an instance of a field in a structure of a
    certain type, return a pointer to the containing structure. */
#define GM_STRUCT_CONTAINING(type,field,field_instance)			\
((type *)((char *)(field_instance) - GM_OFFSETOF (type, field)))

/** Given an array, return the number of elements in the array. */
#define GM_NUM_ELEM(ar) (sizeof (ar) / sizeof (*ar))

/** Return nonzero if the input is neither a power of two nor zero.
    Otherwise, return zero. */
#define GM_POWER_OF_TWO(n) (!((n)&((n)-1)))

/** Stores at *node_id the local connection ID corresponding to
the connection to global_id. (@ref gm_global_id_to_node_id
"Details"). */
GM_ENTRY_POINT gm_status_t gm_global_id_to_node_id (struct gm_port *port,
						    unsigned int global_id,
						    unsigned int *node_id);

/** Stores at *global_id the global node ID corresponding to the
connection identified by the local connection ID. (@ref
gm_global_id_to_node_id "Details"). */
GM_ENTRY_POINT gm_status_t gm_node_id_to_global_id (struct gm_port *port,
						    unsigned int node_id,
						    unsigned int *global_id);

/** Store at *name the host name of the host containing the network
interface card with GM node id node_id.  (@ref
gm_node_id_to_host_name_ex "Details"). */
GM_ENTRY_POINT gm_status_t gm_node_id_to_host_name_ex (struct gm_port *port,
						       unsigned int
						       timeout_usecs,
						       unsigned int node_id,
						       char (*name)
						       [GM_MAX_HOST_NAME_LEN+1]
						       );
/** Store at *node_id the node ID of the host containing the network
interface card with GM host name host_name.  (@ref
gm_host_name_to_node_id_ex "Details"). */
GM_ENTRY_POINT gm_status_t gm_host_name_to_node_id_ex (struct gm_port *port,
						       unsigned int
						       timeout_usecs,
						       const char *host_name,
						       unsigned int *node_id);

#endif /* GM_API_VERSION >= GM_API_VERSION_2_0 */

#if GM_API_VERSION >= GM_API_VERSION_2_0_6

/** Registers virtual memory for DMA transfers. (@ref gm_register_memory_ex
"Details"). */
GM_ENTRY_POINT gm_status_t gm_register_memory_ex (struct gm_port *p,
						  void *ptr, gm_size_t length,
						  void *pvma);
#endif /* GM_API_VERSION >= GM_API_VERSION_2_0_6 */

/****************
 * Features added in gm-2.0.16 and gm-2.1.6
 ****************/

#if (GM_API_MAJOR == 2 && GM_API_MINOR == 1 && GM_API_SUBMINOR >= 6) \
|| (GM_API_MAJOR == 2 && GM_API_MINOR == 0 && GM_API_SUBMINOR >= 16) \
|| (GM_API_MAJOR >= 3)
#define GM_HAVE_YP 1
#endif

#ifdef GM_HAVE_YP
#define GM_MAX_YP_STRING_LEN 128 /* including null termination */

/** Performs a generic GM YP query. (@ref gm_yp "Details"). */
GM_ENTRY_POINT gm_status_t gm_yp (struct gm_port *p,
				  unsigned int timeout_usecs,
				  unsigned int *node_id,
				  const char *key,
				  const char *value,
				  char (*answer)[GM_MAX_YP_STRING_LEN]);
#endif

/****************************************************************
 * Warning suppression
 ****************************************************************/

/****************
 * Intel compiler
 ****************/

/* Trick the Intel compiler into not emitting Remark #177 about functions
   being declared but never referenced. */

#if defined __INTEL_COMPILER

extern gm_u8_t  __gm_u8;
extern gm_u16_t __gm_u16;
extern gm_u32_t __gm_u32;
extern gm_u64_t __gm_u64;

extern gm_s8_t  __gm_s8;
extern gm_s16_t __gm_s16;
extern gm_s32_t __gm_s32;
extern gm_s64_t __gm_s64;

extern gm_u8_n_t  __gm_u8_n;
extern gm_u16_n_t __gm_u16_n;
extern gm_u32_n_t __gm_u32_n;
extern gm_u64_n_t __gm_u64_n;

extern gm_s8_n_t  __gm_s8_n;
extern gm_s16_n_t __gm_s16_n;
extern gm_s32_n_t __gm_s32_n;
extern gm_s64_n_t __gm_s64_n;

extern void *__gm_void_p;
extern gm_up_n_t __gm_up_n;

static inline
void
__gm_suppress_intel_compiler_warnings ()
{
  __gm_s8 = gm_ntoh_s8 (__gm_s8_n);
  __gm_s16 = gm_ntoh_s16 (__gm_s16_n);
  __gm_s32 = gm_ntoh_s32 (__gm_s32_n);
  __gm_s64 = gm_ntoh_s64 (__gm_s64_n);

  __gm_u8 = gm_ntoh_u8 (__gm_u8_n);
  __gm_u16 = gm_ntoh_u16 (__gm_u16_n);
  __gm_u32 = gm_ntoh_u32 (__gm_u32_n);
  __gm_u64 = gm_ntoh_u64 (__gm_u64_n);

  __gm_s8_n = gm_hton_s8 (__gm_s8);
  __gm_s16_n = gm_hton_s16 (__gm_s16);
  __gm_s32_n = gm_hton_s32 (__gm_s32);
  __gm_s64_n = gm_hton_s64 (__gm_s64);

  __gm_u8_n = gm_hton_u8 (__gm_u8);
  __gm_u16_n = gm_hton_u16 (__gm_u16);
  __gm_u32_n = gm_hton_u32 (__gm_u32);
  __gm_u64_n = gm_hton_u64 (__gm_u64);

  __gm_u32 = __gm_swap_u32_asm (__gm_u32);
  __gm_u32 = gm_htopci_u32 (__gm_u32);

#if !GM_BUILDING_GM
  __gm_u8 = gm_ntohc (__gm_u8_n);
  __gm_u16 = gm_ntohs (__gm_u16_n);
  __gm_u32 = gm_ntohl (__gm_u32_n);
  __gm_u64 = gm_ntohll (__gm_u64_n);

  __gm_u8_n = gm_htonc (__gm_u8);
  __gm_u16_n = gm_htons (__gm_u16);
  __gm_u32_n = gm_htonl (__gm_u32);
  __gm_u64_n = gm_htonll (__gm_u64);

  __gm_void_p = gm_ntohp (__gm_up_n);
#endif /* !GM_BUILDING_GM */
  
  __gm_suppress_intel_compiler_warnings ();
}

#endif /* defined __INTEL_COMPILER */

/************************************************************************
 * Epilogue
 ************************************************************************/

#ifdef __cplusplus
#if 0
{				/* indent hack */
#endif
}
#endif

#endif /* ifndef _gm_h_ */

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
