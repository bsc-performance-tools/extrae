/*************************************************************************
 * The contents of this file are subject to the MYRICOM MYRINET          *
 * EXPRESS (MX) NETWORKING SOFTWARE AND DOCUMENTATION LICENSE (the       *
 * "License"); User may not use this file except in compliance with the  *
 * License.  The full text of the License can found in LICENSE.TXT       *
 *                                                                       *
 * Software distributed under the License is distributed on an "AS IS"   *
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied.  See  *
 * the License for the specific language governing rights and            *
 * limitations under the License.                                        *
 *                                                                       *
 * Copyright 2005 by Myricom, Inc.  All rights reserved.                 *
 *************************************************************************/

#ifndef MX_AUTODETECT_H
#define MX_AUTODETECT_H

#ifdef	__cplusplus
extern "C"
{
#if 0
}				/* indent hack */
#endif
#endif

/************
 * Determine CPU based on what compiler is being used.
 ************/

#if defined MX_CPU_alpha
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_lanai
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_mips
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_powerpc
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_powerpc64
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_sparc
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_sparc64
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_x86
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_x86_64
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_hppa
#  define MX_CPU_DEFINED 1
#elif defined MX_CPU_ia64
#  define MX_CPU_DEFINED 1
#else
#  define MX_CPU_DEFINED 0
#endif

#if !MX_CPU_DEFINED
#  if defined _MSC_VER		/* Microsoft compiler */
#    if defined _M_IX86
#      define MX_CPU_x86 1
#    elif defined _M_IA64
#      define MX_CPU_ia64 1
#    elif defined _M_AMD64
#      define MX_CPU_x86_64 1
#    elif defined _M_ALPHA
#      define MX_CPU_alpha 1
#    else
#      error Could not determine CPU type.  You need to modify mx_autodetect.h.
#    endif
#  elif defined __APPLE_CC__	/* Apple OSX compiler defines __GNUC__ */
#    if defined __ppc__ 	/* but doesn't support #cpu syntax     */
#      define MX_CPU_powerpc 1
#    elif defined __ppc64__
#      define MX_CPU_powerpc64 1
#    elif defined __i386__
#      define MX_CPU_x86 1
#    else
#      error Could not determine CPU type.  You need to modify mx_autodetect.h.
#    endif
#  elif defined mips
#    define MX_CPU_mips 1
#  elif defined(__GNUC__) || defined __linux__ || defined __PGI
#    if #cpu(alpha) || defined __alpha || defined __alpha__
#      define MX_CPU_alpha 1
#    elif #cpu(hppa)
#      define MX_CPU_hppa 1
#    elif defined lanai || defined lanai3 || defined lanai7
#      define MX_CPU_lanai 1
#    elif #cpu(powerpc64) || defined(powerpc64) || defined __powerpc64__
#      define MX_CPU_powerpc64 1
#    elif #cpu(powerpc) || defined(__ppc__) || defined __powerpc__ || defined powerpc
#      define MX_CPU_powerpc 1
#    elif defined(_POWER) || defined _IBMR2
#      define MX_CPU_powerpc 1
#    elif #cpu(ia64) || defined __ia64__
#      define MX_CPU_ia64 1
#    elif #cpu(sparc64) || defined sparc64 || defined __sparcv9
#      define MX_CPU_sparc64 1
#    elif #cpu(sparc) || defined sparc || defined __sparc__
#      define MX_CPU_sparc 1
#    elif #cpu(i386) || defined __i386 || defined i386 || defined __i386__
#      define MX_CPU_x86 1
#    elif #cpu(x86_64) || defined __x86_64__
#      define MX_CPU_x86_64 1
#    elif defined(CPU)   /* This is how vxWorks defines their CPUs */
#      if (CPU==PPC603)
#	 define MX_CPU_powerpc 1
#      elif (CPU==PPC604)
#	 define MX_CPU_powerpc 1
#      elif (CPU==PPC405)
#        define MX_CPU_powerpc 1
#      else
#        error Could not determine CPU type.  If this is VxWorks, you will need to modify mx_autodetect.h to add your cpu type.
#      endif
#    else
#      error Could not determine CPU type.  You need to modify mx_autodetect.h.
#    endif
#  elif (defined (_POWER) && defined(_AIX))
#      define MX_CPU_powerpc 1
#  elif (defined __powerpc__)
#      define MX_CPU_powerpc 1
#  elif (defined (__DECC) || defined (__DECCXX)) && defined(__alpha)
#      define MX_CPU_alpha 1
#  elif (defined (__SUNPRO_C) || defined(__SUNPRO_CC)) && (defined(sparc64) || defined(__sparcv9))
#      define MX_CPU_sparc64 1
#  elif (defined (__SUNPRO_C) || defined(__SUNPRO_CC)) && defined(sparc)
#      define MX_CPU_sparc 1
#  elif (defined (__SUNPRO_C) || defined(__SUNPRO_CC)) && defined i386
#      define MX_CPU_x86 1
#  elif defined(__hppa) || defined(_PA_RISC1_1)
#      define MX_CPU_hppa 1
#  else
#    error Could not determine CPU type.  You need to modify mx_autodetect.h.
#  endif
#  undef MX_CPU_DEFINED
#  define MX_CPU_DEFINED 1
#endif

/** Define all undefined MX_CPU switches to 0 to prevent problems
   with "gcc -Wundef" */

#ifndef MX_CPU_alpha
#define MX_CPU_alpha 0
#endif
#ifndef MX_CPU_ia64
#define MX_CPU_ia64 0
#endif
#ifndef MX_CPU_hppa
#define MX_CPU_hppa 0
#endif
#ifndef MX_CPU_lanai
#define MX_CPU_lanai 0
#endif
#ifndef MX_CPU_mips
#define MX_CPU_mips 0
#endif
#ifndef MX_CPU_powerpc
#define MX_CPU_powerpc 0
#endif
#ifndef MX_CPU_powerpc64
#define MX_CPU_powerpc64 0
#endif
#ifndef MX_CPU_sparc
#define MX_CPU_sparc 0
#endif
#ifndef MX_CPU_sparc64
#define MX_CPU_sparc64 0
#endif
#ifndef MX_CPU_x86
#define MX_CPU_x86 0
#endif
#ifndef MX_CPU_x86_64
#define MX_CPU_x86_64 0
#endif

/****************************************************************
 * Endianess
 ****************************************************************/

/* Determine endianness */

#if MX_CPU_alpha
#  define MX_CPU_BIGENDIAN 0
#elif MX_CPU_lanai
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_mips
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_powerpc
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_powerpc64
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_sparc
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_sparc64
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_hppa
#  define MX_CPU_BIGENDIAN 1
#elif MX_CPU_x86
#  define MX_CPU_BIGENDIAN 0
#elif MX_CPU_x86_64
#  define MX_CPU_BIGENDIAN 0
#elif MX_CPU_ia64
#  define MX_CPU_BIGENDIAN 0
#else
#  error Could not determine endianness.  You need to modify mx_autodetect.h.
#endif


/****************************************************************
 * OS
 ****************************************************************/

#ifndef MX_OS_UDRV
#define MX_OS_UDRV 0
#endif

#if MX_CPU_lanai || MX_OS_UDRV
/* nothing */
#elif defined __linux__ || defined __gnu_linux__
#define MX_OS_LINUX 1
#elif defined _WIN32
#define MX_OS_WINNT 1
#elif defined __FreeBSD__
#define MX_OS_FREEBSD 1
#elif defined __APPLE__
#define MX_OS_MACOSX 1
#elif defined sun || defined __sun__
#define MX_OS_SOLARIS 1
#elif defined AIX || defined _AIX
#define MX_OS_AIX 1
#else
#error cannot autodetect your Operating system
#endif

#ifndef MX_OS_LINUX 
#define MX_OS_LINUX 0
#endif
#ifndef MX_OS_WINNT 
#define MX_OS_WINNT 0
#endif
#ifndef MX_OS_FREEBSD
#define MX_OS_FREEBSD 0
#endif
#ifndef MX_OS_MACOSX 
#define MX_OS_MACOSX 0
#endif
#ifndef MX_OS_SOLARIS
#define MX_OS_SOLARIS 0
#endif
#ifndef MX_OS_AIX
#define MX_OS_AIX 0
#endif

/**********************************************************************/
/* Control import/export of symbols and calling convention.           */
/**********************************************************************/
#if !MX_OS_WINNT
#  define MX_FUNC(type) type
#  define MX_VAR(type) type
#else
#  ifdef MX_BUILDING_LIB
#    ifdef __cplusplus
#      define MX_FUNC(type) extern "C" type __cdecl
#      define MX_VAR(type) extern "C" type
#    else
#      define MX_FUNC(type) type __cdecl
#      define MX_VAR(type) type
#    endif
#  else
#    ifdef __cplusplus
#      define MX_FUNC(type) extern "C" __declspec(dllimport) type __cdecl
#      define MX_VAR(type) extern "C" __declspec(dllimport) type
#    else
#      define MX_FUNC(type) __declspec(dllimport) type __cdecl
#      define MX_VAR(type) __declspec(dllimport) type
#    endif
#  endif
#endif

#ifdef __cplusplus
#if 0
{				/* indent hack */
#endif
}
#endif

#endif /* ifndef MX_AUTODETECT_H */
