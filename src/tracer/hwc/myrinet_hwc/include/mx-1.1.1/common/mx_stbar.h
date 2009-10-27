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
 * Copyright 2003 - 2004 by Myricom, Inc.  All rights reserved.          *
 *************************************************************************/

#ifndef MX_STBAR_H
#define MX_STBAR_H

/****************************************************************
 * Store synchronization barrier Macro
 ****************************************************************/

/******************
 * lanai
 ******************/

#if defined lanai3 || defined lanai7 || defined lanai

 /* ": : :" for C++ compat. */
#define MX_STBAR() __asm__ volatile ("! MX_STBAR" : : :"memory")
#define MX_READBAR() __asm__ volatile ("! MX_READBAR" : : :"memory")
#define MX_WRITEBAR() __asm__ volatile ("! MX_WRITEBAR" : : :"memory")

/****************
 * sparc
 ****************/

#elif MX_CPU_sparc || MX_CPU_sparc64
/* asm_memory() is like asm(), but tells the compiler that the
   asm effects memory, if possible.  This macro is useful for hiding
   "#if" processing below. */
#ifdef __GNUC__
#define mx_asm_memory(x) asm (x : : : "memory")
#else
#define mx_asm_memory(x) asm (x)
#endif

/* Specify store barrier and read barrier asms for sparcv9 and sparcv8 */
#if defined __sparcv9
#define __MX_STBAR() mx_asm_memory ("membar #MemIssue")
#define __MX_READBAR() mx_asm_memory ("membar #LoadLoad | #LoadStore")
#define __MX_WRITEBAR() mx_asm_memory ("membar #StoreLoad | #StoreStore")
#else
#define __MX_STBAR() mx_asm_memory ("stbar")
#define __MX_READBAR() /* Do nothing */
#define __MX_WRITEBAR() mx_asm_memory ("stbar");
#endif

/* Use the barrier asms directly with the Gnu C compiler, but call a
   function instead with the Sun compiler for performance, since it
   disables optimization for any function containing an asm. */
#ifdef __GNUC__
#define MX_STBAR( ) __MX_STBAR ();
#define MX_READBAR() __MX_READBAR ()
#define MX_WRITEBAR() __MX_WRITEBAR ()
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC))
void mx__stbar(void);
void mx__readbar(void);
void mx__writebar(void);
#define MX_STBAR( ) mx__stbar ()
#define MX_READBAR() mx__readbar ()
#define MX_WRITEBAR() mx__writebar ()
#endif

/****************
 * x86
 ****************/

#elif MX_CPU_x86 || MX_CPU_x86_64
#if defined _MSC_VER && defined _M_IX86
__inline void MX_STBAR(void) { long l; __asm { xchg l, eax } }
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#elif defined _MSC_VER && defined _M_AMD64
#define MX_STBAR _mm_mfence
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#elif defined __GNUC__ || defined __INTEL_COMPILER
#define MX_STBAR() __asm__ __volatile__ ("sfence;": : :"memory")
#if MX_CPU_x86_64 || MX_ENABLE_SSE2
#define MX_READBAR() __asm__ __volatile__ ("lfence;": : :"memory")
#else
#define MX_READBAR() __asm__ __volatile__ ("lock;addl $0,0(%%esp);": : : "memory");
#endif
#define MX_WRITEBAR() __asm__ __volatile__ ("": : :"memory")
static inline int mx__cpu_has_sse2(void)
{ 
  uint32_t eax,ecx,edx;
  __asm__("push %%ebx;cpuid;pop %%ebx" : "=a" (eax), "=c" (ecx), "=d" (edx) : "0" (1));
  return (edx & (1 << 26)) != 0;
}
#elif defined __PGI
#error still need to implement sfence in for compiler.
#define MX_STBAR()
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#error Do not know how to emit an sfence instruction with this compiler
#endif

/****************
 * ia64
 ****************/

#elif MX_CPU_ia64
#if defined _MSC_VER
#define MX_STBAR( ) __mf ()
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#elif defined __INTEL_COMPILER
#include <ia64intrin.h>
#define MX_STBAR( ) __mf ()
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#elif defined __GNUC__
#define MX_STBAR() __asm__ volatile ("mf": : :"memory") /* ": : :" for C++ */
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#else
#error Do not know how to emit the "mf" instruction with this compiler.
#endif

/****************
 * alpha
 ****************/

#elif MX_CPU_alpha
#ifdef __GNUC__
#define MX_STBAR()  __asm__ volatile ("mb": : :"memory") /* ": : :" for C++ */
#define MX_READBAR() __asm__ volatile ("mb": : :"memory")
#define MX_WRITEBAR() __asm__ volatile ("wmb": : :"memory")
#elif defined __DECC || defined __DECCXX
#ifndef MX_KERNEL
#include <c_asm.h>
#define MX_STBAR() asm ("mb")
#define MX_READBAR() asm ("mb")
#define MX_WRITEBAR() asm ("wmb")
#else
#include <sys/types.h>
#define MX_STBAR() mb()
#define MX_READBAR() mb()
#define MX_WRITEBAR() mb()
#endif
#else
#error Do not know how to emit the "mb" instruction with this compiler.
#endif

/****************
 * powerpc 
 * powerpc64
 ****************/

#elif MX_CPU_powerpc || MX_CPU_powerpc64
#ifdef __GNUC__
/* can't use -ansi for vxworks ccppc or this will fail with a syntax error */
#define MX_STBAR()  __asm__ volatile ("sync": : :"memory") /* ": : :" for C++ */
#define MX_READBAR() __asm__ volatile ("isync": : :"memory")
#define MX_WRITEBAR() __asm__ volatile ("eieio": : :"memory")
#else
#if	MX_OS_AIX
extern void __iospace_eieio(void); 
extern void __iospace_sync(void);  
#define MX_STBAR()   __iospace_sync ()
#define MX_READBAR() __iospace_sync ()
#define MX_WRITEBAR() __iospace_eieio ()
#else	/* MX_OS_AIX */
#error Do not know how to make a store barrier for this system
#endif	/* MX_OS_AIX */
#endif

/****************
 * mips
 ****************/

#elif MX_CPU_mips
#ifdef MX_KERNEL
void flushbus(void);		/* hack to avoid including <sys/systm.h> */
#define MX_STBAR() flushbus();	/* kernel */
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#else
void gm_sync(void);		/* the myricom-provided flushbus() equiv. */
#define MX_STBAR() gm_sync();	/* user */
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()
#endif

/*****************
 * HP-PA RISC 
 *****************/
#elif MX_CPU_hppa
#define MX_STBAR()
#define MX_READBAR() MX_STBAR()
#define MX_WRITEBAR() MX_STBAR()

#endif /* various architectures */

#endif /* MX_STBAR_H */

