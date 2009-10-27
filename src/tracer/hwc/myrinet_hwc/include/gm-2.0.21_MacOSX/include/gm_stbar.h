#ifndef _gm_stbar_h_
#define _gm_stbar_h_

/* find out what architecture we are compiling for */

#include "gm_config.h"

/****************************************************************
 * asm abstraction
 ****************************************************************/

/* asm_memory() is like asm(), but tells the compiler that the
   asm effects memory, if possible.  This macro is useful for hiding
   "#if" processing below. */

#ifdef __GNUC__
#define gm_asm_memory(x) asm (x : : : "memory")
#else
#define gm_asm_memory(x) asm (x)
#endif

/****************************************************************
 * Store synchronization barrier Macro
 ****************************************************************/

/* In some cases, we put the barrier code inside the following functions: */
GM_ENTRY_POINT extern void __gm_stbar (void);
GM_ENTRY_POINT extern void __gm_readbar (void);
GM_ENTRY_POINT extern void __gm_writebar (void);

/******************
 * lanai
 ******************/

#if defined lanai3 || defined lanai7 || defined lanai

 /* ": : :" for C++ compat. */
#define GM_STBAR() asm volatile ("! GM_STBAR" : : :"memory")
#define GM_READBAR() asm volatile ("! GM_READBAR" : : :"memory")
#define GM_WRITEBAR() asm volatile ("! GM_WRITEBAR" : : :"memory")

/****************
 * sparc
 ****************/

#elif GM_CPU_sparc || GM_CPU_sparc64

/* Specify store barrier and read barrier asms for sparcv9 and sparcv8 */

#if defined __sparcv9
#define __GM_STBAR() gm_asm_memory ("membar #MemIssue")
#define __GM_READBAR() gm_asm_memory ("membar #LoadLoad | #LoadStore")
#define __GM_WRITEBAR() gm_asm_memory ("membar #StoreLoad | #StoreStore")
#elif defined __sparcv8
#define __GM_STBAR() gm_asm_memory ("stbar")
#define __GM_READBAR() /* Do nothing */
#define __GM_WRITEBAR() gm_asm_memory ("stbar");
#else
#error Do not know how to implement store barrier.
#endif

/* Use the barrier asms directly with the Gnu C compiler, but call a
   function instead with the Sun compiler for performance, since it
   disables optimization for any function containing an asm. */
  
#ifdef __GNUC__
#define GM_STBAR( ) __GM_STBAR ()
#define GM_READBAR() __GM_READBAR ()
#define GM_WRITEBAR() __GM_WRITEBAR ()
#elif (defined (__SUNPRO_C) || defined (__SUNPRO_CC))
#define GM_STBAR( ) __gm_stbar ()
#define GM_READBAR() __gm_readbar ()
#define GM_WRITEBAR() __gm_writebar ()
#else
#error Do not know how to emit the "stbar" instruction under this compiler.
#endif

/****************
 * x86
 ****************/

#elif GM_CPU_x86 || GM_CPU_x86_64
#if defined _MSC_VER
#define __GM_STBAR() /* Do nothing (but do it in __gm_stbar()) */
#define __GM_READBAR() /* Do nothing (but do it in __gm_readbar()) */
#define __GM_WRITEBAR() /* Do nothing (but do it in __gm_writebar()) */
#define GM_STBAR() __gm_stbar ()
#define GM_READBAR() __gm_readbar ()
#define GM_WRITEBAR() __gm_writebar ()
#elif defined __GNUC__
#if GM_OS_SOLARIS
/* No obvious way to disable write-combining mapping on Solaris/i386,
   so use the sfence instruction.  We emit this as machine language
   because some sun assemblers misassemble the sfence instruction. */
#define GM_STBAR() gm_asm_memory (".byte 0x0f,0xae,0xf8")
#else
#define GM_STBAR() asm ("": : :"memory")
#endif
#define GM_READBAR() asm ("": : :"memory")
#define GM_WRITEBAR() asm ("": : :"memory")
#elif defined __PGI
#define GM_STBAR()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#elif defined __INTEL_COMPILER
#define GM_STBAR()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#else
#define GM_STBAR()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#endif

/****************
 * ia64
 ****************/

#elif GM_CPU_ia64
#if defined _MSC_VER
#define GM_STBAR( ) __mf ()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#elif defined __INTEL_COMPILER
#include <ia64intrin.h>
#define GM_STBAR( ) __mf ()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#elif defined __GNUC__
#define GM_STBAR() asm volatile ("mf": : :"memory") /* ": : :" for C++ */
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#else
#error Do not know how to emit the "mf" instruction with this compiler.
#endif

/****************
 * alpha
 ****************/

#elif GM_CPU_alpha
#ifdef __GNUC__
#define GM_STBAR()  asm volatile ("mb": : :"memory") /* ": : :" for C++ */
#define GM_READBAR() asm volatile ("mb": : :"memory")
#define GM_WRITEBAR() asm volatile ("wmb": : :"memory")
#elif defined __DECC || defined __DECCXX
#if !GM_KERNEL
#include <c_asm.h>
#define GM_STBAR() asm ("mb")
#define GM_READBAR() asm ("mb")
#define GM_WRITEBAR() asm ("wmb")
#else
#include <sys/types.h>
#define GM_STBAR() mb()
#define GM_READBAR() mb()
#define GM_WRITEBAR() mb()
#endif
#else
#error Do not know how to emit the "mb" instruction with this compiler.
#endif

/****************
 * powerpc 
 * powerpc64
 ****************/

#elif GM_CPU_powerpc || GM_CPU_powerpc64
#ifdef __GNUC__
/* can't use -ansi for vxworks ccppc or this will fail with a syntax error */
#define GM_STBAR()  asm volatile ("sync": : :"memory") /* ": : :" for C++ */
#define GM_READBAR() asm volatile ("sync": : :"memory")
#define GM_WRITEBAR() asm volatile ("eieio": : :"memory")
#else
#if GM_OS_AIX || defined (__IBMC__) || defined (__IBMCPP__)
extern void __iospace_eieio(void); 
extern void __iospace_sync(void);  
#define GM_STBAR()   __iospace_sync ()
#define GM_READBAR() __iospace_sync ()
#define GM_WRITEBAR() __iospace_eieio ()
#else	/* GM_OS_AIX || defined (__IBMC__) || defined (__IBMCPP__) */
#error Do not know how to make a store barrier for this system
#endif	/* GM_OS_AIX || defined (__IBMC__) || defined (__IBMCPP__) */
#endif

/****************
 * mips
 ****************/

#elif GM_CPU_mips
#if GM_KERNEL
void flushbus(void);		/* hack to avoid including <sys/systm.h> */
#define GM_STBAR() flushbus();	/* kernel */
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#else
void gm_sync(void);		/* the myricom-provided flushbus() equiv. */
#define GM_STBAR() gm_sync();	/* user */
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()
#endif

/*****************
 * HP-PA RISC 
 *****************/
#elif GM_CPU_hppa
#define GM_STBAR()
#define GM_READBAR() GM_STBAR()
#define GM_WRITEBAR() GM_STBAR()

#endif /* various architectures */

#endif /* _gm_stbar_h_ */

