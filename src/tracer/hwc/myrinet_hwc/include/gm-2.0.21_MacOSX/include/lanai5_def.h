/* lanai5_def.h */

/*************************************************************************
 *                                                                       *
 * Myricom Myrinet Software                                              *
 *                                                                       *
 * Copyright (c) 1999 by Myricom, Inc.                                   *
 * All rights reserved.                                                  *
 *                                                                       *
 * Permission to use, copy, modify and distribute this software and its  *
 * documentation in source and binary forms for non-commercial purposes  *
 * and without fee is hereby granted, provided that the modified software*
 * is returned to Myricom, Inc. for redistribution. The above copyright  *
 * notice must appear in all copies.  Both the copyright notice and      *
 * this permission notice must appear in supporting documentation, and   *
 * any documentation, advertising materials and other materials related  *
 * to such distribution and use must acknowledge that the software was   *
 * developed by Myricom, Inc. The name of Myricom, Inc. may not be used  *
 * to endorse or promote products derived from this software without     *
 * specific prior written permission.                                    *
 *                                                                       *
 * Myricom, Inc. makes no representations about the suitability of this  *
 * software for any purpose.                                             *
 *                                                                       *
 * THIS FILE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND, WHETHER   *
 * EXPRESSED OR IMPLIED, INCLUDING THE WARRANTY OF MERCHANTIBILITY OR    *
 * FITNESS FOR A PARTICULAR PURPOSE. MYRICOM, INC. SHALL HAVE NO         *
 * LIABILITY WITH RESPECT TO THE INFRINGEMENT OF COPYRIGHTS, TRADE       *
 * SECRETS OR ANY PATENTS BY THIS FILE OR ANY PART THEREOF.              *
 *                                                                       *
 * In no event will Myricom, Inc. be liable for any lost revenue         *
 * or profits or other special, indirect and consequential damages, even *
 * if Myricom has been advised of the possibility of such damages.       *
 *                                                                       *
 * Other copyrights might apply to parts of this software and are so     *
 * noted when applicable.                                                *
 *                                                                       *
 * Myricom, Inc.                                                         *
 * 325 N. Santa Anita Ave.                                              *
 * Arcadia, CA 91006                                                     *
 * 626 821-5555                                                          *
 * http://www.myri.com                                                   *
 *************************************************************************/


/* LANai5.0 */


#ifndef LANAI5_DEF_H
#define LANAI5_DEF_H

#include "gcc_version.h"

#if GCC_VERSION < GCC(2,95)
#define GM_VOLREG register volatile
#else
#define GM_VOLREG register
#endif

/**********************************
 ** Processor-internal registers **
 **********************************/

register unsigned PC 	asm ("pc");  /* Program Counter */
register unsigned CCR 	asm ("ps");  /* Process Status Word */
register int      *SP 	asm ("sp");  /* Stack Pointer */
register unsigned APS 	asm ("aps"); /* Alternate Program Status Word */
register unsigned APC 	asm ("apc"); /* Alternate Program Counter */
GM_VOLREG unsigned IMR 	asm ("imr"); /* Interrupt Mask Register */
GM_VOLREG unsigned ISR 	asm ("isr"); /* Interrupt Status Register */

/*****************************
 ** Memory-mapped registers **
 *****************************/

#define MM_REG_FULLWORD_P(X) (*((void* volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_FULLWORD(X)   (*((int   volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_HALFWORD(X)   (*((short volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_BYTE(X)       (*((char  volatile * const ) (0xFFFFFE00 + (X))))

#define IPF0	MM_REG_FULLWORD( 0x00 )	/* 5 context-0 state registers */
#define CUR0	MM_REG_FULLWORD( 0x08 )
#define PREV0	MM_REG_FULLWORD( 0x10 )
#define DATA0	MM_REG_FULLWORD( 0x18 )
#define DPF0	MM_REG_FULLWORD( 0x20 )

#define IPF1	MM_REG_FULLWORD( 0x28 )	/* 5 context-1 state registers */
#define CUR1	MM_REG_FULLWORD( 0x30 )
#define PREV1	MM_REG_FULLWORD( 0x38 )
#define DATA1	MM_REG_FULLWORD( 0x40 )
#define DPF1	MM_REG_FULLWORD( 0x48 )

#define EIMR	MM_REG_FULLWORD( 0x58 )	/* external-interrupt mask register */

#define IT	MM_REG_FULLWORD( 0x60 )	/* interrupt timer */
#define RTC	MM_REG_FULLWORD( 0x68 )	/* real-time clock */

#define L2E_LAR MM_REG_FULLWORD_P( 0x80 ) /* L2E-DMA local address */
#define E2L_LAR MM_REG_FULLWORD_P( 0x88 ) /* E2L-DMA local address */

#define L2E_EAR MM_REG_FULLWORD_P( 0x90 ) /* L2E-DMA external address */
#define E2L_EAR MM_REG_FULLWORD_P( 0x98 ) /* E2L-DMA external address */

#define L2E_CTR MM_REG_FULLWORD  ( 0xA0 ) /* L2E-DMA counter */
#define E2L_CTR MM_REG_FULLWORD  ( 0xA8 ) /* E2L-DMA counter */

#define PULSE	MM_REG_FULLWORD( 0xB8 ) /* E2L-DMA external address, FPGA use only */

#define RMW	MM_REG_FULLWORD_P( 0xD0 ) /* the same as RMC, but does not trigger CRC-32 */
#define RMC	MM_REG_FULLWORD_P( 0xD8 ) /* pointer to receive-DMA header CRC-32 */
#define RMP	MM_REG_FULLWORD_P( 0xE0 ) /* receive-DMA pointer */
#define RML	MM_REG_FULLWORD_P( 0xE8 ) /* receive-DMA limit */

#define SMP	MM_REG_FULLWORD_P( 0xF0 ) /* send-DMA pointer */
#define SMH	MM_REG_FULLWORD_P( 0xF8 ) /* pointer to send-DMA routing header end */
#define SML	MM_REG_FULLWORD_P( 0x100 ) /* send-DMA limit */
#define SMLT	MM_REG_FULLWORD_P( 0x108 ) /* send-DMA limit with tail */
#define SMC	MM_REG_FULLWORD_P( 0x110 ) /* pointer to send-DMA header CRC-32 */

#define SA      MM_REG_FULLWORD( 0x118 )	/* send align */

#define BURST	MM_REG_FULLWORD( 0x120 )	/* EBUS-DMA modes */
#define		PCI_STATUS_BIT			(0x1)
#define		PCI_WITH_8B_CACHE_LINE		(0x1)
#define		PCI_WITH_16B_CACHE_LINE		(0x3)
#define		PCI_WITH_32B_CACHE_LINE		(0x5)
#define		PCI_WITH_64B_CACHE_LINE		(0x9)
#define		PCI_WITH_128B_CACHE_LINE	(0x11)
#define		PCI_WITH_256B_CACHE_LINE	(0x21)
#define		PCI_WITH_512B_CACHE_LINE	(0x41)
#define		PCI_WITH_1024B_CACHE_LINE	(0x81)
#define		SBUS_ENABLE_2_WORD_BURST	(0x2)
#define		SBUS_ENABLE_4_WORD_BURST	(0x4)
#define		SBUS_ENABLE_8_WORD_BURST	(0x8)
#define		SBUS_ENABLE_16_WORD_BURST	(0x10)

#define TIMEOUT	MM_REG_FULLWORD( 0x128 )	/* NRES-period selection */
#define MYRINET	MM_REG_FULLWORD( 0x130 )
#define		NRES_ENABLE_BIT		(0x1)
#define		CRC8_ENABLE_BIT		(0x2)
#define		CRC32_ENABLE_BIT	(0x4)

#define DEBUG   MM_REG_FULLWORD( 0x138 ) /* hardware debugging */
#define LED	MM_REG_FULLWORD( 0x140 ) /* LED */
#define	VERSION	MM_REG_FULLWORD( 0x148 ) /* the ex-window-pins register */
#define	WRITE_ENABLE MM_REG_FULLWORD( 0x150 ) /* memory protection */

/*************************
 ** Interrupt bit names **
 *************************/

#define	DBG_BIT		0x80000000
#define	DEBUG_BIT	DBG_BIT
#define	HOST_SIG_BIT	0x40000000

#define	LAN7_SIG_BIT	0x00800000
#define	LAN6_SIG_BIT	0x00400000
#define	LAN5_SIG_BIT	0x00200000
#define	LAN4_SIG_BIT	0x00100000
#define	LAN3_SIG_BIT	0x00080000
#define	LAN2_SIG_BIT	0x00040000
#define	LAN1_SIG_BIT	0x00020000
#define	LAN0_SIG_BIT	0x00010000
#define WAKE_INT_BIT	0x00001000
#define NRES_INT_BIT	0x00000800
#define OFF_BY_4_BIT	0x00000400
#define OFF_BY_2_BIT	0x00000200
#define OFF_BY_1_BIT	0x00000100
#define WDOG_INT_BIT	0x00000080
#define TIME_INT_BIT	0x00000040
#define L2E_INT_BIT	0x00000020
#define E2L_INT_BIT	0x00000010
#define SEND_INT_BIT 	0x00000008
#define BUFF_INT_BIT 	0x00000004
#define RECV_INT_BIT 	0x00000002
#define HEAD_INT_BIT 	0x00000001

#define ORUN4_BIT	OFF_BY_4_BIT
#define ORUN2_BIT	OFF_BY_2_BIT
#define ORUN1_BIT	OFF_BY_1_BIT

#define ORUN4_INT_BIT	OFF_BY_4_BIT
#define ORUN2_INT_BIT	OFF_BY_2_BIT
#define ORUN1_INT_BIT	OFF_BY_1_BIT


/**********************************
 ** Macros to set/clear ISR bits **
 **********************************/

#define touch(VAR)		do {asm("":"=r"(VAR):"r"(VAR));} while (0)

#define	set_HOST_SIG_BIT()	do {ISR = HOST_SIG_BIT;} while (0)

#define	LAN7_SIG_BIT	0x00800000
#define	LAN6_SIG_BIT	0x00400000
#define	LAN5_SIG_BIT	0x00200000
#define	LAN4_SIG_BIT	0x00100000
#define	LAN3_SIG_BIT	0x00080000
#define	LAN2_SIG_BIT	0x00040000
#define	LAN1_SIG_BIT	0x00020000
#define	LAN0_SIG_BIT	0x00010000
#define WAKE_INT_BIT	0x00001000
#define NRES_INT_BIT	0x00000800
#define OFF_BY_4_BIT	0x00000400
#define OFF_BY_2_BIT	0x00000200
#define OFF_BY_1_BIT	0x00000100
#define WDOG_INT_BIT	0x00000080
#define TIME_INT_BIT	0x00000040
#define L2E_INT_BIT	0x00000020
#define E2L_INT_BIT	0x00000010
#define SEND_INT_BIT 	0x00000008
#define BUFF_INT_BIT 	0x00000004
#define RECV_INT_BIT 	0x00000002
#define HEAD_INT_BIT 	0x00000001

#define	clear_LAN7_SIG_BIT()    do {ISR = LAN7_SIG_BIT;} while (0)
#define	clear_LAN6_SIG_BIT()    do {ISR = LAN6_SIG_BIT;} while (0)
#define	clear_LAN5_SIG_BIT()    do {ISR = LAN5_SIG_BIT;} while (0)
#define	clear_LAN4_SIG_BIT()    do {ISR = LAN4_SIG_BIT;} while (0)
#define	clear_LAN3_SIG_BIT()    do {ISR = LAN3_SIG_BIT;} while (0)
#define	clear_LAN2_SIG_BIT()    do {ISR = LAN2_SIG_BIT;} while (0)
#define	clear_LAN1_SIG_BIT()    do {ISR = LAN1_SIG_BIT;} while (0)
#define	clear_LAN0_SIG_BIT()    do {ISR = LAN0_SIG_BIT;} while (0)
#define clear_WAKE_INT_BIT()	do {ISR = WAKE_INT_BIT;} while (0)
#define clear_NRES_INT_BIT()	do {ISR = NRES_INT_BIT;} while (0)
#define clear_OFF_BY_4_BIT()	do {ISR = OFF_BY_4_BIT;} while (0)
#define clear_OFF_BY_2_BIT()	do {ISR = OFF_BY_2_BIT;} while (0)
#define clear_OFF_BY_1_BIT()	do {ISR = OFF_BY_1_BIT;} while (0)
#define clear_ORUN4_BIT()	do {ISR = ORUN4_BIT;} while (0)
#define clear_ORUN2_BIT()	do {ISR = ORUN2_BIT;} while (0)
#define clear_ORUN1_BIT()	do {ISR = ORUN1_BIT;} while (0)
#define clear_ORUN4_INT_BIT()	do {ISR = ORUN4_INT_BIT;} while (0)
#define clear_ORUN2_INT_BIT()	do {ISR = ORUN2_INT_BIT;} while (0)
#define clear_ORUN1_INT_BIT()	do {ISR = ORUN1_INT_BIT;} while (0)
#define clear_WDOG_INT_BIT()	do {ISR = WDOG_INT_BIT;} while (0)
#define clear_TIME_INT_BIT()	do {ISR = TIME_INT_BIT;} while (0)
#define clear_L2E_INT_BIT()	do {ISR = L2E_INT_BIT;} while (0)

#define clear_E2L_INT_BIT()	do {ISR = E2L_INT_BIT;} while (0)

#define clear_SEND_INT_BIT()	do {ISR = SEND_INT_BIT;} while (0)
#define clear_BUFF_INT_BIT()	do {ISR = BUFF_INT_BIT;} while (0)
#define clear_RECV_INT_BIT()	do {ISR = RECV_INT_BIT;} while (0)
#define clear_HEAD_INT_BIT()	do {ISR = HEAD_INT_BIT;} while (0)

/* The volatile below keeps this from being optimized away. */
#define START_USER(user_function,stack_pointer) do {			     \
    int START_USER_OUT;							     \
    asm volatile (							     \
       "st %%r0,[0xffffff14] !clear the other context's         \n	     \
        st %%r0,[0xffffff18] !pipeline state                    \n	     \
        st %%r0,[0xffffff1c]                                    \n	     \
        st %%r0,[0xffffff20]                                    \n	     \
        st %%r0,[0xffffff24]                                    \n	     \
        ! The following two instructions must be done in the    \n	     \
        ! supervisor context, otherwise %%2 and %%1 will expand \n	     \
        ! to the wrong register                                 \n	     \
        mov %2,%%r27         !setup %%sp of other context       \n	     \
        mov %1,%%r17         !store function to call where the  \n	     \
                             !  other context can find it       \n	     \
        add %%pc,2*4,%%apc   !apc<-User context trampoline address-4\n	     \
        punt                 !run in user context               \n	     \
        bt.r 4*7             !when user context stops, continue \n	     \
	nop		                             		\n	     \
        ! here is the trampoline, which calls the fuction       \n	     \
        ! specified by %%1 above                                \n	     \
        mov %%sp,%%fp        !set up frame pointer              \n	     \
        add %%pc,8,%%rca     !compute return address            \n	     \
        mov %%r14,%%pc       !call the specified function       \n	     \
        st %%rca,[--%%sp]    !push return address in shadow     \n	     \
        bt .                 !When done, return to system context \n	     \
        punt                 !return to system context when user done."	     \
        : "=r" (START_USER_OUT) : "r" (user_function), "r" (stack_pointer)); \
} while (0)

#define RESUME_USER   asm volatile ("nop\n\tpunt")
#define RESUME_SYSTEM asm volatile ("nop\n\tpunt")

/* The following are faster versions of the above, but do not guarantee that
   all writes to memory are completed before the context switch. */

#define RISKY_RESUME_USER   asm volatile ("punt")
#define RISKY_RESUME_SYSTEM asm volatile ("punt")

extern void reset_Myrinet_ifc();


struct DMA_BLOCK {
	unsigned int next;	/* next pointer */
	unsigned short csum[2];	/* 2 16-bit ones complement checksums of this block */
	unsigned int len;	/* byte count */
	unsigned int lar;	/* lanai address */
	unsigned int eah;	/* high 32bit PCI address */
	unsigned int eal;	/*  low 32bit PCI address */
};

#define DMA_L2E		0x0
#define DMA_E2L		0x1
#define DMA_TERMINAL	0x2	/* terminal block? */
#define DMA_WAKE	0x4	/* wake when block completes */

#endif /* LANAI5_DEF_H */

