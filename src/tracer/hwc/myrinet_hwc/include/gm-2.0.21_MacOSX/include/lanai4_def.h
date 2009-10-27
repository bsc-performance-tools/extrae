/* lanai4_def.h */

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
 * 325 N. Santa Anita Ave.                                               *
 * Arcadia, CA 91024                                                     *
 *************************************************************************/


/* LANai4.1 */


#ifndef LANAI4_DEF_H
#define LANAI4_DEF_H

#include "gm_gcc_version.h"

/**********************************
 ** Processor-internal registers **
 **********************************/

register unsigned PC 	asm ("pc");  /* Program Counter */
register unsigned CCR 	asm ("ps");  /* Process Status Word */
register int      *SP 	asm ("sp");  /* Stack Pointer */
register unsigned APS 	asm ("aps"); /* Alternate Program Status Word */
register unsigned APC 	asm ("apc"); /* Alternate Program Counter */

#if GCC_VERSION < GCC(2,95)
#define GM_VOLREG register volatile
#else
#define GM_VOLREG register
#endif

GM_VOLREG unsigned IMR 	asm ("imr"); /* Interrupt Mask Register */
GM_VOLREG unsigned ISR 	asm ("isr"); /* Interrupt Status Register */

/*****************************
 ** Memory-mapped registers **
 *****************************/

#define MM_REG_FULLWORD_P(X) (*((void* volatile * const ) (0xFFFFFF00 + (X))))
#define MM_REG_FULLWORD(X)   (*((int   volatile * const ) (0xFFFFFF00 + (X))))
#define MM_REG_HALFWORD(X)   (*((short volatile * const ) (0xFFFFFF00 + (X))))
#define MM_REG_BYTE(X)       (*((char  volatile * const ) (0xFFFFFF00 + (X))))

#define IPF0	MM_REG_FULLWORD( 0x00 )	/* 5 context-0 state registers */
#define CUR0	MM_REG_FULLWORD( 0x04 )
#define PREV0	MM_REG_FULLWORD( 0x08 )
#define DATA0	MM_REG_FULLWORD( 0x0c )
#define DPF0	MM_REG_FULLWORD( 0x10 )

#define IPF1	MM_REG_FULLWORD( 0x14 )	/* 5 context-1 state registers */
#define CUR1	MM_REG_FULLWORD( 0x18 )
#define PREV1	MM_REG_FULLWORD( 0x1c )
#define DATA1	MM_REG_FULLWORD( 0x20 )
#define DPF1	MM_REG_FULLWORD( 0x24 )

#define EIMR	MM_REG_FULLWORD( 0x2c )	/* external-interrupt mask register */

#define IT	MM_REG_FULLWORD( 0x30 )	/* interrupt timer */
#define RTC	MM_REG_FULLWORD( 0x34 )	/* real-time clock */

#define CKS	MM_REG_FULLWORD  ( 0x38 ) /* checksum */
#define EAR	MM_REG_FULLWORD_P( 0x3c ) /* SBus-DMA exteral address */
#define LAR	MM_REG_FULLWORD_P( 0x40 ) /* SBus-DMA local address */
#define DMA_CTR	MM_REG_FULLWORD  ( 0x44 ) /* SBus-DMA counter */

#define RMP	MM_REG_FULLWORD_P( 0x48 ) /* receive-DMA pointer */
#define RML	MM_REG_FULLWORD_P( 0x4c ) /* receive-DMA limit */

#define SMP	MM_REG_FULLWORD_P( 0x50 ) /* send-DMA pointer */
#define SML	MM_REG_FULLWORD_P( 0x54 ) /* send-DMA limit */
#define SMLT	MM_REG_FULLWORD_P( 0x58 ) /* send-DMA limit with tail */

#define RB	MM_REG_BYTE( 0x60 )	/* single-receive commands */
#define RH	MM_REG_HALFWORD( 0x64 )
#define RW	MM_REG_FULLWORD( 0x68 )

#define SA      MM_REG_FULLWORD( 0x6c )	/* send align */

#define SB      MM_REG_FULLWORD( 0x70 )	/* single-send commands */
#define SH      MM_REG_FULLWORD( 0x74 )
#define SW      MM_REG_FULLWORD( 0x78 )
#define ST      MM_REG_FULLWORD( 0x7c )

#define DMA_DIR	MM_REG_FULLWORD( 0x80 )	/* SBus-DMA direction */
#define DMA_STS	MM_REG_FULLWORD( 0x84 )	/* SBus-DMA modes */
#define TIMEOUT	MM_REG_FULLWORD( 0x88 )	/* NRES-period selection */
#define MYRINET	MM_REG_FULLWORD( 0x8c )
#define		NRES_ENABLE_BIT		(0x1)
#define		CRC_ENABLE_BIT		(0x2)

#define HW_DBUG MM_REG_FULLWORD( 0x90 ) /* hardware debugging */
#define LED	MM_REG_FULLWORD( 0x94 )	/* LED */
#define	VERSION	MM_REG_FULLWORD( 0x98 ) /* the ex-window-pins register */
#define	WRITE_ENABLE MM_REG_FULLWORD( 0x9C ) /* memory protection */

/*************************
 ** Interrupt bit names **
 *************************/

#define	DBG_BIT		0x80000000
#define	HOST_SIG_BIT	0x40000000

#define	LAN7_SIG_BIT	0x00800000
#define	LAN6_SIG_BIT	0x00400000
#define	LAN5_SIG_BIT	0x00200000
#define	LAN4_SIG_BIT	0x00100000
#define	LAN3_SIG_BIT	0x00080000
#define	LAN2_SIG_BIT	0x00040000
#define	LAN1_SIG_BIT	0x00020000
#define	LAN0_SIG_BIT	0x00010000
#define WORD_RDY_BIT    0x00008000
#define HALF_RDY_BIT    0x00004000
#define SEND_RDY_BIT    0x00002000
#define LINK_INT_BIT	0x00001000
#define NRES_INT_BIT	0x00000800
#define WAKE_INT_BIT	0x00000400
#define OFF_BY_2_BIT	0x00000200
#define OFF_BY_1_BIT	0x00000100
#define TAIL_INT_BIT    0x00000080
#define WDOG_INT_BIT	0x00000040
#define TIME_INT_BIT	0x00000020
#define DMA_INT_BIT	0x00000010
#define SEND_INT_BIT 	0x00000008
#define BUFF_INT_BIT 	0x00000004
#define RECV_INT_BIT 	0x00000002
#define BYTE_RDY_BIT    0x00000001

#define ORUN2_BIT	OFF_BY_2_BIT
#define ORUN1_BIT	OFF_BY_1_BIT

#define ORUN2_INT_BIT	OFF_BY_2_BIT
#define ORUN1_INT_BIT	OFF_BY_1_BIT

#define RECV_RDY_BITS	(BYTE_RDY_BIT | HALF_RDY_BIT | WORD_RDY_BIT)


/**********************************
 ** Macros to set/clear ISR bits **
 **********************************/

#define touch(VAR) asm ("nop ! touch %1->%0" : "=r" (VAR) : "r" (VAR))

#define	set_HOST_SIG_BIT()	do {ISR = HOST_SIG_BIT;}	 while (0)     

#define	clear_LAN7_SIG_BIT()    do {ISR = LAN7_SIG_BIT;}	 while (0)     
#define	clear_LAN6_SIG_BIT()    do {ISR = LAN6_SIG_BIT;}	 while (0)     
#define	clear_LAN5_SIG_BIT()    do {ISR = LAN5_SIG_BIT;}	 while (0)     
#define	clear_LAN4_SIG_BIT()    do {ISR = LAN4_SIG_BIT;}	 while (0)     
#define	clear_LAN3_SIG_BIT()    do {ISR = LAN3_SIG_BIT;}	 while (0)     
#define	clear_LAN2_SIG_BIT()    do {ISR = LAN2_SIG_BIT;}	 while (0)     
#define	clear_LAN1_SIG_BIT()    do {ISR = LAN1_SIG_BIT;}	 while (0)     
#define	clear_LAN0_SIG_BIT()    do {ISR = LAN0_SIG_BIT;}	 while (0)     
#define clear_HALF_RDY_BIT()   	do {ISR = HALF_RDY_BIT;}	 while (0)     
#define clear_SEND_RDY_BIT() 	do {ISR = SEND_RDY_BIT;}	 while (0)     
#define clear_LINK_INT_BIT()	do {ISR = LINK_INT_BIT;}	 while (0)     
#define clear_NRES_INT_BIT()	do {ISR = NRES_INT_BIT;}	 while (0)     
#define clear_WAKE_INT_BIT()	do {ISR = WAKE_INT_BIT;}	 while (0)     
#define clear_OFF_BY_2_BIT()	do {ISR = OFF_BY_2_BIT;}	 while (0)     
#define clear_OFF_BY_1_BIT()	do {ISR = OFF_BY_1_BIT;}	 while (0)     
#define clear_ORUN2_BIT()	do {ISR = ORUN2_BIT;}		 while (0)     
#define clear_ORUN1_BIT()	do {ISR = ORUN1_BIT;}		 while (0)     
#define clear_ORUN2_INT_BIT()	do {ISR = ORUN2_INT_BIT;}	 while (0)     
#define clear_ORUN1_INT_BIT()	do {ISR = ORUN1_INT_BIT;}	 while (0)     
#define clear_TAIL_INT_BIT()   	do {ISR = TAIL_INT_BIT;}	 while (0)     
#define clear_WDOG_INT_BIT()	do {ISR = WDOG_INT_BIT;}	 while (0)     
#define clear_TIME_INT_BIT()	do {ISR = TIME_INT_BIT;}	 while (0)     
#define clear_DMA_INT_BIT()	do {ISR = DMA_INT_BIT;}		 while (0)     
#define clear_SEND_INT_BIT()	do {ISR = SEND_INT_BIT;}	 while (0)     
#define clear_BUFF_INT_BIT()	do {ISR = BUFF_INT_BIT;}	 while (0)     
#define clear_RECV_INT_BIT()	do {ISR = RECV_INT_BIT;}	 while (0)     
#define clear_BYTE_RDY_BIT()  	do {ISR = BYTE_RDY_BIT;}         while (0)     

/* The volatile below keeps this from being optimized away. */
#define START_USER(user_function,stack_pointer) do {			      \
    int START_USER_OUT;							      \
    asm volatile (							      \
       "st %%r0,[0xffffff14] !clear the other context's         \n	      \
        st %%r0,[0xffffff18] !pipeline state                    \n	      \
        st %%r0,[0xffffff1c]                                    \n	      \
        st %%r0,[0xffffff20]                                    \n	      \
        st %%r0,[0xffffff24]                                    \n	      \
        ! The following two instructions must be done in the    \n	      \
        ! supervisor context, otherwise %%2 and %%1 will expand \n	      \
        ! to the wrong register                                 \n	      \
        mov %2,%%r27         !setup %%sp of other context       \n	      \
        mov %1,%%r17         !store function to call where the  \n	      \
                             !  other context can find it       \n	      \
        add %%pc,2*4,%%apc   !apc<-User context trampoline address-4\n	      \
        punt                 !run in user context               \n	      \
        bt.r 4*7             !when user context stops, continue \n	      \
	nop		                             		\n	      \
        ! here is the trampoline, which calls the fuction       \n	      \
        ! specified by %%1 above                                \n	      \
        mov %%sp,%%fp        !set up frame pointer              \n	      \
        add %%pc,8,%%rca     !compute return address            \n	      \
        mov %%r14,%%pc       !call the specified function       \n	      \
        st %%rca,[--%%sp]    !push return address in shadow     \n	      \
        bt .                 !When done, return to system context \n	      \
        punt                 !return to system context when user done."	      \
        : "=r" (START_USER_OUT) : "r" (user_function), "r" (stack_pointer));  \
} while (0)

#define RESUME_USER   asm volatile ("nop\n\tpunt")
#define RESUME_SYSTEM asm volatile ("nop\n\tpunt")

/* The following are faster versions of the above, but do not guarantee that
   all writes to memory are completed before the context switch. */

#define RISKY_RESUME_USER   asm volatile ("punt")
#define RISKY_RESUME_SYSTEM asm volatile ("punt")

extern void reset_Myrinet_ifc(void);

#endif /* LANAI4_DEF_H */

