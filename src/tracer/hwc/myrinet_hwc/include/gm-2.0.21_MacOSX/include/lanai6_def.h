/* lanai6_def.h */

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


/* LANai6.0 */


#ifndef LANAI6_DEF_H
#define LANAI6_DEF_H


/**********************************
 ** Processor-internal registers **
 **********************************/

register unsigned PC 	asm ("pc");  /* Program Counter */
register unsigned CCR 	asm ("ps");  /* Process Status Word */
register int      *SP 	asm ("sp");  /* Stack Pointer */
register unsigned APS 	asm ("aps"); /* Alternate Program Status Word */
register unsigned APC 	asm ("apc"); /* Alternate Program Counter */
register volatile unsigned IMR 	asm ("imr"); /* Interrupt Mask Register */
register volatile unsigned ISR 	asm ("isr"); /* Interrupt Status Register */

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

#define IT0	MM_REG_FULLWORD( 0x60 )	/* interrupt timer */
#define IT1	MM_REG_FULLWORD( 0xC0 )	/* interrupt timer */
#define IT2	MM_REG_FULLWORD( 0xC8 )	/* interrupt timer */
#define RTC	MM_REG_FULLWORD( 0x68 )	/* real-time clock */

#define LAR	MM_REG_FULLWORD_P( 0x70 ) /* EBUS-DMA local address, FPGA access only */
#define CTR	MM_REG_FULLWORD  ( 0x78 ) /* EBUS-DMA counter, FPGA access only */

#define RMP	MM_REG_FULLWORD_P( 0xE0 ) /* receive-DMA pointer */
#define RMC	MM_REG_FULLWORD_P( 0xD8 ) /* pointer to receive-DMA header CRC-32 */
#define RMW	MM_REG_FULLWORD_P( 0xD0 ) /* the same as RMC, but does not trigger CRC-32 */
#define RML	MM_REG_FULLWORD_P( 0xE8 ) /* receive-DMA limit */

#define SMP	MM_REG_FULLWORD_P( 0xF0 ) /* send-DMA pointer */
#define SMH	MM_REG_FULLWORD_P( 0xF8 ) /* pointer to send-DMA routing header end */
#define SMC	MM_REG_FULLWORD_P( 0x110 ) /* pointer to send-DMA header CRC-32 */
#define SML	MM_REG_FULLWORD_P( 0x100 ) /* send-DMA limit */
#define SMLT	MM_REG_FULLWORD_P( 0x108 ) /* send-DMA limit with tail */

#define SA      MM_REG_FULLWORD( 0x118 )	/* send align */

#define LINK	MM_REG_FULLWORD( 0x128 ) /* LINK */
#define MYRINET	MM_REG_FULLWORD( 0x130 )
#define		NRES_ENABLE_BIT		(0x1)
#define		CRC32_ENABLE_BIT	(0x2)

#define DEBUG   MM_REG_FULLWORD( 0x138 ) /* hardware debugging */
#define LED	MM_REG_FULLWORD( 0x140 ) /* LED */
#define PULSE	MM_REG_FULLWORD( 0xB8  ) /* P0,P1,P2 output pins */
#define	WRITE_ENABLE MM_REG_FULLWORD( 0x150 ) /* memory protection */

/*************************
 ** Interrupt bit names **
 *************************/

#define	DBG_BIT		0x80000000
#define	DEBUG_BIT	DBG_BIT
#define	HOST_SIG_BIT	0x40000000

#define	LINK2_INT_BIT	0x04000000
#define	LINK1_INT_BIT	0x02000000
#define	LINK0_INT_BIT	0x01000000
#define	LAN7_SIG_BIT	0x00800000
#define	LAN6_SIG_BIT	0x00400000
#define	LAN5_SIG_BIT	0x00200000
#define	LAN4_SIG_BIT	0x00100000
#define	LAN3_SIG_BIT	0x00080000
#define	LAN2_SIG_BIT	0x00040000
#define	LAN1_SIG_BIT	0x00020000
#define	LAN0_SIG_BIT	0x00010000
#define PAR_INT_BIT	0x00008000
#define MEM_INT_BIT	0x00004000
#define TIME2_INT_BIT	0x00002000
#define WAKE_INT_BIT	0x00001000
#define NRES_INT_BIT	0x00000800
#define OFF_BY_4_BIT	0x00000400
#define OFF_BY_2_BIT	0x00000200
#define OFF_BY_1_BIT	0x00000100
#define TIME1_INT_BIT	0x00000080
#define TIME0_INT_BIT	0x00000040
#define LAN9_SIG_BIT	0x00000020
#define LAN8_SIG_BIT	0x00000010
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

#define	clear_LINK2_INT_BIT()   {ISR = LINK2_INT_BIT;}
#define	clear_LINK1_INT_BIT()   {ISR = LINK1_INT_BIT;}
#define	clear_LINK0_INT_BIT()   {ISR = LINK0_INT_BIT;}
#define	clear_LAN7_SIG_BIT()    {ISR = LAN7_SIG_BIT;}
#define	clear_LAN6_SIG_BIT()    {ISR = LAN6_SIG_BIT;}
#define	clear_LAN5_SIG_BIT()    {ISR = LAN5_SIG_BIT;}
#define	clear_LAN4_SIG_BIT()    {ISR = LAN4_SIG_BIT;}
#define	clear_LAN3_SIG_BIT()    {ISR = LAN3_SIG_BIT;}
#define	clear_LAN2_SIG_BIT()    {ISR = LAN2_SIG_BIT;}
#define	clear_LAN1_SIG_BIT()    {ISR = LAN1_SIG_BIT;}
#define	clear_LAN0_SIG_BIT()    {ISR = LAN0_SIG_BIT;}
#define clear_PARITY_INT_BIT()	{ISR = PARITY_INT_BIT;}
#define clear_MEMORY_INT_BIT()	{ISR = MEMORY_INT_BIT;}
#define clear_TIME2_INT_BIT()	{ISR = TIME2_INT_BIT;}
#define clear_WAKE_INT_BIT()	{ISR = WAKE_INT_BIT;}
#define clear_NRES_INT_BIT()	{ISR = NRES_INT_BIT;}
#define clear_OFF_BY_4_BIT()	{ISR = OFF_BY_4_BIT;}
#define clear_OFF_BY_2_BIT()	{ISR = OFF_BY_2_BIT;}
#define clear_OFF_BY_1_BIT()	{ISR = OFF_BY_1_BIT;}
#define clear_ORUN4_BIT()	{ISR = ORUN4_BIT;}
#define clear_ORUN2_BIT()	{ISR = ORUN2_BIT;}
#define clear_ORUN1_BIT()	{ISR = ORUN1_BIT;}
#define clear_ORUN4_INT_BIT()	{ISR = ORUN4_INT_BIT;}
#define clear_ORUN2_INT_BIT()	{ISR = ORUN2_INT_BIT;}
#define clear_ORUN1_INT_BIT()	{ISR = ORUN1_INT_BIT;}
#define clear_TIME1_INT_BIT()	{ISR = TIME1_INT_BIT;}
#define clear_TIME0_INT_BIT()	{ISR = TIME0_INT_BIT;}
#define	clear_LINK9_SIG_BIT()   {ISR = LINK9_SIG_BIT;}
#define	clear_LINK8_SIG_BIT()   {ISR = LINK8_SIG_BIT;}
#define clear_SEND_INT_BIT()	{ISR = SEND_INT_BIT;}
#define clear_BUFF_INT_BIT()	{ISR = BUFF_INT_BIT;}
#define clear_RECV_INT_BIT()	{ISR = RECV_INT_BIT;}
#define clear_HEAD_INT_BIT()	{ISR = HEAD_INT_BIT;}

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
    unsigned int next;  /* next pointer */
    unsigned short csum[2]; /* 2 16-bit ones complement checksums of this block */
    unsigned int len;   /* byte count */
    unsigned int lar;   /* lanai address */
    unsigned int eah;   /* high 32bit PCI address */
    unsigned int eal;   /*  low 32bit PCI address */
};

#define DMA_L2E     0x0
#define DMA_E2L     0x1
#define DMA_TERMINAL    0x2 /* terminal block? */
#define DMA_WAKE    0x4 /* wake when block completes */




#endif /* LANAI6_DEF_H */

