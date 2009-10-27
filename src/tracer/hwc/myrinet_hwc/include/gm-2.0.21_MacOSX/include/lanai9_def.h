/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* lanai9_def.h */

/* WARNING: There may be some extra definitions in this file.  Only
   the documented registers (from http://www.myri.com/vlsi/Lanai8.pdf)
   should be used. */

#ifndef LANAI9_DEF_H
#define LANAI9_DEF_H

/*****************************
 ** Memory-mapped registers **
 *****************************/

#define MM_REG_FULLWORD_P(X) (*((void* volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_FULLWORD(X)   (*((int   volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_HALFWORD(X)   (*((short volatile * const ) (0xFFFFFE00 + (X))))
#define MM_REG_BYTE(X)       (*((char  volatile * const ) (0xFFFFFE00 + (X))))

#define CPUC	MM_REG_FULLWORD   (0x078)	  
#define DEBUG   MM_REG_FULLWORD   (0x138)
#define EIMR	MM_REG_FULLWORD   (0x058)
#define ISR	MM_REG_FULLWORD   (0x050)
#define IT0	MM_REG_FULLWORD   (0x060)
#define IT1	MM_REG_FULLWORD   (0x0C0)
#define IT2	MM_REG_FULLWORD   (0x0C8)
#define LAR	MM_REG_FULLWORD_P (0x070)
#define LED	MM_REG_FULLWORD   (0x140)
#define MYRINET	MM_REG_FULLWORD   (0x130)
#define PULSE	MM_REG_FULLWORD   (0x0B8)
#define RMC	MM_REG_FULLWORD_P (0x0D8)
#define RML	MM_REG_FULLWORD_P (0x0E8)
#define RMP	MM_REG_FULLWORD_P (0x0E0)
#define RMW	MM_REG_FULLWORD_P (0x0D0)
#define RTC	MM_REG_FULLWORD   (0x068)
#define SA      MM_REG_FULLWORD   (0x118)
#define SMC	MM_REG_FULLWORD_P (0x110)
#define SMH	MM_REG_FULLWORD_P (0x0F8)
#define SML	MM_REG_FULLWORD_P (0x100)
#define SMLT	MM_REG_FULLWORD_P (0x108)
#define SMP	MM_REG_FULLWORD_P (0x0F0)

/* Bits in the IMR and ISR */

#define DEBUG_BIT	(1<<31)
#define HOST_SIG_BIT	(1<<30)
#define LANAI_SIG_BIT	(1<<29)

#define LINK_DOWN_BIT	(1<<16)
#define PAR_INT_BIT	(1<<15)
#define MEM_INT_BIT	(1<<14)
#define TIME2_INT_BIT	(1<<13)
#define WAKE0_INT_BIT	(1<<12)
#define NRES_INT_BIT	(1<<11)
#define ORUN4_INT_BIT	(1<<10)
#define ORUN2_INT_BIT	(1<<9)
#define ORUN1_INT_BIT	(1<<8)
#define TIME1_INT_BIT	(1<<7)
#define TIME0_INT_BIT	(1<<6)
#define WAKE2_INT_BIT	(1<<5)
#define WAKE1_INT_BIT	(1<<4)
#define SEND_INT_BIT 	(1<<3)
#define BUFF_INT_BIT 	(1<<2)
#define RECV_INT_BIT 	(1<<1)
#define HEAD_INT_BIT 	(1<<0)

/* Bits in the MYRINET register */

#define		NRES_ENABLE_BIT		(1<<0)
#define		CRC32_ENABLE_BIT	(1<<1)
#define		TX_CRC8_ENABLE_BIT	(1<<2)
#define		RX_CRC8_ENABLE_BIT	(1<<3)
#define		ILLEGAL_ENABLE		(1<<4)
#define		BEAT_ENABLE		(1<<5)
#define		TIMEOUT0		(1<<6)
#define		TIMEOUT1		(1<<7)
#define		WINDOW0			(1<<8)
#define		WINDOW1			(1<<9)

/****************
 * DMA descriptor
 ****************/

struct DMA_BLOCK {
    unsigned int next;  /* next pointer */
    unsigned short csum[2]; /* 2 16-bit ones complement checksums of this block */
    unsigned int len;   /* byte count */
    unsigned int lar;   /* lanai address register */
    unsigned int eah;   /* high 32bit external (PCI) address */
    unsigned int eal;   /*  low 32bit external (PCI) address */
};

/* bits in the NEXT field */

#define DMA_L2E		0x0	/* LBUS to EBUS (lanai to host) */
#define DMA_E2L		0x1	/* EBUS to LBUS (host to lanai) */
#define DMA_TERMINAL    0x2	/* terminal block? */
#define DMA_WAKE	0x4	/* wake when block completes */

#endif /* LANAI9_DEF_H */
