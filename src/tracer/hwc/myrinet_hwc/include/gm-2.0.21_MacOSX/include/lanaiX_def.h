/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 2001 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

#ifndef LANAIX_DEF_H
#define LANAIX_DEF_H




/*****************************
 ** Memory-mapped registers **
 *****************************/

/* Port 0 receive special registers, 0-8 */
#define	P0_RECV_REG_OFFSET	0
#define	P0_RECV_REG(A)		(((P0_RECV_REG_OFFSET+A)<<3))
#define	P0_RECV_REG_MSH(A)	(((P0_RECV_REG_OFFSET+A)<<3)+0)
#define	P0_RECV_REG_LSH(A)	(((P0_RECV_REG_OFFSET+A)<<3)+2)
#define	P0_RECV_REG_MSB(A)	(((P0_RECV_REG_OFFSET+A)<<3)+0)
#define	P0_RECV_REG_msb(A)	(((P0_RECV_REG_OFFSET+A)<<3)+1)
#define	P0_RECV_REG_lsb(A)	(((P0_RECV_REG_OFFSET+A)<<3)+2)
#define	P0_RECV_REG_LSB(A)	(((P0_RECV_REG_OFFSET+A)<<3)+3)

/* Port 0 send special registers, 0-4 */
#define	P0_SEND_REG_OFFSET	9
#define	P0_SEND_REG(A)		(((P0_SEND_REG_OFFSET+A)<<3))
#define	P0_SEND_REG_MSH(A)	(((P0_SEND_REG_OFFSET+A)<<3)+0)
#define	P0_SEND_REG_LSH(A)	(((P0_SEND_REG_OFFSET+A)<<3)+2)
#define	P0_SEND_REG_MSB(A)	(((P0_SEND_REG_OFFSET+A)<<3)+0)
#define	P0_SEND_REG_msb(A)	(((P0_SEND_REG_OFFSET+A)<<3)+1)
#define	P0_SEND_REG_lsb(A)	(((P0_SEND_REG_OFFSET+A)<<3)+2)
#define	P0_SEND_REG_LSB(A)	(((P0_SEND_REG_OFFSET+A)<<3)+3)

/* Port-access special registers, 0-1 */
#define	C_PORT_REG_OFFSET	14
#define	C_PORT_REG(A)		(((C_PORT_REG_OFFSET+A)<<3))
#define	C_PORT_REG_MSH(A)	(((C_PORT_REG_OFFSET+A)<<3)+0)
#define	C_PORT_REG_LSH(A)	(((C_PORT_REG_OFFSET+A)<<3)+2)
#define	C_PORT_REG_MSB(A)	(((C_PORT_REG_OFFSET+A)<<3)+0)
#define	C_PORT_REG_msb(A)	(((C_PORT_REG_OFFSET+A)<<3)+1)
#define	C_PORT_REG_lsb(A)	(((C_PORT_REG_OFFSET+A)<<3)+2)
#define	C_PORT_REG_LSB(A)	(((C_PORT_REG_OFFSET+A)<<3)+3)

/* Port 1 receive special registers, 0-8 */
#define	P1_RECV_REG_OFFSET	16
#define	P1_RECV_REG(A)		(((P1_RECV_REG_OFFSET+A)<<3))
#define	P1_RECV_REG_MSH(A)	(((P1_RECV_REG_OFFSET+A)<<3)+0)
#define	P1_RECV_REG_LSH(A)	(((P1_RECV_REG_OFFSET+A)<<3)+2)
#define	P1_RECV_REG_MSB(A)	(((P1_RECV_REG_OFFSET+A)<<3)+0)
#define	P1_RECV_REG_msb(A)	(((P1_RECV_REG_OFFSET+A)<<3)+1)
#define	P1_RECV_REG_lsb(A)	(((P1_RECV_REG_OFFSET+A)<<3)+2)
#define	P1_RECV_REG_LSB(A)	(((P1_RECV_REG_OFFSET+A)<<3)+3)

/* Port 1 send special registers, 0-4 */
#define	P1_SEND_REG_OFFSET	25
#define	P1_SEND_REG(A)		(((P1_SEND_REG_OFFSET+A)<<3))
#define	P1_SEND_REG_MSH(A)	(((P1_SEND_REG_OFFSET+A)<<3)+0)
#define	P1_SEND_REG_LSH(A)	(((P1_SEND_REG_OFFSET+A)<<3)+2)
#define	P1_SEND_REG_MSB(A)	(((P1_SEND_REG_OFFSET+A)<<3)+0)
#define	P1_SEND_REG_msb(A)	(((P1_SEND_REG_OFFSET+A)<<3)+1)
#define	P1_SEND_REG_lsb(A)	(((P1_SEND_REG_OFFSET+A)<<3)+2)
#define	P1_SEND_REG_LSB(A)	(((P1_SEND_REG_OFFSET+A)<<3)+3)

/* Copy-Engine special registers, 0-2 */
#define	COPY_REG_OFFSET		30
#define	COPY_REG(A)		(((COPY_REG_OFFSET+A)<<3))
#define	COPY_REG_MSH(A)		(((COPY_REG_OFFSET+A)<<3)+0)
#define	COPY_REG_LSH(A)		(((COPY_REG_OFFSET+A)<<3)+2)
#define	COPY_REG_MSB(A)		(((COPY_REG_OFFSET+A)<<3)+0)
#define	COPY_REG_msb(A)		(((COPY_REG_OFFSET+A)<<3)+1)
#define	COPY_REG_lsb(A)		(((COPY_REG_OFFSET+A)<<3)+2)
#define	COPY_REG_LSB(A)		(((COPY_REG_OFFSET+A)<<3)+3)

/* CRC32-Engine special registers, 0-3 */
#define	CRC32_REG_OFFSET	33
#define	CRC32_REG(A)		(((CRC32_REG_OFFSET+A)<<3))
#define	CRC32_REG_MSH(A)	(((CRC32_REG_OFFSET+A)<<3)+0)
#define	CRC32_REG_LSH(A)	(((CRC32_REG_OFFSET+A)<<3)+2)
#define	CRC32_REG_MSB(A)	(((CRC32_REG_OFFSET+A)<<3)+0)
#define	CRC32_REG_msb(A)	(((CRC32_REG_OFFSET+A)<<3)+1)
#define	CRC32_REG_lsb(A)	(((CRC32_REG_OFFSET+A)<<3)+2)
#define	CRC32_REG_LSB(A)	(((CRC32_REG_OFFSET+A)<<3)+3)

/* Regular special registers, 0-8 */
#define	SPEC_REG_OFFSET		37
#define	SPEC_REG(A)		(((SPEC_REG_OFFSET+A)<<3))
#define	SPEC_REG_MSH(A)		(((SPEC_REG_OFFSET+A)<<3)+0)
#define	SPEC_REG_LSH(A)		(((SPEC_REG_OFFSET+A)<<3)+2)
#define	SPEC_REG_MSB(A)		(((SPEC_REG_OFFSET+A)<<3)+0)
#define	SPEC_REG_msb(A)		(((SPEC_REG_OFFSET+A)<<3)+1)
#define	SPEC_REG_lsb(A)		(((SPEC_REG_OFFSET+A)<<3)+2)
#define	SPEC_REG_LSB(A)		(((SPEC_REG_OFFSET+A)<<3)+3)

/* Dispatch special registers, 0-5 */
#define	DISPATCH_REG_OFFSET	46
#define	DISPATCH_REG(A)		(((DISPATCH_REG_OFFSET+A)<<3))
#define	DISPATCH_REG_MSH(A)	(((DISPATCH_REG_OFFSET+A)<<3)+0)
#define	DISPATCH_REG_LSH(A)	(((DISPATCH_REG_OFFSET+A)<<3)+2)
#define	DISPATCH_REG_MSB(A)	(((DISPATCH_REG_OFFSET+A)<<3)+0)
#define	DISPATCH_REG_msb(A)	(((DISPATCH_REG_OFFSET+A)<<3)+1)
#define	DISPATCH_REG_lsb(A)	(((DISPATCH_REG_OFFSET+A)<<3)+2)
#define	DISPATCH_REG_LSB(A)	(((DISPATCH_REG_OFFSET+A)<<3)+3)

/* Arbiter special registers, 0-1 */
#define	ARBITER_REG_OFFSET	52
#define	ARBITER_REG(A)		(((ARBITER_REG_OFFSET+A)<<3))
#define	ARBITER_REG_MSH(A)	(((ARBITER_REG_OFFSET+A)<<3)+0)
#define	ARBITER_REG_LSH(A)	(((ARBITER_REG_OFFSET+A)<<3)+2)
#define	ARBITER_REG_MSB(A)	(((ARBITER_REG_OFFSET+A)<<3)+0)
#define	ARBITER_REG_msb(A)	(((ARBITER_REG_OFFSET+A)<<3)+1)
#define	ARBITER_REG_lsb(A)	(((ARBITER_REG_OFFSET+A)<<3)+2)
#define	ARBITER_REG_LSB(A)	(((ARBITER_REG_OFFSET+A)<<3)+3)

/* Memory special registers, 0-1 */
#define	MP_REG_OFFSET		54
#define	MP_REG(A)		(((MP_REG_OFFSET+A)<<3))
#define	MP_REG_MSH(A)		(((MP_REG_OFFSET+A)<<3)+0)
#define	MP_REG_LSH(A)		(((MP_REG_OFFSET+A)<<3)+2)
#define	MP_REG_MSB(A)		(((MP_REG_OFFSET+A)<<3)+0)
#define	MP_REG_msb(A)		(((MP_REG_OFFSET+A)<<3)+1)
#define	MP_REG_lsb(A)		(((MP_REG_OFFSET+A)<<3)+2)
#define	MP_REG_LSB(A)		(((MP_REG_OFFSET+A)<<3)+3)

/* JTAG special registers, 0 */
#define	JTAG_REG_OFFSET		56
#define	JTAG_REG(A)		(((JTAG_REG_OFFSET+A)<<3))
#define	JTAG_REG_MSH(A)		(((JTAG_REG_OFFSET+A)<<3)+0)
#define	JTAG_REG_LSH(A)		(((JTAG_REG_OFFSET+A)<<3)+2)
#define	JTAG_REG_MSB(A)		(((JTAG_REG_OFFSET+A)<<3)+0)
#define	JTAG_REG_msb(A)		(((JTAG_REG_OFFSET+A)<<3)+1)
#define	JTAG_REG_lsb(A)		(((JTAG_REG_OFFSET+A)<<3)+2)
#define	JTAG_REG_LSB(A)		(((JTAG_REG_OFFSET+A)<<3)+3)

/* ISR-related special registers, 0-3 */
#define	ISR_REG_OFFSET		60
#define	ISR_REG(A)		(((ISR_REG_OFFSET+A)<<3))
#define	ISR_REG_MSH(A)		(((ISR_REG_OFFSET+A)<<3)+0)
#define	ISR_REG_LSH(A)		(((ISR_REG_OFFSET+A)<<3)+2)
#define	ISR_REG_MSB(A)		(((ISR_REG_OFFSET+A)<<3)+0)
#define	ISR_REG_msb(A)		(((ISR_REG_OFFSET+A)<<3)+1)
#define	ISR_REG_lsb(A)		(((ISR_REG_OFFSET+A)<<3)+2)
#define	ISR_REG_LSB(A)		(((ISR_REG_OFFSET+A)<<3)+3)

/* PCI/DMA special registers, 0-9 */
#define	PCIDMA_REG_OFFSET	(16+64)
#define	PCIDMA_REG(A)		(((PCIDMA_REG_OFFSET+A)<<3))
#define	PCIDMA_REG_MSH(A)	(((PCIDMA_REG_OFFSET+A)<<3)+0)
#define	PCIDMA_REG_LSH(A)	(((PCIDMA_REG_OFFSET+A)<<3)+2)
#define	PCIDMA_REG_MSB(A)	(((PCIDMA_REG_OFFSET+A)<<3)+0)
#define	PCIDMA_REG_msb(A)	(((PCIDMA_REG_OFFSET+A)<<3)+1)
#define	PCIDMA_REG_lsb(A)	(((PCIDMA_REG_OFFSET+A)<<3)+2)
#define	PCIDMA_REG_LSB(A)	(((PCIDMA_REG_OFFSET+A)<<3)+3)




#define MMREG_VOID_PTR(A)      (*((void* volatile * const ) (0xFFFFFC00 + (A))))
#define MMREG_WORD(A) (*((unsigned int   volatile * const ) (0xFFFFFC00 + (A))))
#define MMREG_HALF(A) (*((unsigned short volatile * const ) (0xFFFFFC00 + (A))))
#define MMREG_BYTE(A) (*((unsigned char  volatile * const ) (0xFFFFFC00 + (A))))




#define	CRC32			MMREG_WORD(CRC32_REG(0))
#define	CRC32_BYTE		MMREG_BYTE(CRC32_REG_lsb(1))
#define	CRC32_HALF		MMREG_HALF(CRC32_REG_MSH(2))
#define	CRC32_WORD		MMREG_WORD(CRC32_REG(3))

#define	COPY			MMREG_WORD(COPY_REG(0))
#define	COPY_FROM		MMREG_VOID_PTR(COPY_REG(0))
#define	COPY_SIZE		MMREG_WORD(COPY_REG(0))
#define	COPY_TO			MMREG_VOID_PTR(COPY_REG(0))
#define		COPY_TO_CRC32		(0x80000000)
#define		COPY_LENGTH		(0x7FFFFFFF)

#define COPY_ABORT		MMREG_BYTE(COPY_REG_lsb(1))
#define CRC32_CONFIG		MMREG_BYTE(COPY_REG_lsb(2))

#define	CPUC			MMREG_WORD(SPEC_REG(0))
#define	RTC			MMREG_WORD(SPEC_REG(1))
#define	IT0			MMREG_WORD(SPEC_REG(2))
#define	IT1			MMREG_WORD(SPEC_REG(3))
#define	IT2			MMREG_WORD(SPEC_REG(4))
#define IT(n)			MMREG_WORD(SPEC_REG(2+(n)))


#define	LED			MMREG_BYTE(SPEC_REG_MSB(5))

#define	MDI			MMREG_BYTE(SPEC_REG_MSB(6))
#define		MDC			(0x04)
#define		MDEN			(0x02)
#define		MDIO			(0x01)

#define	SYNC			MMREG_WORD(SPEC_REG(7))
#define		SYNC_DEBUG		(0xFC000000)
#define		SYNC_P1_P1		(0x00038000)
#define		SYNC_P1_CPU		(0x00007000)
#define		SYNC_CPU_P1		(0x00000E00)
#define		SYNC_P0_P0		(0x000001C0)
#define		SYNC_P0_CPU		(0x00000038)
#define		SYNC_CPU_P0		(0x00000007)

#define	PLL			MMREG_WORD(SPEC_REG(8))
#define		P1_SDET_INV		(0x80000000)
#define		P1_TX_RATE		(0x40000000)
#define		P1_TX_PLL		(0x38000000)
#define		P1_TX_DN		(0x04000000)
#define		P1_TX_DBL		(0x02000000)
#define		P1_TX_INV		(0x01000000)
#define		P1_RX_SYNC		(0x00800000)
#define		P1_RX_RATE		(0x00400000)
#define		P1_RX_PLL		(0x00380000)
#define		P1_RX_DN		(0x00040000)
#define		P1_RX_DBL		(0x00020000)
#define		P1_RX_INV		(0x00010000)
#define		P0_SDET_INV		(0x00008000)
#define		P0_TX_RATE		(0x00004000)
#define		P0_TX_PLL		(0x00003800)
#define		P0_TX_DN		(0x00000400)
#define		P0_TX_DBL		(0x00000200)
#define		P0_TX_INV		(0x00000100)
#define		P0_RX_SYNC		(0x00000080)
#define		P0_RX_RATE		(0x00000040)
#define		P0_RX_PLL		(0x00000038)
#define		P0_RX_DN		(0x00000004)
#define		P0_RX_DBL		(0x00000002)
#define		P0_RX_INV		(0x00000001)

#define		PLL_DISABLED		(0x0)
#define		PLL_MULT1_DIV1		(0x1)
#define		PLL_MULT2_DIV1		(0x2)
#define		PLL_MULT4_DIV1		(0x3)
#define		PLL_MULT2_DIV2		(0x4)
#define		PLL_MULT4_DIV2		(0x5)
#define		PLL_MULT4_DIV4		(0x6)

#define	PLL_XM_M				\
		( P0_RX_DBL			\
		| (PLL_MULT2_DIV1 <<  3)	\
		| P0_RX_RATE			\
		| P0_TX_DBL			\
		| (PLL_MULT2_DIV1 << 11)	\
		| P0_TX_RATE			\
		| (PLL_MULT2_DIV1 << 27)	\
		| P0_TX_INV			\
		| P0_RX_INV			\
		)
#define	PLL_XM_IB	PLL_XM_M
#define	PLL_XM_GEX				\
		( P0_RX_DN			\
		| (PLL_MULT2_DIV2 <<  3)	\
		| P0_RX_SYNC			\
		| (PLL_MULT2_DIV2 << 11)	\
		| (PLL_MULT2_DIV1 << 27)	\
		| P0_TX_INV			\
		| P0_RX_INV			\
		)
#define	PLL_XM_GMII				\
		( (PLL_MULT2_DIV2 <<  3)	\
		| P0_TX_DN			\
		| (PLL_MULT2_DIV2 << 11)	\
		| (PLL_MULT2_DIV1 << 27)	\
		)

#define	PLL_XP_M				\
		( P0_RX_DBL			\
		| (PLL_MULT2_DIV1 <<  3)	\
		| P0_RX_RATE			\
		| P0_TX_DBL			\
		| (PLL_MULT2_DIV1 << 11)	\
		| P0_TX_RATE			\
		| P0_RX_INV			\
		)
#define	PLL_XP_IB	PLL_XP_M
#define	PLL_XP_GEX				\
		( P0_RX_DN			\
		| (PLL_MULT2_DIV2 <<  3)	\
		| P0_RX_SYNC			\
		| (PLL_MULT2_DIV2 << 11)	\
		| P0_RX_INV			\
		)
#define	PLL_XP_GMII				\
		( (PLL_MULT2_DIV2 <<  3)	\
		| P0_TX_DN			\
		| (PLL_MULT2_DIV2 << 11)	\
		)

#define	PLL_2XP_M	(PLL_XP_M + (PLL_XP_M << 16))

#define	PLL_M					\
		( P0_RX_DBL			\
		| (PLL_MULT2_DIV1 <<  3)	\
		| P0_RX_RATE			\
		| P0_TX_DBL			\
		| (PLL_MULT2_DIV1 << 11)	\
		| P0_TX_RATE			\
		| P0_RX_INV			\
		)
#define	PLL_IB		PLL_M
#define	PLL_GEX					\
		( P0_RX_DN			\
		| (PLL_MULT2_DIV2 <<  3)	\
		| P0_RX_SYNC			\
		| (PLL_MULT2_DIV2 << 11)	\
		| P0_RX_INV			\
		| P0_SDET_INV			\
		)
#define	PLL_GMII				\
		( (PLL_MULT2_DIV2 <<  3)	\
		| P0_TX_DN			\
		| (PLL_MULT2_DIV2 << 11)	\
		)

#define SYNC_TABLE_XM_M_SIZE	300
#define SYNC_TABLE_XM_GEX_SIZE	300

extern	unsigned int	SYNC_TABLE_XM_M[SYNC_TABLE_XM_M_SIZE];
extern	unsigned int	SYNC_TABLE_XM_GEX[SYNC_TABLE_XM_GEX_SIZE];

#define SYNC_TABLE_XM_IB	SYNC_TABLE_XM_M
#define SYNC_TABLE_XM_GMII	SYNC_TABLE_XM_GEX

#define	SYNC_TABLE_XP_M		SYNC_TABLE_XM_M
#define SYNC_TABLE_XP_IB	SYNC_TABLE_XM_IB
#define	SYNC_TABLE_XP_GEX	SYNC_TABLE_XM_GEX
#define SYNC_TABLE_XP_GMII	SYNC_TABLE_XM_GMII

#define SYNC_TABLE_M_SIZE	400
#define SYNC_TABLE_GEX_SIZE	400

extern	const unsigned char	SYNC_TABLE_M[SYNC_TABLE_M_SIZE];
extern	unsigned short	SYNC_TABLE_GEX[SYNC_TABLE_GEX_SIZE];

#define SYNC_TABLE_IB		SYNC_TABLE_M
#define SYNC_TABLE_GMII		SYNC_TABLE_GEX

#define	ARBITER_SLOT		MMREG_BYTE(ARBITER_REG_MSB(0))
#define	ARBITER_CODE		MMREG_BYTE(ARBITER_REG_MSB(1))
#define		CODE_JTAG		(0x0)
#define		CODE_P0_R		(0x1)
#define		CODE_P0_S		(0x2)
#define		CODE_P1_R		(0x3)
#define		CODE_P1_S		(0x4)
#define		CODE_PCIX		(0x5)
#define		CODE_DRAM		(0x6)
#define		CODE_COPY		(0x7)
#define		CODE_CPU		(0x8)
#define		CODE_WRAP		(0xF)

#define	MP_LOWER		MMREG_WORD(MP_REG(0))
#define	MP_UPPER		MMREG_WORD(MP_REG(1))

#define	PORT_ADDR		MMREG_HALF(C_PORT_REG_MSH(0))
#define		PORT_READ		(0x8000)
#define		PORT_REG		(0x003F)

#define	PORT_DATA		MMREG_HALF(C_PORT_REG_MSH(1))

#define	P0_SEND			MMREG_VOID_PTR(P0_SEND_REG(0))
#define	P0_SEND_POINTER		MMREG_VOID_PTR(P0_SEND_REG(0))
#define	P0_SEND_LENGTH		MMREG_WORD(P0_SEND_REG(0))
#define	P1_SEND			MMREG_VOID_PTR(P1_SEND_REG(0))
#define	P1_SEND_POINTER		MMREG_VOID_PTR(P1_SEND_REG(0))
#define	P1_SEND_LENGTH		MMREG_WORD(P1_SEND_REG(0))

#define	P0_SEND_FREE_COUNT	MMREG_BYTE(P0_SEND_REG_LSB(0))
#define	P1_SEND_FREE_COUNT	MMREG_BYTE(P1_SEND_REG_LSB(0))

#define	P0_SEND_FREE_LIMIT	MMREG_BYTE(P0_SEND_REG_LSB(1))
#define	P1_SEND_FREE_LIMIT	MMREG_BYTE(P1_SEND_REG_LSB(1))

#define	P0_SEND_COUNT		MMREG_BYTE(P0_SEND_REG_LSB(2))
#define	P1_SEND_COUNT		MMREG_BYTE(P1_SEND_REG_LSB(2))

#define	P0_PAUSE_COUNT		MMREG_BYTE(P0_SEND_REG_LSB(3))
#define	P1_PAUSE_COUNT		MMREG_BYTE(P1_SEND_REG_LSB(3))

#define	P0_SEND_CONTROL		MMREG_BYTE(P0_SEND_REG_LSB(4))
#define	P1_SEND_CONTROL		MMREG_BYTE(P1_SEND_REG_LSB(4))
#define		SEND_ON			(0x00)
#define		SEND_PAUSE		(0x01)
#define		SEND_OFF		(0x02)
#define		SEND_FLUSH		(0x03)

#define	P0_MEMORY_DROP		MMREG_HALF(P0_RECV_REG_MSH(8))
#define	P1_MEMORY_DROP		MMREG_HALF(P1_RECV_REG_MSH(8))

#define	P0_RB			MMREG_VOID_PTR(P0_RECV_REG(1))
#define	P0_A_RB			MMREG_VOID_PTR(P0_RECV_REG(4))
#define	P1_RB			MMREG_VOID_PTR(P1_RECV_REG(1))
#define	P1_A_RB			MMREG_VOID_PTR(P1_RECV_REG(4))
#define		DESC_COMPLETE		(0x80000000)
#define		DESC_POINTER		(0x7FFFFFFF)

#define	P0_RPL_CONFIG		MMREG_WORD(P0_RECV_REG(0))
#define	P0_A_RPL_CONFIG		MMREG_WORD(P0_RECV_REG(3))
#define	P1_RPL_CONFIG		MMREG_WORD(P1_RECV_REG(0))
#define	P1_A_RPL_CONFIG		MMREG_WORD(P1_RECV_REG(3))
#define		PORT_ENABLE		(0x80000000)
#define		ENABLE_PAUSE_TIMER	(0x10000000)
#define		ENABLE_BUFFER_DROP	(0x08000000)
#define		MATCH_FLAGS		(0x07000000)
#define		MATCH_LINK		(0x04000000)
#define		MATCH_OR		(0x02000000)
#define		MATCH_VL15		(0x01000000)
#define		HEADER_LENGTH		(0x00FF0000)
#define		STOP_SIZE		(0x0000F000)
#define		MAX_PACKET		(0x00000F00)
#define		BLOCK_SIZE		(0x000000F0)
#define		BUFFER_SIZE		(0x0000000F)

#define STOP_SIZE_512    0x0000
#define STOP_SIZE_1K     0x1000
#define STOP_SIZE_2K     0x2000
#define STOP_SIZE_4K     0x3000
#define STOP_SIZE_8K     0x4000
#define STOP_SIZE_16K    0x5000
#define STOP_SIZE_32K    0x6000
#define STOP_SIZE_64K    0x7000
#define STOP_SIZE_128K   0x8000
#define STOP_SIZE_256K   0x9000
#define STOP_SIZE_512K   0xa000
#define STOP_SIZE_1M     0xb000
#define STOP_SIZE_2M     0xc000
#define STOP_SIZE_4M     0xd000
#define STOP_SIZE_8M     0xe000
#define STOP_SIZE_16M    0xf000

#define MAX_PACKET_512    0x000
#define MAX_PACKET_1K     0x100
#define MAX_PACKET_2K     0x200
#define MAX_PACKET_4K     0x300
#define MAX_PACKET_8K     0x400
#define MAX_PACKET_16K    0x500
#define MAX_PACKET_32K    0x600
#define MAX_PACKET_64K    0x700
#define MAX_PACKET_128K   0x800
#define MAX_PACKET_256K   0x900
#define MAX_PACKET_512K   0xa00
#define MAX_PACKET_1M     0xb00
#define MAX_PACKET_2M     0xc00
#define MAX_PACKET_4M     0xd00
#define MAX_PACKET_8M     0xe00
#define MAX_PACKET_16M    0xf00

#define BLOCK_SIZE_8       0x00
#define BLOCK_SIZE_16      0x10
#define BLOCK_SIZE_32      0x20
#define BLOCK_SIZE_64      0x30
#define BLOCK_SIZE_128     0x40
#define BLOCK_SIZE_256     0x50
#define BLOCK_SIZE_512     0x60
#define BLOCK_SIZE_1K      0x70
#define BLOCK_SIZE_2K      0x80
#define BLOCK_SIZE_4K      0x90
#define BLOCK_SIZE_8K      0xa0
#define BLOCK_SIZE_16K     0xb0
#define BLOCK_SIZE_32K     0xc0
#define BLOCK_SIZE_64K     0xd0
#define BLOCK_SIZE_128K    0xe0
#define BLOCK_SIZE_256K    0xf0

#define BUFFER_SIZE_512     0x0
#define BUFFER_SIZE_1K      0x1
#define BUFFER_SIZE_2K      0x2
#define BUFFER_SIZE_4K      0x3
#define BUFFER_SIZE_8K      0x4
#define BUFFER_SIZE_16K     0x5
#define BUFFER_SIZE_32K     0x6
#define BUFFER_SIZE_64K     0x7
#define BUFFER_SIZE_128K    0x8
#define BUFFER_SIZE_256K    0x9
#define BUFFER_SIZE_512K    0xa
#define BUFFER_SIZE_1M      0xb
#define BUFFER_SIZE_2M      0xc
#define BUFFER_SIZE_4M      0xd
#define BUFFER_SIZE_8M      0xe
#define BUFFER_SIZE_16M     0xf

#define	P0_RPL_INDEX		MMREG_HALF(P0_RECV_REG_MSH(2))
#define	P0_A_RPL_INDEX		MMREG_HALF(P0_RECV_REG_MSH(5))
#define	P1_RPL_INDEX		MMREG_HALF(P1_RECV_REG_MSH(2))
#define	P1_A_RPL_INDEX		MMREG_HALF(P1_RECV_REG_MSH(5))

#define	P0_RECV_COUNT		MMREG_HALF(P0_RECV_REG_LSH(2))
#define	P0_A_RECV_COUNT		MMREG_HALF(P0_RECV_REG_LSH(5))
#define	P1_RECV_COUNT		MMREG_HALF(P1_RECV_REG_LSH(2))
#define	P1_A_RECV_COUNT		MMREG_HALF(P1_RECV_REG_LSH(5))

#define	P0_RPL_SNAPSHOT	MMREG_WORD(P0_RECV_REG(2))
#define	P0_A_RPL_SNAPSHOT	MMREG_WORD(P0_RECV_REG(5))
#define	P1_RPL_SNAPSHOT	MMREG_WORD(P1_RECV_REG(2))
#define	P1_A_RPL_SNAPSHOT	MMREG_WORD(P1_RECV_REG(5))
#define		RPL_INDEX		(0xFFFF0000)
#define		RECV_COUNT		(0x0000FFFF)

#define	P0_BUFFER_DROP		MMREG_HALF(P0_RECV_REG_LSH(6))
#define	P0_A_BUFFER_DROP	MMREG_HALF(P0_RECV_REG_LSH(7))
#define	P1_BUFFER_DROP		MMREG_HALF(P1_RECV_REG_LSH(6))
#define	P1_A_BUFFER_DROP	MMREG_HALF(P1_RECV_REG_LSH(7))

#define	ISR			MMREG_WORD(ISR_REG(3))
#define		REQ_ACK_1		(0x20000000)
#define		REQ_ACK_0		(0x10000000)
#define		DMA3_DONE		(0x08000000)
#define		DMA2_DONE		(0x04000000)
#define		DMA1_DONE		(0x02000000)
#define		DMA0_DONE		(0x01000000)
#define		DMA_DONE(n)		(0x01000000<<(n))
#define		DMA_ERROR_INT		(0x00800000)
#define		WAKE_INT		(0x00400000)
#define		MEMORY_INT		(0x00200000)
#define		PARITY_INT		(0x00100000)
#define		TIME2_INT		(0x00080000)
#define		TIME1_INT		(0x00040000)
#define		TIME0_INT		(0x00020000)
#define		TIME_INT(n)		(0x00020000<<(n))
#define		COPY_BUSY		(0x00010000)
#define		P1_EVENT		(0x00008000)
#define		P1_A_PACKET_RCVD	(0x00004000)
#define		P1_NO_BACKUP		(0x00002000)
#define		P1_PACKET_RCVD		(0x00001000)
#define		P1_PACKET_HEAD		(0x00000800)
#define		P1_PACKET_SENT		(0x00000400)
#define		P1_SEND_READY		(0x00000200)
#define		P1_LINK_READY		(0x00000100)
#define		P0_EVENT		(0x00000080)
#define		P0_A_PACKET_RCVD	(0x00000040)
#define		P0_NO_BACKUP		(0x00000020)
#define		P0_PACKET_RCVD		(0x00000010)
#define		P0_PACKET_HEAD		(0x00000008)
#define		P0_PACKET_SENT		(0x00000004)
#define		P0_SEND_READY		(0x00000002)
#define		P0_LINK_READY		(0x00000001)

#define	AISR_ON			MMREG_WORD(ISR_REG(1))
#define	AISR_OFF		MMREG_WORD(ISR_REG(0))
#define	AISR			MMREG_WORD(ISR_REG(2))
#define		SAN_DATA_SENT_INT	(0x04000000)
#define		SAN_DATA_RCVD_INT	(0x02000000)
#define		SAN_TX_TOO_LONG_INT	(0x01000000)
#define		SAN_RX_TOO_LONG_INT	(0x00800000)
#define		SAN_TX_BLOCKED_INT	(0x00400000)
#define		SAN_RX_BLOCKED_INT	(0x00200000)
#define		P1_GEX_LATE_COLL_INT	(0x02000000)
#define		P1_GEX_REG_COLL_INT	(0x01000000)
#define		P1_GEX_NO_COLL_INT	(0x00800000)
#define		P1_GEX_RESOLVE_INT	(0x00400000)
#define		P1_MYRI_STOP		(0x00200000)
#define		P1_MYRI_ILGL_INT	(0x00100000)
#define		P1_MYRI_BEAT_INT	(0x00080000)
#define		P1_MYRI_NO_BEAT_INT	(0x00040000)
#define		P1_A_NO_BACKUP		(0x00020000)
#define		P1_A_NO_BUFFER		(0x00010000)
#define		P1_NO_BUFFER		(0x00008000)
#define		P1_SEND_STOPPED		(0x00004000)
#define		P1_PAUSE_RCVD		(0x00002000)
#define		P0_GEX_LATE_COLL_INT	(0x00001000)
#define		P0_GEX_REG_COLL_INT	(0x00000800)
#define		P0_GEX_NO_COLL_INT	(0x00000400)
#define		P0_GEX_RESOLVE_INT	(0x00000200)
#define		P0_MYRI_STOP		(0x00000100)
#define		P0_MYRI_ILGL_INT	(0x00000080)
#define		P0_MYRI_BEAT_INT	(0x00000040)
#define		P0_MYRI_NO_BEAT_INT	(0x00000020)
#define		P0_A_NO_BACKUP		(0x00000010)
#define		P0_A_NO_BUFFER		(0x00000008)
#define		P0_NO_BUFFER		(0x00000004)
#define		P0_SEND_STOPPED		(0x00000002)
#define		P0_PAUSE_RCVD		(0x00000001)

#define	DISPATCH_STATE		MMREG_WORD(DISPATCH_REG(0))
#define	DISPATCH_INDEX		MMREG_HALF(DISPATCH_REG_MSH(0))
#define DISPATCH_ISR_ON		MMREG_WORD(DISPATCH_REG(1))
#define DISPATCH_ISR_OFF	MMREG_WORD(DISPATCH_REG(2))
#define DISPATCH_STATE_ON	MMREG_WORD(DISPATCH_REG(3))
#define DISPATCH_STATE_OFF	MMREG_WORD(DISPATCH_REG(4))
#define DISPATCH_CONFIG		MMREG_WORD(DISPATCH_REG(5))

#define		DISPATCH_AND		(0x00000100)
#define		DISPATCH_OR		(0x00000080)
#define		DISPATCH_INVERT		(0x00000040)
#define		DISPATCH_SELECT		(0x0000001F)

#define JTAG_MASTER     	MMREG_WORD(JTAG_REG(0))

#define DMA_CONFIG		MMREG_WORD(PCIDMA_REG(0))
#define		DMA_ENABLE		(0x80000000)
#define		DMA_PRIORITY		(0x70000000)
#define		PCI_OFFSET		(0x0FFFFC00)
#define		DMA_PIPELINE		(0x00000300)
#define		PCI_DELAY		(0x000000F0)
#define		PCI_SYNCH		(0x0000000F)
#define PCI_CLOCK		MMREG_HALF(PCIDMA_REG_LSH(1))

#define DMA0_COUNT		MMREG_BYTE(PCIDMA_REG_MSB(2))
#define DMA1_COUNT		MMREG_BYTE(PCIDMA_REG_msb(3))
#define DMA2_COUNT		MMREG_BYTE(PCIDMA_REG_lsb(4))
#define DMA3_COUNT		MMREG_BYTE(PCIDMA_REG_LSB(5))
#define DMA_COUNT(n)		MMREG_BYTE(PCIDMA_REG_MSB(2)+(9*(n)))

#define DMA0_POINTER		MMREG_VOID_PTR(PCIDMA_REG(6))
#define DMA1_POINTER		MMREG_VOID_PTR(PCIDMA_REG(7))
#define DMA2_POINTER		MMREG_VOID_PTR(PCIDMA_REG(8))
#define DMA3_POINTER		MMREG_VOID_PTR(PCIDMA_REG(9))
#define DMA_POINTER(n)		MMREG_VOID_PTR(PCIDMA_REG(6+(n)))

struct DRD
{
  unsigned int		len;		// length of the DMA transfer
  unsigned int		eal;		// PCI address, low 32 bits

  void*			lar;		// LANai address
  unsigned int		eah;		// PCI address, high 32 bits

  struct DRD*		next;		// next pointer
  unsigned short	offset;		// checksum offset
  unsigned short	csum;		// checksum value
};
typedef	struct DRD	DRD;

#define		DMA_READ		(0x80000000)
#define		DMA_END			(0x40000000)
#define		DMA_FLUSH		(0x20000000)
#define		DMA_NO_SNOOP		(0x10000000)
#define		DMA_NO_CACHE		(0x08000000)
#define		DMA_LENGTH		(0x00FFFFFF)
#define		DMA_PCI_ADDR		(0xFFFFFFFF)

#define		DMA_LANAI_ADDR		(0x0FFFFFFF)
#define		DMA_PCI64_ADDR		(0xFFFFFFFF)

#define		DMA_APPEND		(0x20000000)
#define		DMA_VALID		(0x10000000)
#define		DMA_NEXT		(0x0FFFFFFF)
#define		DMA_OFFSET		(0xFFFF0000)
#define		DMA_CHECKSUM		(0x0000FFFF)




struct	RPD_LENGTH
{
    unsigned	invalid		: 1;
    unsigned	link		: 1;
    unsigned	marked_bad	: 1;
    unsigned	last		: 1;
    unsigned	zeroes		: 3;
    unsigned	length		: 25;
};
typedef	struct RPD_LENGTH	RPD_LENGTH;

struct	RPD
{
    void*	pointer;
    unsigned	length;
};
typedef	struct RPD		RPD;

#define		DESC_INVALID		(0x80000000)
#define		DESC_CUT_THROUGH	(0x80000000)
#define		DESC_ROUTE		(0x40000000)
#define		DESC_LINK		(0x40000000)
#define		DESC_MARKED_BAD		(0x20000000)
#define		DESC_INVALIDATE		(0x20000000)
#define		DESC_LAST		(0x10000000)
#define		DESC_ZEROES		(0x0E000000)
#define		DESC_FLAGS		(0xFE000000)
#define		DESC_LENGTH		(0x01FFFFFF)




/***********************************************************
 ** Port registers, accessed through PORT_ADDR, PORT_DATA **
 ***********************************************************/

#define	P0_MODE				(0x10)
#define	P1_MODE				(0x40 + P0_MODE)
#define		MODE_GMII			(0x0008)
#define		MODE_GEX			(0x0004)
#define		MODE_IB				(0x0002)
#define		MODE_M				(0x0001)

#define	P0_TEST				(0x20)
#define	P1_TEST				(0x40 + P0_TEST)
#define		TEST_DEBUG			(0x0010)
#define		TEST_DISCDET			(0x0008)
#define		TEST_MIX			(0x0004)
#define		TEST_LOW			(0x0002)
#define		TEST_HIGH			(0x0001)

#define	P0_STATE			(0x11)
#define	P1_STATE			(0x40 + P0_STATE)
#define		FORCE_STATE			(0x0020)
#define		CLSM_STATE			(0x001F)

#define	P0_RECV_CONFIG			(0x09)
#define	P1_RECV_CONFIG			(0x40 + P0_RECV_CONFIG)
#define	P0_SEND_CONFIG			(0x23)
#define	P1_SEND_CONFIG			(0x40 + P0_SEND_CONFIG)
#define		SAN_INCLUDE			(0x2000)
#define		SAN_DEBUG			(0x1000)
#define		SAN_BEAT_ENABLE			(0x0800)
#define		SAN_ILGL_ENABLE			(0x0400)
#define		SAN_WINDOW			(0x0300)
#define		SAN_TIMEOUT			(0x00C0)
#define		INVALIDATE_MARK_BAD		(0x0200)
#define		INVALIDATE_CRC8			(0x0100)
#define		INVALIDATE_CRC16		(0x0080)
#define		INVALIDATE_CRC32		(0x0040)
#define		ENABLE_CRC8			(0x0020)
#define		ENABLE_CRC16			(0x0010)
#define		ENABLE_CRC32			(0x0008)
#define		CRC32_BIT_ORDER			(0x0006)
#define		CRC32_INFINIBAND		(0x0001)

#define	P0_SHORT_DROP			(0x0A)
#define	P1_SHORT_DROP			(0x40 + P0_SHORT_DROP)

#define	P0_8B10B_ERROR			(0x12)
#define	P1_8B10B_ERROR			(0x40 + P0_8B10B_ERROR)

#define	P0_MYRI_SEND			(0x22)
#define	P1_MYRI_SEND			(0x40 + P0_MYRI_SEND)
#define		M_AUTO_BEAT			(0x0004)
#define		M_SEND_BEAT			(0x0002)
#define		M_SEND_ILGL			(0x0001)

#define	P0_GEX_CONTROL			(0x00)
#define	P1_GEX_CONTROL			(0x40 + P0_GEX_CONTROL)
#define		GEX_RESET			(0x8000)
#define		GEX_LOOPBACK			(0x4000)
#define		GEX_SPEED_SELECTION_LSB		(0x2000)
#define		GEX_AN_ENABLE			(0x1000)
#define		GEX_POWER_DOWN			(0x0800)
#define		GEX_ISOLATE			(0x0400)
#define		GEX_RESTART_AN			(0x0200)
#define		GEX_DUPLEX_MODE			(0x0100)
#define		GEX_COLLISION_TEST		(0x0080)
#define		GEX_SPEED_SELECTION_MSB		(0x0040)

#define	P0_GEX_STATUS			(0x01)
#define	P1_GEX_STATUS			(0x40 + P0_GEX_STATUS)
#define		GEX_100BASE_T4			(0x8000)
#define		GEX_100BASE_X_FULL_DUPLEX	(0x4000)
#define		GEX_100BASE_X_HALF_DUPLEX	(0x2000)
#define		GEX_10_MBPS_FULL_DUPLEX		(0x1000)
#define		GEX_10_MBPS_HALF_DUPLEX		(0x0800)
#define		GEX_100BASE_T2_FULL_DUPLEX	(0x0400)
#define		GEX_100BASE_T2_HALF_DUPLEX	(0x0200)
#define		GEX_EXTENDED_STATUS		(0x0100)
#define		GEX_MF_PREAMBLE_SUSPENSION	(0x0040)
#define		GEX_AN_COMPLETE			(0x0020)
#define		GEX_REMOTE_FAULT_DETECTED	(0x0010)
#define		GEX_AN_ABILITY			(0x0008)
#define		GEX_LINK_STATUS			(0x0004)
#define		GEX_JABBER_DETECT		(0x0002)
#define		GEX_EXTENDED_CAPABILITY		(0x0001)

#define	P0_GEX_AN_ADVERTISEMENT		(0x02)
#define	P1_GEX_AN_ADVERTISEMENT		(0x40 + P0_GEX_AN_ADVERTISEMENT)
#define	P0_GEX_AN_LP_ABILITY_BP		(0x03)
#define	P1_GEX_AN_LP_ABILITY_BP		(0x40 + P0_GEX_AN_LP_ABILITY_BP)
#define		GEX_NEXT_PAGE			(0x8000)
#define		GEX_ACKNOWLEDGE			(0x4000)
#define		GEX_REMOTE_FAULT_CODE		(0x3000)
#define		GEX_PAUSE			(0x0180)
#define		GEX_HALF_DUPLEX			(0x0040)
#define		GEX_FULL_DUPLEX			(0x0020)

#define	P0_GEX_AN_EXPANSION		(0x04)
#define	P1_GEX_AN_EXPANSION		(0x40 + P0_GEX_AN_EXPANSION)
#define		GEX_NEXT_PAGE_ABLE		(0x0004)
#define		GEX_PAGE_RECEIVED		(0x0002)

#define	P0_GEX_AN_NP_TRANSMIT		(0x05)
#define	P1_GEX_AN_NP_TRANSMIT		(0x40 + P0_GEX_AN_NP_TRANSMIT)

#define	P0_GEX_AN_LP_ABILITY_NP		(0x06)
#define	P1_GEX_AN_LP_ABILITY_NP		(0x40 + P0_GEX_AN_LP_ABILITY_NP)

#define	P0_GEX_EXTENDED_STATUS		(0x07)
#define	P1_GEX_EXTENDED_STATUS		(0x40 + P0_GEX_EXTENDED_STATUS)
#define		GEX_1000BASE_X_FULL_DUPLEX	(0x8000)
#define		GEX_1000BASE_X_HALF_DUPLEX	(0x4000)
#define		GEX_1000BASE_T_FULL_DUPLEX	(0x2000)
#define		GEX_1000BASE_T_HALF_DUPLEX	(0x1000)

#define	P0_GE_CONFIG			(0x21)
#define	P1_GE_CONFIG			(0x40 + P0_GE_CONFIG)
#define		GE_REPEATER_MODE		(0x0004)
#define		GE_BURST_MODE			(0x0002)
#define		GE_HALF_DUPLEX_MODE		(0x0001)

#define	P0_GE_FALSE_CARRIER		(0x08)
#define	P1_GE_FALSE_CARRIER		(0x40 + P0_GE_FALSE_CARRIER)

#define	P0_IB_CONFIG			(0x17)
#define	P1_IB_CONFIG			(0x40 + P0_IB_CONFIG)
#define		IB_RETRAIN			(0x0008)
#define		IB_DEFAULT			(0x0004)
#define		IB_RATE				(0x0002)
#define		IB_WIDTH			(0x0001)

#define	P0_IB_LDC			(0x14)
#define	P1_IB_LDC			(0x40 + P0_IB_LDC)

#define	P0_IB_LERC			(0x13)
#define	P1_IB_LERC			(0x40 + P0_IB_LERC)

#define	P0_IB_SEC			(0x12)
#define	P1_IB_SEC			(0x40 + P0_IB_SEC)

#define	P0_IB_LEC			(0x15)
#define	P1_IB_LEC			(0x40 + P0_IB_LEC)

#define	P0_IB_REC			(0x16)
#define	P1_IB_REC			(0x40 + P0_IB_REC)


#define	WAIT_1_CYCLE	\
	{		\
	    asm("nop"); \
	}

#define	WAIT_2_CYCLES	\
	{		\
	    asm("nop"); \
	    asm("nop"); \
	}

#define	WAIT_5_CYCLES	\
	{		\
	    asm("nop"); \
	    asm("nop"); \
	    asm("nop"); \
	    asm("nop"); \
	    asm("nop"); \
	}


static
inline
void
port_reg_write (unsigned short offset, unsigned short data)
{
    PORT_ADDR = offset;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    PORT_DATA = data;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
}


static
inline
unsigned short
port_reg_read (unsigned short offset)
{
    PORT_ADDR = offset | PORT_READ;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    return PORT_DATA;
}


static
inline
void
port_slow_write (unsigned short offset, unsigned short data)
{
    PORT_ADDR = offset;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    PORT_DATA = data;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
}


static
inline
unsigned short
port_slow_read (unsigned short offset)
{
    PORT_ADDR = offset | PORT_READ;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    WAIT_5_CYCLES;
    return PORT_DATA;
}


#endif /* LANAIX_DEF_H */
