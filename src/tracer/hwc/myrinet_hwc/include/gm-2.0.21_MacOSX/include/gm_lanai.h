/******************************************************************-*-c-*-
 * Myricom GM networking software and documentation			 *
 * Copyright (c) 1999 by Myricom, Inc.					 *
 * All rights reserved.	 See the file `COPYING' for copyright notice.	 *
 *************************************************************************/

/* author: glenn@myri.com */

#ifndef _gm_lanai_h_
#define _gm_lanai_h_

/************************************************************************
 * This file includes information about the layout of Myrinet
 * interface cards for use by GM.
 *
 * ONLY OS-INDEPENDENT INFORMATION SHOULD BE IN THIS FILE.
 ************************************************************************/

#include "gm.h"
#include "gm_lanaiX_specials.h"

/********************
 * Myrinet EEPROM definitions
 ********************/

enum gm_bus_type
{
  GM_MYRINET_BUS_UNKNOWN = 0,
  GM_MYRINET_BUS_SBUS = 1,
  GM_MYRINET_BUS_PCI = 2,
  GM_MYRINET_BUS_GSC = 3,
  GM_MYRINET_BUS_FPGA = 4,
  GM_MYRINET_BUS_FIBER = 5,
  GM_MYRINET_BUS_MAP26xx = 6,
  GM_MYRINET_BUS_NONE = 0xffff
};

enum gm_board_type
{
  GM_MYRINET_BOARD_TYPE_1MEG_SRAM = 1,
  GM_MYRINET_BOARD_TYPE_FPGA = 2,
  GM_MYRINET_BOARD_TYPE_L5 = 3,
  GM_MYRINET_BOARD_TYPE_FIBER = 4,
  GM_MYRINET_BOARD_TYPE_SKIP5 = 5,
  GM_MYRINET_BOARD_TYPE_MAP26xx = 6,
  GM_MYRINET_BOARD_TYPE_NONE = 0xffff
};

#define GM_EEPROM(name,suffix)						\
struct name								\
{									\
  gm_u32##suffix lanai_clockval;	/* 00-03 */			\
  gm_u16##suffix lanai_cpu_version;	/* 04-05 */			\
  gm_u8##suffix	 lanai_board_id[6];	/* 06-0B */			\
  gm_u32##suffix lanai_sram_size;	/* 0C-0F */			\
  gm_u8##suffix	 fpga_version[32];	/* 10-2F */			\
  gm_u8##suffix	 more_version[16];	/* 30-3F */			\
		 							\
  gm_u16##suffix delay_line_value;	/* 40-41 */			\
  gm_u16##suffix board_type;		/* 42-43 */			\
  gm_u16##suffix bus_type;		/* 44-45 */			\
  gm_u16##suffix product_code;		/* 46-47 */			\
  gm_u32##suffix serial_number;		/* 48-4B */			\
  gm_u8##suffix  board_label[32];	/* 4C-6B */			\
  gm_u16##suffix max_lanai_speed;	/* 6C-6D */			\
  gm_u16##suffix future_use[7];		/* 6E-7B */			\
  gm_u32##suffix unused_4_bytes;	/* 7C-7F */			\
}

/* Myrinet eeprom in network order */

GM_EEPROM (gm_myrinet_eeprom_n, _n_t);
typedef struct gm_myrinet_eeprom_n gm_myrinet_eeprom_n_t;

/* Myrinet eeprom in host order */

GM_EEPROM (gm_myrinet_eeprom, _t);
typedef struct gm_myrinet_eeprom gm_myrinet_eeprom_t;

/********************
 * PCI configuration registers definitions (as described in the PCI 2.1 spec)
 ********************/

typedef struct gm_pci_config
{
  gm_u16_t Vendor_ID;
  gm_u16_t Device_ID;
  gm_u16_t Command;
  gm_u16_t Status;
  gm_u8_t Revision_ID;
  gm_u8_t Class_Code_Programming_Interface;
  gm_u8_t Class_Code_Subclass;
  gm_u8_t Class_Code_Base_Class;
  gm_u8_t Cache_Line_Size;
  gm_u8_t Latency_Timer;
  gm_u8_t Header_Type;
  gm_u8_t bist;
  gm_u32_t Base_Addresses_Registers[6];
  gm_u32_t Cardbus_CIS_Pointer;
  gm_u16_t Subsystem_Vendor_ID;
  gm_u16_t Subsystem_ID;
  gm_u32_t Expansion_ROM_Base_Address;
  gm_u8_t Capabilities_Pointer;	/* bottom 2 bits reserved. */
  gm_u8_t Reserved[7];
  gm_u8_t Interrupt_Line;
  gm_u8_t Interrupt_Pin;
  gm_u8_t Min_Gnt;
  gm_u8_t Max_Lat;
  gm_u8_t device_specific[192];
}
gm_pci_config_t;

#define  GM_PCI_COMMAND_IO          0x1	/* Enable response in I/O space */
#define  GM_PCI_COMMAND_MEMORY      0x2	/* Enable response in Memory space */
#define  GM_PCI_COMMAND_MASTER      0x4	/* Enable bus mastering */
#define  GM_PCI_COMMAND_SPECIAL     0x8	/* Enable response to special cycles */
#define  GM_PCI_COMMAND_INVALIDATE  0x10	/* Use memory write and invalidate */
#define  GM_PCI_COMMAND_VGA_PALETTE 0x20	/* Enable palette snooping */
#define  GM_PCI_COMMAND_PARITY      0x40	/* Enable parity checking */
#define  GM_PCI_COMMAND_WAIT        0x80	/* Enable address/data stepping */
#define  GM_PCI_COMMAND_SERR        0x100	/* Enable SERR */
#define  GM_PCI_COMMAND_FAST_BACK   0x200	/* Enable back-to-back writes */

#define  GM_PCI_STATUS_PERR         0x8000	/* Dev detected parity err  */
/* Extract the Capabilities_List bit from the PCI config Status field value. */
#define  GM_PCI_STATUS_CAPABILITIES_LIST(x) ((x>>4)&1)

#define GM_PCI_VENDOR_MYRICOM	0x10e8
#define GM_PCI_VENDOR_MYRICOM2	0x14c1
#define GM_PCI_DEVICE_MYRINET	0x8043 

/* The following declarations permit extraction of the information
 * contained in bits 1 and 2 of the low-order byte of a BAR; in the
 * case of a myrinet card, BAR[0] is always used.  See the section
 * "Address Maps" in the PCI 2.1 specification.
 */
enum gm_pci_base_addr_type
{
   GM_PCI_BASE_ADDRESS_32BIT_TYPE,	/* 00 => simple 32-bit base addr */
   GM_PCI_BASE_ADDRESS_20BIT_TYPE,	/* 01 => base addr below 1 Meg   */
   GM_PCI_BASE_ADDRESS_64BIT_TYPE,	/* 10 => simple 64-bit base addr */
   GM_PCI_BASE_ADDRESS_RESERVED_TYPE	/* 11 is reserved for future use */
};

#define GM_PCI_BASE_ADDRESS_TYPE(val) (((val) >> 1) & 3)

/* PCI config space capabilities structure.  These are chained with
   the Next_Pointer field, which is, ignoring the bottom 2 bits, the
   offset to the next capability in the PCI config space. */

typedef union gm_pci_capability
{
  struct
  {
    gm_u8_t Capability_ID;
    gm_u8_t Next_Pointer;	/* bottom 2 bits are reserved. */
  } common;
  
  /* Message signalled interrupt capability with 64-bit address. */
  struct 
  {
    gm_u8_t Capability_ID;
    gm_u8_t Next_Pointer;	/* bottom 2 bits are reserved. */
    gm_u16_t Message_Control;
    /* ... */
  } msi;

  /* Message signalled interrupt capability with 64-bit address. */
  struct 
  {
    gm_u8_t Capability_ID;
    gm_u8_t Next_Pointer;	/* bottom 2 bits are reserved. */
    gm_u16_t Message_Control;
    gm_u32_t Message_Address;
    gm_u32_t Message_Upper_Address;
    gm_u16_t Message_Data;
  } msi64;

  /* PCI-X capability */
  struct
  {
    gm_u8_t Capability_ID;
    gm_u8_t Next_Pointer;
    gm_u16_t Command;
    gm_u16_t Status;
  } pci_x;
  
} gm_pci_capability_t;

/* Strip the reserved bits from the bottom of a capability pointer. */
   
#define GM_PCI_CAP_POINTER_CLEANUP(x) ((x)&0xfc)

/* Extract the bit fields from a PCI capability's "Message_Control" field. */

#define GM_PCI_CAP_MESSAGE_CONTROL_64_BIT_ADDRESS_CAPABLE(x) (((x)>>7)&1)
#define GM_PCI_CAP_MESSAGE_CONTROL_MULTIPLE_MESSAGE_ENABLE(x) (((x)>>4)&0x7)
#define GM_PCI_CAP_MESSAGE_CONTROL_MULTIPLE_MESSAGE_CAPABLE(x) (((x)>>1)&0x7)
#define GM_PCI_CAP_MESSAGE_CONTROL_MSI_ENABLE(x) (((x)>>0)&1)
#define GM_PCI_CAP_PCI_X_COMMAND_MAX_MEM_READ_BYTE_CNT_MASK (0xC)
#define GM_PCI_CAP_PCI_X_COMMAND_MAX_MEM_READ_BYTE_CNT_4096 (0xC)
#define GM_PCI_CAP_PCI_X_COMMAND_MAX_MEM_READ_BYTE_CNT_2048 (0x8)
#define GM_PCI_CAP_PCI_X_COMMAND_MAX_MEM_READ_BYTE_CNT_1024 (0x4)
#define GM_PCI_CAP_PCI_X_COMMAND_MAX_MEM_READ_BYTE_CNT_512 (0x0)

/* Do we want to force PCI-X to use large read blocks?  By default
   we do this, but it is conceivable that some very small fraction of
   customers will not want to do this if other cards in their system
   have real-time constraints that cannot be met with large block
   sizes.  If we don't force large reads, receive DMA performance has
   been observed to drop by up to 50%! */
#define GM_PCI_X_FORCE_LARGE_READ 1

enum gm_pci_capability_id
  {
    GM_PCI_CAP_RESERVED = 0,
    GM_PCI_CAP_PCI_POWER_MANAGEMENT_INTERFACE = 1,
    GM_PCI_CAP_AGP = 2,
    GM_PCI_CAP_VPD = 3,
    GM_PCI_CAP_SLOT_IDENTIFICATION = 4,
    GM_PCI_CAP_MESSAGE_SIGNALLED_INTERRUPTS = 5,
    GM_PCI_CAP_COMPACTPCI_HOT_SWAP = 6,
    GM_PCI_CAP_PCI_X = 7
  };

/************************************
 ** LANai Memory-mapped registers **
 ************************************/

struct LANai4_special_registers
{
  volatile gm_u32_n_t IPF0;	/* 5 context-0 state registers */
  volatile gm_u32_n_t CUR0;
  volatile gm_u32_n_t PREV0;
  volatile gm_u32_n_t DATA0;
  volatile gm_u32_n_t DPF0;
  
  volatile gm_u32_n_t IPF1;	/* 5 context-1 state registers */
  volatile gm_u32_n_t CUR1;
  volatile gm_u32_n_t PREV1;
  volatile gm_u32_n_t DATA1;
  volatile gm_u32_n_t DPF1;
  
  volatile gm_u32_n_t ISR;	/* interrupt status register */
  volatile gm_u32_n_t EIMR;	/* external-interrupt mask register */
  
  volatile gm_u32_n_t IT;	/* interrupt timer */
  volatile gm_s32_n_t RTC;	/* real-time clock */
  
  volatile gm_u32_n_t CKS;	/* checksum */
  volatile gm_u32_n_t gmEAR;	/* SBus-DMA exteral address */
  volatile gm_u32_n_t LAR;	/* SBus-DMA local address */
  volatile gm_u32_n_t DMA_CTR;	/* SBus-DMA counter */
  
  volatile gm_u32_n_t RMP;	/* receive-DMA pointer */
  volatile gm_u32_n_t RML;	/* receive-DMA limit */
  
  volatile gm_u32_n_t SMP;	/* send-DMA pointer */
  volatile gm_u32_n_t SML;	/* send-DMA limit */
  volatile gm_u32_n_t SMLT;	/* send-DMA limit with tail */
  
  volatile gm_u32_n_t skip_0x5c;	/* skipped one word */
  
  volatile gm_u8_n_t RB;	/* receive byte */
  volatile gm_u8_n_t skip_0x61;
  volatile gm_u8_n_t skip_0x62;
  volatile gm_u8_n_t skip_0x63;
  
  volatile gm_u16_n_t RH;	/* receive half-word */
  volatile gm_u8_n_t skip_0x66;
  volatile gm_u8_n_t skip_0x67;
  volatile gm_u32_n_t RW;	/* receive word */
  
  volatile gm_u32_n_t SA;	/* send align */
  
  volatile gm_u32_n_t SB;	/* single-send commands */
  volatile gm_u32_n_t SH;
  volatile gm_u32_n_t SW;
  volatile gm_u32_n_t ST;
  
  volatile gm_u32_n_t DMA_DIR;	/* SBus-DMA direction */
  volatile gm_u32_n_t DMA_STS;	/* SBus-DMA modes */
  volatile gm_u32_n_t TIMEOUT;
  volatile gm_u32_n_t MYRINET;
  
  volatile gm_u32_n_t HW_DEBUG;	/* hardware debugging */
  volatile gm_u32_n_t LED;	/* LED pin(s) */
  volatile gm_u32_n_t VERSION;	/* the ex-window-pins register */
  volatile gm_u32_n_t ACTIVATE;	/* activate Myrinet-link *//* 0x9C */
  
  volatile gm_u32_n_t pad_a[(0xfc - 0xa0) / sizeof (gm_u32_n_t)];
  
  volatile gm_u32_n_t clock_val;	/* clock register 0xFC */
};


#if GM_SUPPORT_OLD_L5
struct Old_LANai5_special_registers
{
  volatile gm_u32_n_t IPF0;	/* 0x00 */
  volatile gm_u32_n_t pad0x4;
  volatile gm_u32_n_t CUR0;	/* 0x08 */
  volatile gm_u32_n_t pad0xc;
  volatile gm_u32_n_t PREV0;	/* 0x10 */
  volatile gm_u32_n_t pad0x14;
  volatile gm_u32_n_t DATA0;	/* 0x18 */
  volatile gm_u32_n_t pad0x1c;
  volatile gm_u32_n_t DPF0;	/* 0x20 */
  volatile gm_u32_n_t pad0x24;
  volatile gm_u32_n_t IPF1;	/* 0x28 */
  volatile gm_u32_n_t pad0x2c;
  volatile gm_u32_n_t CUR1;	/* 0x30 */
  volatile gm_u32_n_t pad0x34;
  volatile gm_u32_n_t PREV1;	/* 0x38 */
  volatile gm_u32_n_t pad0x3c;
  volatile gm_u32_n_t DATA1;	/* 0x40 */
  volatile gm_u32_n_t pad0x44;
  volatile gm_u32_n_t DPF1;	/* 0x48 */
  volatile gm_u32_n_t pad0x4c;
  volatile gm_u32_n_t ISR;	/* 0x50 */
  volatile gm_u32_n_t pad0x54;
  volatile gm_u32_n_t EIMR;	/* 0x58 */
  volatile gm_u32_n_t pad0x5c;
  volatile gm_u32_n_t IT;	/* 0x60 */
  volatile gm_u32_n_t pad0x64;
  volatile gm_s32_n_t RTC;	/* 0x68 */
  volatile gm_u32_n_t pad0x6c;
  volatile gm_u32_n_t LAR;	/* 0x70 */
  volatile gm_u32_n_t pad0x74;
  volatile gm_u32_n_t gmCTR;	/* 0x78 */
  volatile gm_u32_n_t pad0x7c;
  volatile gm_u32_n_t L2E_LAR;	/* 0x80 */
  volatile gm_u32_n_t pad0x84;
  volatile gm_u32_n_t E2L_LAR;	/* 0x88 */
  volatile gm_u32_n_t pad0x8c;
  volatile gm_u32_n_t L2E_EAR;	/* 0x90 */
  volatile gm_u32_n_t pad0x94;
  volatile gm_u32_n_t E2L_EAR;	/* 0x98 */
  volatile gm_u32_n_t pad0x9c;
  volatile gm_u32_n_t L2E_CTR;	/* 0xA0 */
  volatile gm_u32_n_t pad0xa4;
  volatile gm_u32_n_t E2L_CTR;	/* 0xA8 */
  volatile gm_u32_n_t pad0xac;
  volatile gm_u32_n_t L2E_EAR_BL;	/* 0xB0 */
  volatile gm_u32_n_t pad0xb4;
  volatile gm_u32_n_t E2L_EAR_BL;	/* 0xB8 */
  volatile gm_u32_n_t pad0xbc;
  volatile gm_u32_n_t L2E_CTR_BL;	/* 0xC0 */
  volatile gm_u32_n_t pad0xc4;
  volatile gm_u32_n_t E2L_CTR_BL;	/* 0xC8 */
  volatile gm_u32_n_t pad0xcc;
  volatile gm_u32_n_t RMW;	/* 0xD0 */
  volatile gm_u32_n_t pad0xd4;
  volatile gm_u32_n_t RMC;	/* 0xD8 */
  volatile gm_u32_n_t pad0xdc;
  volatile gm_u32_n_t RMP;	/* 0xE0 */
  volatile gm_u32_n_t pad0xe4;
  volatile gm_u32_n_t RML;	/* 0xE8 */
  volatile gm_u32_n_t pad0xec;
  volatile gm_u32_n_t SMP;	/* 0xF0 */
  volatile gm_u32_n_t pad0xf4;
  volatile gm_u32_n_t SMH;	/* 0xF8 */
  volatile gm_u32_n_t pad0xfc;
  volatile gm_u32_n_t SML;	/* 0x100 */
  volatile gm_u32_n_t pad0x104;
  volatile gm_u32_n_t SMLT;	/* 0x108 */
  volatile gm_u32_n_t pad0x10c;
  volatile gm_u32_n_t SMC;	/* 0x110 */
  volatile gm_u32_n_t pad0x114;
  volatile gm_u32_n_t SA;	/* 0x118 */
  volatile gm_u32_n_t pad0x11c;
  volatile gm_u32_n_t BURST;	/* 0x120 */
  volatile gm_u32_n_t pad0x124;
  volatile gm_u32_n_t TIMEOUT;	/* 0x128 */
  volatile gm_u32_n_t pad0x12c;
  volatile gm_u32_n_t MYRINET;	/* 0x130 */
  volatile gm_u32_n_t pad0x134;
  volatile gm_u32_n_t HW_DEBUG;	/* 0x138 */
  volatile gm_u32_n_t pad0x13c;
  volatile gm_u32_n_t LED;	/* 0x140 */
  volatile gm_u32_n_t pad0x144;
  volatile gm_u32_n_t WINDOW;	/* 0x148 */
  volatile gm_u32_n_t pad0x14c;
  volatile gm_u32_n_t WRITE_ENABLE;	/* 0x150 */
  volatile gm_u32_n_t pad0x154;
  volatile gm_u32_n_t pad0x158;
  volatile gm_u32_n_t pad0x15c;
  volatile gm_u32_n_t pad0x160;
  volatile gm_u32_n_t pad0x164;
  volatile gm_u32_n_t pad0x168;
  volatile gm_u32_n_t pad0x16c;
  volatile gm_u32_n_t pad0x170;
  volatile gm_u32_n_t pad0x174;
  volatile gm_u32_n_t pad0x178;
  volatile gm_u32_n_t pad0x17c;
  volatile gm_u32_n_t pad0x180;
  volatile gm_u32_n_t pad0x184;
  volatile gm_u32_n_t pad0x188;
  volatile gm_u32_n_t pad0x18c;
  volatile gm_u32_n_t pad0x190;
  volatile gm_u32_n_t pad0x194;
  volatile gm_u32_n_t pad0x198;
  volatile gm_u32_n_t pad0x19c;
  volatile gm_u32_n_t pad0x1a0;
  volatile gm_u32_n_t pad0x1a4;
  volatile gm_u32_n_t pad0x1a8;
  volatile gm_u32_n_t pad0x1ac;
  volatile gm_u32_n_t pad0x1b0;
  volatile gm_u32_n_t pad0x1b4;
  volatile gm_u32_n_t pad0x1b8;
  volatile gm_u32_n_t pad0x1bc;
  volatile gm_u32_n_t pad0x1c0;
  volatile gm_u32_n_t pad0x1c4;
  volatile gm_u32_n_t pad0x1c8;
  volatile gm_u32_n_t pad0x1cc;
  volatile gm_u32_n_t pad0x1d0;
  volatile gm_u32_n_t pad0x1d4;
  volatile gm_u32_n_t pad0x1d8;
  volatile gm_u32_n_t pad0x1dc;
  volatile gm_u32_n_t pad0x1e0;
  volatile gm_u32_n_t pad0x1e4;
  volatile gm_u32_n_t pad0x1e8;
  volatile gm_u32_n_t pad0x1ec;
  volatile gm_u32_n_t pad0x1f0;
  volatile gm_u32_n_t pad0x1f4;
  volatile gm_u32_n_t clock_val;	/* 0x1F8 */
}
#endif

struct LANai5_special_registers
{
  volatile gm_u32_n_t IPF0;	/* 0x00 */
  volatile gm_u32_n_t pad0x4;
  volatile gm_u32_n_t CUR0;	/* 0x08 */
  volatile gm_u32_n_t pad0xc;
  volatile gm_u32_n_t PREV0;	/* 0x10 */
  volatile gm_u32_n_t pad0x14;
  volatile gm_u32_n_t DATA0;	/* 0x18 */
  volatile gm_u32_n_t pad0x1c;
  volatile gm_u32_n_t DPF0;	/* 0x20 */
  volatile gm_u32_n_t pad0x24;
  volatile gm_u32_n_t IPF1;	/* 0x28 */
  volatile gm_u32_n_t pad0x2c;
  volatile gm_u32_n_t CUR1;	/* 0x30 */
  volatile gm_u32_n_t pad0x34;
  volatile gm_u32_n_t PREV1;	/* 0x38 */
  volatile gm_u32_n_t pad0x3c;
  volatile gm_u32_n_t DATA1;	/* 0x40 */
  volatile gm_u32_n_t pad0x44;
  volatile gm_u32_n_t DPF1;	/* 0x48 */
  volatile gm_u32_n_t pad0x4c;
  volatile gm_u32_n_t ISR;	/* 0x50 */
  volatile gm_u32_n_t pad0x54;
  volatile gm_u32_n_t EIMR;	/* 0x58 */
  volatile gm_u32_n_t pad0x5c;
  volatile gm_u32_n_t IT;	/* 0x60 */
  volatile gm_u32_n_t pad0x64;
  volatile gm_s32_n_t RTC;	/* 0x68 */
  volatile gm_u32_n_t pad0x6c;
  volatile gm_u32_n_t LAR;	/* 0x70 */
  volatile gm_u32_n_t pad0x74;
  volatile gm_u32_n_t gmCTR;	/* 0x78 */
  volatile gm_u32_n_t pad0x7c;
  volatile gm_u32_n_t pad0x80;	/* 0x80 */
  volatile gm_u32_n_t pad0x84;
  volatile gm_u32_n_t pad0x88;	/* 0x88 */
  volatile gm_u32_n_t pad0x8c;
  volatile gm_u32_n_t pad0x90;	/* 0x90 */
  volatile gm_u32_n_t pad0x94;
  volatile gm_u32_n_t pad0x97;	/* 0x98 */
  volatile gm_u32_n_t pad0x9c;
  volatile gm_u32_n_t pad0xa0;	/* 0xA0 */
  volatile gm_u32_n_t pad0xa4;
  volatile gm_u32_n_t pad0xa8;	/* 0xA8 */
  volatile gm_u32_n_t pad0xac;
  volatile gm_u32_n_t pad0xb0;	/* 0xB0 */
  volatile gm_u32_n_t pad0xb4;
  volatile gm_u32_n_t PULSE;	/* 0xB8 */
  volatile gm_u32_n_t pad0xbc;
  volatile gm_u32_n_t pad0xc0;	/* 0xC0 */
  volatile gm_u32_n_t pad0xc4;
  volatile gm_u32_n_t pad0xc8;	/* 0xC8 */
  volatile gm_u32_n_t pad0xcc;
  volatile gm_u32_n_t RMW;	/* 0xD0 */
  volatile gm_u32_n_t pad0xd4;
  volatile gm_u32_n_t RMC;	/* 0xD8 */
  volatile gm_u32_n_t pad0xdc;
  volatile gm_u32_n_t RMP;	/* 0xE0 */
  volatile gm_u32_n_t pad0xe4;
  volatile gm_u32_n_t RML;	/* 0xE8 */
  volatile gm_u32_n_t pad0xec;
  volatile gm_u32_n_t SMP;	/* 0xF0 */
  volatile gm_u32_n_t pad0xf4;
  volatile gm_u32_n_t SMH;	/* 0xF8 */
  volatile gm_u32_n_t pad0xfc;
  volatile gm_u32_n_t SML;	/* 0x100 */
  volatile gm_u32_n_t pad0x104;
  volatile gm_u32_n_t SMLT;	/* 0x108 */
  volatile gm_u32_n_t pad0x10c;
  volatile gm_u32_n_t SMC;	/* 0x110 */
  volatile gm_u32_n_t pad0x114;
  volatile gm_u32_n_t SA;	/* 0x118 */
  volatile gm_u32_n_t pad0x11c;
  volatile gm_u32_n_t BURST;	/* 0x120 */
  volatile gm_u32_n_t pad0x124;
  volatile gm_u32_n_t TIMEOUT;	/* 0x128 */
  volatile gm_u32_n_t pad0x12c;
  volatile gm_u32_n_t MYRINET;	/* 0x130 */
  volatile gm_u32_n_t pad0x134;
  volatile gm_u32_n_t HW_DEBUG;	/* 0x138 */
  volatile gm_u32_n_t pad0x13c;
  volatile gm_u32_n_t LED;	/* 0x140 */
  volatile gm_u32_n_t pad0x144;
  volatile gm_u32_n_t WINDOW;	/* 0x148 */
  volatile gm_u32_n_t pad0x14c;
  volatile gm_u32_n_t WRITE_ENABLE;	/* 0x150 */
  volatile gm_u32_n_t pad0x154;
  volatile gm_u32_n_t pad0x158;
  volatile gm_u32_n_t pad0x15c;
  volatile gm_u32_n_t pad0x160;
  volatile gm_u32_n_t pad0x164;
  volatile gm_u32_n_t pad0x168;
  volatile gm_u32_n_t pad0x16c;
  volatile gm_u32_n_t pad0x170;
  volatile gm_u32_n_t pad0x174;
  volatile gm_u32_n_t pad0x178;
  volatile gm_u32_n_t pad0x17c;
  volatile gm_u32_n_t pad0x180;
  volatile gm_u32_n_t pad0x184;
  volatile gm_u32_n_t pad0x188;
  volatile gm_u32_n_t pad0x18c;
  volatile gm_u32_n_t pad0x190;
  volatile gm_u32_n_t pad0x194;
  volatile gm_u32_n_t pad0x198;
  volatile gm_u32_n_t pad0x19c;
  volatile gm_u32_n_t pad0x1a0;
  volatile gm_u32_n_t pad0x1a4;
  volatile gm_u32_n_t pad0x1a8;
  volatile gm_u32_n_t pad0x1ac;
  volatile gm_u32_n_t pad0x1b0;
  volatile gm_u32_n_t pad0x1b4;
  volatile gm_u32_n_t pad0x1b8;
  volatile gm_u32_n_t pad0x1bc;
  volatile gm_u32_n_t pad0x1c0;
  volatile gm_u32_n_t pad0x1c4;
  volatile gm_u32_n_t pad0x1c8;
  volatile gm_u32_n_t pad0x1cc;
  volatile gm_u32_n_t pad0x1d0;
  volatile gm_u32_n_t pad0x1d4;
  volatile gm_u32_n_t pad0x1d8;
  volatile gm_u32_n_t pad0x1dc;
  volatile gm_u32_n_t pad0x1e0;
  volatile gm_u32_n_t pad0x1e4;
  volatile gm_u32_n_t pad0x1e8;
  volatile gm_u32_n_t pad0x1ec;
  volatile gm_u32_n_t pad0x1f0;
  volatile gm_u32_n_t pad0x1f4;
  volatile gm_u32_n_t clock_val;	/* 0x1F8 */
};

struct LANai6_special_registers
{
  volatile gm_u32_n_t IPF0;	/* 0x00 */
  volatile gm_u32_n_t pad0x4;
  volatile gm_u32_n_t CUR0;	/* 0x08 */
  volatile gm_u32_n_t pad0xc;
  volatile gm_u32_n_t PREV0;	/* 0x10 */
  volatile gm_u32_n_t pad0x14;
  volatile gm_u32_n_t DATA0;	/* 0x18 */
  volatile gm_u32_n_t pad0x1c;
  volatile gm_u32_n_t DPF0;	/* 0x20 */
  volatile gm_u32_n_t pad0x24;
  volatile gm_u32_n_t IPF1;	/* 0x28 */
  volatile gm_u32_n_t pad0x2c;
  volatile gm_u32_n_t CUR1;	/* 0x30 */
  volatile gm_u32_n_t pad0x34;
  volatile gm_u32_n_t PREV1;	/* 0x38 */
  volatile gm_u32_n_t pad0x3c;
  volatile gm_u32_n_t DATA1;	/* 0x40 */
  volatile gm_u32_n_t pad0x44;
  volatile gm_u32_n_t DPF1;	/* 0x48 */
  volatile gm_u32_n_t pad0x4c;
  volatile gm_u32_n_t ISR;	/* 0x50 */
  volatile gm_u32_n_t pad0x54;
  volatile gm_u32_n_t EIMR;	/* 0x58 */
  volatile gm_u32_n_t pad0x5c;
  volatile gm_u32_n_t IT0;	/* 0x60 */
  volatile gm_u32_n_t pad0x64;
  volatile gm_s32_n_t RTC;	/* 0x68 */
  volatile gm_u32_n_t pad0x6c;
  volatile gm_u32_n_t LAR;	/* 0x70 */
  volatile gm_u32_n_t pad0x74;
  volatile gm_u32_n_t gmCTR;	/* 0x78 */
  volatile gm_u32_n_t pad0x7c;
  volatile gm_u32_n_t pad0x80;	/* 0x80 */
  volatile gm_u32_n_t pad0x84;
  volatile gm_u32_n_t pad0x88;	/* 0x88 */
  volatile gm_u32_n_t pad0x8c;
  volatile gm_u32_n_t pad0x90;	/* 0x90 */
  volatile gm_u32_n_t pad0x94;
  volatile gm_u32_n_t pad0x98E2L_EAR;	/* 0x98 */
  volatile gm_u32_n_t pad0x9c;
  volatile gm_u32_n_t pad0xa0;	/* 0xA0 */
  volatile gm_u32_n_t pad0xa4;
  volatile gm_u32_n_t pad0xa8;	/* 0xA8 */
  volatile gm_u32_n_t pad0xac;
  volatile gm_u32_n_t pad0xb0;	/* 0xB0 */
  volatile gm_u32_n_t pad0xb4;
  volatile gm_u32_n_t PULSE;	/* 0xB8 */
  volatile gm_u32_n_t pad0xbc;
  volatile gm_u32_n_t IT1;	/* 0xC0 */
  volatile gm_u32_n_t pad0xc4;
  volatile gm_u32_n_t IT2;	/* 0xC8 */
  volatile gm_u32_n_t pad0xcc;
  volatile gm_u32_n_t RMW;	/* 0xD0 */
  volatile gm_u32_n_t pad0xd4;
  volatile gm_u32_n_t RMC;	/* 0xD8 */
  volatile gm_u32_n_t pad0xdc;
  volatile gm_u32_n_t RMP;	/* 0xE0 */
  volatile gm_u32_n_t pad0xe4;
  volatile gm_u32_n_t RML;	/* 0xE8 */
  volatile gm_u32_n_t pad0xec;
  volatile gm_u32_n_t SMP;	/* 0xF0 */
  volatile gm_u32_n_t pad0xf4;
  volatile gm_u32_n_t SMH;	/* 0xF8 */
  volatile gm_u32_n_t pad0xfc;
  volatile gm_u32_n_t SML;	/* 0x100 */
  volatile gm_u32_n_t pad0x104;
  volatile gm_u32_n_t SMLT;	/* 0x108 */
  volatile gm_u32_n_t pad0x10c;
  volatile gm_u32_n_t SMC;	/* 0x110 */
  volatile gm_u32_n_t pad0x114;
  volatile gm_u32_n_t SA;	/* 0x118 */
  volatile gm_u32_n_t pad0x11c;
  volatile gm_u32_n_t pad0x120;	/* 0x120 */
  volatile gm_u32_n_t pad0x124;
  volatile gm_u32_n_t LINK;	/* 0x128 */
  volatile gm_u32_n_t pad0x12c;
  volatile gm_u32_n_t MYRINET;	/* 0x130 */
  volatile gm_u32_n_t pad0x134;
  volatile gm_u32_n_t gm_DEBUG;	/* 0x138 */
  volatile gm_u32_n_t pad0x13c;
  volatile gm_u32_n_t LED;	/* 0x140 */
  volatile gm_u32_n_t pad0x144;
  volatile gm_u32_n_t WINDOW;	/* 0x148 */
  volatile gm_u32_n_t pad0x14c;
  volatile gm_u32_n_t WRITE_ENABLE;	/* 0x150 */
  volatile gm_u32_n_t pad0x154;
  volatile gm_u32_n_t pad0x158;
  volatile gm_u32_n_t pad0x15c;
  volatile gm_u32_n_t pad0x160;
  volatile gm_u32_n_t pad0x164;
  volatile gm_u32_n_t pad0x168;
  volatile gm_u32_n_t pad0x16c;
  volatile gm_u32_n_t pad0x170;
  volatile gm_u32_n_t pad0x174;
  volatile gm_u32_n_t pad0x178;
  volatile gm_u32_n_t pad0x17c;
  volatile gm_u32_n_t pad0x180;
  volatile gm_u32_n_t pad0x184;
  volatile gm_u32_n_t pad0x188;
  volatile gm_u32_n_t pad0x18c;
  volatile gm_u32_n_t pad0x190;
  volatile gm_u32_n_t pad0x194;
  volatile gm_u32_n_t pad0x198;
  volatile gm_u32_n_t pad0x19c;
  volatile gm_u32_n_t pad0x1a0;
  volatile gm_u32_n_t pad0x1a4;
  volatile gm_u32_n_t pad0x1a8;
  volatile gm_u32_n_t pad0x1ac;
  volatile gm_u32_n_t pad0x1b0;
  volatile gm_u32_n_t pad0x1b4;
  volatile gm_u32_n_t pad0x1b8;
  volatile gm_u32_n_t pad0x1bc;
  volatile gm_u32_n_t pad0x1c0;
  volatile gm_u32_n_t pad0x1c4;
  volatile gm_u32_n_t pad0x1c8;
  volatile gm_u32_n_t pad0x1cc;
  volatile gm_u32_n_t pad0x1d0;
  volatile gm_u32_n_t pad0x1d4;
  volatile gm_u32_n_t pad0x1d8;
  volatile gm_u32_n_t pad0x1dc;
  volatile gm_u32_n_t pad0x1e0;
  volatile gm_u32_n_t pad0x1e4;
  volatile gm_u32_n_t pad0x1e8;
  volatile gm_u32_n_t pad0x1ec;
  volatile gm_u32_n_t pad0x1f0;
  volatile gm_u32_n_t pad0x1f4;
  volatile gm_u32_n_t CLOCK;	/* 0x1F8 */
};

struct LANai7_special_registers
{
  volatile gm_u32_n_t pad0x0;	/* 0x00 */
  volatile gm_u32_n_t pad0x4;
  volatile gm_u32_n_t pad0x8;	/* 0x08 */
  volatile gm_u32_n_t pad0xc;
  volatile gm_u32_n_t pad0x10;	/* 0x10 */
  volatile gm_u32_n_t pad0x14;
  volatile gm_u32_n_t pad0x18;	/* 0x18 */
  volatile gm_u32_n_t pad0x1c;
  volatile gm_u32_n_t pad0x20;	/* 0x20 */
  volatile gm_u32_n_t pad0x24;
  volatile gm_u32_n_t pad0x28;	/* 0x28 */
  volatile gm_u32_n_t pad0x2c;
  volatile gm_u32_n_t pad0x30;	/* 0x30 */
  volatile gm_u32_n_t pad0x34;
  volatile gm_u32_n_t pad0x38;	/* 0x38 */
  volatile gm_u32_n_t pad0x3c;
  volatile gm_u32_n_t pad0x40;	/* 0x40 */
  volatile gm_u32_n_t pad0x44;
  volatile gm_u32_n_t pad0x48;	/* 0x48 */
  volatile gm_u32_n_t pad0x4c;
  volatile gm_u32_n_t ISR;	/* 0x50 */
  volatile gm_u32_n_t pad0x54;
  volatile gm_u32_n_t EIMR;	/* 0x58 */
  volatile gm_u32_n_t pad0x5c;
  volatile gm_u32_n_t IT0;	/* 0x60 */
  volatile gm_u32_n_t pad0x64;
  volatile gm_s32_n_t RTC;	/* 0x68 */
  volatile gm_u32_n_t pad0x6c;
  volatile gm_u32_n_t LAR;	/* 0x70 */
  volatile gm_u32_n_t pad0x74;
  volatile gm_u32_n_t gmCTR;	/* 0x78 */
  volatile gm_u32_n_t pad0x7c;
  volatile gm_u32_n_t pad0x80;	/* 0x80 */
  volatile gm_u32_n_t pad0x84;
  volatile gm_u32_n_t pad0x88;	/* 0x88 */
  volatile gm_u32_n_t pad0x8c;
  volatile gm_u32_n_t pad0x90;	/* 0x90 */
  volatile gm_u32_n_t pad0x94;
  volatile gm_u32_n_t pad0x98;	/* 0x98 */
  volatile gm_u32_n_t pad0x9c;
  volatile gm_u32_n_t pad0xa0;	/* 0xA0 */
  volatile gm_u32_n_t pad0xa4;
  volatile gm_u32_n_t pad0xa8;	/* 0xA8 */
  volatile gm_u32_n_t pad0xac;
  volatile gm_u32_n_t pad0xb0;	/* 0xB0 */
  volatile gm_u32_n_t pad0xb4;
  volatile gm_u32_n_t PULSE;	/* 0xB8 */
  volatile gm_u32_n_t pad0xbc;
  volatile gm_u32_n_t IT1;	/* 0xC0 */
  volatile gm_u32_n_t pad0xc4;
  volatile gm_u32_n_t IT2;	/* 0xC8 */
  volatile gm_u32_n_t pad0xcc;
  volatile gm_u32_n_t RMW;	/* 0xD0 */
  volatile gm_u32_n_t pad0xd4;
  volatile gm_u32_n_t RMC;	/* 0xD8 */
  volatile gm_u32_n_t pad0xdc;
  volatile gm_u32_n_t RMP;	/* 0xE0 */
  volatile gm_u32_n_t pad0xe4;
  volatile gm_u32_n_t RML;	/* 0xE8 */
  volatile gm_u32_n_t pad0xec;
  volatile gm_u32_n_t SMP;	/* 0xF0 */
  volatile gm_u32_n_t pad0xf4;
  volatile gm_u32_n_t SMH;	/* 0xF8 */
  volatile gm_u32_n_t pad0xfc;
  volatile gm_u32_n_t SML;	/* 0x100 */
  volatile gm_u32_n_t pad0x104;
  volatile gm_u32_n_t SMLT;	/* 0x108 */
  volatile gm_u32_n_t pad0x10c;
  volatile gm_u32_n_t SMC;	/* 0x110 */
  volatile gm_u32_n_t pad0x114;
  volatile gm_u32_n_t SA;	/* 0x118 */
  volatile gm_u32_n_t pad0x11c;
  volatile gm_u32_n_t pad0x120;	/* 0x120 */
  volatile gm_u32_n_t pad0x124;
  volatile gm_u32_n_t TIMEOUT;	/* 0x128 */
  volatile gm_u32_n_t pad0x12c;
  volatile gm_u32_n_t MYRINET;	/* 0x130 */
  volatile gm_u32_n_t pad0x134;
  volatile gm_u32_n_t gm_DEBUG;	/* 0x138 */
  volatile gm_u32_n_t pad0x13c;
  volatile gm_u32_n_t LED;	/* 0x140 */
  volatile gm_u32_n_t pad0x144;
  volatile gm_u32_n_t pad0x148;	/* 0x148 */
  volatile gm_u32_n_t pad0x14c;
  volatile gm_u32_n_t GM_MP;	/* 0x150 */
  volatile gm_u32_n_t pad0x154;
  volatile gm_u32_n_t pad0x158;
  volatile gm_u32_n_t pad0x15c;
  volatile gm_u32_n_t pad0x160;
  volatile gm_u32_n_t pad0x164;
  volatile gm_u32_n_t pad0x168;
  volatile gm_u32_n_t pad0x16c;
  volatile gm_u32_n_t pad0x170;
  volatile gm_u32_n_t pad0x174;
  volatile gm_u32_n_t pad0x178;
  volatile gm_u32_n_t pad0x17c;
  volatile gm_u32_n_t pad0x180;
  volatile gm_u32_n_t pad0x184;
  volatile gm_u32_n_t pad0x188;
  volatile gm_u32_n_t pad0x18c;
  volatile gm_u32_n_t pad0x190;
  volatile gm_u32_n_t pad0x194;
  volatile gm_u32_n_t pad0x198;
  volatile gm_u32_n_t pad0x19c;
  volatile gm_u32_n_t pad0x1a0;
  volatile gm_u32_n_t pad0x1a4;
  volatile gm_u32_n_t pad0x1a8;
  volatile gm_u32_n_t pad0x1ac;
  volatile gm_u32_n_t pad0x1b0;
  volatile gm_u32_n_t pad0x1b4;
  volatile gm_u32_n_t pad0x1b8;
  volatile gm_u32_n_t pad0x1bc;
  volatile gm_u32_n_t pad0x1c0;
  volatile gm_u32_n_t pad0x1c4;
  volatile gm_u32_n_t pad0x1c8;
  volatile gm_u32_n_t pad0x1cc;
  volatile gm_u32_n_t pad0x1d0;
  volatile gm_u32_n_t pad0x1d4;
  volatile gm_u32_n_t pad0x1d8;
  volatile gm_u32_n_t pad0x1dc;
  volatile gm_u32_n_t pad0x1e0;
  volatile gm_u32_n_t pad0x1e4;
  volatile gm_u32_n_t pad0x1e8;
  volatile gm_u32_n_t pad0x1ec;
  volatile gm_u32_n_t pad0x1f0;
  volatile gm_u32_n_t pad0x1f4;
  volatile gm_u32_n_t CLOCK;	/* 0x1F8 */
};

struct LANai8_special_registers
{
  volatile gm_u32_n_t pad0x0;	/*  0x00 */
  volatile gm_u32_n_t pad0x4;
  volatile gm_u32_n_t pad0x8;	/*  0x08 */
  volatile gm_u32_n_t pad0xc;
  volatile gm_u32_n_t pad0x10;	/*  0x10 */
  volatile gm_u32_n_t pad0x14;
  volatile gm_u32_n_t pad0x18;	/*  0x18 */
  volatile gm_u32_n_t pad0x1c;
  volatile gm_u32_n_t pad0x20;	/*  0x20 */
  volatile gm_u32_n_t pad0x24;
  volatile gm_u32_n_t pad0x28;	/*  0x28 */
  volatile gm_u32_n_t pad0x2c;
  volatile gm_u32_n_t pad0x30;	/*  0x30 */
  volatile gm_u32_n_t pad0x34;
  volatile gm_u32_n_t pad0x38;	/*  0x38 */
  volatile gm_u32_n_t pad0x3c;
  volatile gm_u32_n_t pad0x40;	/*  0x40 */
  volatile gm_u32_n_t pad0x44;
  volatile gm_u32_n_t pad0x48;	/*  0x48 */
  volatile gm_u32_n_t pad0x4c;
  volatile gm_u32_n_t ISR;	/*  0x50 */
  volatile gm_u32_n_t pad0x54;
  volatile gm_u32_n_t EIMR;	/*  0x58 */
  volatile gm_u32_n_t pad0x5c;
  volatile gm_u32_n_t IT0;	/*  0x60 */
  volatile gm_u32_n_t pad0x64;
  volatile gm_s32_n_t RTC;	/*  0x68 */
  volatile gm_u32_n_t pad0x6c;
  volatile gm_u32_n_t LAR;	/*  0x70 */
  volatile gm_u32_n_t pad0x74;
  volatile gm_u32_n_t CPUC;	/*  0x78 */
  volatile gm_u32_n_t pad0x7c;
  volatile gm_u32_n_t pad0x80;	/*  0x80 */
  volatile gm_u32_n_t pad0x84;
  volatile gm_u32_n_t pad0x88;	/*  0x88 */
  volatile gm_u32_n_t pad0x8c;
  volatile gm_u32_n_t pad0x90;	/*  0x90 */
  volatile gm_u32_n_t pad0x94;
  volatile gm_u32_n_t pad0x98;	/*  0x98 */
  volatile gm_u32_n_t pad0x9c;
  volatile gm_u32_n_t pad0xa0;	/*  0xA0 */
  volatile gm_u32_n_t pad0xa4;
  volatile gm_u32_n_t pad0xa8;	/*  0xA8 */
  volatile gm_u32_n_t pad0xac;
  volatile gm_u32_n_t pad0xb0;	/*  0xB0 */
  volatile gm_u32_n_t pad0xb4;
  volatile gm_u32_n_t PULSE;	/*  0xB8 */
  volatile gm_u32_n_t pad0xbc;
  volatile gm_u32_n_t IT1;	/*  0xC0 */
  volatile gm_u32_n_t pad0xc4;
  volatile gm_u32_n_t IT2;	/*  0xC8 */
  volatile gm_u32_n_t pad0xcc;
  volatile gm_u32_n_t pad0xd0;	/*  0xD0 */
  volatile gm_u32_n_t pad0xd4;
  volatile gm_u32_n_t pad0xd8;	/*  0xD8 */
  volatile gm_u32_n_t pad0xdc;
  volatile gm_u32_n_t RMP;	/*  0xE0 */
  volatile gm_u32_n_t pad0xe4;
  volatile gm_u32_n_t RML;	/*  0xE8 */
  volatile gm_u32_n_t pad0xec;
  volatile gm_u32_n_t SMP;	/*  0xF0 */
  volatile gm_u32_n_t pad0xf4;
  volatile gm_u32_n_t SMH;	/*  0xF8 */
  volatile gm_u32_n_t pad0xfc;
  volatile gm_u32_n_t SML;	/*  0x100 */
  volatile gm_u32_n_t pad0x104;
  volatile gm_u32_n_t SMLT;	/*  0x108 */
  volatile gm_u32_n_t pad0x10c;
  volatile gm_u32_n_t SMC;	/*  0x110 */
  volatile gm_u32_n_t pad0x114;
  volatile gm_u32_n_t SA;	/*  0x118 */
  volatile gm_u32_n_t pad0x11c;
  volatile gm_u32_n_t pad0x120;	/*  0x120 */
  volatile gm_u32_n_t pad0x124;
  volatile gm_u32_n_t pad0x128;	/*  0x128 */
  volatile gm_u32_n_t pad0x12c;
  volatile gm_u32_n_t MYRINET;	/*  0x130 */
  volatile gm_u32_n_t pad0x134;
  volatile gm_u32_n_t gm_DEBUG;	/*  0x138 */
  volatile gm_u32_n_t pad0x13c;
  volatile gm_u32_n_t LED;	/*  0x140 */
  volatile gm_u32_n_t pad0x144;
  volatile gm_u32_n_t pad0x148;	/*  0x148 */
  volatile gm_u32_n_t pad0x14c;
  volatile gm_u32_n_t GM_MP;	/*  0x150 */
  volatile gm_u32_n_t pad0x154;
  volatile gm_u32_n_t pad0x158;
  volatile gm_u32_n_t pad0x15c;
  volatile gm_u32_n_t pad0x160;
  volatile gm_u32_n_t pad0x164;
  volatile gm_u32_n_t pad0x168;
  volatile gm_u32_n_t pad0x16c;
  volatile gm_u32_n_t pad0x170;
  volatile gm_u32_n_t pad0x174;
  volatile gm_u32_n_t pad0x178;
  volatile gm_u32_n_t pad0x17c;
  volatile gm_u32_n_t pad0x180;
  volatile gm_u32_n_t pad0x184;
  volatile gm_u32_n_t pad0x188;
  volatile gm_u32_n_t pad0x18c;
  volatile gm_u32_n_t pad0x190;
  volatile gm_u32_n_t pad0x194;
  volatile gm_u32_n_t pad0x198;
  volatile gm_u32_n_t pad0x19c;
  volatile gm_u32_n_t pad0x1a0;
  volatile gm_u32_n_t pad0x1a4;
  volatile gm_u32_n_t pad0x1a8;
  volatile gm_u32_n_t pad0x1ac;
  volatile gm_u32_n_t pad0x1b0;
  volatile gm_u32_n_t pad0x1b4;
  volatile gm_u32_n_t pad0x1b8;
  volatile gm_u32_n_t pad0x1bc;
  volatile gm_u32_n_t pad0x1c0;
  volatile gm_u32_n_t pad0x1c4;
  volatile gm_u32_n_t pad0x1c8;
  volatile gm_u32_n_t pad0x1cc;
  volatile gm_u32_n_t pad0x1d0;
  volatile gm_u32_n_t pad0x1d4;
  volatile gm_u32_n_t pad0x1d8;
  volatile gm_u32_n_t pad0x1dc;
  volatile gm_u32_n_t pad0x1e0;
  volatile gm_u32_n_t pad0x1e4;
  volatile gm_u32_n_t pad0x1e8;
  volatile gm_u32_n_t pad0x1ec;
  volatile gm_u32_n_t pad0x1f0;
  volatile gm_u32_n_t pad0x1f4;
  volatile gm_u32_n_t CLOCK;	/*  0x1F8 */
};


typedef union gm_lanai_special_registers
{
  union
  {
    struct LANai4_special_registers l4;
    struct LANai5_special_registers l5;
    struct LANai6_special_registers l6;
    struct LANai7_special_registers l7;
    struct LANai8_special_registers l8;
    struct LANai8_special_registers l9;	/* same as l8 */
    struct LANaiX_readable_specials lX;
  } read;
  
  union
  {
    struct LANai4_special_registers l4;
    struct LANai5_special_registers l5;
    struct LANai6_special_registers l6;
    struct LANai7_special_registers l7;
    struct LANai8_special_registers l8;
    struct LANai8_special_registers l9;	/* same as l8 */
    struct LANaiX_writable_specials lX;
  } write;
}
gm_lanai_special_registers_t;

#define	GM_HOST_SIG_BIT		(1U<<30)
#define	GM_DEBUG_SIG_BIT 	(1U<<31)

#define GM_LX_REQ_ACK_0		0x10000000 /* REQ_ACK_0 */

#if 0
#define GM_L4_ORUN2_BIT		GM_OFF_BY_2_BIT
#define GM_L4_ORUN1_BIT		GM_OFF_BY_1_BIT
#define GM_L4_ORUN2_INT_BIT 	GM_OFF_BY_2_BIT
#define GM_L4_ORUN1_INT_BIT 	GM_OFF_BY_1_BIT
#define GM_L4_RECV_RDY_BITS (GM_BYTE_RDY_BIT |GM_HALF_RDY_BIT |GM_WORD_RDY_BIT)
#endif

#endif /* _gm_lanai_h_ */

/*
  Local Variables:
  comment-column:40
  tab-width:8
  End:
*/

/*
  This file uses GM standard indentation:

  Local Variables:
  c-file-style:"gnu"
  tab-width:8
  End:
*/
