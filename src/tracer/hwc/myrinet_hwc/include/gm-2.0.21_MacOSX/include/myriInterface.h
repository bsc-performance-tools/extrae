/* Interface.h */

/*************************************************************************
 *                                                                       *
 * Myricom Myrinet Software                                              *
 *                                                                       *
 * Copyright (c) 1996-1999 by Myricom, Inc.                              *
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
 * 325B N. Santa Anita Ave.                                              *
 * Arcadia, CA 91006                                                     *
 * 818 821-5555                                                          *
 * http://www.myri.com                                                   *
 *************************************************************************/

#ifndef M_INTERFACE_H_INCLUDED
#define M_INTERFACE_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#define OFF 0
#define ON  1

#ifndef _WIN32
typedef enum enum_board_type_number {
        unknown   = 0x0000,
        lanai_2_3 = 0x0203,

        lanai_3_0 = 0x0300,
        lanai_3_1 = 0x0301,
        lanai_3_2 = 0x0302,

        lanai_4_0 = 0x0400,
        lanai_4_1 = 0x0401,
        lanai_4_2 = 0x0402,
        lanai_4_3 = 0x0403,
        lanai_4_4 = 0x0404,
        lanai_4_5 = 0x0405,

        lanai_5_0 = 0x0500,
        lanai_5_1 = 0x0501,
        lanai_5_2 = 0x0502,
        lanai_5_3 = 0x0503,

        lanai_6_0 = 0x0600,
        lanai_6_1 = 0x0601,
        lanai_6_2 = 0x0602,
        lanai_6_3 = 0x0603,

        lanai_7_0 = 0x0700,
        lanai_7_1 = 0x0701,
        lanai_7_2 = 0x0702,
        lanai_7_3 = 0x0703,
        lanai_7_4 = 0x0704,
        lanai_7_5 = 0x0705,

        lanai_8_0 = 0x0800,
        lanai_8_1 = 0x0801,
        lanai_8_2 = 0x0802,
        lanai_8_3 = 0x0803,
        lanai_8_4 = 0x0804,

        lanai_9_0 = 0x0900,
        lanai_9_1 = 0x0901,
        lanai_9_2 = 0x0902,
        lanai_9_3 = 0x0903,
        lanai_9_4 = 0x0904


} board_type_number;
#endif /*_WIN32*/


#define MYRINET_BUS_SBUS	((unsigned short)1)
#define MYRINET_BUS_PCI		((unsigned short)2)
#define MYRINET_BUS_GSC		((unsigned short)3)
#define MYRINET_BUS_FPGA	((unsigned short)4)
#define MYRINET_BUS_FIBER	((unsigned short)5)
#define MYRINET_BUS_NONE	((unsigned short)0xFFFF)

#define MYRINET_BOARDTYPE_1MEG_SRAM	((unsigned short)1)
#define MYRINET_BOARDTYPE_FPGA		((unsigned short)2)
#define MYRINET_BOARDTYPE_L5		((unsigned short)3)
#define MYRINET_BOARDTYPE_FIBER		((unsigned short)4)
#define MYRINET_BOARDTYPE_NONE	((unsigned short)0xFFFF)


/* NOTE: the EEPROMs are programmed big-endian */
/*       so the cpu version will look like 0x04 0x00 etc. */
struct MYRINET_EEPROM {
    unsigned int lanai_clockval;	/*  0 */
    unsigned short lanai_cpu_version;	/*  4 */
    unsigned char lanai_board_id[6];    /*  6 */
    unsigned int lanai_sram_size;       /* 12 */
    unsigned char fpga_version[32];     /* 16 */
    unsigned char more_version[16];     /* 48 */

    unsigned short delay_line_value;	/* 64 = 0x40 */
    unsigned short board_type;		/* 66 */
    unsigned short bus_type;		/* 68 */
    unsigned short product_code;	/* 70 */
    unsigned int serial_number;		/* 72 */
    unsigned char board_label[32];	/* 76 */
    unsigned short max_lanai_speed;	/*108 */
    unsigned char voltage_code;		/*110 */
    unsigned char pad_voltage_code; /*111 */
    unsigned short future_use[6];	/*112 */
    unsigned int unused_4_bytes;	/*124 */
  };


/* used to save pointers to the board and pass it to user space etc. */
struct board_info {
    unsigned long lanai_memory;  /* ptr to LANai SRAM */
    unsigned long lanai_eeprom;  /* ptr to board eeprom */
    unsigned long lanai_registers;   /* ptr to LANai3.x registers */
    unsigned long lanai_control; /* ptr to board control registers */
    unsigned long copy_blockD;   /* DMA ptr to kernel allocated block */
    unsigned long copy_blockK;   /* ptr to kernel allocated block */
    unsigned long copy_blockU;   /* user ptr to kernel allocated block */
    unsigned int copy_block_size;   /* length of the allocated block */
    unsigned int lanai_memory_size;   /* length of the lanai SRAM */
};


int myrinet_init_pointers(int unit, void *board_base, unsigned char revision);

unsigned char lanai_read_byte(int unit, unsigned byte_offset);
unsigned short lanai_read_half(int unit, unsigned byte_offset);
unsigned int lanai_read_word(int unit, unsigned byte_offset);
unsigned int lanai_read_special(int unit, unsigned byte_offset);
unsigned int lanai_read_control(int unit);

void lanai_write_byte(int unit, unsigned byte_offset, unsigned char c);
void lanai_write_half(int unit, unsigned byte_offset, unsigned short s);
void lanai_write_word(int unit, unsigned byte_offset, unsigned int i);
void lanai_write_special(int unit, unsigned byte_offset, unsigned int i);
void lanai_write_control(int unit, unsigned int i);

void lanai_get(int unit, void *dest, unsigned byte_offset, unsigned len_in_bytes);
void lanai_put(int unit, void *source, unsigned byte_offset, unsigned len_in_bytes);

unsigned int *lanai_get_pointer(int unit, int page, unsigned byte_offset);

void lanai_reset_unit(int unit, int n);
void lanai_ereset_unit(int unit, int n);
void lanai_breset_unit(int unit, int n);
void lanai_dma_master(int unit, int n);
int lanai_dma_master_read(int unit);
void lanai_set_ex1_unit(int unit, int n);
void lanai_wake_unit(int unit, int n);
void lanai_interrupt_unit(int unit, int n);
int lanai_interrupt_pending(int unit);
void lanai_interrupt_clear(int unit);
#ifdef hp_dino
unsigned int lanai_read_control(int unit);
void lanai_write_control(int unit, unsigned int value);
#endif


#ifndef _WIN32
board_type_number lanai_board_type(unsigned int unit);
#endif /*_WIN32*/
int lanai_get_board_id(int unit, void *id);
unsigned int lanai_get_clockval(unsigned int unit);

#ifdef __cplusplus
}
#endif


#endif
