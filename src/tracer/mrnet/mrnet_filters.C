/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/mrnet_filters.C,v $
 | 
 | @last_commit: $Date: 2009/06/10 17:41:56 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#include "common.h"

static char UNUSED rcsid[] = "$Id: mrnet_filters.C,v 1.5 2009/06/10 17:41:56 gllort Exp $";

#include "mrnet/MRNet.h"
#include "mrnet_commands.h"

using namespace MRN;

extern "C" 
{

#if 0
const char * IntegerAdd_format_string = "%d";
void IntegerAdd( const std::vector< PacketPtr >& packets_in,
                 std::vector< PacketPtr >& packets_out,
                 std::vector< PacketPtr >&,
                 void ** /* client data */, PacketPtr& )
{
	int sum = 0;

	for (unsigned int i=0; i<packets_in.size(); i++)
	{
		PacketPtr cur_packet = packets_in[i];
		int val;
		cur_packet->unpack("%d", &val);
		sum += val;
	}
	PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
	                                 packets_in[0]->get_Tag(), 
	                                 IntegerAdd_format_string, sum) );
	packets_out.push_back( new_packet );
}
#endif

const char * IntegerAdd_format_string = "%d";
const char * UllMinMax_format_string = "%uld";
const char * MinMaxPositive_format_string = "%ld";
const char * DoubleAdd_format_string = "%lf";

const char * BigFilter_format_string = "";
void BigFilter( const std::vector< PacketPtr >& packets_in,
                 std::vector< PacketPtr >& packets_out,
                 std::vector< PacketPtr >&,
                 void ** /* client data */, PacketPtr& )
{
#if 1
	int tag = packets_in[0]->get_Tag();

	switch(tag)
	{
		case REDUCE_INT_ADD:
		{
			int sum = 0;

			for (unsigned int i=0; i<packets_in.size(); i++)
			{
				PacketPtr cur_packet = packets_in[i];
				int val;
				cur_packet->unpack(IntegerAdd_format_string, &val);
				sum += val;
			}
			PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
			                      packets_in[0]->get_Tag(), 
			                      IntegerAdd_format_string, sum) );
			packets_out.push_back( new_packet );
			break;
		}
		case REDUCE_DOUBLE_ADD:
		{
			double sum = 0;

			for (unsigned int i=0; i<packets_in.size(); i++)
			{
                PacketPtr cur_packet = packets_in[i];
				double val;
				cur_packet->unpack(DoubleAdd_format_string, &val);
				sum += val;
			}
            PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
                                  packets_in[0]->get_Tag(),
                                  DoubleAdd_format_string, sum) );
            packets_out.push_back( new_packet );
			break;
		}
/*
		case REDUCE_ULL_MIN:
		{
			unsigned long long min;
			PacketPtr cur_packet = packets_in[0];
			cur_packet->unpack("%uld", &min);

			for (unsigned int i=1; i<packets_in.size(); i++)
			{
                PacketPtr cur_packet = packets_in[i];
                unsigned long long val;
                cur_packet->unpack("%uld", &val);
                min = (val < min ? val : min);
			}
            PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
                                  packets_in[0]->get_Tag(),
                                  UllMinMax_format_string, min) );
            packets_out.push_back( new_packet );
			break;
		}
		case REDUCE_ULL_MAX:
		{
            unsigned long long max;
            PacketPtr cur_packet = packets_in[0];
            cur_packet->unpack("%uld", &max);

			for (unsigned int i=1; i<packets_in.size(); i++)
			{
                PacketPtr cur_packet = packets_in[i];
                unsigned long long val;
                cur_packet->unpack("%uld", &val);
                max = (val > max ? val : max);
			}
            PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
                                  packets_in[0]->get_Tag(),
                                  UllMinMax_format_string, max) );
            packets_out.push_back( new_packet );
			break;
		}
*/
        case REDUCE_LL_MIN_POSITIVE:
        {
            long long min;
            PacketPtr cur_packet = packets_in[0];
            cur_packet->unpack("%ld", &min);

            for (unsigned int i=1; i<packets_in.size(); i++)
            {
                long long val;
                PacketPtr cur_packet = packets_in[i];
                cur_packet->unpack("%ld", &val);

				if (min < 0) min = val;
				else if (val >= 0) min = (val < min ? val : min);
            }
            PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
                                  packets_in[0]->get_Tag(),
                                  MinMaxPositive_format_string, min) );
            packets_out.push_back( new_packet );
            break;
        }
        case REDUCE_LL_MAX_POSITIVE:
        {
            long long max;
            PacketPtr cur_packet = packets_in[0];
            cur_packet->unpack("%ld", &max);

            for (unsigned int i=1; i<packets_in.size(); i++)
            {
                long long val;
                PacketPtr cur_packet = packets_in[i];
                cur_packet->unpack("%ld", &val);

				max = (val > max ? val : max);
            }
            PacketPtr new_packet( new Packet(packets_in[0]->get_StreamId(),
                                  packets_in[0]->get_Tag(),
                                  MinMaxPositive_format_string, max) );
            packets_out.push_back( new_packet );
            break;
        }

		default:
		{
			/* Bypass all messages to the parent node */
			for (unsigned int i=0; i<packets_in.size(); i++)
			{
				packets_out.push_back( packets_in[i] );
			}
			break;
		}
	}
#endif
}


} /* extern "C" */
