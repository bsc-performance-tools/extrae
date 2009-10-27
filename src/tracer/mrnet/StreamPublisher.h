#ifndef __STREAMPUBLISHER_H__
#define __STREAMPUBLISHER_H__

#include <mrnet/MRNet.h>
#include "mrnet_commands.h"

using namespace MRN;
using namespace std;

#define NULL_STREAM -1

class StreamPublisher
{
	int BackEndID;
	Network *Net;
	Stream *BCastStream;
	std::map<int, int> StreamIDs;
	typedef std::map<int, int>::iterator MapIterator;

	public:
		StreamPublisher (Network *n, Stream *bcast_stream);
		StreamPublisher (int be, Network *n, Stream *bcast_stream);
		~StreamPublisher ();
		void Set (int be, Stream *s);
		void SetAll (Stream *s);
		int get_StreamID (int be);
		Stream * get_Stream (int be);
		void Send ();
		Stream * Recv ();
		Stream * Announce (std::set<int> be_list, int up_transfilter_id=TFILTER_NULL, int up_syncfilter_id=SFILTER_DONTWAIT);
		std::vector<Stream *> * AnnounceP2P (Stream *stream, int up_transfilter_id=TFILTER_NULL, int up_syncfilter_id=SFILTER_DONTWAIT);

};

#endif /* __STREAMPUBLISHER_H__ */
