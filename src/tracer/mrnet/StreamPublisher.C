#include <stdlib.h>
#include "StreamPublisher.h"

/* FE */
StreamPublisher::StreamPublisher (Network *n, Stream *bcast_stream) 
	: Net(n),
	  BCastStream(bcast_stream)
{

    std::set<Rank> ep = BCastStream->get_EndPoints();
    std::set<Rank>::iterator it;

    for( it = ep.begin(); it != ep.end(); it++ ) 
    {
		Rank r = *it;

		StreamIDs[BE_RANK(r)] = NULL_STREAM;
	}
}

/* BE */
StreamPublisher::StreamPublisher (int be_id, Network *n, Stream *bcast_stream)
	: BackEndID(be_id),
	  Net(n),
	  BCastStream(bcast_stream)
{
}

StreamPublisher::~StreamPublisher ()
{
}

void StreamPublisher::Set (int be_id, Stream *s)
{
	if (StreamIDs.find(be_id) != StreamIDs.end())
	{
		StreamIDs[be_id] = s->get_Id();		
		/* fprintf(stderr, "StreamPublisher::Set %d %d\n", be_id, StreamIDs[be_id]); */
	}
}

void StreamPublisher::SetAll (Stream *s)
{
	MapIterator it;

	for(it = StreamIDs.begin(); it != StreamIDs.end(); it++)
	{
		StreamIDs[it->first] = s->get_Id();
	}
}

int StreamPublisher::get_StreamID (int be_id)
{
	if (StreamIDs.find(be_id) != StreamIDs.end())
	{
		return StreamIDs[be_id];
	}
	else 
	{
		return NULL_STREAM;
	}
}

Stream * StreamPublisher::get_Stream (int be_id)
{
	int stream_id = StreamPublisher::get_StreamID (be_id);

	if (stream_id != NULL_STREAM)
	{
		return Net->get_Stream(stream_id);
	}
	else
	{
		return NULL;
	}
}

void StreamPublisher::Send ()
{
    MapIterator it;
	int *be_ids = NULL, *stream_ids = NULL;
	int size = StreamIDs.size();

	be_ids = (int *)malloc( size * sizeof(int) );
	stream_ids = (int *)malloc( size * sizeof(int) );

	/* fprintf(stderr, "StreamPublisher::Send "); */
	int i = 0;
    for(it = StreamIDs.begin(); it != StreamIDs.end(); it++)
    {
        be_ids[i] = it->first;
		stream_ids[i] = it->second;
		/* fprintf(stderr, "(%d,%d) ", be_ids[i], stream_ids[i]); */
		i ++;
    }
	/* fprintf(stderr, "\n"); */

	MRN_STREAM_SEND (BCastStream, MRN_REGISTER_STREAM, "%ad %ad", be_ids, size, stream_ids, size);

	free (be_ids);
	free (stream_ids);
}

Stream * StreamPublisher::Recv ()
{
	int tag, *be_ids=NULL, *stream_ids=NULL, num_be=0;
	PacketPtr data;

	MRN_STREAM_RECV (BCastStream, &tag, data, MRN_REGISTER_STREAM);
	data->unpack("%ad %ad", &be_ids, &num_be, &stream_ids, &num_be);
	
	/* fprintf(stderr, "(be_id, st_id) "); */
	for (int i=0; i<num_be; i++)
	{
		/* fprintf(stderr, "(%d, %d) ", be_ids[i], stream_ids[i]); */
		StreamIDs[be_ids[i]] = stream_ids[i];
	}
	/* fprintf(stderr, "\n"); */
	free (be_ids);
	free (stream_ids);
	return StreamPublisher::get_Stream(BackEndID);
}

Stream * StreamPublisher::Announce ( std::set<int> be_list, int up_transfilter_id, int up_syncfilter_id )
{
	Stream *new_stream = NULL;
	Communicator *new_comm = Net->new_Communicator();
	std::set<int>::iterator it;
	
	/* Build the communicator */
	for (it = be_list.begin(); it != be_list.end(); it++)
	{
		int be_id = *it;
		new_comm->add_EndPoint(MRN_RANK(be_id));
/* fprintf(stderr, "StreamPublisher::Announce adding to communicator %d\n", MRN_RANK(be_id)); */
	}

	/* Create the new stream */
/* fprintf(stderr, "StreamPublisher::Announce creating new stream (up_transfilter_id=%d, up_syncfilter_id=%d)\n", up_transfilter_id, up_syncfilter_id); */
	new_stream = Net->new_Stream(new_comm, up_transfilter_id, up_syncfilter_id);

	/* Mark which back-ends should receive the new stream */
    for (it = be_list.begin(); it != be_list.end(); it++)
    {
		int be_id = *it;
		StreamPublisher::Set (be_id, new_stream);
/* fprintf(stderr, "StreamPublisher::Announce setting (%d,%d)\n", be_id, new_stream->get_Id()); */
	}	
	StreamPublisher::Send();
	return new_stream;
}

std::vector<Stream *> * StreamPublisher::AnnounceP2P ( Stream * stream, int up_transfilter_id, int up_syncfilter_id )
{
	/* Build p2p streams with every back-end in the given stream */
	std::vector<Stream *> *stream_list = new std::vector<Stream *>();
	
	std::set<Rank> ep = stream->get_EndPoints();
	std::set<Rank>::iterator it;

	for (it = ep.begin(); it != ep.end(); it++)
	{
		int mrn_id = *it;
		int be_id = BE_RANK(mrn_id);
		Communicator *new_comm = Net->new_Communicator();
		Stream *new_stream = NULL;
		
		new_comm->add_EndPoint(mrn_id);
		new_stream = Net->new_Stream(new_comm, up_transfilter_id, up_syncfilter_id);
		stream_list->push_back(new_stream);
		
		StreamPublisher::Set(be_id, new_stream);
	}
	StreamPublisher::Send();
	return stream_list;
}

