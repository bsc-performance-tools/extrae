#include "timesync.h"
#include "BE_ProtScope.h"
#include "trace_buffers.h"

int BE_ProtScope::run(Stream * stream)
{
    /* Common region defined from MAX(min_time) to MAX(max_time) */
    Buffer_t *buffer = TRACING_BUFFER(0);
    BufferIterator_t *itf, *itb, *itr;
    long long min_local_time = -1, max_local_time = -1;
    unsigned long long min_common_time, max_common_time;
    int num_common_events = 0;
    int tag;
    PacketPtr data;

    Buffer_Lock(buffer);

    itf = BIT_NewForward(buffer);
    itb = BIT_NewBackward(buffer);

    if (!BIT_OutOfBounds(itf)) min_local_time = TIMESYNC(TASKID, Get_EvTime(BIT_GetEvent(itf)));

    if (!BIT_OutOfBounds(itb)) max_local_time = TIMESYNC(TASKID, Get_EvTime(BIT_GetEvent(itb)));

    MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", min_local_time);
    MRN_STREAM_SEND(stream, REDUCE_LL_MAX_POSITIVE, "%ld", max_local_time);

    MRN_STREAM_RECV(stream, &tag, data, MRN_CALC_MB_MIN)
    data->unpack("%ld %ld", &min_common_time, &max_common_time);
    min_common_time = TIMEDESYNC(TASKID, min_common_time);
    max_common_time = TIMEDESYNC(TASKID, max_common_time);

    /* Now calc Mb/min ratio */
    itr = BIT_NewRange(buffer, min_common_time, max_common_time);
    num_common_events = 0;
    while (!BIT_OutOfBounds(itr))
    {
        num_common_events ++;
        BIT_Next(itr);
    }
    MRN_STREAM_SEND(stream, REDUCE_INT_ADD, "%d", num_common_events);

    double ns = max_common_time - min_common_time;
    double bytes = (num_common_events * sizeof(event_t));
    double mb, secs, mins, mb_min, bytes_ns;

    mb = bytes / (1024*1024);
    secs = ns / 1000000000;
    mins = secs / 60;
    mb_min = mb / mins;
    bytes_ns = bytes / ns;

    fprintf(stderr, "[BE %d] MRN_CALC_MB_MIN min_local=%lld max_local=%lld n_events=%d -- min_common_time=%llu max_common_time=%llu n_events=%d -- bytes=%llu ns=%llu mb_min=%.3lf\n",
        TASKID, min_local_time, max_local_time, Buffer_GetFillCount(buffer), min_common_time, max_common_time, num_common_events, bytes, ns, mb_min);

    MRN_STREAM_SEND(stream, REDUCE_DOUBLE_ADD, "%lf", mb_min);
    MRN_STREAM_SEND(stream, REDUCE_DOUBLE_ADD, "%lf", bytes_ns);

    Buffer_Unlock(buffer);
    return 0;
}

