#include "FE_ProtScope.h"

int FE_ProtScope::run(Stream * stream)
{
    int tag;
    PacketPtr data;
    long long min_common, max_common;
    double total_mb_min, total_bytes_ns;
    int total_num_common_events = 0;

    /* Reduce the common region */
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &min_common);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_LL_MAX_POSITIVE);
    data->unpack("%ld", &max_common);

    /* Broadcast the common region */
    MRN_STREAM_SEND(stream, MRN_CALC_MB_MIN, "%ld %ld", min_common, max_common);

    MRN_STREAM_RECV(stream, &tag, data, REDUCE_INT_ADD);
    data->unpack("%d", &total_num_common_events);

    MRN_STREAM_RECV(stream, &tag, data, REDUCE_DOUBLE_ADD);
    data->unpack("%lf", &total_mb_min);
    MRN_STREAM_RECV(stream, &tag, data, REDUCE_DOUBLE_ADD);
    data->unpack("%lf", &total_bytes_ns);
    fprintf(stderr, "[FE] total_num_common_events=%d total_mb_min=%lf\n", total_num_common_events, total_mb_min);

    #define RATIO 1
    int desired_mb = 100;
    double freq_secs = ((desired_mb * RATIO) / total_mb_min) * 60;
    double freq_ns = ((desired_mb * 1024 * 1024 * RATIO) / total_bytes_ns);
    fprintf(stderr, "[FE] freq_secs=%lf freq_ns=%lf\n", freq_secs, freq_ns);

    return 0;
}
