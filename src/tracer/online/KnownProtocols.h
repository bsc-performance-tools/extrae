#ifndef __KNOWN_PROTOCOLS_H__
#define __KNOWN_PROTOCOLS_H__

#if defined(HAVE_SPECTRAL)
# include "SpectralRoot.h"
# include "SpectralWorker.h"
#endif

#if defined(HAVE_CLUSTERING)
# include "ClusteringRoot.h"
# include "ClusteringWorker.h"
#endif

#endif /* __KNOWN_PROTOCOLS_H__ */
