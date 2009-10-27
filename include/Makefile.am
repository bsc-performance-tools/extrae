include_HEADERS = mpitracef_user_events.h mpitrace_user_events.h 

if IS_CELL_MACHINE
include_HEADERS += spu_trace.h ppu_trace_sdk1.h ppu_trace_sdk2.h
endif

#
# install-exec-hook is executed before the regular objectives are performed
# install-data-hook is executed after the regular objectives are performed
#

install-data-hook:
if CELL_SDK_1
	cd $(includedir); $(LN_S) -f ppu_trace_sdk1.h ppu_trace.h
else
if CELL_SDK_2
	cd $(includedir); $(LN_S) -f ppu_trace_sdk2.h ppu_trace.h
endif
endif
