#
# Unix/Linux makefile
# Abulhair Saparov
#

#
# List of source files
#

HDPMCMC_CPP_SRCS=mcmc.cpp
HDPMCMC_DBG_OBJS=$(HDPMCMC_CPP_SRCS:.cpp=.debug.o)
HDPMCMC_OBJS=$(HDPMCMC_CPP_SRCS:.cpp=.release.o)


#
# Compile and link options
#

CPP=g++
cc-option = $(shell $(CPP) -Werror $(1) -c -x c /dev/null -o /dev/null 2>/dev/null; echo $$?)

LIBRARY_PKG_LIBS=
PKG_LIBS=-pthread
NO_AS_NEEDED=-Wl,--no-as-needed
ifeq ($(call cc-option, $(NO_AS_NEEDED)),0)
	PKG_LIBS += $(NO_AS_NEEDED)
endif
GLIBC := $(word 2,$(shell getconf GNU_LIBC_VERSION 2>/dev/null))
ifeq "$(.SHELLSTATUS)" "0"
	GLIBC_HAS_RT := $(shell expr $(GLIBC) \>= 2.17)
	ifeq "$(GLIBC_HAS_RT)" "0"
		LIBRARY_PKG_LIBS += -lrt
		PKG_LIBS += -lrt
	endif
endif

WARNING_FLAGS=-Wall -Wpedantic
override CPPFLAGS_DBG += $(WARNING_FLAGS) -I. -g -march=native -mtune=native -std=c++11
override CPPFLAGS += $(WARNING_FLAGS) -I. -Ofast -DNDEBUG -march=native -mtune=native -std=c++11 -fno-stack-protector
override LDFLAGS_DBG += -g $(LIB_PATHS) $(PKG_LIBS)
override LDFLAGS += $(LIB_PATHS) -fwhole-program $(PKG_LIBS)


#
# GNU Make: targets that don't build files
#

.PHONY: all debug clean distclean

#
# Make targets
#

all: hdp_mcmc

debug: hdp_mcmc_dbg

-include $(HDPMCMC_OBJS:.release.o=.release.d)
-include $(HDPMCMC_DBG_OBJS:.debug.o=.debug.d)

define make_dependencies
	$(1) $(2) -c $(3).$(4) -o $(3).$(5).o
	$(1) -MM $(2) $(3).$(4) > $(3).$(5).d
	@mv -f $(3).$(5).d $(3).$(5).d.tmp
	@sed -e 's|.*:|$(3).$(5).o:|' < $(3).$(5).d.tmp > $(3).$(5).d
	@sed -e 's/.*://' -e 's/\\$$//' < $(3).$(5).d.tmp | fmt -1 | \
		sed -e 's/^ *//' -e 's/$$/:/' >> $(3).$(5).d
	@rm -f $(3).$(5).d.tmp
endef

%.release.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS),$*,cpp,release)
%.release.pic.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS),$*,cpp,release.pic)
%.debug.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS_DBG),$*,cpp,debug)
%.debug.pic.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS_DBG),$*,cpp,debug.pic)

hdp_mcmc: $(LIBS) $(HDPMCMC_OBJS)
		$(CPP) -o hdp_mcmc $(CPPFLAGS) $(LDFLAGS) $(HDPMCMC_OBJS)

hdp_mcmc_dbg: $(LIBS) $(HDPMCMC_DBG_OBJS)
		$(CPP) -o hdp_mcmc_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(HDPMCMC_DBG_OBJS)

clean:
	    ${RM} -f *.o */*.o */*/*.o *.d */*.d */*/*.d hdp_mcmc hdp_mcmc.exe hdp_mcmc_dbg hdp_mcmc_dbg.exe $(LIBS)
