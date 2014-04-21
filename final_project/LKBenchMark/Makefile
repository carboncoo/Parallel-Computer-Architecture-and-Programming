# Common binaries
GPLUS           ?= g++

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      TARGET    := debug
else
      TARGET    := release
endif

# Common includes and paths
LDFLAGS       := -lm -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_video

CFLAGS  = -O3

# Target rules
all: build

build: vectors

LK.o: LK.cpp
	$(GPLUS) $(CCFLAGS) -o $@ -c $<

vectors.o: vectors.cpp
	$(GPLUS) $(CCFLAGS) -o $@ -c $<

vectors: vectors.o LK.o
	$(GPLUS) -o $@ vectors.o LK.o $(LDFLAGS) $(EXTRA_LDFLAGS)

run: build
	./vectors

clean:
	rm -f vectors *.o core.* *.a out.bin
