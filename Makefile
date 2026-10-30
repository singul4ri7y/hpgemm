CC      ?= cc
CFLAGS  := -std=c99 -O3 -march=native -fopenmp -Wall -Wextra

# Intel MKL (single dynamic library). Override `MKLROOT` if needed.
MKLROOT     ?= /opt/intel/oneapi/mkl/latest
MKL_CFLAGS  := -I$(MKLROOT)/include
MKL_LDFLAGS := -L$(MKLROOT)/lib -L$(MKLROOT)/lib/intel64 \
               -Wl,-rpath,$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib/intel64 \
               -lmkl_rt

OBJ := hpgemm.o kernel.o

all: bench empirical
	mkdir -p data/perf data/empirical

bench: bench.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(MKL_LDFLAGS) -lm

empirical: empirical.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

bench.o: CFLAGS += $(MKL_CFLAGS)

%.o: %.c hpgemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o bench empirical

.PHONY: all clean
