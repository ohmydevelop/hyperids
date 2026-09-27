# HyperIDs static single-binary build.
# Weights/vocab are embedded at link time via `ld -r -b binary`.
CC ?= cc
CFLAGS ?= -O3 -march=x86-64-v2
STATIC ?= 1

.PHONY: all clean

all: build/hyperids

build/model_f32.o: ci_assets/model_f32.bin
	mkdir -p build
	ld -r -b binary ci_assets/model_f32.bin -o build/model_f32.o

build/vocab.o: ci_assets/vocab.txt
	mkdir -p build
	ld -r -b binary ci_assets/vocab.txt -o build/vocab.o

build/hyperids: hyperids.c hyperids_labels.h build/model_f32.o build/vocab.o
	mkdir -p build
ifeq ($(STATIC),1)
	$(CC) $(CFLAGS) -static hyperids.c build/model_f32.o build/vocab.o -lm -o build/hyperids
else
	$(CC) $(CFLAGS) hyperids.c build/model_f32.o build/vocab.o -lm -o build/hyperids
endif

clean:
	rm -rf build
