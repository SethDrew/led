LIB := ../../../lib
CXX ?= c++
CXXFLAGS := -O2 -std=c++14 -Wall -I$(LIB)/fast_math -I$(LIB)/delta_sigma

bloom_dither_sim: bloom_dither_sim.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f bloom_dither_sim

.PHONY: clean
