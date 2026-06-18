# Host-compiled 3D ledsim for helix-canopy.
# Includes the REAL effect math (../../src/effects.h) and generated geometry
# (../../geometry/topology.h) — no copies, so nothing can drift.

CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall
INCLUDES  = -I../../src -I../../geometry

effect_sim: effect_sim.cpp ../../src/effects.h ../../geometry/topology.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) effect_sim.cpp -o effect_sim

clean:
	rm -f effect_sim *.csv

.PHONY: clean
