CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wno-unused-function
TARGET = sim

SRCS = sim_main.cpp effects.cpp oklch_lut_data.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: clean
