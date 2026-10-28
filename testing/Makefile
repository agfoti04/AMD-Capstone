CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread

all: link linktest

link: link.cpp net.h
	$(CXX) $(CXXFLAGS) link.cpp -o link

linktest: linktest.cpp net.h
	$(CXX) $(CXXFLAGS) linktest.cpp -o linktest

clean:
	rm -f link linktest