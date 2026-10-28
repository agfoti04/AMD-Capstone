CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread
 
all: drive_sender
 
drive_sender: drive_sender.cpp net.h drive_cmd.h
	$(CXX) $(CXXFLAGS) drive_sender.cpp -o drive_sender
 
 
clean:
	rm -f drive_sender