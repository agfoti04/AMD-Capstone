CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread
 
all: drive_sender
 
drive_sender: drive_sender.cpp net.h drive_cmd.h
	$(CXX) $(CXXFLAGS) drive_sender.cpp -o drive_sender
 
drive_receiver: drive_receiver.cpp net.h drive_cmd.h
	$(CXX) $(CXXFLAGS) drive_receiver.cpp -o drive_receiver
 
clean:
	rm -f link linktest cmd_sender cmd_receiver drive_sender drive_receiver