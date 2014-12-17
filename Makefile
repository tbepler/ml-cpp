CC=g++
CFLAGS= -c -Wall -Ofast -fopenmp -std=c++11
SRCS=Main.cpp
OBJS=$(SRCS:.cpp=.o)
EXE=main.out

all: $(SRCS) $(EXE)

$(EXE): $(OBJS)
	$(CC) $(OBJS) -o $@

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $< -o $@

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^>> ./.depend;

include .depend
