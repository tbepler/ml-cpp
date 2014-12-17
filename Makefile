CC=g++
CFLAGS= -Wall -Ofast -fopenmp -std=c++11

SRCDIR=src
BUILDDIR=build
TESTDIR=tests
TARGET=bin/pkkridge.out

SRCS=$(shell find $(SRCDIR) -type f -name *.cpp)
OBJS=$(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SRCS:.cpp=.o))
TESTS=$(shell find $(TESTDIR) -type f -name *.cpp)
INC= -I include
LIB= -lgomp

all: $(SRCS) $(TARGET) tests

$(TARGET): $(OBJS)
	$(CC) $^ $(LIB) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $(INC) $< -o $@

tests:
	@mkdir -p bin/$(TESTDIR)
	$(foreach var, $(TESTS), $(CC) $(CFLAGS) $(INC) var -o bin/$(TESTDIR)/$(var:.cpp=.out))

clean:
	@rm -r $(BUILDDIR) $(TARGET)

depend: .depend

.depend: $(SRCS) $(TESTS)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^>> ./.depend;

include .depend
