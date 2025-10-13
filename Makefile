CC=gcc
CFLAGS=-Wall
LFLAGS=
FLAGS ?=

SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)
FILENAME = player

all: $(FILENAME)

# Debug mode with extra flags
debug: CFLAGS += -DDEBUG -g
debug: $(FILENAME)

# Debug mode without optimizations
debugNoOpt: CFLAGS += -DDEBUG -g -O0
debugNoOpt: $(FILENAME)

$(FILENAME): $(OBJECTS)
	$(CC) $(OBJECTS) $(LFLAGS) $(FLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(FILENAME)
