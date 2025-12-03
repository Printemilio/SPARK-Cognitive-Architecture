CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -pedantic -fPIC
LDFLAGS = -lm -lpthread

SRC = cg18.c qb16.c spark18.c
OBJ = $(SRC:.c=.o)

all: libspark_core.a libspark_core.so bench_memory

libspark_core.a: $(OBJ)
	ar rcs $@ $(OBJ)

libspark_core.so: $(OBJ)
	$(CC) $(CFLAGS) -shared $(OBJ) -o $@ $(LDFLAGS)

bench_memory: bench_memory.c libspark_core.a
	$(CC) $(CFLAGS) bench_memory.c -L. -lspark_core -o bench_memory $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) libspark_core.a libspark_core.so bench_memory

.PHONY: all clean
