CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lm

SRC = cg19.c qb16.c spark19.c spark_bridge.c spark_cortex.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean

all: libspark_core.a libspark_core.so

libspark_core.a: $(OBJ)
	ar rcs $@ $^

libspark_core.so: $(OBJ)
	$(CC) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)


%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(OBJ) libspark_core.a libspark_core.so
