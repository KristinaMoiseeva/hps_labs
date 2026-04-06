NVCC ?= nvcc
TARGET ?= vectorsum
SRC := src/main.cu
CUDAFLAGS ?= -O3 -std=c++17

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
