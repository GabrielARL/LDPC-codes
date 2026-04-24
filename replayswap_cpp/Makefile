CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic -Iinclude

SRC := src/replayswap_bundle.cpp src/replayswap_runner.cpp
BIN := run_replayswap_cpp

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)
