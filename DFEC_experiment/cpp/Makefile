CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic

all: run_experiment_cpp

run_experiment_cpp: run_experiment.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f run_experiment_cpp
