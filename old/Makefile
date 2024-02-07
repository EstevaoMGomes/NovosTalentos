### definitions

CXX = g++-13 -std=c++20 # $(shell root-config --cflags) -I src -I ~/eigen-3.4.0
#ROOT_INC = -I $(shell root-config --incdir)
#ROOT_LIB = $(shell root-config --libs)

#SRC = $(wildcard src/*.cpp)
MAIN = $(wildcard *.cpp)

#SRC_OBJ = $(patsubst %.cpp, bin/%.o, $(notdir $(SRC)))
MAIN_OBJ = $(patsubst %.cpp, bin/%.o, $(notdir $(MAIN)))
MAIN_EXE = $(patsubst %.cpp, bin/%.exe, $(notdir $(MAIN)))

### rules (target: dependencies / actions)

exe: $(SRC_OBJ) $(MAIN_EXE)
obj: $(SRC_OBJ) $(MAIN_OBJ)

bin/%.exe: bin/%.o $(SRC_OBJ)
	@$(CXX) $< -o $@ $(ROOT_INC) $(SRC_OBJ) $(ROOT_LIB)

#bin/%.o: src/%.cpp
#	@echo compiling... [$< -o $@]
#	@$(CXX) -c $< -o $@ $(ROOT_INC)

bin/%.o: %.cpp
	@echo compiling... [$< -o $@]
	@$(CXX) -c $< -o $@ -I src $(ROOT_INC)

clean:
	@echo deleting...[$(wildcard bin/*)]
	@rm -f $(wildcard bin/*)