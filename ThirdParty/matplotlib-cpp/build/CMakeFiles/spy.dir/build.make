# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/hongfeng/Storage/Ubuntu/matplotlib-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/spy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/spy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/spy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/spy.dir/flags.make

CMakeFiles/spy.dir/examples/spy.cpp.o: CMakeFiles/spy.dir/flags.make
CMakeFiles/spy.dir/examples/spy.cpp.o: ../examples/spy.cpp
CMakeFiles/spy.dir/examples/spy.cpp.o: CMakeFiles/spy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/spy.dir/examples/spy.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/spy.dir/examples/spy.cpp.o -MF CMakeFiles/spy.dir/examples/spy.cpp.o.d -o CMakeFiles/spy.dir/examples/spy.cpp.o -c /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/examples/spy.cpp

CMakeFiles/spy.dir/examples/spy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spy.dir/examples/spy.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/examples/spy.cpp > CMakeFiles/spy.dir/examples/spy.cpp.i

CMakeFiles/spy.dir/examples/spy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spy.dir/examples/spy.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/examples/spy.cpp -o CMakeFiles/spy.dir/examples/spy.cpp.s

# Object files for target spy
spy_OBJECTS = \
"CMakeFiles/spy.dir/examples/spy.cpp.o"

# External object files for target spy
spy_EXTERNAL_OBJECTS =

bin/spy: CMakeFiles/spy.dir/examples/spy.cpp.o
bin/spy: CMakeFiles/spy.dir/build.make
bin/spy: /media/hongfeng/Storage/Ubuntu/anaconda3/lib/libpython3.9.so
bin/spy: CMakeFiles/spy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/spy"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/spy.dir/build: bin/spy
.PHONY : CMakeFiles/spy.dir/build

CMakeFiles/spy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/spy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/spy.dir/clean

CMakeFiles/spy.dir/depend:
	cd /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/hongfeng/Storage/Ubuntu/matplotlib-cpp /media/hongfeng/Storage/Ubuntu/matplotlib-cpp /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build /media/hongfeng/Storage/Ubuntu/matplotlib-cpp/build/CMakeFiles/spy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/spy.dir/depend
