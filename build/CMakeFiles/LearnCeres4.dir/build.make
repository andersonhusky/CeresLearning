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
CMAKE_SOURCE_DIR = /media/hongfeng/Storage/Code/CeresLearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/hongfeng/Storage/Code/CeresLearning/build

# Include any dependencies generated for this target.
include CMakeFiles/LearnCeres4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LearnCeres4.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LearnCeres4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LearnCeres4.dir/flags.make

CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o: CMakeFiles/LearnCeres4.dir/flags.make
CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o: ../src/No4_CurveFitting.cpp
CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o: CMakeFiles/LearnCeres4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/hongfeng/Storage/Code/CeresLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o -MF CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o.d -o CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o -c /media/hongfeng/Storage/Code/CeresLearning/src/No4_CurveFitting.cpp

CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/hongfeng/Storage/Code/CeresLearning/src/No4_CurveFitting.cpp > CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.i

CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/hongfeng/Storage/Code/CeresLearning/src/No4_CurveFitting.cpp -o CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.s

# Object files for target LearnCeres4
LearnCeres4_OBJECTS = \
"CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o"

# External object files for target LearnCeres4
LearnCeres4_EXTERNAL_OBJECTS =

LearnCeres4: CMakeFiles/LearnCeres4.dir/src/No4_CurveFitting.cpp.o
LearnCeres4: CMakeFiles/LearnCeres4.dir/build.make
LearnCeres4: /usr/local/lib/x86_64-linux-gnu/libceres.a
LearnCeres4: /usr/lib/x86_64-linux-gnu/libpython2.7.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libglog.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
LearnCeres4: /usr/lib/x86_64-linux-gnu/libspqr.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libcholmod.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libamd.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libcamd.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libccolamd.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libcolamd.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libcxsparse.so
LearnCeres4: /usr/local/cuda/lib64/libcudart_static.a
LearnCeres4: /usr/lib/x86_64-linux-gnu/librt.so
LearnCeres4: /usr/local/cuda/lib64/libcublas.so
LearnCeres4: /usr/local/cuda/lib64/libcusolver.so
LearnCeres4: /usr/local/cuda/lib64/libcusparse.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/liblapack.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libblas.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libf77blas.so
LearnCeres4: /usr/lib/x86_64-linux-gnu/libatlas.so
LearnCeres4: CMakeFiles/LearnCeres4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/hongfeng/Storage/Code/CeresLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LearnCeres4"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LearnCeres4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LearnCeres4.dir/build: LearnCeres4
.PHONY : CMakeFiles/LearnCeres4.dir/build

CMakeFiles/LearnCeres4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LearnCeres4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LearnCeres4.dir/clean

CMakeFiles/LearnCeres4.dir/depend:
	cd /media/hongfeng/Storage/Code/CeresLearning/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/hongfeng/Storage/Code/CeresLearning /media/hongfeng/Storage/Code/CeresLearning /media/hongfeng/Storage/Code/CeresLearning/build /media/hongfeng/Storage/Code/CeresLearning/build /media/hongfeng/Storage/Code/CeresLearning/build/CMakeFiles/LearnCeres4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LearnCeres4.dir/depend

