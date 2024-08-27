# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build

# Include any dependencies generated for this target.
include CMakeFiles/reduction.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/reduction.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/reduction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduction.dir/flags.make

CMakeFiles/reduction.dir/src/main.cpp.o: CMakeFiles/reduction.dir/flags.make
CMakeFiles/reduction.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/reduction.dir/src/main.cpp.o: CMakeFiles/reduction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reduction.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reduction.dir/src/main.cpp.o -MF CMakeFiles/reduction.dir/src/main.cpp.o.d -o CMakeFiles/reduction.dir/src/main.cpp.o -c /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/main.cpp

CMakeFiles/reduction.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reduction.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/main.cpp > CMakeFiles/reduction.dir/src/main.cpp.i

CMakeFiles/reduction.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reduction.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/main.cpp -o CMakeFiles/reduction.dir/src/main.cpp.s

CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o: CMakeFiles/reduction.dir/flags.make
CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o: ../src/reduce_cpu.cpp
CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o: CMakeFiles/reduction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o -MF CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o.d -o CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o -c /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/reduce_cpu.cpp

CMakeFiles/reduction.dir/src/reduce_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reduction.dir/src/reduce_cpu.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/reduce_cpu.cpp > CMakeFiles/reduction.dir/src/reduce_cpu.cpp.i

CMakeFiles/reduction.dir/src/reduce_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reduction.dir/src/reduce_cpu.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/reduce_cpu.cpp -o CMakeFiles/reduction.dir/src/reduce_cpu.cpp.s

CMakeFiles/reduction.dir/src/reduce_gpu.cu.o: CMakeFiles/reduction.dir/flags.make
CMakeFiles/reduction.dir/src/reduce_gpu.cu.o: ../src/reduce_gpu.cu
CMakeFiles/reduction.dir/src/reduce_gpu.cu.o: CMakeFiles/reduction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/reduction.dir/src/reduce_gpu.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/reduction.dir/src/reduce_gpu.cu.o -MF CMakeFiles/reduction.dir/src/reduce_gpu.cu.o.d -x cu -c /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/reduce_gpu.cu -o CMakeFiles/reduction.dir/src/reduce_gpu.cu.o

CMakeFiles/reduction.dir/src/reduce_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/reduction.dir/src/reduce_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/reduction.dir/src/reduce_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/reduction.dir/src/reduce_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/reduction.dir/src/timer.cpp.o: CMakeFiles/reduction.dir/flags.make
CMakeFiles/reduction.dir/src/timer.cpp.o: ../src/timer.cpp
CMakeFiles/reduction.dir/src/timer.cpp.o: CMakeFiles/reduction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/reduction.dir/src/timer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reduction.dir/src/timer.cpp.o -MF CMakeFiles/reduction.dir/src/timer.cpp.o.d -o CMakeFiles/reduction.dir/src/timer.cpp.o -c /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/timer.cpp

CMakeFiles/reduction.dir/src/timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reduction.dir/src/timer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/timer.cpp > CMakeFiles/reduction.dir/src/timer.cpp.i

CMakeFiles/reduction.dir/src/timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reduction.dir/src/timer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/timer.cpp -o CMakeFiles/reduction.dir/src/timer.cpp.s

CMakeFiles/reduction.dir/src/utils.cpp.o: CMakeFiles/reduction.dir/flags.make
CMakeFiles/reduction.dir/src/utils.cpp.o: ../src/utils.cpp
CMakeFiles/reduction.dir/src/utils.cpp.o: CMakeFiles/reduction.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/reduction.dir/src/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reduction.dir/src/utils.cpp.o -MF CMakeFiles/reduction.dir/src/utils.cpp.o.d -o CMakeFiles/reduction.dir/src/utils.cpp.o -c /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/utils.cpp

CMakeFiles/reduction.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/reduction.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/utils.cpp > CMakeFiles/reduction.dir/src/utils.cpp.i

CMakeFiles/reduction.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/reduction.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/src/utils.cpp -o CMakeFiles/reduction.dir/src/utils.cpp.s

# Object files for target reduction
reduction_OBJECTS = \
"CMakeFiles/reduction.dir/src/main.cpp.o" \
"CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o" \
"CMakeFiles/reduction.dir/src/reduce_gpu.cu.o" \
"CMakeFiles/reduction.dir/src/timer.cpp.o" \
"CMakeFiles/reduction.dir/src/utils.cpp.o"

# External object files for target reduction
reduction_EXTERNAL_OBJECTS =

reduction: CMakeFiles/reduction.dir/src/main.cpp.o
reduction: CMakeFiles/reduction.dir/src/reduce_cpu.cpp.o
reduction: CMakeFiles/reduction.dir/src/reduce_gpu.cu.o
reduction: CMakeFiles/reduction.dir/src/timer.cpp.o
reduction: CMakeFiles/reduction.dir/src/utils.cpp.o
reduction: CMakeFiles/reduction.dir/build.make
reduction: /usr/local/cuda/lib64/libcudart_static.a
reduction: /usr/lib/x86_64-linux-gnu/librt.a
reduction: CMakeFiles/reduction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable reduction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduction.dir/build: reduction
.PHONY : CMakeFiles/reduction.dir/build

CMakeFiles/reduction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduction.dir/clean

CMakeFiles/reduction.dir/depend:
	cd /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2 /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2 /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build /home/lim/Desktop/other/GPU/GPU_LC/lesson3/ReduceInteger_2/build/CMakeFiles/reduction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduction.dir/depend

