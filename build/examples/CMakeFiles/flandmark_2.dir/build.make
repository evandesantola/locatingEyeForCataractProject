# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/evan/eyeDetection/flandmark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/evan/eyeDetection/flandmark/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/flandmark_2.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/flandmark_2.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/flandmark_2.dir/flags.make

examples/CMakeFiles/flandmark_2.dir/example2.cpp.o: examples/CMakeFiles/flandmark_2.dir/flags.make
examples/CMakeFiles/flandmark_2.dir/example2.cpp.o: ../examples/example2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/evan/eyeDetection/flandmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/flandmark_2.dir/example2.cpp.o"
	cd /home/evan/eyeDetection/flandmark/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/flandmark_2.dir/example2.cpp.o -c /home/evan/eyeDetection/flandmark/examples/example2.cpp

examples/CMakeFiles/flandmark_2.dir/example2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flandmark_2.dir/example2.cpp.i"
	cd /home/evan/eyeDetection/flandmark/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/evan/eyeDetection/flandmark/examples/example2.cpp > CMakeFiles/flandmark_2.dir/example2.cpp.i

examples/CMakeFiles/flandmark_2.dir/example2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flandmark_2.dir/example2.cpp.s"
	cd /home/evan/eyeDetection/flandmark/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/evan/eyeDetection/flandmark/examples/example2.cpp -o CMakeFiles/flandmark_2.dir/example2.cpp.s

examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.requires:

.PHONY : examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.requires

examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.provides: examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/flandmark_2.dir/build.make examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.provides.build
.PHONY : examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.provides

examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.provides.build: examples/CMakeFiles/flandmark_2.dir/example2.cpp.o


# Object files for target flandmark_2
flandmark_2_OBJECTS = \
"CMakeFiles/flandmark_2.dir/example2.cpp.o"

# External object files for target flandmark_2
flandmark_2_EXTERNAL_OBJECTS =

examples/flandmark_2: examples/CMakeFiles/flandmark_2.dir/example2.cpp.o
examples/flandmark_2: examples/CMakeFiles/flandmark_2.dir/build.make
examples/flandmark_2: libflandmark/libflandmark_static.a
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
examples/flandmark_2: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
examples/flandmark_2: examples/CMakeFiles/flandmark_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/evan/eyeDetection/flandmark/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable flandmark_2"
	cd /home/evan/eyeDetection/flandmark/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flandmark_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/flandmark_2.dir/build: examples/flandmark_2

.PHONY : examples/CMakeFiles/flandmark_2.dir/build

examples/CMakeFiles/flandmark_2.dir/requires: examples/CMakeFiles/flandmark_2.dir/example2.cpp.o.requires

.PHONY : examples/CMakeFiles/flandmark_2.dir/requires

examples/CMakeFiles/flandmark_2.dir/clean:
	cd /home/evan/eyeDetection/flandmark/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/flandmark_2.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/flandmark_2.dir/clean

examples/CMakeFiles/flandmark_2.dir/depend:
	cd /home/evan/eyeDetection/flandmark/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/evan/eyeDetection/flandmark /home/evan/eyeDetection/flandmark/examples /home/evan/eyeDetection/flandmark/build /home/evan/eyeDetection/flandmark/build/examples /home/evan/eyeDetection/flandmark/build/examples/CMakeFiles/flandmark_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/flandmark_2.dir/depend

