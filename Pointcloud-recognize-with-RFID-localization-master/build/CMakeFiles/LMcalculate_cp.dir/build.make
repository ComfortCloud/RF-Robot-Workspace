# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/tzq/Pointcloud-recognize-with-RFID-localization-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tzq/Pointcloud-recognize-with-RFID-localization-master/build

# Include any dependencies generated for this target.
include CMakeFiles/LMcalculate_cp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LMcalculate_cp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LMcalculate_cp.dir/flags.make

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o: CMakeFiles/LMcalculate_cp.dir/flags.make
CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o: ../src/preprocessing_cp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tzq/Pointcloud-recognize-with-RFID-localization-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o -c /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/preprocessing_cp.cpp

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/preprocessing_cp.cpp > CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.i

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/preprocessing_cp.cpp -o CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.s

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.requires:

.PHONY : CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.requires

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.provides: CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.requires
	$(MAKE) -f CMakeFiles/LMcalculate_cp.dir/build.make CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.provides.build
.PHONY : CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.provides

CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.provides.build: CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o


CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o: CMakeFiles/LMcalculate_cp.dir/flags.make
CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o: ../src/main_cp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tzq/Pointcloud-recognize-with-RFID-localization-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o -c /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/main_cp.cpp

CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/main_cp.cpp > CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.i

CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tzq/Pointcloud-recognize-with-RFID-localization-master/src/main_cp.cpp -o CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.s

CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.requires:

.PHONY : CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.requires

CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.provides: CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.requires
	$(MAKE) -f CMakeFiles/LMcalculate_cp.dir/build.make CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.provides.build
.PHONY : CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.provides

CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.provides.build: CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o


# Object files for target LMcalculate_cp
LMcalculate_cp_OBJECTS = \
"CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o" \
"CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o"

# External object files for target LMcalculate_cp
LMcalculate_cp_EXTERNAL_OBJECTS =

LMcalculate_cp: CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o
LMcalculate_cp: CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o
LMcalculate_cp: CMakeFiles/LMcalculate_cp.dir/build.make
LMcalculate_cp: CMakeFiles/LMcalculate_cp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tzq/Pointcloud-recognize-with-RFID-localization-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable LMcalculate_cp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LMcalculate_cp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LMcalculate_cp.dir/build: LMcalculate_cp

.PHONY : CMakeFiles/LMcalculate_cp.dir/build

CMakeFiles/LMcalculate_cp.dir/requires: CMakeFiles/LMcalculate_cp.dir/src/preprocessing_cp.cpp.o.requires
CMakeFiles/LMcalculate_cp.dir/requires: CMakeFiles/LMcalculate_cp.dir/src/main_cp.cpp.o.requires

.PHONY : CMakeFiles/LMcalculate_cp.dir/requires

CMakeFiles/LMcalculate_cp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LMcalculate_cp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LMcalculate_cp.dir/clean

CMakeFiles/LMcalculate_cp.dir/depend:
	cd /home/tzq/Pointcloud-recognize-with-RFID-localization-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tzq/Pointcloud-recognize-with-RFID-localization-master /home/tzq/Pointcloud-recognize-with-RFID-localization-master /home/tzq/Pointcloud-recognize-with-RFID-localization-master/build /home/tzq/Pointcloud-recognize-with-RFID-localization-master/build /home/tzq/Pointcloud-recognize-with-RFID-localization-master/build/CMakeFiles/LMcalculate_cp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LMcalculate_cp.dir/depend

