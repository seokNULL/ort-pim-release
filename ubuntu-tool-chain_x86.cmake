SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
#SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
#SET(CMAKE_C_FLAGS "-march=armv8-a -mtune=generic -Wno-unused-parameter -Wno-type-limits")
#SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
#SET(CMAKE_CXX_FLAGS "-march=armv8-a -mtune=generic -Wno-unused-parameter -Wno-type-limits")

SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_C_FLAGS "-march=x86-64 -Wno-unused-parameter -Wno-type-limits")
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_CXX_FLAGS "-march=x86-64 -Wno-unused-parameter -Wno-type-limits")

#SET(CMAKE_FIND_ROOT_PATH /mnt/sde2)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

