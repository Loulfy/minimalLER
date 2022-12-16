cmake_minimum_required(VERSION 3.24)
project(minimalRT)

set(CMAKE_CXX_STANDARD 20)
set(SPIRV_REFLECT_EXAMPLES OFF)
set(SPIRV_REFLECT_EXECUTABLE OFF)
set(SPIRV_REFLECT_STATIC_LIB ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-std=c++20 -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++")
endif()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Vulkan)

#add_subdirectory(libs/KickstartRT)
#include_directories(libs/KickstartRT/include)KickstartRT_core_VULKAN
add_subdirectory(libs/SPIRV-Reflect)
include_directories(libs/VulkanMemoryAllocator-Hpp/include)
include_directories(libs/VulkanMemoryAllocator-Hpp/VulkanMemoryAllocator/include)

add_definitions(-DPROJECT_DIR=\"${PROJECT_SOURCE_DIR}\")

add_executable(minimalRT src/main.cpp src/ler.hpp src/ler.cpp src/format.cpp)
target_link_libraries(minimalRT Vulkan::Vulkan ${CONAN_LIBS} spirv-reflect-static)
