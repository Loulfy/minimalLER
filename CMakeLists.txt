cmake_minimum_required(VERSION 3.22)
project(minimalRT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
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
include_directories(libs/imgui)
include_directories(libs/imgui/backends)
include_directories(libs/ImGuizmo)
include_directories(libs/VulkanMemoryAllocator-Hpp/include)
include_directories(libs/VulkanMemoryAllocator-Hpp/VulkanMemoryAllocator/include)

add_definitions(-DPROJECT_DIR=\"${PROJECT_SOURCE_DIR}\")

set(IMGUI
    "libs/imgui/imconfig.h"
    "libs/imgui/imgui.h"
    "libs/imgui/imgui_internal.h"
    "libs/imgui/imstb_rectpack.h"
    "libs/imgui/imstb_textedit.h"
    "libs/imgui/imstb_truetype.h"
    "libs/imgui/imgui.cpp"
    "libs/imgui/imgui_demo.cpp"
    "libs/imgui/imgui_draw.cpp"
    "libs/imgui/imgui_tables.cpp"
    "libs/imgui/imgui_widgets.cpp"
    "libs/imgui/backends/imgui_impl_vulkan.h"
    "libs/imgui/backends/imgui_impl_vulkan.cpp"
    "libs/imgui/backends/imgui_impl_glfw.h"
    "libs/imgui/backends/imgui_impl_glfw.cpp"
    "libs/ImGuizmo/ImGuizmo.h"
    "libs/ImGuizmo/ImGuizmo.cpp"
)

add_executable(minimalRT src/main.cpp src/ler.hpp src/ler.cpp src/format.cpp src/camera.h ${IMGUI})
target_link_libraries(minimalRT Vulkan::Vulkan ${CONAN_LIBS} spirv-reflect-static)
