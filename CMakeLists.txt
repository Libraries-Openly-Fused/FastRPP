# FastRPP: AMD RPP's tensor API (rppt_*) as operations of the Fused Kernel Library (FKL), HIP only.
# Header only: the library target carries the include directories; the tests are HIP executables.
cmake_minimum_required(VERSION 3.28)

# The RX 9060 (gfx1200) unless the configure says otherwise
set(CMAKE_HIP_ARCHITECTURES "gfx1200" CACHE STRING "HIP architectures")
if(NOT DEFINED CMAKE_HIP_PLATFORM)
    set(CMAKE_HIP_PLATFORM "amd")
endif()

project(FastRPP VERSION 0.1.0 DESCRIPTION "AMD RPP's tensor API as FKL operations (HIP)" LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FASTRPP_BUILD_TESTS "Build the tests (they need RPP's library and an AMD GPU to run)" ON)
option(FASTRPP_CHECK_FKL_PIN "Fail the configure unless the fkl submodule is at the pinned commit" ON)

# ---- FKL: the git submodule fkl, pinned to the LTS-C++17 branch (C++17). Nothing from FKL's main branch (C++20).
set(FASTRPP_FKL_COMMIT "64ad9195cd785d0c93a70b35e02d5a6c7c4a5048")
set(FASTRPP_FKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fkl")
if(NOT EXISTS "${FASTRPP_FKL_DIR}/include/fused_kernel/fused_kernel.h")
    message(FATAL_ERROR "FKL is missing in ${FASTRPP_FKL_DIR}: git submodule update --init")
endif()
if(FASTRPP_CHECK_FKL_PIN)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${FASTRPP_FKL_DIR}"
                    OUTPUT_VARIABLE FASTRPP_FKL_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE FASTRPP_FKL_RESULT)
    if(NOT FASTRPP_FKL_RESULT EQUAL 0 OR NOT FASTRPP_FKL_HEAD STREQUAL FASTRPP_FKL_COMMIT)
        message(FATAL_ERROR "fkl must be at ${FASTRPP_FKL_COMMIT} (FKL LTS-C++17); it is at '${FASTRPP_FKL_HEAD}'")
    endif()
endif()

# ---- ROCm: the HIP runtime and RPP (its headers for the library, its library for the tests)
foreach(ROCM_DIR "${CMAKE_HIP_COMPILER_ROCM_ROOT}" "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
    if(NOT "${ROCM_DIR}" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_DIR}")
    endif()
endforeach()
find_package(hip CONFIG REQUIRED)
find_package(rpp CONFIG REQUIRED)

# ---- the library
add_library(FastRPP INTERFACE)
add_library(FastRPP::FastRPP ALIAS FastRPP)
target_include_directories(FastRPP INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
# FKL and RPP's headers are dependencies, included as system headers: the strict flags below check FastRPP's own code.
# FKL's array.h uses anonymous structs in unions (a GNU extension that -pedantic-errors rejects); FKL's CI checks FKL.
target_include_directories(FastRPP SYSTEM INTERFACE "${FASTRPP_FKL_DIR}/include")
get_target_property(FASTRPP_RPP_INCLUDE_DIRS rpp::rpp INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(FastRPP SYSTEM INTERFACE ${FASTRPP_RPP_INCLUDE_DIRS})
target_link_libraries(FastRPP INTERFACE hip::host)

# Strict C++17 for every FastRPP target: no compiler extensions, C++20 features are errors, and so is anything the
# standard does not allow (-pedantic-errors). The template depth of FKL's own builds.
set(FASTRPP_STRICT_FLAGS -Werror=c++20-extensions -pedantic-errors)
function(fastrpp_setup_target TARGET)
    set_target_properties(${TARGET} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF
                                               CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
                                               HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_options(${TARGET} PRIVATE ${FASTRPP_STRICT_FLAGS} -ftemplate-depth=1000)
endfunction()

if(FASTRPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "FastRPP ${PROJECT_VERSION}: HIP ${CMAKE_HIP_COMPILER_VERSION} (${CMAKE_HIP_COMPILER}), architectures "
               "${CMAKE_HIP_ARCHITECTURES}, RPP ${rpp_VERSION}, FKL ${FASTRPP_FKL_COMMIT}, flags ${FASTRPP_STRICT_FLAGS}")
