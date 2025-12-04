cmake_minimum_required(VERSION 3.10)
project(Raycaster)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

# GLEW
find_package(GLEW REQUIRED)

# GLFW
find_package(glfw3 REQUIRED)

# GLM (optional but recommended)
find_package(glm QUIET)

add_executable(raycaster
    main.cpp
)

target_include_directories(raycaster PRIVATE
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(raycaster PRIVATE
    glfw
    ${GLEW_LIBRARIES}
    OpenGL::GL
)

