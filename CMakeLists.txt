cmake_minimum_required(VERSION 3.10)
project(Visualizer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)

find_package(glfw3 REQUIRED)

find_package(glm QUIET)

add_executable(vis
    main.cpp
)

target_include_directories(vis PRIVATE
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(vis PRIVATE
    glfw
    ${GLEW_LIBRARIES}
    OpenGL::GL
)

