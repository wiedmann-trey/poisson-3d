#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

struct VolumeHeader {
    int nx, ny, nz;
    double start_x, start_y, start_z;
    double end_x, end_y, end_z;
};

// --------------------------------------------------------
// Load entire shader file into a string
// --------------------------------------------------------
std::string loadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open file " << path << "\n";
        return {};
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// --------------------------------------------------------
// Compile shader stage
// --------------------------------------------------------
GLuint compileShader(const std::string& src, GLenum type) {
    GLuint id = glCreateShader(type);
    const char* s = src.c_str();
    glShaderSource(id, 1, &s, nullptr);
    glCompileShader(id);

    GLint ok;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(id, 4096, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return id;
}

// --------------------------------------------------------
// Link shader program
// --------------------------------------------------------
GLuint makeProgram(const std::string& vert, const std::string& frag)
{
    GLuint vs = compileShader(vert, GL_VERTEX_SHADER);
    GLuint fs = compileShader(frag, GL_FRAGMENT_SHADER);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(prog, 4096, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return prog;
}

// --------------------------------------------------------
// Load volume with double data and convert to float32 texture
// --------------------------------------------------------
bool loadVolume(const std::string& filename,
                VolumeHeader& hdr,
                std::vector<double>& dataD)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;

    fread(&hdr.nx, sizeof(int), 1, f);
    fread(&hdr.ny, sizeof(int), 1, f);
    fread(&hdr.nz, sizeof(int), 1, f);

    fread(&hdr.start_x, sizeof(double), 1, f);
    fread(&hdr.start_y, sizeof(double), 1, f);
    fread(&hdr.start_z, sizeof(double), 1, f);

    fread(&hdr.end_x, sizeof(double), 1, f);
    fread(&hdr.end_y, sizeof(double), 1, f);
    fread(&hdr.end_z, sizeof(double), 1, f);

    size_t total = (size_t)hdr.nx * hdr.ny * hdr.nz;
    dataD.resize(total);
    fread(dataD.data(), sizeof(double), total, f);

    fclose(f);
    return true;
}

// --------------------------------------------------------
// Create GL texture (always float32 in OpenGL)
// --------------------------------------------------------
GLuint create3DTexture(const VolumeHeader& hdr,
                       const std::vector<double>& dataD)
{
    std::vector<float> dataF(dataD.size());
    for (size_t i = 0; i < dataD.size(); i++)
        dataF[i] = (float)dataD[i];   // convert double → float

    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_3D, id);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glTexImage3D(GL_TEXTURE_3D,
                 0,
                 GL_R32F,
                 hdr.nx, hdr.ny, hdr.nz,
                 0,
                 GL_RED,
                 GL_FLOAT,
                 dataF.data());

    glBindTexture(GL_TEXTURE_3D, 0);
    return id;
}

// --------------------------------------------------------
// Cube VAO (a cube in [0,1]^3)
// --------------------------------------------------------
GLuint createCubeVAO()
{
    float verts[] = {
        // x y z (0 to 1 cube)
        0,0,0, 1,0,0, 1,1,0, 0,1,0,
        0,0,1, 1,0,1, 1,1,1, 0,1,1
    };
    unsigned int idx[] = {
        0,1,2, 2,3,0,
        4,5,6, 6,7,4,
        0,4,7, 7,3,0,
        1,5,6, 6,2,1,
        3,2,6, 6,7,3,
        0,1,5, 5,4,0
    };

    GLuint vao,vbo,ebo;
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(idx),idx,GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);

    glBindVertexArray(0);
    return vao;
}

// Add these global variables at the top of main.cpp
float camYaw = 1.0f;
float camPitch = 0.6f;
double lastMouseX = 0.0;
double lastMouseY = 0.0;
bool firstMouse = true;
bool mousePressed = false;  // Track mouse button state

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            firstMouse = true;  // Reset on new press to avoid jump
        } else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (!mousePressed) return;  // Only rotate when mouse is pressed

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
        return;  // Don't apply movement on first frame
    }

    double xoffset = xpos - lastMouseX;
    double yoffset = lastMouseY - ypos;
    lastMouseX = xpos;
    lastMouseY = ypos;

    float sensitivity = 0.005f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    camYaw += xoffset;
    camPitch += -yoffset;

    // Constrain pitch
    if (camPitch > 1.57f) camPitch = 1.57f;
    if (camPitch < -1.57f) camPitch = -1.57f;
}

int main()
{
    glfwInit();
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Raycaster", nullptr, nullptr);
    glfwMakeContextCurrent(win);
    glewInit();

    // Set up mouse input
    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);  // Add button callback

    // Load volume
    VolumeHeader hdr;
    std::vector<double> dataD;
    if (!loadVolume("volume.bin", hdr, dataD)) {
        std::cerr << "Failed loading volume\n";
        return 1;
    }
    glm::vec3 boxMin(hdr.start_x, hdr.start_y, hdr.start_z);
    glm::vec3 boxMax(hdr.end_x, hdr.end_y, hdr.end_z);
    glm::vec3 boxCenter = (boxMax+boxMin)/glm::vec3(2.0);
    glm::vec3 size = boxMax - boxMin;
    float maxDim = std::max(size.x, std::max(size.y, size.z));
    float volumeMax = (float)*std::max_element(dataD.begin(), dataD.end());
    float volumeMin = (float)*std::min_element(dataD.begin(), dataD.end());

    GLuint tex3D = create3DTexture(hdr, dataD);
    GLuint cube = createCubeVAO();

    std::string vs = loadFile("shader.vert");
    std::string fs = loadFile("raycast.frag");
    GLuint prog = makeProgram(vs, fs);

    float fovY = glm::radians(60.0f);
    float camDist = 1.5 * 0.5f * maxDim / std::tan(fovY / 2.0f);

    glm::vec3 target = boxCenter;
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    int width, height;
    glfwGetFramebufferSize(win, &width, &height);

    glm::mat4 projection = glm::perspective(
        fovY,
        float(width) / float(height),
        0.01f,
        100.0f
    );

    // Main loop
    while (!glfwWindowShouldClose(win)) {
        // Update camera position based on yaw and pitch
        glm::vec3 camPos(
            camDist * cos(camPitch) * cos(camYaw),
            camDist * sin(camPitch),
            camDist * cos(camPitch) * sin(camYaw)
        );
        camPos += target;

        glm::mat4 view = glm::lookAt(camPos, target, up);

        // Transform camera to texture space
        glm::vec3 camPosTexture = (camPos - boxMin) / (boxMax - boxMin);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(prog);

        glUniform3f(glGetUniformLocation(prog, "boxMin"),
                   hdr.start_x, hdr.start_y, hdr.start_z);
        glUniform3f(glGetUniformLocation(prog, "boxMax"),
                   hdr.end_x, hdr.end_y, hdr.end_z);

        glUniformMatrix4fv(glGetUniformLocation(prog, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(prog, "view"), 1, GL_FALSE, &view[0][0]);

        // Send camera position in texture space
        glUniform3f(glGetUniformLocation(prog, "cameraPos"),
                    camPosTexture.x, camPosTexture.y, camPosTexture.z);
        glUniform1f(glGetUniformLocation(prog, "stepSize"), 0.01f);
        glUniform1f(glGetUniformLocation(prog, "densityScale"), 2.0f);
        glUniform1f(glGetUniformLocation(prog, "volumeMin"), volumeMin);
        glUniform1f(glGetUniformLocation(prog, "volumeMax"), volumeMax);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, tex3D);
        glUniform1i(glGetUniformLocation(prog, "volumeTex"), 0);

        glBindVertexArray(cube);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
