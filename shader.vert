#version 330 core

layout(location = 0) in vec3 inPos;

uniform mat4 projection;
uniform mat4 view;

uniform vec3 boxMin;
uniform vec3 boxMax;

out vec3 vPos;

void main()
{
    // Map cube [0,1]^3 to physical bounding box
    vec3 worldPos = mix(boxMin, boxMax, inPos);
    vPos = inPos;              // pass texture coords
    gl_Position = projection * view * vec4(worldPos, 1.0);
}

