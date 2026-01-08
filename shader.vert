#version 330 core

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inColor;

uniform mat4 gViewProjection;

out vec3 fragPos;
out vec3 fragNormal;
out vec3 fragColor;

void main()
{
    fragPos = inPos;
    fragNormal = inNormal;
    fragColor = inColor;
    
    gl_Position = gViewProjection  * vec4(inPos, 1.0);
}