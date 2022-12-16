#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Varying
layout (location = 0) in vec2 inCoord;

//layout(set = 0, binding = 0) uniform sampler2D tex;

// Return Output
layout (location = 0) out vec4 outFragColor;

void main()
{
    //outFragColor = vec4(texture(tex, inCoord).xyz, 1.0);
    outFragColor = vec4(1.0, 0.0, 1.0, 1.0);
}