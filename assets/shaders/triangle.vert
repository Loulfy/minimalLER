#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: enable

// Attributes
layout (location = 0) in vec3 inPos;
//layout (location = 1) in vec3 inNormal;
layout (location = 1) in vec3 inTex;

struct Instance
{
    mat4 model;
    vec3 bbmin;
    vec3 bbmax;
    uint matId;
};

layout(set = 0, binding = 0) readonly buffer inInstBuffer { Instance props[]; };

layout (push_constant) uniform constants
{
    mat4 proj;
    mat4 view;
} PushConstants;

// Varyings
layout (location = 0) out vec2 outCoord;
layout (location = 1) out uint outMatId;

out gl_PerVertex
{
    vec4 gl_Position;
};


void main()
{
    Instance inst = props[gl_DrawID];
    outCoord = inTex.xy;
    outMatId = inst.matId;
    gl_Position = PushConstants.proj * PushConstants.view * inst.model * vec4(inPos.xyz, 1.0);
}