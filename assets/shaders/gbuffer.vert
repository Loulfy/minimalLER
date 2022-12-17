#version 460

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inUV;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
//layout (location = 4) in vec3 inColor;

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

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outWorldPos;
layout (location = 4) out vec3 outTangent;
layout (location = 5) out uint outMatId;

void main()
{
    Instance inst = props[gl_DrawID];
    vec4 tmpPos = vec4(inPos.xyz, 1.0);

    gl_Position = PushConstants.proj * PushConstants.view * inst.model * tmpPos;

    outUV = inUV.xy;

    // Vertex position in world space
    outWorldPos = vec3(inst.model * tmpPos);

    // Normal in world space
    mat3 mNormal = transpose(inverse(mat3(inst.model)));
    outNormal = mNormal * normalize(inNormal);
    outTangent = mNormal * normalize(inTangent);

    // Currently just vertex color
    //outColor = inColor;
    outColor = vec3(1,0,1);

    // Material
    outMatId = inst.matId;
}