#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_nonuniform_qualifier : require


// Varying
layout (location = 0) in vec2 inCoord;
layout (location = 1) in flat uint inMatId;

struct Material
{
    uint texId;
    uint norId;
    vec3 color;
};

layout(set = 1, binding = 0) readonly buffer inMatBuffer { Material mats[]; };
layout(set = 1, binding = 1) uniform sampler2D textures[];

// Return Output
layout (location = 0) out vec4 outFragColor;

void main()
{
    Material m = mats[inMatId];
    //vec2 t = inCoord;
    //t.x = 1 - t.x;
    //outFragColor = vec4(texture(textures[nonuniformEXT(m.texId)], inCoord).xyz * m.color, 1.0);
    outFragColor = texture(textures[nonuniformEXT(m.texId)], inCoord) * vec4(m.color, 1.0);
    //outFragColor = vec4(m.color, 1.0);
    //outFragColor = vec4(1.0, 0.0, 0.0, 1.0);
}