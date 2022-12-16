#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Varyings
layout (location = 0) out vec2 outTex;

out gl_PerVertex
{
    vec4 gl_Position;
};

const vec2 positions[4] = vec2[](
    vec2(-1, -1),
    vec2(+1, -1),
    vec2(-1, +1),
    vec2(+1, +1)
);
const vec2 coords[4] = vec2[](
    vec2(0, 0),
    vec2(1, 0),
    vec2(0, 1),
    vec2(1, 1)
);

/*struct Mdi_cmd {
    uint idx_count;
    uint inst_count; // visibility
    uint idx_base;
    int vert_offset;
    uint inst_idx;
    float paddings[7];
};

layout(set = 0, binding = 2)  buffer Mdi_cmd_buffer_out
{
    Mdi_cmd cmds[];
};*/

void main()
{
    outTex = coords[gl_VertexIndex];
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}