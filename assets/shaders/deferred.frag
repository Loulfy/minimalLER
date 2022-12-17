#version 460

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput samplerPosition;
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput samplerNormal;
layout (input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput samplerAlbedo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

struct Light
{
    vec4 position;
    vec3 color;
    float radius;
};

layout (push_constant) uniform constants
{
    vec3 viewPos;
} pc;

/*layout (binding = 4) uniform UBO
{
    Light lights[6];
    vec4 viewPos;
    int displayDebugTarget;
} ubo;*/

const Light lights[1] = Light[](
    Light(vec4(0.f, 1.f, 0.3f, 0.f), vec3(1,1,1), 20) // -0.2f, -1.0f, -0.3f // 0.f, 15.f, 0.3f, 1.f
);

void main()
{
    // Get G-Buffer values
    /*vec3 fragPos = subpassLoad(samplerPosition, gl_SampleID).rgb;
    vec3 normal = subpassLoad(samplerNormal, gl_SampleID).rgb;
    vec4 albedo = subpassLoad(samplerAlbedo, gl_SampleID);*/

    vec3 fragPos = subpassLoad(samplerPosition).rgb;
    vec3 normal = subpassLoad(samplerNormal).rgb;
    vec4 albedo = subpassLoad(samplerAlbedo);

    //outFragcolor = vec4(normal, 1);
    //outFragcolor = albedo;
    //outFragcolor = vec4(1,0,1,1);
    //return;

    // Debug display
    /*if (ubo.displayDebugTarget > 0) {
        switch (ubo.displayDebugTarget) {
            case 1:
            outFragcolor.rgb = fragPos;
            break;
            case 2:
            outFragcolor.rgb = normal;
            break;
            case 3:
            outFragcolor.rgb = albedo.rgb;
            break;
            case 4:
            outFragcolor.rgb = albedo.aaa;
            break;
        }
        outFragcolor.a = 1.0;
        return;
    }*/

        // Render-target composition

        #define lightCount 1
        #define ambient 0.0

    // Ambient part
    vec3 fragcolor  = albedo.rgb * ambient;

    for(int i = 0; i < lightCount; ++i)
    {
        // Vector to light
        vec3 L = lights[i].position.xyz - fragPos;
        //vec3 L = -lights[i].position.xyz;
        // Distance from light to fragment position
        float dist = length(L);

        // Viewer to fragment
        vec3 V = pc.viewPos.xyz - fragPos;
        V = normalize(V);

        //if(dist < ubo.lights[i].radius)
        {
            // Light to fragment
            L = normalize(L);

            // Attenuation
            float atten = lights[i].radius / (pow(dist, 2.0) + 1.0);

            // Diffuse part
            vec3 N = normalize(normal);
            float NdotL = max(0.0, dot(N, L));
            vec3 diff = lights[i].color * albedo.rgb * NdotL * atten;

            // Specular part
            // Specular map values are stored in alpha of albedo mrt
            vec3 R = reflect(-L, N);
            float NdotR = max(0.0, dot(R, V));
            vec3 spec = lights[i].color * albedo.a * pow(NdotR, 16.0) * atten;

            fragcolor += diff;// + spec;
        }
    }

    outFragcolor = vec4(fragcolor, 1.0);
}