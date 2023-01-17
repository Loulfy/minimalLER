#version 460
#define ambient 0.f

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInputMS samplerPosition;
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS samplerNormal;
layout (input_attachment_index = 2, set = 0, binding = 2) uniform subpassInputMS samplerAlbedo;

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outFragcolor;

struct Light
{
    vec3 position;
    vec3 color;
    float radius;
};

layout (push_constant) uniform constants
{
    vec3 viewPos;
    uint viewMode;
    uint lightCount;
    uint shadowMode;
} pc;

layout (set = 1, binding = 0) uniform UBO
{
    Light lights[6];
} ubo;

// Debug
const Light lights[1] = Light[](
Light(vec3(0.f, 1.f, 0.3f), vec3(1,1,1), 20) // -0.2f, -1.0f, -0.3f // 0.f, 15.f, 0.3f, 1.f
);

layout (set = 2, binding = 0) uniform accelerationStructureEXT topLevelAS;

/*vec3 uniformSampleSphere(uint seed)
{
    float r1 = rnd(seed);
    float r2 = rnd(seed);

    const float kPI = 3.14159265f;
    float z = 1.0 - 2.0 * r1;
    float r = sqrt(max(0.f, 1.0 - z * z));
    float phi = 2.0 * kPI * r2;

    float x = r * cos(phi);
    float y = r * sin(phi);

    return vec3(x, y, z);
}*/

void main()
{
    // Get G-Buffer values
    vec3 fragPos = subpassLoad(samplerPosition, gl_SampleID).rgb;
    vec3 normal = subpassLoad(samplerNormal, gl_SampleID).rgb;
    vec4 albedo = subpassLoad(samplerAlbedo, gl_SampleID);

    if (pc.viewMode > 0) {
        switch (pc.viewMode) {
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
    }

    // Ambient part
    vec3 fragcolor  = albedo.rgb * ambient;

    for(int i = 0; i < pc.lightCount; ++i)
    {
        // Vector to light
        vec3 L = ubo.lights[i].position.xyz - fragPos;
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
            float atten = ubo.lights[i].radius / (pow(dist, 2.0) + 1.0);

            // Diffuse part
            vec3 N = normalize(normal);
            float NdotL = max(0.0, dot(N, L));
            vec3 diff = ubo.lights[i].color * albedo.rgb * NdotL * atten;

            // Specular part
            // Specular map values are stored in alpha of albedo mrt
            vec3 R = reflect(-L, N);
            float NdotR = max(0.0, dot(R, V));
            vec3 spec = ubo.lights[i].color * albedo.a * pow(NdotR, 16.0) * atten;

            fragcolor += diff;// + spec;
        }

        if(pc.shadowMode == 0)
            continue;

        vec3 rayDir = normalize(ubo.lights[i].position - fragPos);
        float distToLight = length(ubo.lights[i].position - fragPos);

        // Ray Query for shadow
        vec3  origin    = fragPos;
        vec3  direction = rayDir;  // vector to light
        float tMin      = 0.01f;
        float tMax      = distToLight;

        // Initializes a ray query object but does not start traversal
        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, tMin, direction, tMax);

        // Start traversal: return false if traversal is complete
        while(rayQueryProceedEXT(rayQuery))
        {
        }

        // Returns type of committed (true) intersection
        if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        {
            // Got an intersection == Shadow
            fragcolor *= 0.1;
        }
    }

    outFragcolor = vec4(fragcolor, 1.0);
}