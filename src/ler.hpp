//
// Created by loulfy on 14/12/2022.
//

#ifndef LER_HPP
#define LER_HPP

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <rtxmu/VkAccelStructManager.h>
#include <taskflow/taskflow.hpp>

#include <set>
#include <map>
#include <span>
#include <list>
#include <mutex>
#include <memory>
#include <limits>
#include <fstream>
#include <iostream>
#include <functional>
#include <filesystem>
namespace fs = std::filesystem;

namespace ler
{
    struct Buffer
    {
        vk::Buffer handle;
        vk::BufferCreateInfo info;
        vma::Allocation allocation;

        [[nodiscard]] uint32_t length() const { return info.size; }
    };

    using BufferPtr = std::shared_ptr<Buffer>;

    struct Texture
    {
        vk::Image handle;
        vk::ImageCreateInfo info;
        vk::UniqueImageView view;
        vma::Allocation allocation;
    };

    using TexturePtr = std::shared_ptr<Texture>;

    struct SwapChain
    {
        vk::UniqueSwapchainKHR handle;
        vk::Format format = vk::Format::eB8G8R8A8Unorm;
        vk::Extent2D extent;
    };

    struct RenderPass
    {
        vk::UniqueRenderPass handle;
        std::vector<vk::AttachmentDescription2> attachments;
        std::array<std::set<uint32_t>, 4> subPass;
    };

    struct FrameBuffer
    {
        vk::UniqueFramebuffer handle;
        std::vector<TexturePtr> images;
    };

    struct DescriptorSetLayoutData
    {
        uint32_t set_number = 0;
        VkDescriptorSetLayoutCreateInfo create_info;
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
    };

    struct Shader
    {
        vk::UniqueShaderModule shaderModule;
        vk::ShaderStageFlagBits stageFlagBits = {};
        vk::PipelineVertexInputStateCreateInfo pvi;
        std::vector<vk::PushConstantRange> pushConstants;
        std::map<uint32_t, DescriptorSetLayoutData> descriptorMap;
        std::vector<vk::VertexInputBindingDescription> bindingDesc;
        std::vector<vk::VertexInputAttributeDescription> attributeDesc;
    };

    using ShaderPtr = std::shared_ptr<Shader>;

    struct DescriptorAllocator
    {
        std::vector<vk::DescriptorSetLayoutBinding> layoutBinding;
        vk::UniqueDescriptorSetLayout layout;
        vk::UniqueDescriptorPool pool;
    };

    struct PipelineInfo
    {
        vk::Extent2D extent;
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
        vk::PolygonMode polygonMode = vk::PolygonMode::eFill;
        vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e1;
        uint32_t textureCount = 1;
        bool writeDepth = true;
        uint32_t subPass = 0;
    };

    class BasePipeline
    {
    public:

        void reflectPipelineLayout(vk::Device device, const std::vector<ShaderPtr>& shaders);
        vk::DescriptorSet createDescriptorSet(vk::Device& device, uint32_t set);

        vk::UniquePipeline handle;
        vk::UniquePipelineLayout pipelineLayout;
        vk::PipelineBindPoint bindPoint = vk::PipelineBindPoint::eGraphics;
        std::unordered_map<uint32_t,DescriptorAllocator> descriptorAllocMap;
    };

    using PipelinePtr = std::shared_ptr<BasePipeline>;

    class GraphicsPipeline : public BasePipeline
    {

    };

    class ComputePipeline : public BasePipeline
    {

    };

    struct Instance
    {
        alignas(16) glm::mat4 model = glm::mat4(1.f);
        alignas(16) glm::vec3 bMin = glm::vec3(0.f);
        alignas(16) glm::vec3 bMax = glm::vec3(0.f);
        alignas(4) glm::uint matId = 0;
    };

    struct Material
    {
        alignas(4) glm::uint texId = 0;
        alignas(4) glm::uint norId = 0;
        alignas(16) glm::vec3 color = glm::vec3(1.f);
    };

    struct MeshIndirect
    {
        uint32_t countIndex = 0;
        uint32_t firstIndex = 0;
        uint32_t countVertex = 0;
        uint32_t firstVertex = 0;
        uint32_t materialId = 0;
        glm::vec3 bMin = glm::vec3(0.f);
        glm::vec3 bMax = glm::vec3(0.f);
    };

    struct DeferredConstant
    {
        alignas(16) glm::vec3 viewPos = glm::vec3(1.f);
        alignas(4) glm::uint viewMode = 0;
        alignas(4) glm::uint lightCount = 1;
        alignas(4) glm::uint shadowMode = 1;
    };

    struct SceneConstant
    {
        glm::mat4 proj = glm::mat4(1.f);
        glm::mat4 view = glm::mat4(1.f);
    };

    struct Frustum
    {
        alignas(16) glm::vec4 planes[6];
        alignas(16) glm::vec4 corners[8];
        alignas(4) glm::uint num = 0;
    };

    struct Light
    {
        alignas(16) glm::vec3 pos = glm::vec3(1.f);
        alignas(16) glm::vec3 color = glm::vec3(1.f);
        alignas(4) float radius = 10.f;
    };

    struct Scene
    {
        Buffer staging;
        Buffer indexBuffer;
        Buffer vertexBuffer;
        Buffer normalBuffer;
        Buffer tangentBuffer;
        Buffer texcoordBuffer;
        Buffer indirectBuffer;
        Buffer instanceBuffer;
        Buffer materialBuffer;
        Buffer transformBuffer;
        Buffer tlasBuffer;

        Buffer aabbBuffer;

        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t drawCount = 0;
        uint32_t lineCount = 0;
        uint32_t matCount = 0;

        std::vector<uint32_t> mapping;
        std::vector<TexturePtr> textures;
        std::vector<Instance> instances;
        std::vector<Material> materials;
        std::vector<MeshIndirect> geometries;
        std::vector<vk::DrawIndexedIndirectCommand> commands;
    };

    struct LerSettings
    {
        vk::Instance instance;
        vk::PhysicalDevice physicalDevice;
        vk::Device device;
        uint32_t graphicsQueueFamily = UINT32_MAX;
        uint32_t transferQueueFamily = UINT32_MAX;
        vk::PipelineCache pipelineCache;
    };

    struct TrackedCmd
    {
        vk::UniqueCommandPool pool;
        vk::CommandBuffer cmd;
    };

    using TrackedCmdPtr = std::shared_ptr<TrackedCmd>;

    class LerContext
    {
    public:

        ~LerContext();
        explicit LerContext(const LerSettings& settings);

        // Buffer
        void destroyBuffer(Buffer& buffer);
        void getFromBuffer(Buffer& buffer, uint32_t* ptr);
        Buffer createBuffer(uint32_t byteSize, vk::BufferUsageFlags usages = vk::BufferUsageFlagBits(), bool staging = false);
        Buffer createBufferWithAlign(uint32_t byteSize, uint64_t minAlign);
        void uploadBuffer(Buffer& staging, const void* src, uint32_t byteSize);
        void copyBuffer(Buffer& staging, Buffer& dst, uint64_t byteSize = VK_WHOLE_SIZE);
        static void copyBufferToTexture(vk::CommandBuffer& cmd, const Buffer& buffer, const TexturePtr& texture);

        // Texture
        TexturePtr createTexture(vk::Format format, const vk::Extent2D& extent, vk::SampleCountFlagBits sampleCount, bool isRenderTarget = false);
        TexturePtr createTextureFromNative(vk::Image image, vk::Format format, const vk::Extent2D& extent);
        vk::UniqueSampler createSampler(const vk::SamplerAddressMode& addressMode, bool filter);
        TexturePtr loadTextureFromFile(const fs::path& path);
        TexturePtr loadTextureFromFileAsync(const fs::path& path);
        TexturePtr loadTextureFromMemory(const unsigned char* buffer, uint32_t size);
        static vk::ImageAspectFlags guessImageAspectFlags(vk::Format format);

        SwapChain createSwapChain(vk::SurfaceKHR surface, uint32_t width, uint32_t height, bool vSync = true);

        // RenderPass
        RenderPass createDefaultRenderPass(vk::Format surfaceFormat);
        std::vector<FrameBuffer> createFrameBuffers(const RenderPass& renderPass, const SwapChain& swapChain);

        // Pipeline
        ShaderPtr createShader(const fs::path& path);
        PipelinePtr createGraphicsPipeline(const RenderPass& renderPass, const std::vector<ShaderPtr>& shaders, const PipelineInfo& info);
        PipelinePtr createComputePipeline(const ShaderPtr& shader);
        void updateSampler(vk::DescriptorSet descriptorSet, uint32_t binding, vk::Sampler& sampler, const std::vector<TexturePtr>& textures);
        void updateStorage(vk::DescriptorSet descriptorSet, uint32_t binding, const Buffer& buffer, uint64_t byteSize, bool uniform = false);
        void updateAttachment(vk::DescriptorSet descriptorSet, uint32_t binding, const TexturePtr& texture);
        void updateAccelerationStructure(vk::DescriptorSet descriptorSet, uint32_t binding, vk::AccelerationStructureKHR tlas);

        // Scene
        Scene fromFile(const fs::path& path);
        void destroyScene(const Scene& scene);
        static void transformBoundingBox(const glm::mat4& t, glm::vec3& min, glm::vec3& max);
        static void getFrustumPlanes(glm::mat4 mvp, glm::vec4* planes);
        static void getFrustumCorners(glm::mat4 mvp, glm::vec4* points);

        // Execution
        vk::CommandBuffer getCommandBuffer();
        void submitAndWait(vk::CommandBuffer& cmd);
        TrackedCmdPtr getCommandTracked();
        void submitTracked(TrackedCmdPtr& tracked);

        // Ray tracing
        void sceneToBlas(const Scene& scene, std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& ranges, std::vector<vk::AccelerationStructureGeometryKHR>& geometries);
        void sceneToTlas(Scene& scene, vk::AccelerationStructureKHR blas, std::vector<vk::AccelerationStructureGeometryKHR>& geometries);
        vk::AccelerationStructureKHR convertSceneToTLAS(Scene& scene);

    private:

        void populateTexture(const TexturePtr& texture, vk::Format format, const vk::Extent2D& extent, vk::SampleCountFlagBits sampleCount, bool isRenderTarget = false);
        void mergeSceneBuffer(Scene& scene, Buffer& dest, const aiScene* aiScene, const std::function<bool(aiMesh*)>& predicate, const std::function<void*(aiMesh*)>& provider);
        static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes, bool vSync);
        static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
        static vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height);
        static std::vector<char> loadBinaryFromFile(const fs::path& path);
        static uint32_t formatSize(VkFormat format);
        vk::Format chooseDepthFormat();
        uint32_t loadTexture(Scene& scene, const aiScene* aiScene, const aiString& filename, const fs::path& path);

        LerSettings m_settings;
        std::mutex m_mutexCmd;
        std::mutex m_mutexQueue;
        tf::Executor m_executor;
        vk::PhysicalDevice m_physicalDevice;
        vk::Device m_device;
        vk::Queue m_queue;
        vk::Queue m_transfer;
        vma::Allocator m_allocator;
        vk::PipelineCache m_pipelineCache;
        vk::UniqueCommandPool m_commandPool;
        std::list<vk::CommandBuffer> m_commandBuffersPool;
        std::list<TrackedCmdPtr> m_commandTracker;

        std::vector<TexturePtr> m_textures;
        std::map<std::string, uint64_t> m_cache;
        std::vector<uint64_t> m_asIds;
        std::unique_ptr<rtxmu::VkAccelStructManager> m_rtxMemUtil;
    };
}

#endif //LER_HPP
