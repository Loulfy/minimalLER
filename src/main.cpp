#include "ler.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define GLFW_INCLUDE_NONE // Do not include any OpenGL/Vulkan headers
#include <GLFW/glfw3.h>

#include "camera.h"

#define KickstartRT_Graphics_API_Vulkan
//#include <KickstartRT.h>
//namespace KS = KickstartRT::VK;

static const fs::path ASSETS = fs::path(PROJECT_DIR) / "assets";
static const uint32_t WIDTH = 1280;
static const uint32_t HEIGHT = 720;

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* manager = reinterpret_cast<Camera*>(glfwGetWindowUserPointer(window));
    if(action == GLFW_PRESS || action == GLFW_REPEAT)
        manager->keyboardCallback(key, action, 0.002);
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main()
{
    if (!glfwInit())
        throw std::runtime_error("failed to init glfw");

    // Create window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "MinimalRT", nullptr, nullptr);
    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instanceExtensions(extensions, extensions + count);

    static const vk::DynamicLoader dl;
    const auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    std::initializer_list<const char*> layers = {
        "VK_LAYER_KHRONOS_validation"
    };

    std::initializer_list<const char*> devices = {
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        //VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        //VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        //VK_EXT_DEBUG_MARKER_EXTENSION_NAME
    };

    // Create instance
    vk::ApplicationInfo appInfo;
    appInfo.setApiVersion(VK_API_VERSION_1_3);
    appInfo.setPEngineName("minimalKS");

    vk::InstanceCreateInfo instInfo;
    instInfo.setPApplicationInfo(&appInfo);
    instInfo.setPEnabledLayerNames(layers);
    instInfo.setPEnabledExtensionNames(instanceExtensions);
    auto instance = vk::createInstanceUnique(instInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

    // Pick First GPU
    auto physicalDevice = instance->enumeratePhysicalDevices().front();
    auto test = instance->enumeratePhysicalDevices();
    for(auto& phy : test)
        std::cout << phy.getProperties().deviceName << std::endl;

    // Device Features
    auto features = physicalDevice.getFeatures();

    // Find Graphics Queue
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    const auto family = std::find_if(queueFamilies.begin(), queueFamilies.end(), [](const vk::QueueFamilyProperties& queueFamily) {
         return queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics;
    });

    uint32_t graphicsQueueFamily = std::distance(queueFamilies.begin(), family);

    // Create queues
    float queuePriority = 1.0f;
    std::initializer_list<vk::DeviceQueueCreateInfo> queueCreateInfos = {
        { {}, graphicsQueueFamily, 1, &queuePriority }
    };

    // Create Device
    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(queueCreateInfos);
    deviceInfo.setPEnabledExtensionNames(devices);
    deviceInfo.setPEnabledLayerNames(layers);
    deviceInfo.setPEnabledFeatures(&features);

    vk::PhysicalDeviceVulkan11Features vulkan11Features;
    vulkan11Features.setShaderDrawParameters(true);
    vk::PhysicalDeviceVulkan12Features vulkan12Features;

    vulkan12Features.setDescriptorIndexing(true);
    vulkan12Features.setRuntimeDescriptorArray(true);
    vulkan12Features.setDescriptorBindingPartiallyBound(true);
    vulkan12Features.setDescriptorBindingVariableDescriptorCount(true);
    vulkan12Features.setTimelineSemaphore(true);
    vulkan12Features.setBufferDeviceAddress(true);
    vulkan12Features.setShaderSampledImageArrayNonUniformIndexing(true);

    vulkan12Features.setBufferDeviceAddress(true);
    vulkan12Features.setRuntimeDescriptorArray(true);
    vulkan12Features.setDescriptorBindingVariableDescriptorCount(true);
    vulkan12Features.setShaderSampledImageArrayNonUniformIndexing(true);
    vk::StructureChain<vk::DeviceCreateInfo,
    /*vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,*/
    vk::PhysicalDeviceVulkan11Features,
    vk::PhysicalDeviceVulkan12Features> createInfoChain(deviceInfo, /*{false}, {false},*/ vulkan11Features, vulkan12Features);
    auto device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

    auto queue = device->getQueue(graphicsQueueFamily, 0);

    // Create surface
    VkSurfaceKHR glfwSurface;
    auto res = glfwCreateWindowSurface(instance.get(), window, nullptr, &glfwSurface);
    if (res != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface!");

    vk::UniqueSurfaceKHR surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(glfwSurface), { instance.get() });

    vk::UniquePipelineCache pipelineCache = device->createPipelineCacheUnique({});

    ler::LerSettings config;
    config.device = device.get();
    config.instance = instance.get();
    config.physicalDevice = physicalDevice;
    config.pipelineCache = pipelineCache.get();
    config.graphicsQueueFamily = graphicsQueueFamily;
    ler::LerContext engine(config);

    auto scene = engine.fromFile("C:/Users/loulfy/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");
    //auto scene = engine.fromFile(ASSETS / "Lantern.glb");

    uint32_t swapChainIndex = 0;
    auto swapChain = engine.createSwapChain(glfwSurface, WIDTH, HEIGHT);
    auto renderPass = engine.createDefaultRenderPass(swapChain.format);
    auto frameBuffers = engine.createFrameBuffers(renderPass, swapChain);

    vk::Viewport viewport(0, 0, static_cast<float>(swapChain.extent.width), static_cast<float>(swapChain.extent.height), 0, 1.0f);
    vk::Rect2D renderArea(vk::Offset2D(), swapChain.extent);

    // FIRST PASS (Fill GBuffer)
    std::vector<ler::ShaderPtr> gbufferShaders;
    gbufferShaders.push_back(engine.createShader(ASSETS / "shaders" / "gbuffer.vert.spv"));
    gbufferShaders.push_back(engine.createShader(ASSETS / "shaders" / "gbuffer.frag.spv"));

    ler::PipelineInfo info;
    info.subPass = 0;
    info.extent = swapChain.extent;
    info.textureCount = scene.textures.size();
    info.sampleCount = vk::SampleCountFlagBits::e8;
    info.topology = vk::PrimitiveTopology::eTriangleList;
    auto gbuffer = engine.createGraphicsPipeline(renderPass, gbufferShaders, info);

    vk::UniqueSampler sampler = engine.createSampler(vk::SamplerAddressMode::eRepeat, true);

    std::array<vk::DescriptorSet, 2> gbufferDescriptorSet;
    gbufferDescriptorSet[0] = gbuffer->createDescriptorSet(device.get(), 0);
    engine.updateStorage(gbufferDescriptorSet[0], 0, scene.instanceBuffer, scene.drawCount * sizeof(ler::Instance));
    gbufferDescriptorSet[1] = gbuffer->createDescriptorSet(device.get(), 1);
    engine.updateStorage(gbufferDescriptorSet[1], 0, scene.materialBuffer, scene.matCount * sizeof(ler::Material));
    engine.updateSampler(gbufferDescriptorSet[1], 1, sampler.get(), scene.textures);

    // SECOND PASS (Composition)
    std::vector<ler::ShaderPtr> deferredShaders;
    deferredShaders.push_back(engine.createShader(ASSETS / "shaders" / "quad.vert.spv"));
    deferredShaders.push_back(engine.createShader(ASSETS / "shaders" / "deferred.frag.spv"));

    info.subPass = 1;
    info.textureCount = 0;
    info.writeDepth = false;
    info.topology = vk::PrimitiveTopology::eTriangleStrip;
    auto deferred = engine.createGraphicsPipeline(renderPass, deferredShaders, info);

    std::array<vk::DescriptorSet, 2> inputColor;
    for(size_t i = 0; i < inputColor.size(); i++)
    {
        inputColor[i] = deferred->createDescriptorSet(device.get(), 0);
        engine.updateAttachment(inputColor[i], 0, frameBuffers[i].images[0]);
        engine.updateAttachment(inputColor[i], 1, frameBuffers[i].images[1]);
        engine.updateAttachment(inputColor[i], 2, frameBuffers[i].images[2]);
    }

    // THIRD PASS (Bounding Box)
    std::vector<ler::ShaderPtr> aabbShaders;
    aabbShaders.push_back(engine.createShader(ASSETS / "shaders" / "aabb.vert.spv"));
    aabbShaders.push_back(engine.createShader(ASSETS / "shaders" / "aabb.frag.spv"));

    info.subPass = 1;
    info.textureCount = 0;
    info.topology = vk::PrimitiveTopology::eLineList;
    auto quad = engine.createGraphicsPipeline(renderPass, aabbShaders, info);

    std::array<float, 4> color = {1.f, 1.f, 1.f, 1.f};
    std::vector<vk::ClearValue> clearValues;
    for(const auto& attachment : renderPass.attachments)
    {
        auto aspect = ler::LerContext::guessImageAspectFlags(attachment.format);
        if(aspect == vk::ImageAspectFlagBits::eColor)
            clearValues.emplace_back(vk::ClearColorValue(color));
        else
            clearValues.emplace_back(vk::ClearDepthStencilValue(1.0f, 0));
    }

    auto presentSemaphore = device->createSemaphoreUnique({});

    Camera camera;
    glfwSetWindowUserPointer(window, &camera);
    double xpos, ypos;
    float deltaTime = 0.f;
    float lastFrame = 0.f;
    ler::SceneConstant constant;
    constant.proj = glm::perspective(glm::radians(55.f), 1920.f / 1080.f, 0.01f, 10000.0f); // 0, 50, -50
    constant.view = camera.getViewMatrix();
    constant.proj[1][1] *= -1;

    vk::Result result;
    vk::CommandBuffer cmd;
    vk::DeviceSize offset = 0;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        auto currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwGetCursorPos(window, &xpos, &ypos);
        camera.mouseCallback(xpos, ypos);
        constant.view = camera.getViewMatrix();

        result = device->acquireNextImageKHR(swapChain.handle.get(), std::numeric_limits<uint64_t>::max(), presentSemaphore.get(), vk::Fence(), &swapChainIndex);
        assert(result == vk::Result::eSuccess);

        cmd = engine.getCommandBuffer();
        vk::RenderPassBeginInfo beginInfo;
        beginInfo.setRenderPass(renderPass.handle.get());
        beginInfo.setFramebuffer(frameBuffers[swapChainIndex].handle.get());
        beginInfo.setRenderArea(renderArea);
        beginInfo.setClearValues(clearValues);
        cmd.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
        cmd.pushConstants(quad->pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(ler::SceneConstant), &constant);
        cmd.setScissor(0, 1, &renderArea);
        cmd.setViewport(0, 1, &viewport);

        cmd.bindPipeline(gbuffer->bindPoint, gbuffer->handle.get());
        cmd.bindDescriptorSets(gbuffer->bindPoint, gbuffer->pipelineLayout.get(), 0, gbufferDescriptorSet, nullptr);
        cmd.bindIndexBuffer(scene.indexBuffer.handle, offset, vk::IndexType::eUint32);
        cmd.bindVertexBuffers(0, 1, &scene.vertexBuffer.handle, &offset);
        cmd.bindVertexBuffers(1, 1, &scene.texcoordBuffer.handle, &offset);
        cmd.bindVertexBuffers(2, 1, &scene.normalBuffer.handle, &offset);
        cmd.bindVertexBuffers(3, 1, &scene.tangentBuffer.handle, &offset);
        cmd.drawIndexedIndirect(scene.indirectBuffer.handle, offset, scene.drawCount, sizeof(vk::DrawIndexedIndirectCommand));

        cmd.nextSubpass(vk::SubpassContents::eInline);
        cmd.bindPipeline(deferred->bindPoint, deferred->handle.get());
        cmd.bindDescriptorSets(deferred->bindPoint, deferred->pipelineLayout.get(), 0, inputColor[swapChainIndex], nullptr);
        cmd.draw(4, 1, 0, 0);
        /*cmd.bindPipeline(quad->bindPoint, quad->handle.get());
        cmd.bindVertexBuffers(0, 1, &scene.aabbBuffer.handle, &offset);
        cmd.draw(scene.lineCount, 1, 0, 0);*/

        cmd.endRenderPass();
        engine.submitAndWait(cmd);

        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphoreCount(1);
        presentInfo.setPWaitSemaphores(&presentSemaphore.get());
        presentInfo.setSwapchainCount(1);
        presentInfo.setPSwapchains(&swapChain.handle.get());
        presentInfo.setPImageIndices(&swapChainIndex);

        result = queue.presentKHR(&presentInfo);
        assert(result == vk::Result::eSuccess);

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    device->waitIdle();
    engine.destroyScene(scene);

    // Clean
    //context->DestroyAllInstanceHandles();
    //context->DestroyAllGeometryHandles();
    //KS::ExecuteContext::Destruct(context);

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}