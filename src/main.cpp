#include "ler.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define GLFW_INCLUDE_NONE // Do not include any OpenGL/Vulkan headers
#include <GLFW/glfw3.h>

#define KickstartRT_Graphics_API_Vulkan
//#include <KickstartRT.h>
//namespace KS = KickstartRT::VK;

static const fs::path ASSETS = fs::path(PROJECT_DIR) / "assets";
static const uint32_t WIDTH = 1280;
static const uint32_t HEIGHT = 720;

int main()
{
    if (!glfwInit())
        throw std::runtime_error("failed to init glfw");

    // Create window
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "MinimalRT", nullptr, nullptr);

    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> instanceExtensions(extensions, extensions + count);

    static const vk::DynamicLoader dl;
    const auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

    std::initializer_list<const char*> layers = {
        "VK_LAYER_KHRONOS_validation"
    };

    std::initializer_list<const char*> devices = {
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
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

    vk::PhysicalDeviceVulkan12Features vulkan12Features;
    vulkan12Features.setBufferDeviceAddress(true);
    vulkan12Features.setDescriptorBindingVariableDescriptorCount(true);
    vk::StructureChain<vk::DeviceCreateInfo,
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    vk::PhysicalDeviceVulkan12Features> createInfoChain(deviceInfo, {true}, {true}, vulkan12Features);
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

    //auto scene = engine.fromFile("C:/Users/loulfy/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");
    //auto scene = engine.fromFile(ASSETS / "Lantern.glb");

    uint32_t swapChainIndex = 0;
    auto swapChain = engine.createSwapChain(glfwSurface, WIDTH, HEIGHT);
    auto renderPass = engine.createDefaultRenderPass(swapChain.format);
    auto frameBuffers = engine.createFrameBuffers(renderPass, swapChain);

    vk::Viewport viewport(0, 0, static_cast<float>(swapChain.extent.width), static_cast<float>(swapChain.extent.height), 0, 1.0f);
    vk::Rect2D renderArea(vk::Offset2D(), swapChain.extent);

    std::vector<ler::ShaderPtr> shaders;
    shaders.push_back(engine.createShader(ASSETS / "shaders" / "quad.vert.spv"));
    shaders.push_back(engine.createShader(ASSETS / "shaders" / "quad.frag.spv"));

    ler::PipelineInfo info;
    info.subPass = 1;
    info.extent = swapChain.extent;
    info.sampleCount = vk::SampleCountFlagBits::e8;
    info.topology = vk::PrimitiveTopology::eTriangleStrip;
    auto quad = engine.createGraphicsPipeline(renderPass, shaders, info);

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

    vk::Result result;
    vk::CommandBuffer cmd;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        result = device->acquireNextImageKHR(swapChain.handle.get(), std::numeric_limits<uint64_t>::max(), presentSemaphore.get(), vk::Fence(), &swapChainIndex);
        assert(result == vk::Result::eSuccess);

        cmd = engine.getCommandBuffer();
        vk::RenderPassBeginInfo beginInfo;
        beginInfo.setRenderPass(renderPass.handle.get());
        beginInfo.setFramebuffer(frameBuffers[swapChainIndex].get());
        beginInfo.setRenderArea(renderArea);
        beginInfo.setClearValues(clearValues);
        cmd.beginRenderPass(beginInfo, vk::SubpassContents::eInline);
        cmd.setScissor(0, 1, &renderArea);
        cmd.setViewport(0, 1, &viewport);
        cmd.nextSubpass(vk::SubpassContents::eInline);

        cmd.bindPipeline(quad->bindPoint, quad->handle.get());
        cmd.draw(4, 1, 0, 0);

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

    // Clean
    //context->DestroyAllInstanceHandles();
    //context->DestroyAllGeometryHandles();
    //KS::ExecuteContext::Destruct(context);

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}