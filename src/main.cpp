#include "ler.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#define GLFW_INCLUDE_NONE // Do not include any OpenGL/Vulkan headers
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>
#include <ImGuizmo.h>

#include "camera.h"

static const fs::path ASSETS = fs::path(PROJECT_DIR) / "assets";
static const uint32_t WIDTH = 1280;
static const uint32_t HEIGHT = 720;

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto* camera = reinterpret_cast<Camera*>(glfwGetWindowUserPointer(window));
    if(action == GLFW_PRESS || action == GLFW_REPEAT)
        camera->keyboardCallback(key, action, 0.002);
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        if(glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            camera->lockMouse = true;
        }
        else
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            camera->lockMouse = false;
        }
    }
}

template<typename ContainerType>
void swapAndPop(ContainerType & container, size_t index)
{
    // ensure that we're not attempting to access out of the bounds of the container.
    assert(index < container.size());

    //Swap the element with the back element, except in the case when we're the last element.
    if (index + 1 != container.size())
        std::swap(container[index], container.back());

    //Pop the back of the container, deleting our old element.
    container.pop_back();
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
    float queuePriority[] = {1.0f, 0.5f};
    std::initializer_list<vk::DeviceQueueCreateInfo> queueCreateInfos = {
        { {}, graphicsQueueFamily, 2, queuePriority }
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

    auto scene = engine.fromFile("C:/Users/loria/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf");
    //auto scene = engine.fromFile(ASSETS / "Lantern.glb");

    std::vector<ler::Light> lighting;
    lighting.push_back({.pos = glm::vec4(0.f, 1.f, 0.3f, 0.f)});

    vk::CommandBuffer cmd;
    uint32_t swapChainIndex = 0;
    auto swapChain = engine.createSwapChain(glfwSurface, WIDTH, HEIGHT);
    auto renderPass = engine.createDefaultRenderPass(swapChain.format);
    auto frameBuffers = engine.createFrameBuffers(renderPass, swapChain);

    vk::Viewport viewport(0, 0, static_cast<float>(swapChain.extent.width), static_cast<float>(swapChain.extent.height), 0, 1.0f);
    vk::Rect2D renderArea(vk::Offset2D(), swapChain.extent);

    std::array<vk::DescriptorPoolSize, 11> pool_sizes =
    {{
         {vk::DescriptorType::eSampler, 1000},
         {vk::DescriptorType::eCombinedImageSampler, 1000},
         {vk::DescriptorType::eSampledImage, 1000},
         {vk::DescriptorType::eStorageImage, 1000},
         {vk::DescriptorType::eUniformTexelBuffer, 1000},
         {vk::DescriptorType::eStorageTexelBuffer, 1000},
         {vk::DescriptorType::eUniformBuffer, 1000},
         {vk::DescriptorType::eStorageBuffer, 1000},
         {vk::DescriptorType::eUniformBufferDynamic, 1000},
         {vk::DescriptorType::eStorageBufferDynamic, 1000},
         {vk::DescriptorType::eInputAttachment, 1000}
     }};

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    poolInfo.setPoolSizes(pool_sizes);
    poolInfo.setMaxSets(1000);

    auto imguiPool = device->createDescriptorPoolUnique(poolInfo);

    int w, h;
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    glfwGetFramebufferSize(window, &w, &h);
    io.DisplaySize = ImVec2(static_cast<float>(w), static_cast<float>(h));
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    //this initializes imgui for Vulkan
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance.get();
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device.get();
    init_info.Queue = queue;
    init_info.DescriptorPool = imguiPool.get();
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.Subpass = 2;
    init_info.MSAASamples = VK_SAMPLE_COUNT_8_BIT;
    ImGui_ImplVulkan_Init(&init_info, renderPass.handle.get());

    cmd = engine.getCommandBuffer();
    ImGui_ImplVulkan_CreateFontsTexture(cmd);
    engine.submitAndWait(cmd);

    // FRUSTUM CULLING
    auto cShader = engine.createShader(ASSETS / "shaders" / "triangle_cull.comp.spv");
    auto culling = engine.createComputePipeline(cShader);
    auto visibleBuffer = engine.createBuffer(256, vk::BufferUsageFlagBits::eStorageBuffer, true);
    auto frustumBuffer = engine.createBuffer(256, vk::BufferUsageFlagBits::eUniformBuffer, true);
    std::array<vk::DescriptorSet,2> cullDescriptorSet;
    for(size_t i = 0; i < cullDescriptorSet.size(); ++i)
        cullDescriptorSet[i] = culling->createDescriptorSet(device.get(), i);

    engine.updateStorage(cullDescriptorSet[0], 0, frustumBuffer, 256, true);
    engine.updateStorage(cullDescriptorSet[1], 0, scene.instanceBuffer, scene.drawCount * sizeof(ler::Instance));
    engine.updateStorage(cullDescriptorSet[1], 1, scene.indirectBuffer, scene.drawCount * sizeof(vk::DrawIndexedIndirectCommand));
    engine.updateStorage(cullDescriptorSet[1], 2, visibleBuffer, 256);

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

    auto lightBuffer = engine.createBuffer(256, vk::BufferUsageFlagBits::eUniformBuffer, true);
    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());

    vk::DescriptorSet inputLight;
    inputLight = deferred->createDescriptorSet(device.get(), 1);
    engine.updateStorage(inputLight, 0, lightBuffer, 256, true);

    // THIRD PASS (Bounding Box)
    std::vector<ler::ShaderPtr> aabbShaders;
    aabbShaders.push_back(engine.createShader(ASSETS / "shaders" / "aabb.vert.spv"));
    aabbShaders.push_back(engine.createShader(ASSETS / "shaders" / "aabb.frag.spv"));

    info.subPass = 2;
    info.textureCount = 0;
    info.topology = vk::PrimitiveTopology::eLineList;
    auto quad = engine.createGraphicsPipeline(renderPass, aabbShaders, info);


    // PREPARE RenderPass
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
    ler::SceneConstant constant;
    constant.proj = glm::perspective(glm::radians(55.f), 1920.f / 1080.f, 0.01f, 10000.0f);
    constant.view = camera.getViewMatrix();
    constant.proj[1][1] *= -1;

    ler::Frustum frustum;
    ler::DeferredConstant defConstant;

    std::array<const char*,4> items = {"Deferred", "Position", "Normal", "Albedo"};
    static const char* current_item = items[0];

    bool showAABB = false;
    vk::Result result;
    uint32_t resetNum = 0;
    vk::DeviceSize offset = 0;
    uint32_t numVisibleMeshes = 0;
    int node_clicked = -1;
    glm::mat4 trans = glm::mat4(1.f);
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glfwGetCursorPos(window, &xpos, &ypos);
        camera.mouseCallback(xpos, ypos);
        constant.view = camera.getViewMatrix();
        defConstant.viewPos = camera.position;
        defConstant.lightCount = lighting.size();

        result = device->acquireNextImageKHR(swapChain.handle.get(), std::numeric_limits<uint64_t>::max(), presentSemaphore.get(), vk::Fence(), &swapChainIndex);
        assert(result == vk::Result::eSuccess);

        cmd = engine.getCommandBuffer();

        // Frustum Culling
        engine.getFromBuffer(visibleBuffer, &numVisibleMeshes);
        frustum.num = scene.instances.size();
        ler::LerContext::getFrustumPlanes(constant.proj * constant.view, frustum.planes);
        ler::LerContext::getFrustumCorners(constant.proj * constant.view, frustum.corners);
        engine.uploadBuffer(visibleBuffer, &resetNum, sizeof(uint32_t));
        engine.uploadBuffer(frustumBuffer, &frustum, sizeof(ler::Frustum));

        cmd.bindPipeline(culling->bindPoint, culling->handle.get());
        cmd.bindDescriptorSets(culling->bindPoint, culling->pipelineLayout.get(), 0, cullDescriptorSet, nullptr);
        cmd.dispatch(1 + scene.instances.size() / 64, 1, 1);

        using ps = vk::PipelineStageFlagBits;
        std::vector<vk::BufferMemoryBarrier> bufferBarriers;
        uint32_t byteSize = scene.commands.size() * sizeof(vk::DrawIndexedIndirectCommand);
        bufferBarriers.emplace_back(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, scene.indirectBuffer.handle, 0, byteSize);
        cmd.pipelineBarrier(ps::eComputeShader, ps::eVertexShader, vk::DependencyFlags(), {}, bufferBarriers, {});

        // Begin RenderPass
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
        cmd.bindDescriptorSets(deferred->bindPoint, deferred->pipelineLayout.get(), 1, inputLight, nullptr);
        cmd.pushConstants(deferred->pipelineLayout.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(ler::DeferredConstant), &defConstant);
        cmd.draw(4, 1, 0, 0);

        cmd.nextSubpass(vk::SubpassContents::eInline);
        cmd.bindPipeline(quad->bindPoint, quad->handle.get());
        cmd.bindVertexBuffers(0, 1, &scene.aabbBuffer.handle, &offset);
        cmd.pushConstants(quad->pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(ler::SceneConstant), &constant);
        cmd.draw(showAABB ? scene.lineCount : 0, 1, 0, 0);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(0, 300), ImGuiCond_Always);
        ImGui::Begin("Scene Renderer", nullptr, ImGuiWindowFlags_NoResize);
        if (ImGui::BeginCombo("##custom combo", current_item))
        {
            for (size_t n = 0; n < items.size(); n++)
            {
                bool is_selected = (current_item == items[n]);
                if (ImGui::Selectable(items[n], is_selected))
                {
                    current_item = items[n];
                    defConstant.viewMode = n;
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        //ImGuizmo uses OpenGL Y-axis
        glm::mat4 proj = constant.proj;
        proj[1][1] *= -1;

        ImGui::Checkbox("Show Bounding Box", &showAABB);
        ImGui::Text("Visible Meshes: %d", numVisibleMeshes);
        ImGui::Text("Light Count: %zu", lighting.size());
        ImGui::AlignTextToFramePadding();
        bool treeOpen = ImGui::TreeNodeEx("Scene", ImGuiTreeNodeFlags_AllowItemOverlap);
        ImGui::SameLine();
        if(ImGui::Button("Add Light") && lighting.size() < 6)
        {
            lighting.emplace_back();
            engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
        }
        if(treeOpen)
        {
            for (int i = 0; i < lighting.size(); i++)
            {
                ImGuiTreeNodeFlags node_flags = 0;
                const bool is_selected = i == node_clicked;
                if (is_selected)
                    node_flags |= ImGuiTreeNodeFlags_Selected;
                node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;
                ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, "Light %d", i);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
                    node_clicked = i;
            }
            if (node_clicked != -1)
            {
                ImGuizmo::BeginFrame();
                ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
                trans = glm::translate(glm::mat4(1.f), glm::vec3(lighting[node_clicked].pos));
                ImGuizmo::Manipulate(glm::value_ptr(constant.view), glm::value_ptr(proj), ImGuizmo::OPERATION::TRANSLATE, ImGuizmo::WORLD, glm::value_ptr(trans));
                if(ImGui::ColorEdit3("Color", glm::value_ptr(lighting[node_clicked].color), ImGuiColorEditFlags_Float))
                    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
                if(ImGui::SliderFloat("Radius", &lighting[node_clicked].radius, 0, 400))
                    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
                if(ImGui::InputFloat3("Position", glm::value_ptr(lighting[node_clicked].pos)))
                    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
                auto lineSize = ImGui::GetContentRegionAvail();
                if(ImGui::Button("Remove", ImVec2(lineSize.x, 0)))
                {
                    swapAndPop(lighting, node_clicked);
                    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
                    node_clicked = -1;
                }
                if(ImGuizmo::IsUsing())
                {
                    lighting[node_clicked].pos = glm::vec3(trans[3]);
                    engine.uploadBuffer(lightBuffer, lighting.data(), sizeof(ler::Light)*lighting.size());
                }
                if(ImGui::GetIO().KeyCtrl)
                    node_clicked = -1;
            }
            ImGui::TreePop();
        }

        ImGui::SliderFloat("Camera Speed", &camera.movementSpeed, 10, 1000);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        ImGui::Render();
        // Record dear imgui primitives into command buffer
        ImDrawData* draw_data = ImGui::GetDrawData();
        ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);

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
    }

    device->waitIdle();
    engine.destroyScene(scene);
    engine.destroyBuffer(lightBuffer);
    engine.destroyBuffer(visibleBuffer);
    engine.destroyBuffer(frustumBuffer);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}