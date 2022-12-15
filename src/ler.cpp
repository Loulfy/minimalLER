//
// Created by loulfy on 14/12/2022.
//

#define VMA_IMPLEMENTATION
#include "ler.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace ler
{
    static const std::array<vk::Format,5> c_depthFormats =
    {
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD32Sfloat,
        vk::Format::eD24UnormS8Uint,
        vk::Format::eD16UnormS8Uint,
        vk::Format::eD16Unorm
    };

    LerContext::LerContext(const LerSettings& settings)
    {
        m_device = settings.device;
        m_physicalDevice = settings.physicalDevice;
        m_queue = settings.device.getQueue(settings.graphicsQueueFamily, 0);

        // Create VMA Allocator
        vma::AllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.physicalDevice = settings.physicalDevice;
        allocatorInfo.instance = settings.instance;
        allocatorInfo.device = settings.device;
        m_allocator = vma::createAllocator(allocatorInfo);

        // Create Command Pool
        auto poolUsage = vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient;
        m_commandPool = m_device.createCommandPoolUnique({ poolUsage, settings.graphicsQueueFamily });
    }

    LerContext::~LerContext()
    {
        for(auto& tex : m_textures)
        {
            if(tex->allocation)
                m_allocator.destroyImage(tex->handle, tex->allocation);
        }

        //for(auto& buf : m_buffers)
            //m_allocator.destroyBuffer(buf.handle, buf.allocation);

        m_allocator.destroy();
    }

    Buffer LerContext::createBuffer(uint32_t byteSize, vk::BufferUsageFlags usages)
    {
        Buffer buffer;
        vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        usageFlags |= usages;
        buffer.info = vk::BufferCreateInfo();
        buffer.info.setSize(byteSize);
        buffer.info.setUsage(usageFlags);
        buffer.info.setSharingMode(vk::SharingMode::eExclusive);

        vma::AllocationCreateInfo allocInfo = {};
        allocInfo.usage = vma::MemoryUsage::eCpuOnly;

        auto [handle, allocation] = m_allocator.createBuffer(buffer.info, allocInfo);

        buffer.handle = handle;
        buffer.allocation = allocation;
        return buffer;
    }

    void LerContext::uploadBuffer(Buffer& staging, const void* src, uint32_t byteSize)
    {
        if(src)
        {
            void* dst = m_allocator.mapMemory(staging.allocation);
            std::memcpy(dst, src, byteSize);
            m_allocator.unmapMemory(staging.allocation);
        }
    }

    void LerContext::copyBuffer(Buffer& staging, Buffer& dst, uint64_t byteSize)
    {
        vk::CommandBuffer cmd = getCommandBuffer();
        vk::BufferCopy copyRegion(0, 0, byteSize);
        cmd.copyBuffer(staging.handle, dst.handle, copyRegion);
        submitAndWait(cmd);
    }

    void LerContext::copyBufferToTexture(vk::CommandBuffer& cmd, const Buffer& buffer, const TexturePtr& texture)
    {
        // prepare texture to transfer layout!
        std::vector<vk::ImageMemoryBarrier> imageBarriersStart;
        vk::PipelineStageFlags beforeStageFlags = vk::PipelineStageFlagBits::eTopOfPipe;
        vk::PipelineStageFlags afterStageFlags = vk::PipelineStageFlagBits::eTransfer;

        imageBarriersStart.emplace_back(
            vk::AccessFlags(),
            vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture->handle,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );

        cmd.pipelineBarrier(beforeStageFlags, afterStageFlags, vk::DependencyFlags(), {}, {}, imageBarriersStart);

        // Copy buffer to texture
        vk::BufferImageCopy copyRegion(0, 0, 0);
        copyRegion.imageExtent = texture->info.extent;
        copyRegion.imageSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        cmd.copyBufferToImage(buffer.handle, texture->handle, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
    }

    static vk::ImageUsageFlags pickImageUsage(vk::Format format, bool isRenderTarget)
    {
        vk::ImageUsageFlags ret = vk::ImageUsageFlagBits::eTransferSrc |
                                  vk::ImageUsageFlagBits::eTransferDst |
                                  vk::ImageUsageFlagBits::eSampled;

        if (isRenderTarget)
        {
            ret |= vk::ImageUsageFlagBits::eInputAttachment;
            switch(format)
            {
                case vk::Format::eS8Uint:
                case vk::Format::eD16Unorm:
                case vk::Format::eD32Sfloat:
                case vk::Format::eD16UnormS8Uint:
                case vk::Format::eD24UnormS8Uint:
                case vk::Format::eD32SfloatS8Uint:
                case vk::Format::eX8D24UnormPack32:
                    ret |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
                    break;

                default:
                    ret |= vk::ImageUsageFlagBits::eColorAttachment;
                    ret |= vk::ImageUsageFlagBits::eStorage;
                    break;
            }
        }
        return ret;
    }

    vk::ImageAspectFlags guessImageAspectFlags(vk::Format format)
    {
        switch(format)
        {
            case vk::Format::eD16Unorm:
            case vk::Format::eX8D24UnormPack32:
            case vk::Format::eD32Sfloat:
                return vk::ImageAspectFlagBits::eDepth;

            case vk::Format::eS8Uint:
                return vk::ImageAspectFlagBits::eStencil;

            case vk::Format::eD16UnormS8Uint:
            case vk::Format::eD24UnormS8Uint:
            case vk::Format::eD32SfloatS8Uint:
                return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;

            default:
                return vk::ImageAspectFlagBits::eColor;
        }
    }

    TexturePtr LerContext::createTexture(vk::Format format, const vk::Extent2D& extent, vk::SampleCountFlagBits sampleCount, bool isRenderTarget)
    {
        auto texture = std::make_shared<Texture>();
        vma::AllocationCreateInfo allocInfo = {};
        allocInfo.usage = vma::MemoryUsage::eGpuOnly;

        texture->info = vk::ImageCreateInfo();
        texture->info.setImageType(vk::ImageType::e2D);
        texture->info.setExtent(vk::Extent3D(extent.width, extent.height, 1));
        texture->info.setMipLevels(1);
        texture->info.setArrayLayers(1);
        texture->info.setFormat(format);
        texture->info.setInitialLayout(vk::ImageLayout::eUndefined);
        texture->info.setUsage(pickImageUsage(format, isRenderTarget));
        texture->info.setSharingMode(vk::SharingMode::eExclusive);
        texture->info.setSamples(sampleCount);
        texture->info.setFlags({});
        texture->info.setTiling(vk::ImageTiling::eOptimal);

        auto [handle, allocation] = m_allocator.createImage(texture->info, allocInfo);

        texture->handle = handle;
        texture->allocation = allocation;

        vk::ImageViewCreateInfo createInfo;
        createInfo.setImage(handle);
        createInfo.setViewType(vk::ImageViewType::e2D);
        createInfo.setFormat(format);
        createInfo.setSubresourceRange(vk::ImageSubresourceRange(guessImageAspectFlags(format), 0, 1, 0, 1));
        texture->view = m_device.createImageViewUnique(createInfo);

        m_textures.push_back(texture);
        return texture;
    }

    TexturePtr LerContext::createTextureFromNative(vk::Image image, vk::Format format, const vk::Extent2D& extent)
    {
        auto texture = std::make_shared<Texture>();

        texture->info = vk::ImageCreateInfo();
        texture->info.setImageType(vk::ImageType::e2D);
        texture->info.setExtent(vk::Extent3D(extent.width, extent.height, 1));
        texture->info.setSamples(vk::SampleCountFlagBits::e1);
        texture->info.setFormat(format);

        texture->handle = image;
        texture->allocation = nullptr;

        vk::ImageViewCreateInfo createInfo;
        createInfo.setImage(image);
        createInfo.setViewType(vk::ImageViewType::e2D);
        createInfo.setFormat(format);
        createInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        texture->view = m_device.createImageViewUnique(createInfo);

        m_textures.push_back(texture);
        return texture;
    }

    vk::PresentModeKHR LerContext::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes, bool vSync)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            if(availablePresentMode == vk::PresentModeKHR::eFifo && vSync)
                return availablePresentMode;
            if (availablePresentMode == vk::PresentModeKHR::eMailbox && !vSync)
                return availablePresentMode;
        }
        return vk::PresentModeKHR::eImmediate;
    }

    vk::SurfaceFormatKHR LerContext::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined)
        {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        for (const auto& format : availableFormats)
        {
            if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return format;
        }

        throw std::runtime_error("found no suitable surface format");
        return {};
    }

    vk::Extent2D LerContext::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height)
    {
        if (capabilities.currentExtent.width == UINT32_MAX)
        {
            vk::Extent2D extent(width, height);
            vk::Extent2D minExtent = capabilities.minImageExtent;
            vk::Extent2D maxExtent = capabilities.maxImageExtent;
            extent.width = std::clamp(extent.width, minExtent.width, maxExtent.width);
            extent.height = std::clamp(extent.height, minExtent.height, maxExtent.height);
            return extent;
        }
        else
        {
            return capabilities.currentExtent;
        }
    }

    vk::Format LerContext::chooseDepthFormat()
    {
        for (const vk::Format& format : c_depthFormats)
        {
            vk::FormatProperties depthFormatProperties = m_physicalDevice.getFormatProperties(format);
            // Format must support depth stencil attachment for optimal tiling
            if (depthFormatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
                return format;
        }
        return vk::Format::eD32Sfloat;
    }

    SwapChain LerContext::createSwapChain(vk::SurfaceKHR surface, uint32_t width, uint32_t height, bool vSync)
    {
        // Setup viewports, vSync
        std::vector<vk::SurfaceFormatKHR> surfaceFormats = m_physicalDevice.getSurfaceFormatsKHR(surface);
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(surface);
        std::vector<vk::PresentModeKHR> surfacePresentModes = m_physicalDevice.getSurfacePresentModesKHR(surface);

        vk::Extent2D extent = chooseSwapExtent(surfaceCapabilities, width, height);
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(surfaceFormats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(surfacePresentModes, vSync);

        uint32_t backBufferCount = std::clamp(surfaceCapabilities.maxImageCount, 1U, 2U);

        // Create swapChain
        using vkIU = vk::ImageUsageFlagBits;
        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.setSurface(surface);
        createInfo.setMinImageCount(backBufferCount);
        createInfo.setImageFormat(surfaceFormat.format);
        createInfo.setImageColorSpace(surfaceFormat.colorSpace);
        createInfo.setImageExtent(extent);
        createInfo.setImageArrayLayers(1);
        createInfo.setImageUsage(vkIU::eColorAttachment | vkIU::eTransferDst | vkIU::eSampled);
        createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        createInfo.setPreTransform(surfaceCapabilities.currentTransform);
        createInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
        createInfo.setPresentMode(presentMode);
        createInfo.setClipped(true);

        SwapChain swapChain;
        swapChain.format = surfaceFormat.format;
        swapChain.extent = extent;
        swapChain.handle = m_device.createSwapchainKHRUnique(createInfo);
        return swapChain;
    }

    RenderPass LerContext::createDefaultRenderPass(vk::Format surfaceFormat)
    {
        RenderPass renderPass;
        renderPass.attachments.resize(6);
        std::array<vk::SubpassDescription2, 2> subPass;
        std::vector<vk::AttachmentReference2> colorAttachmentRef1(3);
        std::vector<vk::AttachmentReference2> colorAttachmentRef2(1);
        std::vector<vk::AttachmentReference2> colorInputRef(3);
        std::array<vk::AttachmentReference2,1> resolveAttachmentRef;
        vk::AttachmentReference2 depthAttachmentRef;

        auto properties = m_physicalDevice.getProperties();
        vk::SampleCountFlagBits sampleCount;
        vk::SampleCountFlags samples = std::min(properties.limits.framebufferColorSampleCounts, properties.limits.framebufferDepthSampleCounts);
        if(samples & vk::SampleCountFlagBits::e1) sampleCount = vk::SampleCountFlagBits::e1;
        if(samples & vk::SampleCountFlagBits::e8) sampleCount = vk::SampleCountFlagBits::e8;

        // Position & Normal
        for(uint32_t i = 0; i < 2; i++)
        {
            renderPass.attachments[i] = vk::AttachmentDescription2()
                .setFormat(vk::Format::eR16G16B16A16Sfloat)
                .setSamples(sampleCount)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

            colorAttachmentRef1[i] = vk::AttachmentReference2()
                .setAttachment(i)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

            colorInputRef[i] = vk::AttachmentReference2()
                .setAttachment(i)
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        // Albedo + Specular
        renderPass.attachments[2] = vk::AttachmentDescription2()
            .setFormat(vk::Format::eR8G8B8A8Unorm)
            .setSamples(sampleCount)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

        colorAttachmentRef1[2] = vk::AttachmentReference2()
            .setAttachment(2)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        colorInputRef[2] = vk::AttachmentReference2()
            .setAttachment(2)
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

        // Depth + Stencil
        vk::ImageLayout depthLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        renderPass.attachments[3] = vk::AttachmentDescription2()
            .setFormat(chooseDepthFormat())
            .setSamples(sampleCount)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eClear)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(depthLayout);

        depthAttachmentRef = vk::AttachmentReference2()
            .setAttachment(3)
            .setLayout(depthLayout);

        // Result Color Image (DIRECT)
        renderPass.attachments[4] = vk::AttachmentDescription2()
            .setFormat(vk::Format::eB8G8R8A8Unorm)
            .setSamples(sampleCount)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

        colorAttachmentRef2[0] = vk::AttachmentReference2()
            .setAttachment(4)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        // Resolve Present Image
        renderPass.attachments[5] = vk::AttachmentDescription2()
            .setFormat(surfaceFormat)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        resolveAttachmentRef[0] = vk::AttachmentReference2()
            .setAttachment(5)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        // FIRST PASS
        subPass[0] = vk::SubpassDescription2()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachments(colorAttachmentRef1)
            .setPDepthStencilAttachment(&depthAttachmentRef);

        // SECOND PASS
        subPass[1] = vk::SubpassDescription2()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachments(colorAttachmentRef2)
            .setResolveAttachments(resolveAttachmentRef)
                    //.setPResolveAttachments(&resolveAttachmentRef)
                    //.setPDepthStencilAttachment(&depthAttachmentRef)
            .setInputAttachments(colorInputRef);

        // DEPENDENCIES
        std::array<vk::SubpassDependency2, 1> dependencies;
        dependencies[0] = vk::SubpassDependency2()
            .setSrcSubpass(0)
            .setDstSubpass(1)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstStageMask(vk::PipelineStageFlagBits::eFragmentShader)
            .setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

        vk::RenderPassCreateInfo2 renderPassInfo;
        renderPassInfo.setAttachments(renderPass.attachments);
        renderPassInfo.setDependencies(dependencies);
        renderPassInfo.setSubpasses(subPass);

        renderPass.handle = m_device.createRenderPass2Unique(renderPassInfo);
        return renderPass;
    }

    std::vector<vk::UniqueFramebuffer> LerContext::createFrameBuffers(const RenderPass& renderPass, const SwapChain& swapChain)
    {
        std::vector<vk::UniqueFramebuffer> frameBuffers;
        std::vector<vk::ImageView> attachmentViews;
        attachmentViews.reserve(renderPass.attachments.size());
        auto swapChainImages = m_device.getSwapchainImagesKHR(swapChain.handle.get());
        for(auto& image : swapChainImages)
        {
            attachmentViews.clear();
            for(size_t i = 0; i < renderPass.attachments.size() - 1; ++i)
            {
                auto texture = createTexture(renderPass.attachments[i].format, swapChain.extent, renderPass.attachments[i].samples, true);
                attachmentViews.emplace_back(texture->view.get());
            }
            auto frame = createTextureFromNative(image, swapChain.format, swapChain.extent);
            attachmentViews.emplace_back(frame->view.get());

            vk::FramebufferCreateInfo framebufferInfo;
            framebufferInfo.setRenderPass(renderPass.handle.get());
            framebufferInfo.setAttachments(attachmentViews);
            framebufferInfo.setWidth(swapChain.extent.width);
            framebufferInfo.setHeight(swapChain.extent.height);
            framebufferInfo.setLayers(1);

            frameBuffers.emplace_back(m_device.createFramebufferUnique(framebufferInfo));
        }

        return frameBuffers;
    }

    uint32_t LerContext::loadTextureFromFile(const fs::path& path)
    {
        int w, h, c;
        if(path.extension() == ".ktx" || path.extension() == ".dds")
            throw std::runtime_error("Can't load image with extension " + path.extension().string());

        unsigned char* image = stbi_load(path.string().c_str(), &w, &h, &c, STBI_rgb_alpha);
        size_t imageSize = w * h * 4;

        auto staging = createBuffer(imageSize);
        uploadBuffer(staging, image, imageSize);

        auto texture = createTexture(vk::Format::eR8G8B8A8Unorm, vk::Extent2D(w, h), vk::SampleCountFlagBits::e1);
        auto cmd = getCommandBuffer();
        copyBufferToTexture(cmd, staging, texture);
        submitAndWait(cmd);

        stbi_image_free(image);
        m_allocator.destroyBuffer(staging.handle, staging.allocation);
        return 0;
    }

    uint32_t LerContext::loadTextureFromMemory(const unsigned char* buffer, uint32_t size)
    {
        int w, h, c;

        unsigned char* image = stbi_load_from_memory(buffer, static_cast<int>(size), &w, &h, &c, STBI_rgb_alpha);
        size_t imageSize = w * h * 4;

        auto staging = createBuffer(imageSize);
        uploadBuffer(staging, image, imageSize);

        auto texture = createTexture(vk::Format::eR8G8B8A8Unorm, vk::Extent2D(w, h), vk::SampleCountFlagBits::e1);
        auto cmd = getCommandBuffer();
        copyBufferToTexture(cmd, staging, texture);
        submitAndWait(cmd);

        stbi_image_free(image);
        m_allocator.destroyBuffer(staging.handle, staging.allocation);
        return 0;
    }

    vk::CommandBuffer LerContext::getCommandBuffer()
    {
        vk::CommandBuffer cmd;
        std::lock_guard lock(m_mutex);
        if (m_commandBuffersPool.empty())
        {
            // Allocate command buffer
            auto allocInfo = vk::CommandBufferAllocateInfo();
            allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
            allocInfo.setCommandPool(m_commandPool.get());
            allocInfo.setCommandBufferCount(1);

            vk::Result res;
            res = m_device.allocateCommandBuffers(&allocInfo, &cmd);
            assert(res == vk::Result::eSuccess);
        }
        else
        {
            cmd = m_commandBuffersPool.front();
            m_commandBuffersPool.pop_front();
        }

        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cmd;
    }

    void LerContext::submitAndWait(vk::CommandBuffer& cmd)
    {
        cmd.end();
        vk::UniqueFence fence = m_device.createFenceUnique({});

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmd);
        m_queue.submit(submitInfo, fence.get());

        auto res = m_device.waitForFences(fence.get(), true, std::numeric_limits<uint64_t>::max());
        assert(res == vk::Result::eSuccess);

        std::lock_guard lock(m_mutex);
        m_commandBuffersPool.push_back(cmd);
    }

    uint32_t LerContext::loadTexture(const aiScene* aiScene, const aiString& filename, const fs::path& path)
    {
        const auto* key = filename.C_Str();
        if(m_cache.contains(key))
            return m_cache.at(key);

        auto em = aiScene->GetEmbeddedTexture(key);
        std::cout << "Load image : " << key << std::endl;
        if(em == nullptr)
        {
            fs::path f = path.parent_path() / fs::path(key);
            loadTextureFromFile(f);
        }
        else
        {
            const auto* buffer = reinterpret_cast<const unsigned char*>(em->pcData);
            loadTextureFromMemory(buffer, em->mWidth);
        }
        m_cache.emplace(key, 0);
        return m_cache.at(key);
    }

    void LerContext::mergeSceneBuffer(Scene& scene, Buffer& dest, const aiScene* aiScene, const std::function<bool(aiMesh*)>& predicate, const std::function<void*(aiMesh*)>& provider)
    {
        auto* data = m_allocator.mapMemory(scene.staging.allocation);
        auto cursor = static_cast<std::byte*>(data);
        for(size_t i = 0; i < aiScene->mNumMeshes; ++i)
        {
            auto* mesh = aiScene->mMeshes[i];
            if(predicate(mesh))
                std::memcpy(cursor + scene.geometries[i].firstVertex * sizeof(glm::vec3), provider(mesh), scene.geometries[i].countVertex * sizeof(glm::vec3));
            else
                std::memset(cursor + scene.geometries[i].firstVertex * sizeof(glm::vec3), 0, scene.geometries[i].countVertex * sizeof(glm::vec3));
        }
        m_allocator.unmapMemory(scene.staging.allocation);
        vk::CommandBuffer cmd = getCommandBuffer();
        vk::BufferCopy copyRegion(0, 0, scene.vertexCount * sizeof(glm::vec3));
        cmd.copyBuffer(scene.staging.handle, dest.handle, copyRegion);
        submitAndWait(cmd);
    }

    void LerContext::transformBoundingBox(const glm::mat4& t, glm::vec3& min, glm::vec3& max)
    {
        std::array<glm::vec3, 8> pts = {
                glm::vec3(min.x, min.y, min.z),
                glm::vec3(min.x, max.y, min.z),
                glm::vec3(min.x, min.y, max.z),
                glm::vec3(min.x, max.y, max.z),
                glm::vec3(max.x, min.y, min.z),
                glm::vec3(max.x, max.y, min.z),
                glm::vec3(max.x, min.y, max.z),
                glm::vec3(max.x, max.y, max.z),
        };

        for (auto& p: pts)
            p = glm::vec3(t * glm::vec4(p, 1.f));

        // create
        min = glm::vec3(std::numeric_limits<float>::max());
        max = glm::vec3(std::numeric_limits<float>::lowest());

        for (auto& p : pts)
        {
            min = glm::min(min, p);
            max = glm::max(max, p);
        }
    }

    void loadNode(Scene& scene, aiNode* aiNode)
    {
        if(aiNode == nullptr)
            return;

        for (size_t i = 0; i < aiNode->mNumMeshes; ++i)
        {
            Instance inst;
            const auto& ind = scene.geometries[aiNode->mMeshes[i]];
            inst.model = glm::make_mat4(aiNode->mTransformation.Transpose()[0]);
            auto* currentParent = aiNode->mParent;
            while (currentParent) {
                inst.model = glm::make_mat4(currentParent->mTransformation.Transpose()[0]) * inst.model;
                currentParent = currentParent->mParent;
            }

            inst.bMin = ind.bMin;
            inst.bMax = ind.bMax;
            LerContext::transformBoundingBox(inst.model, inst.bMin, inst.bMax);

            inst.matId = ind.materialId;
            scene.instances.push_back(inst);
            scene.commands.emplace_back(ind.countIndex, 1, ind.firstIndex, ind.firstVertex, 0);
            scene.drawCount+= 1;
        }
        for(size_t i = 0; i < aiNode->mNumChildren; ++i)
            loadNode(scene, aiNode->mChildren[i]);
    }

    Scene LerContext::fromFile(const fs::path& path)
    {
        Scene scene;
        Assimp::Importer importer;
        unsigned int postProcess = aiProcessPreset_TargetRealtime_Fast;
        postProcess |= aiProcess_ConvertToLeftHanded;
        postProcess |= aiProcess_GenBoundingBoxes;
        const aiScene* aiScene = importer.ReadFile(path.string(), postProcess);
        if(aiScene == nullptr || aiScene->mNumMeshes < 0)
            return scene;

        // Load Materials
        scene.materials.reserve(aiScene->mNumMaterials);
        for(size_t i = 0; i < aiScene->mNumMaterials; ++i)
        {
            aiString filename;
            aiColor3D baseColor;
            Material materialInstance;
            auto* material = aiScene->mMaterials[i];
            material->Get(AI_MATKEY_COLOR_DIFFUSE,baseColor);
            materialInstance.color = glm::vec3(baseColor.r, baseColor.g, baseColor.b);
            if(material->GetTextureCount(aiTextureType_BASE_COLOR) > 0)
            {
                material->GetTexture(aiTextureType_BASE_COLOR, 0, &filename);
                materialInstance.texId = loadTexture(aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_AMBIENT) > 0)
            {
                material->GetTexture(aiTextureType_AMBIENT, 0, &filename);
                materialInstance.texId = loadTexture(aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
            {
                material->GetTexture(aiTextureType_DIFFUSE, 0, &filename);
                materialInstance.texId = loadTexture(aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
            {
                material->GetTexture(aiTextureType_NORMALS, 0, &filename);
                materialInstance.norId = loadTexture(aiScene, filename, path);
            }

            scene.materials.push_back(materialInstance);
        }

        MeshIndirect ind;

        // Prepare indirect data
        scene.geometries.reserve(aiScene->mNumMeshes);
        for(size_t i = 0; i < aiScene->mNumMeshes; ++i)
        {
            auto* mesh = aiScene->mMeshes[i];
            ind.countIndex = mesh->mNumFaces * 3;
            ind.firstIndex = scene.indexCount;
            ind.countVertex = mesh->mNumVertices;
            ind.firstVertex = scene.vertexCount;
            ind.materialId = mesh->mMaterialIndex;
            ind.bMin = glm::make_vec3(&mesh->mAABB.mMin[0]);
            ind.bMax = glm::make_vec3(&mesh->mAABB.mMax[0]);
            scene.geometries.push_back(ind);

            scene.indexCount+= ind.countIndex;
            scene.vertexCount+= ind.countVertex;
        }

        // Load Scene
        loadNode(scene, aiScene->mRootNode);

        // Create Buffer
        size_t indexByteSize = scene.indexCount * sizeof(uint32_t);
        size_t vertexByteSize = scene.vertexCount * sizeof(glm::vec3);
        auto cmdUsage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer;
        scene.staging = createBuffer(std::max(indexByteSize, vertexByteSize));
        scene.indexBuffer = createBuffer(indexByteSize, vk::BufferUsageFlagBits::eIndexBuffer);
        scene.vertexBuffer = createBuffer(vertexByteSize, vk::BufferUsageFlagBits::eVertexBuffer);
        scene.normalBuffer = createBuffer(vertexByteSize, vk::BufferUsageFlagBits::eVertexBuffer);
        scene.texcoordBuffer = createBuffer(vertexByteSize, vk::BufferUsageFlagBits::eVertexBuffer);
        scene.tangentBuffer = createBuffer(vertexByteSize, vk::BufferUsageFlagBits::eVertexBuffer);
        scene.indirectBuffer = createBuffer(scene.drawCount * sizeof(vk::DrawIndexedIndirectCommand), cmdUsage);
        scene.instanceBuffer = createBuffer(scene.drawCount * sizeof(Instance), vk::BufferUsageFlagBits::eStorageBuffer);
        scene.materialBuffer = createBuffer(scene.materials.size() * sizeof(Material), vk::BufferUsageFlagBits::eStorageBuffer);

        // Merge vertex
        mergeSceneBuffer(scene, scene.vertexBuffer, aiScene, [](aiMesh* mesh){ return mesh->HasPositions(); }, [](aiMesh* mesh){ return mesh->mVertices; });

        // Merge normal
        mergeSceneBuffer(scene, scene.normalBuffer, aiScene, [](aiMesh* mesh){ return mesh->HasNormals(); }, [](aiMesh* mesh){ return mesh->mNormals; });

        // Merge texcoord
        mergeSceneBuffer(scene, scene.texcoordBuffer, aiScene, [](aiMesh* mesh){ return mesh->HasTextureCoords(0); }, [](aiMesh* mesh){ return mesh->mTextureCoords[0]; });

        // Merge tangent
        mergeSceneBuffer(scene, scene.tangentBuffer, aiScene, [](aiMesh* mesh){ return mesh->HasTangentsAndBitangents(); }, [](aiMesh* mesh){ return mesh->mTangents; });

        // Merge index
        std::vector<uint32_t> indices;
        indices.reserve(scene.indexCount);
        for(size_t i = 0; i < aiScene->mNumMeshes; ++i)
        {
            auto *mesh = aiScene->mMeshes[i];
            if (mesh->HasFaces())
            {
                for (size_t j = 0; j < mesh->mNumFaces; j++)
                {
                    indices.insert(indices.end(), mesh->mFaces[j].mIndices, mesh->mFaces[j].mIndices + 3);
                }
            }
        }

        size_t byteSize = indices.size() * sizeof(uint32_t);
        uploadBuffer(scene.staging, indices.data(), byteSize);
        copyBuffer(scene.staging, scene.indexBuffer, byteSize);

        // Indirect
        byteSize = scene.commands.size() * sizeof(vk::DrawIndexedIndirectCommand);
        uploadBuffer(scene.staging, scene.commands.data(), byteSize);
        copyBuffer(scene.staging, scene.indirectBuffer, byteSize);

        // Instances
        byteSize = scene.instances.size() * sizeof(Instance);
        uploadBuffer(scene.staging, scene.instances.data(), byteSize);
        copyBuffer(scene.staging, scene.instanceBuffer, byteSize);

        // Materials
        byteSize = scene.materials.size() * sizeof(Material);
        uploadBuffer(scene.staging, scene.materials.data(), byteSize);
        copyBuffer(scene.staging, scene.materialBuffer, byteSize);

        return scene;
    }
}