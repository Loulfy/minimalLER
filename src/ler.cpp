//
// Created by loulfy on 14/12/2022.
//

#define VMA_IMPLEMENTATION
#include "ler.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define SPIRV_REFLECT_HAS_VULKAN_H
#include <spirv_reflect.h>

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

    static const std::array<std::set<std::string>, 5> c_VertexAttrMap =
    {{
             {"inPos"},
             {"inTex", "inUV"},
             {"inNormal"},
             {"inTangent"},
             {"inColor"}
     }};

    LerContext::LerContext(const LerSettings& settings)
    {
        m_device = settings.device;
        m_pipelineCache = settings.pipelineCache;
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

        // prepare texture to color layout
        std::vector<vk::ImageMemoryBarrier> imageBarriersStop;
        beforeStageFlags = vk::PipelineStageFlagBits::eTransfer;
        afterStageFlags = vk::PipelineStageFlagBits::eFragmentShader;
        imageBarriersStop.emplace_back(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture->handle,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );

        cmd.pipelineBarrier(beforeStageFlags, afterStageFlags, vk::DependencyFlags(), {}, {}, imageBarriersStop);
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

    vk::ImageAspectFlags LerContext::guessImageAspectFlags(vk::Format format)
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

    vk::UniqueSampler LerContext::createSampler(const vk::SamplerAddressMode& addressMode, bool filter)
    {
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.setMagFilter(filter ? vk::Filter::eLinear : vk::Filter::eNearest);
        samplerInfo.setMinFilter(filter ? vk::Filter::eLinear : vk::Filter::eNearest);
        samplerInfo.setMipmapMode(filter ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest);
        samplerInfo.setAddressModeU(addressMode);
        samplerInfo.setAddressModeV(addressMode);
        samplerInfo.setAddressModeW(addressMode);
        samplerInfo.setMipLodBias(0.f);
        samplerInfo.setAnisotropyEnable(false);
        samplerInfo.setMaxAnisotropy(1.f);
        samplerInfo.setCompareEnable(false);
        samplerInfo.setCompareOp(vk::CompareOp::eLess);
        samplerInfo.setMinLod(0.f);
        samplerInfo.setMaxLod(std::numeric_limits<float>::max());
        samplerInfo.setBorderColor(vk::BorderColor::eFloatOpaqueBlack);

        return m_device.createSamplerUnique(samplerInfo, nullptr);
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

    std::vector<char> LerContext::loadBinaryFromFile(const fs::path& path)
    {
        std::vector<char> v;
        std::ifstream file(path, std::ios::binary);
        std::stringstream src;
        src << file.rdbuf();
        file.close();

        auto s = src.str();
        std::copy( s.begin(), s.end(), std::back_inserter(v));
        return v;
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

        for(auto& output : colorAttachmentRef1)
            renderPass.subPass[0].insert(output.attachment);

        // SECOND PASS
        subPass[1] = vk::SubpassDescription2()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachments(colorAttachmentRef2)
            .setResolveAttachments(resolveAttachmentRef)
            .setPDepthStencilAttachment(&depthAttachmentRef)
            .setInputAttachments(colorInputRef);

        for(auto& output : colorAttachmentRef2)
            renderPass.subPass[1].insert(output.attachment);

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

    std::vector<FrameBuffer> LerContext::createFrameBuffers(const RenderPass& renderPass, const SwapChain& swapChain)
    {
        std::vector<FrameBuffer> frameBuffers;
        std::vector<vk::ImageView> attachmentViews;
        attachmentViews.reserve(renderPass.attachments.size());
        auto swapChainImages = m_device.getSwapchainImagesKHR(swapChain.handle.get());
        for(auto& image : swapChainImages)
        {
            auto& frameBuffer = frameBuffers.emplace_back();
            attachmentViews.clear();
            for(size_t i = 0; i < renderPass.attachments.size() - 1; ++i)
            {
                auto texture = createTexture(renderPass.attachments[i].format, swapChain.extent, renderPass.attachments[i].samples, true);
                attachmentViews.emplace_back(texture->view.get());
                frameBuffer.images.push_back(texture);
            }
            auto frame = createTextureFromNative(image, swapChain.format, swapChain.extent);
            attachmentViews.emplace_back(frame->view.get());
            frameBuffer.images.push_back(frame);

            vk::FramebufferCreateInfo framebufferInfo;
            framebufferInfo.setRenderPass(renderPass.handle.get());
            framebufferInfo.setAttachments(attachmentViews);
            framebufferInfo.setWidth(swapChain.extent.width);
            framebufferInfo.setHeight(swapChain.extent.height);
            framebufferInfo.setLayers(1);

            frameBuffer.handle = m_device.createFramebufferUnique(framebufferInfo);
        }

        return frameBuffers;
    }

    uint32_t guessVertexInputBinding(const char* name)
    {
        for(size_t i = 0; i < c_VertexAttrMap.size(); ++i)
            if(c_VertexAttrMap[i].contains(name))
                return i;
        throw std::runtime_error("Vertex Input Attribute not reserved");
    }

    ShaderPtr LerContext::createShader(const fs::path& path)
    {
        auto shader = std::make_shared<Shader>();
        auto bytecode = loadBinaryFromFile(path);
        vk::ShaderModuleCreateInfo shaderInfo;
        shaderInfo.setCodeSize(bytecode.size());
        shaderInfo.setPCode(reinterpret_cast<const uint32_t*>(bytecode.data()));
        shader->shaderModule = m_device.createShaderModuleUnique(shaderInfo);

        uint32_t count = 0;
        SpvReflectShaderModule module;
        SpvReflectResult result = spvReflectCreateShaderModule(bytecode.size(), bytecode.data(), &module);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);
        assert(module.generator == SPV_REFLECT_GENERATOR_KHRONOS_GLSLANG_REFERENCE_FRONT_END);

        shader->stageFlagBits = static_cast<vk::ShaderStageFlagBits>(module.shader_stage);
        //log::info("Reflect Shader Stage {}", vk::to_string(handle->stageFlagBits));

        // Input Variables
        result = spvReflectEnumerateInputVariables(&module, &count, nullptr);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        std::vector<SpvReflectInterfaceVariable*> inputs(count);
        result = spvReflectEnumerateInputVariables(&module, &count, inputs.data());
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        std::set<uint32_t> availableBinding;
        if (module.shader_stage == SPV_REFLECT_SHADER_STAGE_VERTEX_BIT)
        {
            for(auto& in : inputs)
            {
                if(in->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN)
                    continue;

                uint32_t binding = guessVertexInputBinding(in->name);
                shader->attributeDesc.emplace_back(in->location, binding, static_cast<vk::Format>(in->format), 0);
                //log::info("location = {}, binding = {}, name = {}", in->location, binding, in->name);
                if(!availableBinding.contains(binding))
                {
                    shader->bindingDesc.emplace_back(binding, 0, vk::VertexInputRate::eVertex);
                    availableBinding.insert(binding);
                }
            }

            std::sort(shader->attributeDesc.begin(), shader->attributeDesc.end(),
                [](const VkVertexInputAttributeDescription& a, const VkVertexInputAttributeDescription& b) {
                    return a.location < b.location;
            });

            // Compute final offsets of each attribute, and total vertex stride.
            for (size_t i = 0; i < shader->attributeDesc.size(); ++i)
            {
                uint32_t format_size = formatSize(static_cast<VkFormat>(shader->attributeDesc[i].format));
                shader->attributeDesc[i].offset = shader->bindingDesc[i].stride;
                shader->bindingDesc[i].stride += format_size;
            }
        }

        shader->pvi = vk::PipelineVertexInputStateCreateInfo();
        shader->pvi.setVertexAttributeDescriptions(shader->attributeDesc);
        shader->pvi.setVertexBindingDescriptions(shader->bindingDesc);

        // Push Constants
        result = spvReflectEnumeratePushConstantBlocks(&module, &count, nullptr);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        std::vector<SpvReflectBlockVariable*> constants(count);
        result = spvReflectEnumeratePushConstantBlocks(&module, &count, constants.data());
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        for(auto& block : constants)
            shader->pushConstants.emplace_back(shader->stageFlagBits, block->offset, block->size);

        // Descriptor Set
        result = spvReflectEnumerateDescriptorSets(&module, &count, nullptr);
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        std::vector<SpvReflectDescriptorSet*> sets(count);
        result = spvReflectEnumerateDescriptorSets(&module, &count, sets.data());
        assert(result == SPV_REFLECT_RESULT_SUCCESS);

        for(auto& set : sets)
        {
            DescriptorSetLayoutData desc;
            desc.set_number = set->set;
            desc.bindings.resize(set->binding_count);
            for(size_t i = 0; i < set->binding_count; ++i)
            {
                auto& binding = desc.bindings[i];
                binding.binding = set->bindings[i]->binding;
                binding.descriptorCount = set->bindings[i]->count;
                binding.descriptorType = static_cast<vk::DescriptorType>(set->bindings[i]->descriptor_type);
                binding.stageFlags = shader->stageFlagBits;
                //log::info("set = {}, binding = {}, count = {:02}, type = {}", set->set, binding.binding, binding.descriptorCount, vk::to_string(binding.descriptorType));
            }
            shader->descriptorMap.insert({set->set, desc});
        }

        spvReflectDestroyShaderModule(&module);
        return shader;
    }

    void BasePipeline::reflectPipelineLayout(vk::Device device, const std::vector<ShaderPtr>& shaders)
    {
        // PIPELINE LAYOUT STATE
        auto layoutInfo = vk::PipelineLayoutCreateInfo();
        std::vector<vk::PushConstantRange> pushConstants;
        for(auto& shader : shaders)
            pushConstants.insert(pushConstants.end(), shader->pushConstants.begin(), shader->pushConstants.end());
        layoutInfo.setPushConstantRanges(pushConstants);

        // SHADER REFLECT
        std::set<uint32_t> sets;
        std::vector<vk::DescriptorPoolSize> descriptorPoolSizeInfo;
        std::multimap<uint32_t,DescriptorSetLayoutData> mergedDesc;
        for(auto& shader : shaders)
            mergedDesc.merge(shader->descriptorMap);

        for(auto& e : mergedDesc)
            sets.insert(e.first);

        std::vector<vk::DescriptorSetLayout> setLayouts;
        setLayouts.reserve(sets.size());
        for(auto& set : sets)
        {
            descriptorPoolSizeInfo.clear();
            auto it = descriptorAllocMap.emplace(set, DescriptorAllocator());
            auto& allocator = std::get<0>(it)->second;

            auto descriptorPoolInfo = vk::DescriptorPoolCreateInfo();
            auto descriptorLayoutInfo = vk::DescriptorSetLayoutCreateInfo();
            auto range = mergedDesc.equal_range(set);
            for (auto e = range.first; e != range.second; ++e)
                allocator.layoutBinding.insert(allocator.layoutBinding.end(), e->second.bindings.begin(), e->second.bindings.end());
            descriptorLayoutInfo.setBindings(allocator.layoutBinding);
            for(auto& b : allocator.layoutBinding)
                descriptorPoolSizeInfo.emplace_back(b.descriptorType, b.descriptorCount+2);
            descriptorPoolInfo.setPoolSizes(descriptorPoolSizeInfo);
            descriptorPoolInfo.setMaxSets(4);
            allocator.pool = device.createDescriptorPoolUnique(descriptorPoolInfo);
            allocator.layout = device.createDescriptorSetLayoutUnique(descriptorLayoutInfo);
            setLayouts.push_back(allocator.layout.get());
        }

        layoutInfo.setSetLayouts(setLayouts);
        pipelineLayout = device.createPipelineLayoutUnique(layoutInfo);
    }

    vk::DescriptorSet BasePipeline::createDescriptorSet(vk::Device& device, uint32_t set)
    {
        if(!descriptorAllocMap.contains(set))
            return {};

        vk::Result res;
        vk::DescriptorSet descriptorSet;
        const auto& allocator = descriptorAllocMap[set];
        vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;
        descriptorSetAllocInfo.setDescriptorSetCount(1);
        descriptorSetAllocInfo.setDescriptorPool(allocator.pool.get());
        descriptorSetAllocInfo.setPSetLayouts(&allocator.layout.get());
        res = device.allocateDescriptorSets(&descriptorSetAllocInfo, &descriptorSet);
        assert(res == vk::Result::eSuccess);
        return descriptorSet;
    }

    void addShaderStage(std::vector<vk::PipelineShaderStageCreateInfo>& stages, const ShaderPtr& shader)
    {
        stages.emplace_back(
            vk::PipelineShaderStageCreateFlags(),
            shader->stageFlagBits,
            shader->shaderModule.get(),
            "main",
            nullptr
        );
    }

    PipelinePtr LerContext::createGraphicsPipeline(const RenderPass& renderPass, const std::vector<ShaderPtr>& shaders, const PipelineInfo& info)
    {
        auto pipeline = std::make_shared<GraphicsPipeline>();
        std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStages;
        for(auto& shader : shaders)
            addShaderStage(pipelineShaderStages, shader);

        // TOPOLOGY STATE
        vk::PipelineInputAssemblyStateCreateInfo pia(vk::PipelineInputAssemblyStateCreateFlags(), info.topology);

        // VIEWPORT STATE
        auto viewport = vk::Viewport(0, 0, static_cast<float>(info.extent.width), static_cast<float>(info.extent.height), 0, 1.0f);
        auto renderArea = vk::Rect2D(vk::Offset2D(), info.extent);

        vk::PipelineViewportStateCreateInfo pv(vk::PipelineViewportStateCreateFlagBits(), 1, &viewport, 1, &renderArea);

        // Multi Sampling STATE
        vk::PipelineMultisampleStateCreateInfo pm(vk::PipelineMultisampleStateCreateFlags(), info.sampleCount);

        // POLYGON STATE
        vk::PipelineRasterizationStateCreateInfo pr;
        pr.setDepthClampEnable(VK_TRUE);
        pr.setRasterizerDiscardEnable(VK_FALSE);
        pr.setPolygonMode(info.polygonMode);
        pr.setFrontFace(vk::FrontFace::eCounterClockwise);
        pr.setDepthBiasEnable(VK_FALSE);
        pr.setDepthBiasConstantFactor(0.f);
        pr.setDepthBiasClamp(0.f);
        pr.setDepthBiasSlopeFactor(0.f);
        pr.setLineWidth(1.f);

        // DEPTH & STENCIL STATE
        vk::PipelineDepthStencilStateCreateInfo pds;
        pds.setDepthTestEnable(VK_TRUE);
        pds.setDepthWriteEnable(info.writeDepth);
        pds.setDepthCompareOp(vk::CompareOp::eLessOrEqual);
        pds.setDepthBoundsTestEnable(VK_FALSE);
        pds.setStencilTestEnable(VK_FALSE);
        pds.setFront(vk::StencilOpState());
        pds.setBack(vk::StencilOpState());
        pds.setMinDepthBounds(0.f);
        pds.setMaxDepthBounds(1.f);

        // BLEND STATE
        std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments;
        vk::PipelineColorBlendAttachmentState pcb;
        pcb.setBlendEnable(VK_TRUE); // false
        pcb.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha); //one //srcAlpha
        pcb.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha); //one //oneminussrcalpha
        pcb.setColorBlendOp(vk::BlendOp::eAdd);
        pcb.setSrcAlphaBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha); //one //oneminussrcalpha
        pcb.setDstAlphaBlendFactor(vk::BlendFactor::eZero); //zero
        pcb.setAlphaBlendOp(vk::BlendOp::eAdd);
        pcb.setColorWriteMask(
                vk::ColorComponentFlagBits::eR |
                vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA);

        for(auto& id : renderPass.subPass[info.subPass])
        {
            auto& attachment = renderPass.attachments[id];
            if(guessImageAspectFlags(attachment.format) == vk::ImageAspectFlagBits::eColor)
                colorBlendAttachments.push_back(pcb);
        }

        vk::PipelineColorBlendStateCreateInfo pbs;
        pbs.setLogicOpEnable(VK_FALSE);
        pbs.setLogicOp(vk::LogicOp::eClear);
        pbs.setAttachments(colorBlendAttachments);

        // DYNAMIC STATE
        std::vector<vk::DynamicState> dynamicStates =
        {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo pdy(vk::PipelineDynamicStateCreateFlags(), dynamicStates);

        // PIPELINE LAYOUT STATE
        auto layoutInfo = vk::PipelineLayoutCreateInfo();
        std::vector<vk::PushConstantRange> pushConstants;
        for(auto& shader : shaders)
            pushConstants.insert(pushConstants.end(), shader->pushConstants.begin(), shader->pushConstants.end());
        layoutInfo.setPushConstantRanges(pushConstants);

        // SHADER REFLECT
        vk::PipelineVertexInputStateCreateInfo pvi;
        for(auto& shader : shaders)
        {
            if(shader->stageFlagBits == vk::ShaderStageFlagBits::eVertex)
                pvi = shader->pvi;
            if(shader->stageFlagBits == vk::ShaderStageFlagBits::eFragment)
            {
                for(auto& e : shader->descriptorMap)
                {
                    for(auto& bind : e.second.bindings)
                        if(bind.descriptorCount == 0 && bind.descriptorType == vk::DescriptorType::eCombinedImageSampler)
                            bind.descriptorCount = info.textureCount;
                }
            }
        }

        pipeline->reflectPipelineLayout(m_device, shaders);

        auto pipelineInfo = vk::GraphicsPipelineCreateInfo();
        pipelineInfo.setRenderPass(renderPass.handle.get());
        pipelineInfo.setLayout(pipeline->pipelineLayout.get());
        pipelineInfo.setStages(pipelineShaderStages);
        pipelineInfo.setPVertexInputState(&pvi);
        pipelineInfo.setPInputAssemblyState(&pia);
        pipelineInfo.setPViewportState(&pv);
        pipelineInfo.setPRasterizationState(&pr);
        pipelineInfo.setPMultisampleState(&pm);
        pipelineInfo.setPDepthStencilState(&pds);
        pipelineInfo.setPColorBlendState(&pbs);
        pipelineInfo.setPDynamicState(&pdy);
        pipelineInfo.setSubpass(info.subPass);

        auto res = m_device.createGraphicsPipelineUnique(m_pipelineCache, pipelineInfo);
        assert(res.result == vk::Result::eSuccess);
        pipeline->handle = std::move(res.value);
        return pipeline;
    }

    PipelinePtr LerContext::createComputePipeline(const ShaderPtr& shader)
    {
        auto pipeline = std::make_shared<ComputePipeline>();
        std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStages;
        addShaderStage(pipelineShaderStages, shader);

        std::vector<ShaderPtr> shaders = { shader };
        pipeline->reflectPipelineLayout(m_device, shaders);

        auto pipelineInfo = vk::ComputePipelineCreateInfo();
        pipelineInfo.setStage(pipelineShaderStages.front());
        pipelineInfo.setLayout(pipeline->pipelineLayout.get());

        auto res = m_device.createComputePipelineUnique(m_pipelineCache, pipelineInfo);
        pipeline->bindPoint = vk::PipelineBindPoint::eCompute;
        assert(res.result == vk::Result::eSuccess);
        pipeline->handle = std::move(res.value);
        return pipeline;
    }

    void LerContext::updateSampler(vk::DescriptorSet descriptorSet, uint32_t binding, vk::Sampler& sampler, const std::vector<TexturePtr>& textures)
    {
        std::vector<vk::WriteDescriptorSet> descriptorWrites;
        std::vector<vk::DescriptorImageInfo> descriptorImageInfo;

        auto descriptorWriteInfo = vk::WriteDescriptorSet();
        descriptorWriteInfo.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
        descriptorWriteInfo.setDstBinding(binding);
        descriptorWriteInfo.setDstSet(descriptorSet);
        descriptorWriteInfo.setDescriptorCount(textures.size());

        for(auto& tex : textures)
        {
            auto& imageInfo = descriptorImageInfo.emplace_back();
            imageInfo = vk::DescriptorImageInfo();
            imageInfo.setSampler(sampler);
            imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
            if(tex)
                imageInfo.setImageView(tex->view.get());
        }

        descriptorWriteInfo.setImageInfo(descriptorImageInfo);
        descriptorWrites.push_back(descriptorWriteInfo);
        m_device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    void LerContext::updateStorage(vk::DescriptorSet descriptorSet, uint32_t binding, const Buffer& buffer, uint64_t byteSize)
    {
        std::vector<vk::WriteDescriptorSet> descriptorWrites;

        auto descriptorWriteInfo = vk::WriteDescriptorSet();
        descriptorWriteInfo.setDescriptorType(vk::DescriptorType::eStorageBuffer);
        descriptorWriteInfo.setDstBinding(binding);
        descriptorWriteInfo.setDstSet(descriptorSet);
        descriptorWriteInfo.setDescriptorCount(1);

        vk::DescriptorBufferInfo buffInfo(buffer.handle, 0, byteSize);

        descriptorWriteInfo.setBufferInfo(buffInfo);
        descriptorWrites.push_back(descriptorWriteInfo);
        m_device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    void LerContext::updateAttachment(vk::DescriptorSet descriptorSet, uint32_t binding, const TexturePtr& texture)
    {
        std::vector<vk::WriteDescriptorSet> descriptorWrites;
        std::vector<vk::DescriptorImageInfo> descriptorImageInfo;

        auto descriptorWriteInfo = vk::WriteDescriptorSet();
        descriptorWriteInfo.setDescriptorType(vk::DescriptorType::eInputAttachment);
        descriptorWriteInfo.setDstBinding(binding);
        descriptorWriteInfo.setDstSet(descriptorSet);
        descriptorWriteInfo.setDescriptorCount(1);

        vk::DescriptorImageInfo imageInfo;
        imageInfo.setSampler(nullptr);
        imageInfo.setImageView(texture->view.get());
        imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

        descriptorWriteInfo.setImageInfo(imageInfo);
        descriptorWrites.push_back(descriptorWriteInfo);
        m_device.updateDescriptorSets(descriptorWrites, nullptr);
    }

    TexturePtr LerContext::loadTextureFromFile(const fs::path& path)
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
        return texture;
    }

    TexturePtr LerContext::loadTextureFromMemory(const unsigned char* buffer, uint32_t size)
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
        return texture;
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

    uint32_t LerContext::loadTexture(Scene& scene, const aiScene* aiScene, const aiString& filename, const fs::path& path)
    {
        const auto* key = filename.C_Str();
        if(m_cache.contains(key))
            return m_cache.at(key);

        TexturePtr tex;
        auto em = aiScene->GetEmbeddedTexture(key);
        std::cout << "Load image : " << key << std::endl;
        if(em == nullptr)
        {
            fs::path f = path.parent_path() / fs::path(key);
            tex = loadTextureFromFile(f);
        }
        else
        {
            const auto* buffer = reinterpret_cast<const unsigned char*>(em->pcData);
            tex = loadTextureFromMemory(buffer, em->mWidth);
        }

        m_cache.emplace(key, scene.textures.size());
        scene.textures.emplace_back(tex);
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

    void addLine(std::vector<glm::vec3>& lines, const glm::vec3& p1, const glm::vec3& p2)
    {
        lines.push_back(p1);
        lines.push_back(p2);
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
                materialInstance.texId = loadTexture(scene, aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_AMBIENT) > 0)
            {
                material->GetTexture(aiTextureType_AMBIENT, 0, &filename);
                materialInstance.texId = loadTexture(scene, aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
            {
                material->GetTexture(aiTextureType_DIFFUSE, 0, &filename);
                materialInstance.texId = loadTexture(scene, aiScene, filename, path);
            }
            if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
            {
                material->GetTexture(aiTextureType_NORMALS, 0, &filename);
                materialInstance.norId = loadTexture(scene, aiScene, filename, path);
            }

            scene.materials.push_back(materialInstance);
        }

        scene.matCount = aiScene->mNumMaterials;
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

        std::vector<glm::vec3> lines;
        lines.reserve(5000);
        for(const auto& obj : scene.instances)
        {
            std::array<glm::vec3, 8> pts = {
                glm::vec3(obj.bMax.x, obj.bMax.y, obj.bMax.z),
                glm::vec3(obj.bMax.x, obj.bMax.y, obj.bMin.z),
                glm::vec3(obj.bMax.x, obj.bMin.y, obj.bMax.z),
                glm::vec3(obj.bMax.x, obj.bMin.y, obj.bMin.z),
                glm::vec3(obj.bMin.x, obj.bMax.y, obj.bMax.z),
                glm::vec3(obj.bMin.x, obj.bMax.y, obj.bMin.z),
                glm::vec3(obj.bMin.x, obj.bMin.y, obj.bMax.z),
                glm::vec3(obj.bMin.x, obj.bMin.y, obj.bMin.z),
            };

            addLine(lines, pts[0], pts[1]);
            addLine(lines, pts[2], pts[3]);
            addLine(lines, pts[4], pts[5]);
            addLine(lines, pts[6], pts[7]);

            addLine(lines, pts[0], pts[2]);
            addLine(lines, pts[1], pts[3]);
            addLine(lines, pts[4], pts[6]);
            addLine(lines, pts[5], pts[7]);

            addLine(lines, pts[0], pts[4]);
            addLine(lines, pts[1], pts[5]);
            addLine(lines, pts[2], pts[6]);
            addLine(lines, pts[3], pts[7]);
        }

        byteSize = lines.size()*sizeof(glm::vec3);
        scene.aabbBuffer = createBuffer(byteSize, vk::BufferUsageFlagBits::eVertexBuffer);
        uploadBuffer(scene.staging, lines.data(), byteSize);
        copyBuffer(scene.staging, scene.aabbBuffer, byteSize);
        scene.lineCount = lines.size();

        return scene;
    }

    void LerContext::destroyScene(const Scene& scene)
    {
        m_allocator.destroyBuffer(scene.staging.handle, scene.staging.allocation);
        m_allocator.destroyBuffer(scene.aabbBuffer.handle, scene.aabbBuffer.allocation);
        m_allocator.destroyBuffer(scene.indexBuffer.handle, scene.indexBuffer.allocation);
        m_allocator.destroyBuffer(scene.vertexBuffer.handle, scene.vertexBuffer.allocation);
        m_allocator.destroyBuffer(scene.normalBuffer.handle, scene.normalBuffer.allocation);
        m_allocator.destroyBuffer(scene.tangentBuffer.handle, scene.tangentBuffer.allocation);
        m_allocator.destroyBuffer(scene.texcoordBuffer.handle, scene.texcoordBuffer.allocation);
        m_allocator.destroyBuffer(scene.indirectBuffer.handle, scene.indirectBuffer.allocation);
        m_allocator.destroyBuffer(scene.instanceBuffer.handle, scene.instanceBuffer.allocation);
        m_allocator.destroyBuffer(scene.materialBuffer.handle, scene.materialBuffer.allocation);
    }
}