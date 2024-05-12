
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <cinttypes>

#include <vulkan/vk_layer.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/layer/vk_layer_settings.hpp>
#include <vulkan/utility/vk_safe_struct.hpp>
#include <vulkan/utility/vk_concurrent_unordered_map.hpp>
#include <vulkan/utility/vk_small_containers.hpp>
#include <vulkan/utility/vk_format_utils.h>

#include "allocator.h"
#include "log.h"
#include "vk_util.h"
#include "vk_api_hash.h"

#define kLayerSettingsForceEnable "force_enable"

namespace {

struct AttachmentSignature final {
    VkFormat format;
    VkSampleCountFlagBits samples;
    VkAttachmentLoadOp loadOp;
    VkAttachmentStoreOp storeOp;
    VkAttachmentLoadOp stencilLoadOp;
    VkAttachmentStoreOp stencilStoreOp;

    friend bool operator==(AttachmentSignature const& lhs, AttachmentSignature const& rhs) {
        return lhs.format == rhs.format && lhs.samples == rhs.samples && lhs.loadOp == rhs.loadOp && lhs.storeOp == rhs.storeOp &&
               lhs.stencilLoadOp == rhs.stencilLoadOp && lhs.stencilStoreOp == rhs.stencilStoreOp;
    }

    struct Hasher final {
        size_t operator()(AttachmentSignature const& v) const {
            size_t res = 17;
            APPEND_HASH(v.format);
            APPEND_HASH(v.samples);
            APPEND_HASH(v.loadOp);
            APPEND_HASH(v.storeOp);
            APPEND_HASH(v.stencilLoadOp);
            APPEND_HASH(v.stencilStoreOp);
            return res;
        }
    };

    // Hasher for pipeline creation that ignores values that don't affect render pass compatibility.
    struct PipelineHasher final {
        size_t operator()(AttachmentSignature const& v) const {
            size_t res = 17;
            APPEND_HASH(v.format);
            APPEND_HASH(v.samples);
            return res;
        }
    };

    // EqualTo for pipeline creation that ignores values that don't affect render pass compatibility.
    struct PipelineEqualTo final {
        bool operator()(AttachmentSignature const& lhs, AttachmentSignature const& rhs) const {
            return lhs.format == rhs.format && lhs.samples == rhs.samples;
        }
    };
};

struct SubpassDesc final {
    vku::small::vector<uint32_t, 1> input_attachments;
    vku::small::vector<uint32_t, 4> color_attachments;
    vku::small::vector<uint32_t, 1> color_resolve_attachments;
    std::optional<uint32_t> depth_stencil_attachment;
    std::optional<uint32_t> depth_stencil_resolve_attachment;

    friend bool operator==(SubpassDesc const& lhs, SubpassDesc const& rhs) {
        return lhs.input_attachments == rhs.input_attachments && lhs.color_attachments == rhs.color_attachments &&
               lhs.color_resolve_attachments == rhs.color_resolve_attachments &&
               lhs.depth_stencil_attachment == rhs.depth_stencil_attachment &&
               lhs.depth_stencil_resolve_attachment == rhs.depth_stencil_resolve_attachment;
    }

    struct Hasher final {
        size_t operator()(SubpassDesc const& v) const {
            size_t res = 17;
            APPEND_HASH_ARRAY(v.input_attachments.data(), v.input_attachments.size(), v.input_attachments.size());
            APPEND_HASH_ARRAY(v.color_attachments.data(), v.color_attachments.size(), v.color_attachments.size());
            APPEND_HASH_ARRAY(v.color_resolve_attachments.data(), v.color_resolve_attachments.size(),
                              v.color_resolve_attachments.size());
            if (v.depth_stencil_attachment.has_value()) {
                APPEND_HASH(v.depth_stencil_attachment.value());
            }
            if (v.depth_stencil_resolve_attachment.has_value()) {
                APPEND_HASH(v.depth_stencil_resolve_attachment.value());
            }
            return res;
        }
    };

    using PipelineHasher = Hasher;
};

struct RenderPassDesc final {
    vku::small::vector<AttachmentSignature, 4> attachment_signatures;
    vku::small::vector<SubpassDesc, 1> subpass_descs;

    friend bool operator==(RenderPassDesc const& lhs, RenderPassDesc const& rhs) {
        return lhs.attachment_signatures == rhs.attachment_signatures && lhs.subpass_descs == rhs.subpass_descs;
    }

    struct Hasher final {
        size_t operator()(RenderPassDesc const& v) const {
            size_t res = 17;
            APPEND_HASH(v.attachment_signatures.size());
            for (AttachmentSignature const& attachment_signature : v.attachment_signatures) {
                res = res * 31 + AttachmentSignature::Hasher()(attachment_signature);
            }
            APPEND_HASH(v.subpass_descs.size());
            for (auto const& subpass : v.subpass_descs) {
                res = res * 31 + SubpassDesc::Hasher()(subpass);
            }
            return res;
        }
    };

    // Hasher for pipeline creation that ignores values that don't affect render pass compatibility.
    struct PipelineHasher final {
        size_t operator()(RenderPassDesc const& v) const {
            size_t res = 17;
            APPEND_HASH(v.attachment_signatures.size());
            for (AttachmentSignature const& attachment_signature : v.attachment_signatures) {
                res = res * 31 + AttachmentSignature::PipelineHasher()(attachment_signature);
            }
            APPEND_HASH(v.subpass_descs.size());
            for (auto const& subpass : v.subpass_descs) {
                res = res * 31 + SubpassDesc::PipelineHasher()(subpass);
            }
            return res;
        }
    };

    // EqualTo for pipeline creation that ignores values that don't affect render pass compatibility.
    struct PipelineEqualTo final {
        bool operator()(RenderPassDesc const& lhs, RenderPassDesc const& rhs) const {
            bool attachments_equal = std::equal(lhs.attachment_signatures.begin(), lhs.attachment_signatures.end(),
                                                rhs.attachment_signatures.begin(), rhs.attachment_signatures.end(),
                                                [](AttachmentSignature const& lhs, AttachmentSignature const& rhs) -> bool {
                                                    return AttachmentSignature::PipelineEqualTo()(lhs, rhs);
                                                });
            return attachments_equal && lhs.subpass_descs == rhs.subpass_descs;
        }
    };
};

struct FramebufferDesc final {
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkRect2D render_area = {{0, 0}, {0, 0}};
    vku::small::vector<VkImageView, 4> attachments;

    friend bool operator==(FramebufferDesc const& lhs, FramebufferDesc const& rhs) {
        return lhs.render_pass == rhs.render_pass && lhs.render_area.offset.x == rhs.render_area.offset.x &&
               lhs.render_area.offset.y == rhs.render_area.offset.y &&
               lhs.render_area.extent.width == rhs.render_area.extent.width &&
               lhs.render_area.extent.height == rhs.render_area.extent.height && lhs.attachments == rhs.attachments;
    }

    struct Hasher final {
        size_t operator()(FramebufferDesc const& v) const {
            size_t res = 17;
            APPEND_HASH(v.render_pass);
            APPEND_HASH(v.render_area);
            APPEND_HASH_ARRAY(v.attachments.data(), v.attachments.size(), v.attachments.size());
            return res;
        }
    };
};

class RenderPassPool final {
  public:
    VkRenderPass FindOrAddForRenderPassInstance(RenderPassDesc const& desc, PFN_vkCreateRenderPass create_render_pass,
                                                VkDevice device) {
        std::lock_guard lock(mutex_);

        return FindOrAdd<false>(desc, create_render_pass, device);
    }
    VkRenderPass FindOrAddForPipeline(RenderPassDesc const& desc, PFN_vkCreateRenderPass create_render_pass, VkDevice device) {
        std::lock_guard lock(mutex_);

        return FindOrAdd<true>(desc, create_render_pass, device);
    }

    void Cleanup(PFN_vkDestroyRenderPass destroy_render_pass, VkDevice device) {
        std::lock_guard lock(mutex_);

        for (auto const& [outer_desc, inner_pool] : outer_pool_) {
            for (auto const& [inner_desc, render_pass] : inner_pool) {
                destroy_render_pass(device, render_pass, nullptr);
            }
        }

        outer_pool_.clear();
    }

  private:
    // `mutex_` MUST be locked.
    template <bool ForPipeline>
    VkRenderPass FindOrAdd(RenderPassDesc const& desc, PFN_vkCreateRenderPass create_render_pass, VkDevice device) {
        InnerPool& inner_pool = outer_pool_[desc];

        if constexpr (ForPipeline) {
            // For pipelines, any `VkRenderPass` that is compatible according to render pass compatibility is fine.
            // i.e. load/store ops don't matter.
            if (!inner_pool.empty()) {
                return inner_pool.begin()->second;
            }
        }

        // For render pass instances, we need exact load/store ops.
        auto [it, inserted] = inner_pool.try_emplace(desc);
        if (inserted) {
            it->second = CreateVkRenderPass(desc, create_render_pass, device);
        }
        return it->second;
    }

    static VkRenderPass CreateVkRenderPass(RenderPassDesc const& desc, PFN_vkCreateRenderPass create_render_pass, VkDevice device) {
        // TODO: If we support depth/stencil resolve, we will be relying on `VK_KHR_depth_stencil_resolve` and hence
        // `VK_KHR_create_renderpass2`.

        // External subpass dependencies such that explicit barriers are always required.
        VkSubpassDependency subpass_dependencies[2];
        {
            VkSubpassDependency& dst = subpass_dependencies[0];
            dst.srcSubpass = VK_SUBPASS_EXTERNAL;
            dst.dstSubpass = 0;
            dst.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dst.dependencyFlags = 0;
        }
        {
            VkSubpassDependency& dst = subpass_dependencies[1];
            dst.srcSubpass = desc.subpass_descs.size() - 1;
            dst.dstSubpass = VK_SUBPASS_EXTERNAL;
            dst.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dst.dependencyFlags = 0;
        }

        std::vector<VkAttachmentDescription> attachment_descriptions;
        for (AttachmentSignature const& attachment_signature : desc.attachment_signatures) {
            VkAttachmentDescription& dst = attachment_descriptions.emplace_back();
            dst.flags = 0;
            dst.format = attachment_signature.format;
            dst.samples = attachment_signature.samples;
            dst.loadOp = attachment_signature.loadOp;
            dst.storeOp = attachment_signature.storeOp;
            dst.stencilLoadOp = attachment_signature.stencilLoadOp;
            dst.stencilStoreOp = attachment_signature.stencilStoreOp;
            // TODO: Support for other layouts:
            // VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
            // VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
            // VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
            //
            // VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
            // VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
            //
            // VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
            // VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
            if (vkuFormatIsColor(attachment_signature.format)) {
                dst.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                dst.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            } else {
                dst.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                dst.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
        }

        struct SubpassStorage final {
            std::vector<VkAttachmentReference> input_attachment_refs;
            std::vector<VkAttachmentReference> color_attachment_refs;
            std::optional<VkAttachmentReference> depth_stencil_attachment_ref;
            std::vector<VkAttachmentReference> color_resolve_attachment_refs;
        };
        std::vector<SubpassStorage> subpass_storages;
        subpass_storages.reserve(desc.subpass_descs.size());

        std::vector<VkSubpassDescription> subpass_descriptions;
        for (SubpassDesc const& subpass_desc : desc.subpass_descs) {
            SubpassStorage& storage = subpass_storages.emplace_back();
            for (uint32_t input_attachment : subpass_desc.input_attachments) {
                VkAttachmentReference& dst = storage.input_attachment_refs.emplace_back();
                dst.attachment = input_attachment;
                dst.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            for (uint32_t color_attachment : subpass_desc.color_attachments) {
                VkAttachmentReference& dst = storage.color_attachment_refs.emplace_back();
                dst.attachment = color_attachment;
                dst.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            if (subpass_desc.depth_stencil_attachment.has_value()) {
                VkAttachmentReference& dst = storage.depth_stencil_attachment_ref.emplace();
                dst.attachment = subpass_desc.depth_stencil_attachment.value();
                dst.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
            for (uint32_t color_resolve_attachment : subpass_desc.color_resolve_attachments) {
                VkAttachmentReference& dst = storage.color_resolve_attachment_refs.emplace_back();
                dst.attachment = color_resolve_attachment;
                dst.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }

            VkSubpassDescription& dst = subpass_descriptions.emplace_back();
            dst.flags = 0;
            dst.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            dst.inputAttachmentCount = uint32_t(storage.input_attachment_refs.size());
            dst.pInputAttachments = storage.input_attachment_refs.data();
            dst.colorAttachmentCount = uint32_t(storage.color_attachment_refs.size());
            dst.pColorAttachments = storage.color_attachment_refs.data();
            dst.pResolveAttachments =
                storage.color_attachment_refs.empty() ? nullptr : storage.color_resolve_attachment_refs.data();
            dst.preserveAttachmentCount = 0;
            dst.pPreserveAttachments = nullptr;
        }

        VkRenderPassCreateInfo create_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        create_info.pNext = nullptr;
        create_info.flags = 0;
        create_info.attachmentCount = uint32_t(attachment_descriptions.size());
        create_info.pAttachments = attachment_descriptions.data();
        create_info.subpassCount = uint32_t(subpass_descriptions.size());
        create_info.pSubpasses = subpass_descriptions.data();
        create_info.dependencyCount = uint32_t(std::size(subpass_dependencies));
        create_info.pDependencies = subpass_dependencies;

        VkRenderPass render_pass;
        VkResult result = create_render_pass(device, &create_info, nullptr, &render_pass);
        RELEASE_ASSERT(result == VK_SUCCESS);

        return render_pass;
    }

    using InnerPool = std::unordered_map<RenderPassDesc, VkRenderPass, RenderPassDesc::Hasher>;
    using OuterPool =
        std::unordered_map<RenderPassDesc, InnerPool, RenderPassDesc::PipelineHasher, RenderPassDesc::PipelineEqualTo>;

    std::mutex mutex_;
    OuterPool outer_pool_;
};

class FramebufferPool final {
  public:
    VkFramebuffer FindOrAdd(FramebufferDesc const& desc, PFN_vkCreateFramebuffer create_framebuffer, VkDevice device) {
        std::lock_guard lock(mutex_);

        auto [it, inserted] = framebuffers_.try_emplace(desc);
        if (!inserted) {
            return it->second;
        }

        VkFramebufferCreateInfo create_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        create_info.pNext = nullptr;
        create_info.flags = 0;
        create_info.renderPass = desc.render_pass;
        create_info.attachmentCount = desc.attachments.size();
        create_info.pAttachments = desc.attachments.data();
        create_info.width = desc.render_area.extent.width;
        create_info.height = desc.render_area.extent.height;
        create_info.layers = 1;  // TODO: layers ?

        VkFramebuffer framebuffer;
        VkResult result = create_framebuffer(device, &create_info, nullptr, &framebuffer);
        RELEASE_ASSERT(result == VK_SUCCESS);

        it->second = framebuffer;
        return framebuffer;
    }

    void Cleanup(PFN_vkDestroyFramebuffer destroy_framebuffer, VkDevice device) {
        std::lock_guard lock(mutex_);

        for (auto const& [desc, framebuffer] : framebuffers_) {
            destroy_framebuffer(device, framebuffer, nullptr);
        }
        framebuffers_.clear();
    }

  private:
    std::mutex mutex_;
    std::unordered_map<FramebufferDesc, VkFramebuffer, FramebufferDesc::Hasher> framebuffers_;
};

}  // namespace

namespace {

struct ImageData {
    VkFormat format;
    VkSampleCountFlagBits samples;
};

struct ImageViewData {
    VkImage image;
};

struct SwapchainData {
    VkFormat format;
    std::unordered_set<VkImage> images;
};

}  // namespace

namespace dynamic_rendering {

static const VkLayerProperties kGlobalLayer = {
    "VK_LAYER_KHRONOS_dynamic_rendering",
    VK_HEADER_VERSION_COMPLETE,
    1,
    "Khronos dynamic rendering layer",
};

static const VkExtensionProperties kDeviceExtension = {VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                                                       VK_KHR_DYNAMIC_RENDERING_SPEC_VERSION};

struct LayerSettings {
    bool force_enable{false};
};

struct PhysicalDeviceData {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    bool lower_has_dynamic_rendering = false;
    uint32_t api_version;
};

#define DECLARE_HOOK(fn) PFN_vk##fn fn
struct InstanceData {
    InstanceData(VkInstance instance, PFN_vkGetInstanceProcAddr gpa, const VkAllocationCallbacks* alloc);
    InstanceData() = delete;
    InstanceData(const InstanceData&) = delete;
    InstanceData& operator=(const InstanceData&) = delete;

    std::shared_ptr<PhysicalDeviceData> GetPhysicalDeviceData(VkPhysicalDevice physical_device) const {
        const auto result = physical_device_map.find(physical_device);
        if (result == physical_device_map.end()) {
            return nullptr;
        }
        return result->second;
    }

    VkInstance instance;
    uint32_t api_version;
    LayerSettings layer_settings;
    const VkAllocationCallbacks* allocator;
    struct InstanceDispatchTable {
        DECLARE_HOOK(GetInstanceProcAddr);
        DECLARE_HOOK(CreateInstance);
        DECLARE_HOOK(DestroyInstance);
        DECLARE_HOOK(CreateDevice);
        DECLARE_HOOK(EnumeratePhysicalDevices);
        DECLARE_HOOK(EnumerateDeviceExtensionProperties);
        DECLARE_HOOK(GetPhysicalDeviceFeatures2);
        DECLARE_HOOK(GetPhysicalDeviceFeatures2KHR);
        DECLARE_HOOK(GetPhysicalDeviceProperties);
    } vtable;

    vku::concurrent::unordered_map<VkPhysicalDevice, std::shared_ptr<PhysicalDeviceData>> physical_device_map;
};

struct DeviceFeatures {
    DeviceFeatures(uint32_t api_version, const VkDeviceCreateInfo* create_info);
    DeviceFeatures() : dynamic_rendering(false) {}

    bool dynamic_rendering;
};

struct DeviceData {
    DeviceData(VkDevice device, PFN_vkGetDeviceProcAddr gpa, const DeviceFeatures& feat, bool enable_layer,
               const VkAllocationCallbacks* alloc);
    DeviceData() = delete;
    DeviceData(const DeviceData&) = delete;
    DeviceData& operator=(const DeviceData&) = delete;

    VkDevice device;
    const VkAllocationCallbacks* allocator;
    DeviceFeatures features;
    bool enable_layer;
    uint32_t api_version;
    vku::concurrent::unordered_map<VkImage, ImageData> image_map;
    vku::concurrent::unordered_map<VkImageView, ImageViewData> image_view_map;
    vku::concurrent::unordered_map<VkSwapchainKHR, SwapchainData> swapchain_map;
    RenderPassPool render_pass_pool;
    FramebufferPool framebuffer_pool;
    struct DeviceDispatchTable {
        DECLARE_HOOK(GetDeviceProcAddr);
        DECLARE_HOOK(DestroyDevice);
        DECLARE_HOOK(CreateGraphicsPipelines);
        DECLARE_HOOK(DestroyPipeline);
        DECLARE_HOOK(CreateImage);
        DECLARE_HOOK(DestroyImage);
        DECLARE_HOOK(CreateImageView);
        DECLARE_HOOK(DestroyImageView);
        DECLARE_HOOK(CreateSwapchainKHR);
        DECLARE_HOOK(DestroySwapchainKHR);
        DECLARE_HOOK(GetSwapchainImagesKHR);
        DECLARE_HOOK(CreateRenderPass);
        DECLARE_HOOK(DestroyRenderPass);
        DECLARE_HOOK(CreateFramebuffer);
        DECLARE_HOOK(DestroyFramebuffer);
        DECLARE_HOOK(CmdBeginRenderPass);
        DECLARE_HOOK(CmdEndRenderPass);
    } vtable;
};
#undef DECLARE_HOOK

#define INIT_HOOK(_vt, _inst, fn) _vt.fn = reinterpret_cast<PFN_vk##fn>(vtable.GetInstanceProcAddr(_inst, "vk" #fn))
InstanceData::InstanceData(VkInstance inst, PFN_vkGetInstanceProcAddr gpa, const VkAllocationCallbacks* alloc)
    : instance(inst), allocator(alloc) {
    vtable.GetInstanceProcAddr = gpa;
    INIT_HOOK(vtable, instance, CreateInstance);
    INIT_HOOK(vtable, instance, DestroyInstance);
    INIT_HOOK(vtable, instance, CreateDevice);
    INIT_HOOK(vtable, instance, EnumeratePhysicalDevices);
    INIT_HOOK(vtable, instance, EnumerateDeviceExtensionProperties);
    INIT_HOOK(vtable, instance, GetPhysicalDeviceFeatures2);
    INIT_HOOK(vtable, instance, GetPhysicalDeviceFeatures2KHR);
    INIT_HOOK(vtable, instance, GetPhysicalDeviceProperties);
}
#undef INIT_HOOK

static vku::concurrent::unordered_map<uintptr_t, std::shared_ptr<InstanceData>> instance_data_map;
static vku::concurrent::unordered_map<uintptr_t, std::shared_ptr<DeviceData>> device_data_map;

uintptr_t DispatchKey(const void* object) {
    auto tmp = reinterpret_cast<const struct VkLayerDispatchTable_* const*>(object);
    return reinterpret_cast<uintptr_t>(*tmp);
}

static std::shared_ptr<InstanceData> GetInstanceData(const void* object) {
    auto result = instance_data_map.find(DispatchKey(object));
    return result != instance_data_map.end() ? result->second : nullptr;
}

static std::shared_ptr<DeviceData> GetDeviceData(const void* object) {
    auto result = device_data_map.find(DispatchKey(object));
    return result != device_data_map.end() ? result->second : nullptr;
}

static VkLayerInstanceCreateInfo* GetChainInfo(const VkInstanceCreateInfo* pCreateInfo, VkLayerFunction func) {
    auto chain_info = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(pCreateInfo->pNext));
    while (chain_info && !(chain_info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain_info->function == func)) {
        chain_info = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(chain_info->pNext));
    }
    ASSERT(chain_info != NULL);
    return chain_info;
}

void InitLayerSettings(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                       LayerSettings* layer_settings) {
    assert(layer_settings != nullptr);

    const VkLayerSettingsCreateInfoEXT* create_info = vkuFindLayerSettingsCreateInfo(pCreateInfo);

    VkuLayerSettingSet layer_setting_set = VK_NULL_HANDLE;
    vkuCreateLayerSettingSet(kGlobalLayer.layerName, create_info, pAllocator, nullptr, &layer_setting_set);

    static const char* setting_names[] = {
        kLayerSettingsForceEnable,
    };
    uint32_t setting_name_count = static_cast<uint32_t>(std::size(setting_names));

    uint32_t unknown_setting_count = 0;
    vkuGetUnknownSettings(create_info, setting_name_count, setting_names, &unknown_setting_count, nullptr);

    if (unknown_setting_count > 0) {
        std::vector<const char*> unknown_settings;
        unknown_settings.resize(unknown_setting_count);

        vkuGetUnknownSettings(create_info, setting_name_count, setting_names, &unknown_setting_count, &unknown_settings[0]);

        for (std::size_t i = 0, n = unknown_settings.size(); i < n; ++i) {
            LOG("Unknown %s setting listed in VkLayerSettingsCreateInfoEXT, this setting is ignored.\n", unknown_settings[i]);
        }
    }

    if (vkuHasLayerSetting(layer_setting_set, kLayerSettingsForceEnable)) {
        vkuGetLayerSettingValue(layer_setting_set, kLayerSettingsForceEnable, layer_settings->force_enable);
    }

    vkuDestroyLayerSettingSet(layer_setting_set, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                                              VkInstance* pInstance) {
    VkLayerInstanceCreateInfo* chain_info = GetChainInfo(pCreateInfo, VK_LAYER_LINK_INFO);

    ASSERT(chain_info->u.pLayerInfo);
    auto gpa = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(gpa(NULL, "vkCreateInstance"));
    if (create_instance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = create_instance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) {
        return result;
    }
    try {
        auto instance_data =
            std::make_shared<InstanceData>(*pInstance, gpa, pAllocator ? pAllocator : &extension_layer::kDefaultAllocator);

        instance_data_map.insert(DispatchKey(*pInstance), instance_data);

        instance_data->api_version = pCreateInfo->pApplicationInfo ? pCreateInfo->pApplicationInfo->apiVersion : 0;

        InitLayerSettings(pCreateInfo, pAllocator, &instance_data->layer_settings);
    } catch (const std::bad_alloc&) {
        auto destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(gpa(NULL, "vkDestroyInstance"));
        destroy_instance(*pInstance, pAllocator);
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    auto key = DispatchKey(instance);
    auto result = instance_data_map.find(key);
    if (result != instance_data_map.end()) {
        auto instance_data = result->second;

        instance_data->vtable.DestroyInstance(instance, pAllocator);

        instance_data_map.erase(key);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount,
                                                        VkPhysicalDevice* pPhysicalDevices) {
    auto instance_data = GetInstanceData(instance);
    VkResult result =
        instance_data->vtable.EnumeratePhysicalDevices(instance_data->instance, pPhysicalDeviceCount, pPhysicalDevices);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && pPhysicalDevices != nullptr) {
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
            VkPhysicalDeviceProperties properties{};
            auto physical_device = pPhysicalDevices[i];

            if (instance_data->physical_device_map.find(physical_device) != instance_data->physical_device_map.end()) {
                continue;
            }
            auto pdd = std::make_shared<PhysicalDeviceData>();
            pdd->physical_device = physical_device;

            instance_data->vtable.GetPhysicalDeviceProperties(physical_device, &properties);
            pdd->api_version = properties.apiVersion;

            instance_data->physical_device_map.insert(physical_device, pdd);
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char* pLayerName,
                                                                  uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    // TODO: What about the required extensions?
    // VK_KHR_get_physical_device_properties2 or 1.1
    // VK_KHR_depth_stencil_resolve or 1.2
    // VK_KHR_create_renderpass2 or 1.2
    // VK_KHR_multiview or 1.1
    // VK_KHR_maintenance2 or 1.1

    // Not the extension we're implementing; delegate down the chain.
    if (pLayerName != nullptr && strncmp(pLayerName, kGlobalLayer.layerName, VK_MAX_EXTENSION_NAME_SIZE) != 0) {
        auto instance_data = GetInstanceData(physicalDevice);
        return instance_data->vtable.EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
    }

    if (pLayerName == nullptr) {
        uint32_t count = 0;
        auto instance_data = GetInstanceData(physicalDevice);
        instance_data->vtable.EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, &count, nullptr);

        // We need to check if the extension is natively supported in order to return the correct count into `pPropertyCount`.
        std::vector<VkExtensionProperties> properties(count);
        instance_data->vtable.EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, &count, properties.data());

        bool has_native_dynamic_rendering = false;
        for (uint32_t i = 0; i < count; ++i) {
            if (strncmp(properties[i].extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE) == 0) {
                has_native_dynamic_rendering = true;
                break;
            }
        }

        uint32_t total_count = !has_native_dynamic_rendering ? count + 1 : count;

        if (!pProperties) {
            *pPropertyCount = total_count;
            return VK_SUCCESS;
        }

        if (*pPropertyCount < total_count) {
            // Spec for `vkEnumerateDeviceLayerProperties` says at most `pPropertyCount` structures will be written in this case,
            // which is the convention we follow for `vkEnumerate*` functions.
            memcpy(pProperties, properties.data(), *pPropertyCount * sizeof(VkExtensionProperties));
            return VK_INCOMPLETE;
        }

        memcpy(pProperties, properties.data(), count * sizeof(VkExtensionProperties));

        if (!has_native_dynamic_rendering) {
            pProperties[count] = kDeviceExtension;
        }

        *pPropertyCount = total_count;

        return VK_SUCCESS;
    }

    // The extension we're implementing.
    VK_OUTARRAY_MAKE(out, pProperties, pPropertyCount);
    vk_outarray_append(&out, prop) { *prop = kDeviceExtension; }
    return vk_outarray_status(&out);
}

static void CheckPhysicalDeviceFeatures(PhysicalDeviceData& pdd, VkPhysicalDeviceFeatures2* pFeatures) {
    // Promoted to core in 1.3.
    if (VK_MAKE_VERSION(1, 3, 0) <= pdd.api_version) {
        pdd.lower_has_dynamic_rendering = true;
        return;
    }

    auto chain = reinterpret_cast<VkBaseInStructure*>(pFeatures->pNext);
    while (chain != nullptr) {
        if (chain->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR) {
            auto feature = reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeaturesKHR*>(chain);
            if (feature->dynamicRendering) {
                pdd.lower_has_dynamic_rendering = true;
            } else {
                pdd.lower_has_dynamic_rendering = false;
                feature->dynamicRendering = true;
            }
        }
        chain = const_cast<VkBaseInStructure*>(chain->pNext);
    }
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
    auto instance_data = GetInstanceData(physicalDevice);
    auto pdd = instance_data->GetPhysicalDeviceData(physicalDevice);

    if (instance_data->vtable.GetPhysicalDeviceFeatures2 != nullptr) {
        instance_data->vtable.GetPhysicalDeviceFeatures2(physicalDevice, pFeatures);
    }

    CheckPhysicalDeviceFeatures(*pdd, pFeatures);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures2KHR(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures) {
    auto instance_data = GetInstanceData(physicalDevice);
    auto pdd = instance_data->GetPhysicalDeviceData(physicalDevice);

    if (instance_data->vtable.GetPhysicalDeviceFeatures2KHR != nullptr) {
        instance_data->vtable.GetPhysicalDeviceFeatures2KHR(physicalDevice, pFeatures);
    }

    CheckPhysicalDeviceFeatures(*pdd, pFeatures);
}

#define INIT_HOOK(_vt, _dev, fn) _vt.fn = reinterpret_cast<PFN_vk##fn>(vtable.GetDeviceProcAddr(_dev, "vk" #fn))
DeviceData::DeviceData(VkDevice device, PFN_vkGetDeviceProcAddr gpa, const DeviceFeatures& feat, bool enable,
                       const VkAllocationCallbacks* alloc)
    : device(device), allocator(alloc), features(feat), enable_layer(enable) {
    vtable.GetDeviceProcAddr = gpa;
    if (enable_layer) {
        INIT_HOOK(vtable, device, DestroyDevice);
        INIT_HOOK(vtable, device, CreateGraphicsPipelines);
        INIT_HOOK(vtable, device, DestroyPipeline);
        INIT_HOOK(vtable, device, CreateImage);
        INIT_HOOK(vtable, device, DestroyImage);
        INIT_HOOK(vtable, device, CreateImageView);
        INIT_HOOK(vtable, device, DestroyImageView);
        INIT_HOOK(vtable, device, CreateSwapchainKHR);
        INIT_HOOK(vtable, device, DestroySwapchainKHR);
        INIT_HOOK(vtable, device, GetSwapchainImagesKHR);
        INIT_HOOK(vtable, device, CreateRenderPass);
        INIT_HOOK(vtable, device, DestroyRenderPass);
        INIT_HOOK(vtable, device, CreateFramebuffer);
        INIT_HOOK(vtable, device, DestroyFramebuffer);
        INIT_HOOK(vtable, device, CmdBeginRenderPass);
        INIT_HOOK(vtable, device, CmdEndRenderPass);
    }
}
#undef INIT_HOOK

static VkLayerDeviceCreateInfo* GetChainInfo(const VkDeviceCreateInfo* pCreateInfo, VkLayerFunction func) {
    auto chain_info = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(pCreateInfo->pNext));
    while (chain_info && !(chain_info->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain_info->function == func)) {
        chain_info = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(chain_info->pNext));
    }
    ASSERT(chain_info != NULL);
    return chain_info;
}

DeviceFeatures::DeviceFeatures(uint32_t api_version, const VkDeviceCreateInfo* create_info) : dynamic_rendering(false) {
    bool dynamic_rendering_feature_requested = false;
    for (auto chain = reinterpret_cast<VkBaseInStructure const*>(create_info->pNext); chain != nullptr; chain = chain->pNext) {
        switch (chain->sType) {
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR: {
                auto feature = reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeaturesKHR const*>(chain);
                dynamic_rendering_feature_requested = feature->dynamicRendering != VK_FALSE;
            } break;
            case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
                auto features = reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(chain);
                dynamic_rendering_feature_requested = features->dynamicRendering != VK_FALSE;
            } break;
            default:
                break;
        }
    }

    bool dynamic_rendering_extension_requested = false;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++) {
        if (strcmp(create_info->ppEnabledExtensionNames[i], VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0) {
            dynamic_rendering_extension_requested = true;
            break;
        }
    }

    this->dynamic_rendering =
        dynamic_rendering_feature_requested && (dynamic_rendering_extension_requested || VK_MAKE_VERSION(1, 3, 0) <= api_version);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
                                            const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    VkResult result;
    auto instance_data = GetInstanceData(physicalDevice);
    auto pdd = instance_data->GetPhysicalDeviceData(physicalDevice);

    VkLayerDeviceCreateInfo* chain_info = GetChainInfo(pCreateInfo, VK_LAYER_LINK_INFO);

    ASSERT(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr instance_proc_addr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateDevice create_device = (PFN_vkCreateDevice)instance_proc_addr(instance_data->instance, "vkCreateDevice");
    PFN_vkGetDeviceProcAddr gdpa = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    if (instance_data->vtable.CreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    uint32_t effective_api_version =
        (instance_data->api_version != 0) ? std::min(instance_data->api_version, pdd->api_version) : pdd->api_version;

    DeviceFeatures features(effective_api_version, pCreateInfo);

    try {
        bool enable_layer =
            (features.dynamic_rendering && (!pdd->lower_has_dynamic_rendering || instance_data->layer_settings.force_enable));
        // Filter out our extension name and feature struct, in a copy of the create info.
        // Only enable device hooks if dynamic rendering extension is enabled AND
        // the physical device doesn't support it already or we are force enabled.
        if (enable_layer) {
            vku::safe_VkDeviceCreateInfo create_info(pCreateInfo);
            vku::RemoveExtension(create_info, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            vku::RemoveFromPnext(create_info, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR);

            for (auto chain = reinterpret_cast<VkBaseInStructure*>(const_cast<void*>(create_info.pNext)); chain;
                 chain = const_cast<VkBaseInStructure*>(chain->pNext)) {
                switch (chain->sType) {
                    case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
                        auto vulkan_13_features = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(chain);
                        vulkan_13_features->dynamicRendering = VK_FALSE;
                    } break;
                    default:
                        break;
                }
            }

            result = create_device(physicalDevice, create_info.ptr(), pAllocator, pDevice);

            LOG("%s enabled.\n", kGlobalLayer.layerName);
        } else {
            result = create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);

            LOG("%s is dormant.\n", kGlobalLayer.layerName);
        }

        if (result != VK_SUCCESS) {
            return result;
        }
        auto alloccb = pAllocator ? pAllocator : instance_data->allocator;
        auto device_data = std::make_shared<DeviceData>(*pDevice, gdpa, features, enable_layer, alloccb);

        device_data_map.insert(DispatchKey(*pDevice), device_data);
    } catch (const std::bad_alloc&) {
        auto destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(gdpa(*pDevice, "vkDestroyDevice"));
        destroy_device(*pDevice, pAllocator);
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    auto key = DispatchKey(device);
    auto result = device_data_map.find(key);
    if (result != device_data_map.end()) {
        auto device_data = result->second;

        device_data->vtable.DestroyDevice(device, pAllocator);

        device_data->framebuffer_pool.Cleanup(device_data->vtable.DestroyFramebuffer, device);
        device_data->render_pass_pool.Cleanup(device_data->vtable.DestroyRenderPass, device);

        device_data_map.erase(key);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo,
                                           const VkAllocationCallbacks* pAllocator, VkImage* pImage) {
    auto device_data = GetDeviceData(device);
    VkResult result = device_data->vtable.CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (result == VK_SUCCESS) {
        ImageData image_data{pCreateInfo->format, pCreateInfo->samples};
        device_data->image_map.insert(*pImage, image_data);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator) {
    auto device_data = GetDeviceData(device);
    device_data->vtable.DestroyImage(device, image, pAllocator);
    device_data->image_map.erase(image);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator, VkImageView* pView) {
    auto device_data = GetDeviceData(device);
    VkResult result = device_data->vtable.CreateImageView(device, pCreateInfo, pAllocator, pView);
    if (result == VK_SUCCESS) {
        ImageViewData image_view_data{pCreateInfo->image};
        device_data->image_view_map.insert(*pView, image_view_data);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator) {
    auto device_data = GetDeviceData(device);
    device_data->vtable.DestroyImageView(device, imageView, pAllocator);
    device_data->image_view_map.erase(imageView);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
                                                  const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    auto device_data = GetDeviceData(device);
    VkResult result = device_data->vtable.CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    if (result == VK_SUCCESS) {
        SwapchainData swapchain_data{pCreateInfo->imageFormat};
        device_data->swapchain_map.insert(*pSwapchain, swapchain_data);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    auto device_data = GetDeviceData(device);
    device_data->vtable.DestroySwapchainKHR(device, swapchain, pAllocator);
    auto iter = device_data->swapchain_map.find(swapchain);
    if (iter != device_data->swapchain_map.end()) {
        for (auto image : iter->second.images) {
            device_data->image_map.erase(image);
        }
        device_data->swapchain_map.erase(swapchain);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount,
                                                     VkImage* pSwapchainImages) {
    auto device_data = GetDeviceData(device);
    VkResult result = device_data->vtable.GetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, pSwapchainImages);
    if (result == VK_SUCCESS && pSwapchainImages) {
        auto iter = device_data->swapchain_map.find(swapchain);
        if (iter != device_data->swapchain_map.end()) {
            ImageData image_data{iter->second.format, VK_SAMPLE_COUNT_1_BIT};
            for (uint32_t i = 0; i < *pSwapchainImageCount; i++) {
                device_data->image_map.insert(pSwapchainImages[i], image_data);
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount,
                                                       const VkGraphicsPipelineCreateInfo* pCreateInfos,
                                                       const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines) {
    auto device_data = GetDeviceData(device);

    VkResult collective_result = VK_SUCCESS;
    for (uint32_t create_info_index = 0; create_info_index < createInfoCount; ++create_info_index) {
        VkGraphicsPipelineCreateInfo const& create_info = pCreateInfos[create_info_index];

        VkPipelineRenderingCreateInfoKHR const* rendering_create_info = nullptr;
        for (auto chain = reinterpret_cast<VkBaseInStructure const*>(pCreateInfos->pNext); chain != nullptr; chain = chain->pNext) {
            if (chain->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                rendering_create_info = reinterpret_cast<VkPipelineRenderingCreateInfoKHR const*>(chain);
                break;
            }
        }

        if (rendering_create_info == nullptr) {
            VkResult result = device_data->vtable.CreateGraphicsPipelines(device, pipelineCache, 1, &create_info, pAllocator,
                                                                          &pPipelines[create_info_index]);
            if (result != VK_SUCCESS) {
                pPipelines[create_info_index] = VK_NULL_HANDLE;
                collective_result = result;

                // Vulkan 1.3.285, 10.1. Multiple Pipeline Creation:
                // > Attempt to create all pipelines, and only return VK_NULL_HANDLE values for those that actually failed.
            }

            continue;
        }

        for (auto chain = reinterpret_cast<VkBaseInStructure const*>(pCreateInfos->pNext); chain != nullptr; chain = chain->pNext) {
            switch (chain->sType) {
                case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
                } break;
                case VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD:
                case VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID:
                case VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT:
                case VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO_NV:
                case VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX:
                case VK_STRUCTURE_TYPE_PIPELINE_COMPILER_CONTROL_CREATE_INFO_AMD:
                case VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR:
                case VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO:
                case VK_STRUCTURE_TYPE_PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO_EXT:
                case VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO_NV:
                case VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR:
                case VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR:
                case VK_STRUCTURE_TYPE_PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO_NV:
                case VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT:
                // TODO: We want to support `VK_KHR_dynamic_rendering_local_read` if possible.
                case VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR:
                case VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR: {
                    LOG_FATAL("CreateGraphicsPipelines: Unsupported sType: %s", string_VkStructureType(chain->sType));
                } break;
                default: {
                    LOG_FATAL("CreateGraphicsPipelines: Unrecognized sType: %s", string_VkStructureType(chain->sType));
                } break;
            }
        }

        // TODO: These are our current assumptions:
        ASSERT(rendering_create_info->viewMask == 0);

        RenderPassDesc render_pass_desc;
        bool uses_color_attachment = false;
        bool uses_depth_stencil_attachment = false;
        {
            // Single subpass.
            render_pass_desc.subpass_descs.emplace_back();
            SubpassDesc& subpass_desc = render_pass_desc.subpass_descs.back();

            VkSampleCountFlagBits samples = create_info.pMultisampleState != nullptr
                                                ? create_info.pMultisampleState->rasterizationSamples
                                                : VK_SAMPLE_COUNT_1_BIT;

            auto SetAttachmentSignature = [samples](AttachmentSignature& dst, VkFormat format) {
                dst.format = format;
                dst.samples = samples;
                dst.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                dst.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                dst.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                dst.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            };

            for (uint32_t i = 0; i < rendering_create_info->colorAttachmentCount; ++i) {
                VkFormat format = rendering_create_info->pColorAttachmentFormats[i];

                uint32_t attachment_index = render_pass_desc.attachment_signatures.size();

                render_pass_desc.attachment_signatures.emplace_back();
                AttachmentSignature& attachment_signature = render_pass_desc.attachment_signatures.back();
                SetAttachmentSignature(attachment_signature, format);

                subpass_desc.color_attachments.emplace_back(attachment_index);

                uses_color_attachment = true;
            }

            if (rendering_create_info->depthAttachmentFormat != VK_FORMAT_UNDEFINED ||
                rendering_create_info->stencilAttachmentFormat != VK_FORMAT_UNDEFINED) {
                VkFormat format = rendering_create_info->depthAttachmentFormat != VK_FORMAT_UNDEFINED
                                      ? rendering_create_info->depthAttachmentFormat
                                      : rendering_create_info->stencilAttachmentFormat;

                uint32_t attachment_index = render_pass_desc.attachment_signatures.size();

                render_pass_desc.attachment_signatures.emplace_back();
                AttachmentSignature& attachment_signature = render_pass_desc.attachment_signatures.back();
                SetAttachmentSignature(attachment_signature, format);

                subpass_desc.depth_stencil_attachment = attachment_index;

                uses_depth_stencil_attachment = true;
            }
        }

        VkRenderPass render_pass = device_data->render_pass_pool.FindOrAddForPipeline(
            render_pass_desc, device_data->vtable.CreateRenderPass, device_data->device);

        vku::safe_VkGraphicsPipelineCreateInfo modified_create_info(&create_info, uses_color_attachment,
                                                                    uses_depth_stencil_attachment);
        modified_create_info.renderPass = render_pass;
        modified_create_info.subpass = 0;

        vku::RemoveFromPnext(modified_create_info, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);

        VkResult result = device_data->vtable.CreateGraphicsPipelines(device, pipelineCache, 1, modified_create_info.ptr(),
                                                                      pAllocator, &pPipelines[create_info_index]);
        if (result != VK_SUCCESS) {
            pPipelines[create_info_index] = VK_NULL_HANDLE;
            collective_result = result;
        }
    }

    return collective_result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator) {
    auto device_data = GetDeviceData(device);

    device_data->vtable.DestroyPipeline(device, pipeline, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfoKHR* pRenderingInfo) {
    // TODO: We don't support any structures in the pNext chain yet.
    for (auto chain = reinterpret_cast<VkBaseInStructure const*>(pRenderingInfo->pNext); chain != nullptr; chain = chain->pNext) {
        switch (chain->sType) {
            case VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO:
            case VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT:
            case VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX:
            case VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO_QCOM:
            case VK_STRUCTURE_TYPE_RENDER_PASS_STRIPE_BEGIN_INFO_ARM:
            case VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT:
            case VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR: {
                LOG_FATAL("CmdBeginRendering: Unsupported sType: %s", string_VkStructureType(chain->sType));
            } break;
            default: {
                LOG_FATAL("CmdBeginRendering: Unrecognized sType: %s", string_VkStructureType(chain->sType));
            } break;
        }
    }

    // TODO: These are our current assumptions.
    {
        ASSERT((pRenderingInfo->flags & (VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR | VK_RENDERING_SUSPENDING_BIT |
                                         VK_RENDERING_RESUMING_BIT)) == 0);
        ASSERT(pRenderingInfo->viewMask == 0);
    }

    // We assume that the image layout is attachment optimal.

    auto device_data = GetDeviceData(commandBuffer);

    RenderPassDesc render_pass_desc;
    {
        // Single subpass.
        render_pass_desc.subpass_descs.emplace_back();
        SubpassDesc& subpass_desc = render_pass_desc.subpass_descs.back();

        auto SetAttachmentSignature = [](AttachmentSignature& dst, VkRenderingAttachmentInfoKHR const& attachment_info,
                                         DeviceData const* device_data) {
            // TODO: We assume that `VK_ATTACHMENT_STORE_OP_NONE_KHR` is not used.
            ASSERT(attachment_info.storeOp != VK_ATTACHMENT_STORE_OP_NONE_KHR);

            VkImage image = VK_NULL_HANDLE;
            {
                auto iter = device_data->image_view_map.find(attachment_info.imageView);
                ASSERT(iter != device_data->image_view_map.end());
                image = iter->second.image;
            }

            auto iter = device_data->image_map.find(image);
            ASSERT(iter != device_data->image_map.end());
            ImageData const& image_data = iter->second;

            dst.format = image_data.format;
            dst.samples = image_data.samples;
            if (vkuFormatIsStencilOnly(image_data.format)) {
                dst.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                dst.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                dst.stencilLoadOp = attachment_info.loadOp;
                dst.stencilStoreOp = attachment_info.storeOp;
            } else {
                dst.loadOp = attachment_info.loadOp;
                dst.storeOp = attachment_info.storeOp;
                dst.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                dst.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
        };

        for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; ++i) {
            VkRenderingAttachmentInfoKHR const& attachment_info = pRenderingInfo->pColorAttachments[i];

            // TODO: Color resolve support.
            ASSERT(attachment_info.resolveImageView == VK_NULL_HANDLE);

            uint32_t attachment_index = render_pass_desc.attachment_signatures.size();

            render_pass_desc.attachment_signatures.emplace_back();
            AttachmentSignature& attachment_signature = render_pass_desc.attachment_signatures.back();
            SetAttachmentSignature(attachment_signature, attachment_info, device_data.get());

            subpass_desc.color_attachments.emplace_back(attachment_index);
        }

        if (pRenderingInfo->pDepthAttachment != nullptr || pRenderingInfo->pStencilAttachment != nullptr) {
            // VUID-VkRenderingInfo-pDepthAttachment-06085:
            // > If neither pDepthAttachment or pStencilAttachment are NULL and the imageView member of either structure is not
            // VK_NULL_HANDLE, > the imageView member of each structure must be the same
            VkRenderingAttachmentInfoKHR const& attachment_info = pRenderingInfo->pDepthAttachment != nullptr
                                                                      ? *pRenderingInfo->pDepthAttachment
                                                                      : *pRenderingInfo->pStencilAttachment;

            // TODO: Depth/stencil resolve support.
            ASSERT(attachment_info.resolveImageView == VK_NULL_HANDLE);

            uint32_t attachment_index = render_pass_desc.attachment_signatures.size();

            render_pass_desc.attachment_signatures.emplace_back();
            AttachmentSignature& attachment_signature = render_pass_desc.attachment_signatures.back();
            SetAttachmentSignature(attachment_signature, attachment_info, device_data.get());

            subpass_desc.depth_stencil_attachment = attachment_index;
        }
    }

    VkRenderPass render_pass = device_data->render_pass_pool.FindOrAddForRenderPassInstance(
        render_pass_desc, device_data->vtable.CreateRenderPass, device_data->device);

    FramebufferDesc framebuffer_desc;
    vku::small::vector<VkClearValue, 4> clear_values;
    {
        framebuffer_desc.render_pass = render_pass;
        framebuffer_desc.render_area = pRenderingInfo->renderArea;

        for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; ++i) {
            VkRenderingAttachmentInfoKHR const& attachment_info = pRenderingInfo->pColorAttachments[i];

            framebuffer_desc.attachments.emplace_back(attachment_info.imageView);
            clear_values.emplace_back(attachment_info.clearValue);
        }

        if (pRenderingInfo->pDepthAttachment != nullptr || pRenderingInfo->pStencilAttachment != nullptr) {
            // See above.
            VkRenderingAttachmentInfoKHR const& attachment_info = pRenderingInfo->pDepthAttachment != nullptr
                                                                      ? *pRenderingInfo->pDepthAttachment
                                                                      : *pRenderingInfo->pStencilAttachment;

            framebuffer_desc.attachments.emplace_back(attachment_info.imageView);
            clear_values.emplace_back(attachment_info.clearValue);
        }
    }

    VkFramebuffer framebuffer =
        device_data->framebuffer_pool.FindOrAdd(framebuffer_desc, device_data->vtable.CreateFramebuffer, device_data->device);

    VkRenderPassBeginInfo begin_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin_info.pNext = nullptr;
    begin_info.renderPass = render_pass;
    begin_info.framebuffer = framebuffer;
    begin_info.renderArea = pRenderingInfo->renderArea;
    begin_info.clearValueCount = clear_values.size();
    begin_info.pClearValues = clear_values.data();

    device_data->vtable.CmdBeginRenderPass(commandBuffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRendering(VkCommandBuffer commandBuffer) {
    auto device_data = GetDeviceData(commandBuffer);

    device_data->vtable.CmdEndRenderPass(commandBuffer);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* pName);

#define ADD_HOOK(fn) \
    { "vk" #fn, (PFN_vkVoidFunction)fn }
#define ADD_HOOK_ALIAS(fn, fn_alias) \
    { "vk" #fn, (PFN_vkVoidFunction)fn_alias }

static const std::unordered_map<std::string, PFN_vkVoidFunction> kInstanceFunctions = {
    ADD_HOOK(CreateInstance),
    ADD_HOOK(DestroyInstance),
    ADD_HOOK(EnumeratePhysicalDevices),
    ADD_HOOK(EnumerateDeviceExtensionProperties),
    ADD_HOOK(GetPhysicalDeviceFeatures2),
    ADD_HOOK(GetPhysicalDeviceFeatures2KHR),
    ADD_HOOK(CreateDevice),
};

static const std::unordered_map<std::string, PFN_vkVoidFunction> kDeviceFunctions = {
    ADD_HOOK(DestroyDevice),

    ADD_HOOK(CreateImage),
    ADD_HOOK(DestroyImage),
    ADD_HOOK(CreateImageView),
    ADD_HOOK(DestroyImageView),
    ADD_HOOK(CreateSwapchainKHR),
    ADD_HOOK(DestroySwapchainKHR),
    ADD_HOOK(GetSwapchainImagesKHR),

    ADD_HOOK(CreateGraphicsPipelines),
    ADD_HOOK(DestroyPipeline),
    ADD_HOOK(CmdBeginRendering),
    ADD_HOOK_ALIAS(CmdBeginRenderingKHR, CmdBeginRendering),
    ADD_HOOK(CmdEndRendering),
    ADD_HOOK_ALIAS(CmdEndRenderingKHR, CmdEndRendering),

    // Needs to point to itself as Android loaders calls vkGet*ProcAddr to itself. Without these hooks, when the app calls
    // vkGetDeviceProcAddr to get layer functions it will fail on Android
    ADD_HOOK(GetInstanceProcAddr),
    ADD_HOOK(GetDeviceProcAddr),
};
#undef ADD_HOOK
#undef ADD_HOOK_ALIAS

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance instance, const char* pName) {
    auto instance_result = kInstanceFunctions.find(pName);
    if (instance_result != kInstanceFunctions.end()) {
        return instance_result->second;
    }
    auto dev_result = kDeviceFunctions.find(pName);
    if (dev_result != kDeviceFunctions.end()) {
        return dev_result->second;
    }
    auto instance_data = GetInstanceData(instance);
    if (instance_data != nullptr && instance_data->vtable.GetInstanceProcAddr) {
        PFN_vkVoidFunction result = instance_data->vtable.GetInstanceProcAddr(instance, pName);
        return result;
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* pName) {
    auto device_data = GetDeviceData(device);
    if (device_data && device_data->enable_layer) {
        auto result = kDeviceFunctions.find(pName);
        if (result != kDeviceFunctions.end()) {
            return result->second;
        }
    }
    if (device_data && device_data->vtable.GetDeviceProcAddr) {
        PFN_vkVoidFunction result = device_data->vtable.GetDeviceProcAddr(device, pName);
        return result;
    }
    return nullptr;
}

}  // namespace dynamic_rendering

#if defined(__GNUC__) && __GNUC__ >= 4
#define VEL_EXPORT __attribute__((visibility("default")))
#else
#define VEL_EXPORT
#endif

extern "C" VEL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return dynamic_rendering::GetInstanceProcAddr(instance, pName);
}

extern "C" VEL_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    return dynamic_rendering::GetDeviceProcAddr(device, pName);
}

extern "C" VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    ASSERT(pVersionStruct != nullptr);
    ASSERT(pVersionStruct->sType == LAYER_NEGOTIATE_INTERFACE_STRUCT);

    // Fill in the function pointers if our version is at least capable of having the structure contain them.
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr = dynamic_rendering::GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = dynamic_rendering::GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }

    return VK_SUCCESS;
}

// loader-layer interface v0 - Needed for Android loader using explicit layers
extern "C" VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* pLayerName,
                                                                                            uint32_t* pPropertyCount,
                                                                                            VkExtensionProperties* pProperties) {
    if (pLayerName && strncmp(pLayerName, dynamic_rendering::kGlobalLayer.layerName, VK_MAX_EXTENSION_NAME_SIZE) == 0) {
        // VK_KHR_dynamic_rendering is a device extension and don't want to have it labeled as both instance and device extension
        *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

// loader-layer interface v0 - Needed for Android loader using explicit layers
extern "C" VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pPropertyCount,
                                                                                        VkLayerProperties* pProperties) {
    if (pProperties == NULL) {
        *pPropertyCount = 1;
        return VK_SUCCESS;
    }
    if (*pPropertyCount < 1) {
        return VK_INCOMPLETE;
    }
    *pPropertyCount = 1;
    pProperties[0] = dynamic_rendering::kGlobalLayer;
    return VK_SUCCESS;
}

// loader-layer interface v0 - Needed for Android loader using explicit layers
extern "C" VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                          const char* pLayerName,
                                                                                          uint32_t* pPropertyCount,
                                                                                          VkExtensionProperties* pProperties) {
    // Want to have this call down chain if multiple layers are enabling extenions
    return dynamic_rendering::EnumerateDeviceExtensionProperties(physicalDevice, pLayerName, pPropertyCount, pProperties);
}

// Deprecated, but needed or else Android loader will not call into the exported vkEnumerateDeviceExtensionProperties
extern "C" VEL_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t* pPropertyCount,
                                                                                      VkLayerProperties* pProperties) {
    return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}
