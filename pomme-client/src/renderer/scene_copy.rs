use std::sync::{Arc, Mutex};

use pomme_gpu_allocator::MemoryLocation;
use pomme_gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use pyronyx::vk;

use crate::renderer::util;

/// Full-resolution copy of the current scene used when a UI effect must
/// reproduce vanilla's UNORM code-value composition exactly.
pub struct SceneCopy {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<Allocation>,
    width: u32,
    height: u32,
    format: vk::Format,
}

impl SceneCopy {
    pub fn new(
        device: &vk::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &Arc<Mutex<Allocator>>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Self {
        let (image, view, allocation) = create_image(device, allocator, width, height, format);
        util::transition_image_to_shader_read(device, queue, command_pool, image);
        let sampler = unsafe { util::create_nearest_sampler(device) };
        Self {
            image,
            view,
            sampler,
            allocation: Some(allocation),
            width,
            height,
            format,
        }
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Copies a swapchain image after the current render pass has ended. Both
    /// images are restored to the layouts expected by subsequent rendering.
    pub fn capture(&self, cmd: vk::CommandBuffer, src_image: vk::Image) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::Color,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let to_transfer = [
            vk::ImageMemoryBarrier {
                image: src_image,
                old_layout: vk::ImageLayout::ColorAttachmentOptimal,
                new_layout: vk::ImageLayout::TransferSrcOptimal,
                src_access_mask: vk::AccessFlags::ColorAttachmentWrite,
                dst_access_mask: vk::AccessFlags::TransferRead,
                subresource_range: range,
                ..Default::default()
            },
            vk::ImageMemoryBarrier {
                image: self.image,
                old_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                new_layout: vk::ImageLayout::TransferDstOptimal,
                src_access_mask: vk::AccessFlags::ShaderRead,
                dst_access_mask: vk::AccessFlags::TransferWrite,
                subresource_range: range,
                ..Default::default()
            },
        ];
        cmd.pipeline_barrier(
            vk::PipelineStageFlags::ColorAttachmentOutput | vk::PipelineStageFlags::FragmentShader,
            vk::PipelineStageFlags::Transfer,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_transfer,
        );

        let copy = vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::Color,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::Color,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            extent: vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            ..Default::default()
        };
        cmd.copy_image(
            src_image,
            vk::ImageLayout::TransferSrcOptimal,
            self.image,
            vk::ImageLayout::TransferDstOptimal,
            &[copy],
        );

        let from_transfer = [
            vk::ImageMemoryBarrier {
                image: src_image,
                old_layout: vk::ImageLayout::TransferSrcOptimal,
                new_layout: vk::ImageLayout::ColorAttachmentOptimal,
                src_access_mask: vk::AccessFlags::TransferRead,
                dst_access_mask: vk::AccessFlags::ColorAttachmentWrite,
                subresource_range: range,
                ..Default::default()
            },
            vk::ImageMemoryBarrier {
                image: self.image,
                old_layout: vk::ImageLayout::TransferDstOptimal,
                new_layout: vk::ImageLayout::ShaderReadOnlyOptimal,
                src_access_mask: vk::AccessFlags::TransferWrite,
                dst_access_mask: vk::AccessFlags::ShaderRead,
                subresource_range: range,
                ..Default::default()
            },
        ];
        cmd.pipeline_barrier(
            vk::PipelineStageFlags::Transfer,
            vk::PipelineStageFlags::ColorAttachmentOutput | vk::PipelineStageFlags::FragmentShader,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &from_transfer,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &mut self,
        device: &vk::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        allocator: &Arc<Mutex<Allocator>>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) {
        if width == self.width && height == self.height && format == self.format {
            return;
        }
        self.destroy_image(device, allocator);
        let (image, view, allocation) = create_image(device, allocator, width, height, format);
        util::transition_image_to_shader_read(device, queue, command_pool, image);
        self.image = image;
        self.view = view;
        self.allocation = Some(allocation);
        self.width = width;
        self.height = height;
        self.format = format;
    }

    fn destroy_image(&mut self, device: &vk::Device, allocator: &Arc<Mutex<Allocator>>) {
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        if let Some(allocation) = self.allocation.take() {
            allocator.lock().unwrap().free(allocation).ok();
        }
    }

    pub fn destroy(&mut self, device: &vk::Device, allocator: &Arc<Mutex<Allocator>>) {
        self.destroy_image(device, allocator);
        device.destroy_sampler(self.sampler, None);
    }
}

fn create_image(
    device: &vk::Device,
    allocator: &Arc<Mutex<Allocator>>,
    width: u32,
    height: u32,
    format: vk::Format,
) -> (vk::Image, vk::ImageView, Allocation) {
    let info = vk::ImageCreateInfo {
        image_type: vk::ImageType::Type2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::Type1,
        tiling: vk::ImageTiling::Optimal,
        usage: vk::ImageUsageFlags::TransferDst | vk::ImageUsageFlags::Sampled,
        ..Default::default()
    };
    let image = device
        .create_image(&info, None)
        .expect("failed to create scene-copy image");
    let requirements = device.get_image_memory_requirements(image);
    let allocation = allocator
        .lock()
        .unwrap()
        .allocate(&AllocationCreateDesc {
            name: "ui_scene_copy",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .expect("failed to allocate scene-copy image");
    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .expect("failed to bind scene-copy image");
    }
    let view = device
        .create_image_view(
            &vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::Type2D,
                format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::Color,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            },
            None,
        )
        .expect("failed to create scene-copy view");
    (image, view, allocation)
}
