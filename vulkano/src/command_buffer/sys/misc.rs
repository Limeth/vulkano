use std::{ops::BitAnd, ptr};

use smallvec::SmallVec;

use vk_sys as vk;

use crate::{
	buffer::{BufferAccess, BufferInner},
	command_buffer::CommandBuffer,
	format::{Format, FormatDesc},
	image::{
		layout::{ImageLayout, ImageLayoutEnd},
		ImageViewAccess
	},
	sync::{AccessFlagBits, PipelineStages},
	VulkanObject
};

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct UnsafeCommandBufferBuilderBindVertexBuffer {
	// Raw handles of the buffers to bind.
	pub raw_buffers: SmallVec<[vk::Buffer; 4]>,
	// Raw offsets of the buffers to bind.
	pub offsets: SmallVec<[vk::DeviceSize; 4]>
}
impl UnsafeCommandBufferBuilderBindVertexBuffer {
	/// Builds a new empty list.
	pub fn new() -> UnsafeCommandBufferBuilderBindVertexBuffer {
		UnsafeCommandBufferBuilderBindVertexBuffer {
			raw_buffers: SmallVec::new(),
			offsets: SmallVec::new()
		}
	}

	/// Adds a buffer to the list.
	pub fn add<B>(&mut self, buffer: &B)
	where
		B: ?Sized + BufferAccess
	{
		let inner = buffer.inner();
		debug_assert!(inner.buffer.usage_vertex_buffer());
		self.raw_buffers.push(inner.buffer.internal_object());
		self.offsets.push(inner.offset as vk::DeviceSize);
	}
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct UnsafeCommandBufferBuilderExecuteCommands {
	// Raw handles of the command buffers to execute.
	pub raw_cbs: SmallVec<[vk::CommandBuffer; 4]>
}
impl UnsafeCommandBufferBuilderExecuteCommands {
	/// Builds a new empty list.
	pub fn new() -> UnsafeCommandBufferBuilderExecuteCommands {
		UnsafeCommandBufferBuilderExecuteCommands { raw_cbs: SmallVec::new() }
	}

	/// Adds a command buffer to the list.
	pub fn add<C>(&mut self, cb: &C)
	where
		C: ?Sized + CommandBuffer
	{
		// TODO: debug assert that it is a secondary command buffer?
		self.raw_cbs.push(cb.inner().internal_object());
	}

	/// Adds a command buffer to the list.
	pub unsafe fn add_raw(&mut self, cb: vk::CommandBuffer) { self.raw_cbs.push(cb); }
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageAspect {
	pub color: bool,
	pub depth: bool,
	pub stencil: bool
}
impl UnsafeCommandBufferBuilderImageAspect {
	pub fn from_format(format: Format) -> Self {
		let mut color = false;
		let mut depth = false;
		let mut stencil = false;

		if format.is_depth_stencil() {
			depth = true;
			stencil = true;
		} else if format.is_depth() {
			depth = true;
		} else if format.is_stencil() {
			stencil = true;
		} else {
			color = true;
			// TODO: Compressed formats?
		}

		UnsafeCommandBufferBuilderImageAspect { color, depth, stencil }
	}

	pub(crate) fn to_vk_bits(&self) -> vk::ImageAspectFlagBits {
		let mut out = 0;
		if self.color {
			out |= vk::IMAGE_ASPECT_COLOR_BIT
		};
		if self.depth {
			out |= vk::IMAGE_ASPECT_DEPTH_BIT
		};
		if self.stencil {
			out |= vk::IMAGE_ASPECT_STENCIL_BIT
		};
		out
	}
}
impl BitAnd for UnsafeCommandBufferBuilderImageAspect {
	type Output = Self;

	fn bitand(self, rhs: Self) -> Self {
		UnsafeCommandBufferBuilderImageAspect {
			color: self.color && rhs.color,
			depth: self.depth && rhs.depth,
			stencil: self.stencil && rhs.stencil
		}
	}
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderColorImageClear {
	pub base_mip_level: u32,
	pub level_count: u32,
	pub base_array_layer: u32,
	pub layer_count: u32
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderBufferImageCopy {
	pub buffer_offset: usize,
	pub buffer_row_length: u32,
	pub buffer_image_height: u32,
	pub image_aspect: UnsafeCommandBufferBuilderImageAspect,
	pub image_mip_level: u32,
	pub image_base_array_layer: u32,
	pub image_layer_count: u32,
	pub image_offset: [i32; 3],
	pub image_extent: [u32; 3]
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageCopy {
	pub aspect: UnsafeCommandBufferBuilderImageAspect,
	pub source_mip_level: u32,
	pub destination_mip_level: u32,
	pub source_base_array_layer: u32,
	pub destination_base_array_layer: u32,
	pub layer_count: u32,
	pub source_offset: [i32; 3],
	pub destination_offset: [i32; 3],
	pub extent: [u32; 3]
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageBlit {
	pub aspect: UnsafeCommandBufferBuilderImageAspect,
	pub source_mip_level: u32,
	pub destination_mip_level: u32,
	pub source_base_array_layer: u32,
	pub destination_base_array_layer: u32,
	pub layer_count: u32,
	pub source_top_left: [i32; 3],
	pub source_bottom_right: [i32; 3],
	pub destination_top_left: [i32; 3],
	pub destination_bottom_right: [i32; 3]
}

/// Command that adds a pipeline barrier to a command buffer builder.
///
/// A pipeline barrier is a low-level system-ish command that is often necessary for safety. By
/// default all commands that you add to a command buffer can potentially run simultaneously.
/// Adding a pipeline barrier separates commands before the barrier from commands after the barrier
/// and prevents them from running simultaneously.
///
/// Please take a look at the Vulkan specifications for more information. Pipeline barriers are a
/// complex topic and explaining them in this documentation would be redundant.
///
/// > **Note**: We use a builder-like API here so that users can pass multiple buffers or images of
/// > multiple different types. Doing so with a single function would be very tedious in terms of
/// > API.
pub struct UnsafeCommandBufferBuilderPipelineBarrier {
	pub src_stage_mask: vk::PipelineStageFlags,
	pub dst_stage_mask: vk::PipelineStageFlags,
	pub dependency_flags: vk::DependencyFlags,
	pub memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
	pub buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
	pub image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>
}

impl UnsafeCommandBufferBuilderPipelineBarrier {
	/// Creates a new empty pipeline barrier command.
	pub fn new() -> UnsafeCommandBufferBuilderPipelineBarrier {
		UnsafeCommandBufferBuilderPipelineBarrier {
			src_stage_mask: 0,
			dst_stage_mask: 0,
			dependency_flags: vk::DEPENDENCY_BY_REGION_BIT,
			memory_barriers: SmallVec::new(),
			buffer_barriers: SmallVec::new(),
			image_barriers: SmallVec::new()
		}
	}

	/// Returns true if no barrier or execution dependency has been added yet.
	pub fn is_empty(&self) -> bool { self.src_stage_mask == 0 || self.dst_stage_mask == 0 }

	/// Merges another pipeline builder into this one.
	pub fn merge(&mut self, other: UnsafeCommandBufferBuilderPipelineBarrier) {
		self.src_stage_mask |= other.src_stage_mask;
		self.dst_stage_mask |= other.dst_stage_mask;
		self.dependency_flags &= other.dependency_flags;

		self.memory_barriers.extend(other.memory_barriers.into_iter());
		self.buffer_barriers.extend(other.buffer_barriers.into_iter());
		self.image_barriers.extend(other.image_barriers.into_iter());
	}

	/// Adds an execution dependency. This means that all the stages in `source` of the previous
	/// commands must finish before any of the stages in `destination` of the following commands can start.
	///
	/// # Safety
	///
	/// - If the pipeline stages include geometry or tessellation stages, then the corresponding
	///   features must have been enabled in the device.
	/// - There are certain rules regarding the pipeline barriers inside render passes.
	pub unsafe fn add_execution_dependency(
		&mut self, source: PipelineStages, destination: PipelineStages, by_region: bool
	) {
		if !by_region {
			self.dependency_flags = 0;
		}

		debug_assert_ne!(source, PipelineStages::none());
		debug_assert_ne!(destination, PipelineStages::none());

		self.src_stage_mask |= source.into_vulkan_bits();
		self.dst_stage_mask |= destination.into_vulkan_bits();
	}

	/// Adds a memory barrier. This means that all the memory writes by the given source stages
	/// for the given source accesses must be visible by the given destination stages for the given
	/// destination accesses.
	///
	/// Also adds an execution dependency similar to `add_execution_dependency`.
	///
	/// # Safety
	///
	/// - Same as `add_execution_dependency`.
	pub unsafe fn add_memory_barrier(
		&mut self, source_stage: PipelineStages, source_access: AccessFlagBits,
		destination_stage: PipelineStages, destination_access: AccessFlagBits, by_region: bool
	) {
		debug_assert!(source_access.is_compatible_with(&source_stage));
		debug_assert!(destination_access.is_compatible_with(&destination_stage));

		self.add_execution_dependency(source_stage, destination_stage, by_region);

		self.memory_barriers.push(vk::MemoryBarrier {
			sType: vk::STRUCTURE_TYPE_MEMORY_BARRIER,
			pNext: ptr::null(),
			srcAccessMask: source_access.into_vulkan_bits(),
			dstAccessMask: destination_access.into_vulkan_bits()
		});
	}

	/// Adds a buffer memory barrier. This means that all the memory writes to the given buffer by
	/// the given source stages for the given source accesses must be visible by the given dest
	/// stages for the given destination accesses.
	///
	/// Also adds an execution dependency similar to `add_execution_dependency`.
	///
	/// Also allows transferring buffer ownership between queues.
	///
	/// # Safety
	///
	/// - Same as `add_execution_dependency`.
	/// - The buffer must be alive for at least as long as the command buffer to which this barrier
	///   is added.
	/// - Queue ownership transfers must be correct.
	pub unsafe fn add_buffer_memory_barrier<B>(
		&mut self, buffer: &B, source_stage: PipelineStages, source_access: AccessFlagBits,
		destination_stage: PipelineStages, destination_access: AccessFlagBits, by_region: bool,
		queue_transfer: Option<(u32, u32)>, offset: usize, size: usize
	) where
		B: ?Sized + BufferAccess
	{
		debug_assert!(source_access.is_compatible_with(&source_stage));
		debug_assert!(destination_access.is_compatible_with(&destination_stage));

		self.add_execution_dependency(source_stage, destination_stage, by_region);

		debug_assert!(size <= buffer.size());
		let BufferInner { buffer, offset: org_offset } = buffer.inner();
		let offset = offset + org_offset;

		let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
			(src_queue, dest_queue)
		} else {
			(vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
		};

		self.buffer_barriers.push(vk::BufferMemoryBarrier {
			sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			pNext: ptr::null(),
			srcAccessMask: source_access.into_vulkan_bits(),
			dstAccessMask: destination_access.into_vulkan_bits(),
			srcQueueFamilyIndex: src_queue,
			dstQueueFamilyIndex: dest_queue,
			buffer: buffer.internal_object(),
			offset: offset as vk::DeviceSize,
			size: size as vk::DeviceSize
		});
	}

	/// Adds an image memory barrier. This is the equivalent of `add_buffer_memory_barrier` but
	/// for images.
	///
	/// In addition to transferring image ownership between queues, it also allows changing the
	/// layout of images.
	///
	/// Also adds an execution dependency similar to `add_execution_dependency`.
	///
	/// # Safety
	///
	/// - Same as `add_execution_dependency`.
	/// - The buffer must be alive for at least as long as the command buffer to which this barrier
	///   is added.
	/// - Queue ownership transfers must be correct.
	/// - Image layouts transfers must be correct.
	/// - Access flags must be compatible with the image usage flags passed at image creation.
	pub unsafe fn add_image_memory_barrier<I>(
		&mut self, image_view: &I, source_stage: PipelineStages, source_access: AccessFlagBits,
		destination_stage: PipelineStages, destination_access: AccessFlagBits, by_region: bool,
		queue_transfer: Option<(u32, u32)>, current_layout: ImageLayout,
		new_layout: ImageLayoutEnd
	) where
		I: ?Sized + ImageViewAccess
	{
		debug_assert!(source_access.is_compatible_with(&source_stage));
		debug_assert!(destination_access.is_compatible_with(&destination_stage));

		self.add_execution_dependency(source_stage, destination_stage, by_region);

		let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
			(src_queue, dest_queue)
		} else {
			(vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
		};

		let aspect_mask = {
			let format = image_view.format();
			if format.is_depth_stencil() {
				vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
			} else if format.is_depth() {
				vk::IMAGE_ASPECT_DEPTH_BIT
			} else if format.is_stencil() {
				vk::IMAGE_ASPECT_STENCIL_BIT
			} else {
				vk::IMAGE_ASPECT_COLOR_BIT
				// TODO: Compressed formats?
			}
		};

		self.image_barriers.push(vk::ImageMemoryBarrier {
			sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			pNext: ptr::null(),
			srcAccessMask: source_access.into_vulkan_bits(),
			dstAccessMask: destination_access.into_vulkan_bits(),
			oldLayout: current_layout as u32,
			newLayout: new_layout as u32,
			srcQueueFamilyIndex: src_queue,
			dstQueueFamilyIndex: dest_queue,
			image: image_view.parent().inner().internal_object(),
			subresourceRange: vk::ImageSubresourceRange {
				aspectMask: aspect_mask,

				baseMipLevel: image_view.subresource_range().mipmap_levels_offset,
				levelCount: image_view.subresource_range().mipmap_levels.get(),

				baseArrayLayer: image_view.subresource_range().array_layers_offset,
				layerCount: image_view.subresource_range().array_layers.get()
			}
		});
	}
}
