// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{
	iter::Empty,
	sync::{
		atomic::{AtomicBool, AtomicUsize, Ordering},
		Arc
	}
};

use crate::{
	buffer::BufferAccess,
	device::Device,
	format::{ClearValue, Format, FormatDesc, FormatTy},
	image::{
		sys::{ImageCreationError, UnsafeImage, UnsafeImageView},
		traits::{ImageAccess, ImageClearValue, ImageContent, ImageViewAccess},
		ImageDimensions,
		ImageInner,
		ImageLayout,
		ImageUsage,
		ViewType
	},
	memory::{
		pool::{
			AllocFromRequirementsFilter,
			AllocLayout,
			MappingRequirement,
			MemoryPool,
			MemoryPoolAlloc,
			PotentialDedicatedAllocation,
			StdMemoryPoolAlloc
		},
		DedicatedAlloc
	},
	sync::{AccessError, Sharing}
};

/// ImageAccess whose purpose is to be used as a framebuffer attachment.
///
/// The image is always two-dimensional and has only one mipmap, but it can have any kind of
/// format. Trying to use a format that the backend doesn't support for rendering will result in
/// an error being returned when creating the image. Once you have an `AttachmentImage`, you are
/// guaranteed that you will be able to draw on it.
///
/// The template parameter of `AttachmentImage` is a type that describes the format of the image.
///
/// # Regular vs transient
///
/// Calling `AttachmentImage::new` will create a regular image, while calling
/// `AttachmentImage::transient` will create a *transient* image. Transient image are only
/// relevant for images that serve as attachments, so `AttachmentImage` is the only type of
/// image in vulkano that provides a shortcut for this.
///
/// A transient image is a special kind of image whose content is undefined outside of render
/// passes. Once you finish drawing, reading from it will returned undefined data (which can be
/// either valid or garbage, depending on the implementation).
///
/// This gives a hint to the Vulkan implementation that it is possible for the image's content to
/// live exclusively in some cache memory, and that no real memory has to be allocated for it.
///
/// In other words, if you are going to read from the image after drawing to it, use a regular
/// image. If you don't need to read from it (for example if it's some kind of intermediary color,
/// or a depth buffer that is only used once) then use a transient image as it may improve
/// performance.
// TODO: forbid reading transient images outside render passes?
#[derive(Debug)]
pub struct AttachmentImage<F = Format, A = PotentialDedicatedAllocation<StdMemoryPoolAlloc>> {
	// Inner implementation.
	image: UnsafeImage,

	// We maintain a view of the whole image since we will need it when rendering.
	view: UnsafeImageView,

	// Memory used to back the image.
	memory: A,

	// Format.
	format: F,

	// Layout to use when the image is used as a framebuffer attachment.
	// Must be either "depth-stencil optimal" or "color optimal".
	attachment_layout: ImageLayout,

	// If true, then the image is in the layout of `attachment_layout` (above). If false, then it
	// is still `Undefined`.
	initialized: AtomicBool,

	// Number of times this image is locked on the GPU side.
	gpu_lock: AtomicUsize
}
impl<F> AttachmentImage<F> {
	/// Creates a new image with the given dimensions, format, usage and samples.
	///
	/// Returns an error if the dimensions are too large or if the backend doesn't support this
	/// format as a framebuffer attachment.
	///
	/// The `color_attachment` or `depth_stencil_attachment` usages are automatically added based
	/// on the format of the usage. Therefore the `usage` parameter allows you specify usages in
	/// addition to these two.
	pub fn new(
		device: Arc<Device>, dimensions: ImageDimensions, format: F, usage: ImageUsage,
		samples: u32
	) -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
	where
		F: FormatDesc
	{
		// TODO: check dimensions against the max_framebuffer_width/height/layers limits

		let is_depth = match format.format().ty() {
			FormatTy::Depth => true,
			FormatTy::DepthStencil => true,
			FormatTy::Stencil => true,
			FormatTy::Compressed => panic!(),
			_ => false
		};

		let usage =
			ImageUsage { color_attachment: !is_depth, depth_stencil_attachment: is_depth, ..usage };

		let (image, mem_reqs) = unsafe {
			UnsafeImage::new(
				device.clone(),
				usage,
				format.format(),
				dimensions,
				samples,
				1,
				Sharing::Exclusive::<Empty<u32>>,
				false,
				false
			)?
		};

		let mem = MemoryPool::alloc_from_requirements(
			&Device::standard_pool(&device),
			&mem_reqs,
			AllocLayout::Optimal,
			MappingRequirement::DoNotMap,
			DedicatedAlloc::Image(&image),
			|t| {
				if t.is_device_local() {
					AllocFromRequirementsFilter::Preferred
				} else {
					AllocFromRequirementsFilter::Allowed
				}
			}
		)?;
		debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
		unsafe {
			image.bind_memory(mem.memory(), mem.offset())?;
		}

		let view = unsafe { UnsafeImageView::raw(&image, ViewType::Dim2D, 0 .. 1, 0 .. 1)? };

		Ok(Arc::new(AttachmentImage {
			image,
			view,
			memory: mem,
			format,
			attachment_layout: if is_depth {
				ImageLayout::DepthStencilAttachmentOptimal
			} else {
				ImageLayout::ColorAttachmentOptimal
			},
			initialized: AtomicBool::new(false),
			gpu_lock: AtomicUsize::new(0)
		}))
	}
}

unsafe impl<F, A> ImageAccess for AttachmentImage<F, A>
where
	F: 'static + Send + Sync
{
	fn inner(&self) -> ImageInner {
		ImageInner {
			image: &self.image,
			first_layer: 0,
			num_layers: self.image.dimensions().array_layers() as usize,
			first_mipmap_level: 0,
			num_mipmap_levels: 1
		}
	}

	fn initial_layout_requirement(&self) -> ImageLayout { self.attachment_layout }

	fn final_layout_requirement(&self) -> ImageLayout { self.attachment_layout }

	fn conflicts_buffer(&self, other: &BufferAccess) -> bool { false }

	fn conflicts_image(&self, other: &ImageAccess) -> bool {
		self.conflict_key() == other.conflict_key()
	}

	fn conflict_key(&self) -> u64 { self.image.key() }

	fn try_gpu_lock(&self, _: bool, expected_layout: ImageLayout) -> Result<(), AccessError> {
		if expected_layout != self.attachment_layout && expected_layout != ImageLayout::Undefined {
			if self.initialized.load(Ordering::SeqCst) {
				return Err(AccessError::UnexpectedImageLayout {
					requested: expected_layout,
					allowed: self.attachment_layout
				})
			} else {
				return Err(AccessError::UnexpectedImageLayout {
					requested: expected_layout,
					allowed: ImageLayout::Undefined
				})
			}
		}

		if expected_layout != ImageLayout::Undefined {
			if !self.initialized.load(Ordering::SeqCst) {
				return Err(AccessError::ImageNotInitialized { requested: expected_layout })
			}
		}

		if self.gpu_lock.compare_and_swap(0, 1, Ordering::SeqCst) == 0 {
			Ok(())
		} else {
			Err(AccessError::AlreadyInUse)
		}
	}

	unsafe fn increase_gpu_lock(&self) {
		let val = self.gpu_lock.fetch_add(1, Ordering::SeqCst);
		debug_assert!(val >= 1);
	}

	unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
		if let Some(new_layout) = new_layout {
			debug_assert_eq!(new_layout, self.attachment_layout);
			self.initialized.store(true, Ordering::SeqCst);
		}

		let prev_val = self.gpu_lock.fetch_sub(1, Ordering::SeqCst);
		debug_assert!(prev_val >= 1);
	}
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for Arc<AttachmentImage<F, A>>
where
	F: FormatDesc + 'static + Send + Sync
{
	fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
		Some(self.format.decode_clear_value(value))
	}
}

unsafe impl<P, F, A> ImageContent<P> for Arc<AttachmentImage<F, A>>
where
	F: 'static + Send + Sync
{
	fn matches_format(&self) -> bool {
		true // FIXME:
	}
}

unsafe impl<F, A> ImageViewAccess for AttachmentImage<F, A>
where
	F: 'static + Send + Sync
{
	fn parent(&self) -> &ImageAccess { self }

	fn dimensions(&self) -> ImageDimensions {
		let dims = self.image.dimensions();
		ImageDimensions::Dim2D { width: dims.width(), height: dims.height() }
	}

	fn inner(&self) -> &UnsafeImageView { &self.view }

	fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
		ImageLayout::ShaderReadOnlyOptimal
	}

	fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
		ImageLayout::ShaderReadOnlyOptimal
	}

	fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
		ImageLayout::ShaderReadOnlyOptimal
	}

	fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
		ImageLayout::ShaderReadOnlyOptimal
	}

	fn identity_swizzle(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
	use super::AttachmentImage;
	use crate::format::Format;

	#[test]
	fn create_regular() {
		let (device, _) = gfx_dev_and_queue!();
		let _img = AttachmentImage::new(device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
	}

	#[test]
	fn create_transient() {
		let (device, _) = gfx_dev_and_queue!();
		let _img = AttachmentImage::transient(device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
	}

	#[test]
	fn d16_unorm_always_supported() {
		let (device, _) = gfx_dev_and_queue!();
		let _img = AttachmentImage::new(device, [32, 32], Format::D16Unorm).unwrap();
	}
}
