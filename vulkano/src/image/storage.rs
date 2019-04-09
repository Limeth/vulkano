// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::sync::{
	atomic::{AtomicUsize, Ordering},
	Arc
};

use crate::{
	buffer::BufferAccess,
	device::Device,
	format::{ClearValue, FormatDesc, FormatTy},
	image::{
		sys::{ImageCreationError, UnsafeImage, UnsafeImageView},
		traits::{ImageAccess, ImageClearValue, ImageContent, ImageViewAccess},
		ImageDimensions,
		ImageInner,
		ImageLayout,
		ImageUsage
	},
	instance::QueueFamily,
	memory::{
		pool::{
			AllocFromRequirementsFilter,
			AllocLayout,
			MappingRequirement,
			MemoryPool,
			MemoryPoolAlloc,
			PotentialDedicatedAllocation,
			StdMemoryPool
		},
		DedicatedAlloc
	},
	sync::{AccessError, Sharing}
};

/// General-purpose image in device memory. Can be used for any usage, but will be slower than a
/// specialized image.
#[derive(Debug)]
pub struct StorageImage<F, A = Arc<StdMemoryPool>>
where
	A: MemoryPool
{
	// Inner implementation.
	image: UnsafeImage,

	// We maintain a view of the whole image.
	view: UnsafeImageView,

	// Memory used to back the image.
	memory: PotentialDedicatedAllocation<A::Alloc>,

	// ImageDimensions of the image view.
	dimensions: ImageDimensions,

	// Format.
	format: F,

	// Queue families allowed to access this image.
	queue_families: SmallVec<[u32; 4]>,

	// Number of times this image is locked on the GPU side.
	gpu_lock: AtomicUsize
}

impl<F> StorageImage<F> {
	/// Creates a new image with the given dimensions and format.
	pub fn new<'a, I>(
		device: Arc<Device>, dimensions: ImageDimensions, format: F, queue_families: I
	) -> Result<Arc<StorageImage<F>>, ImageCreationError>
	where
		F: FormatDesc,
		I: IntoIterator<Item = QueueFamily<'a>>
	{
		let is_depth = match format.format().ty() {
			FormatTy::Depth => true,
			FormatTy::DepthStencil => true,
			FormatTy::Stencil => true,
			FormatTy::Compressed => panic!(),
			_ => false
		};

		let usage = ImageUsage {
			transfer_source: true,
			transfer_destination: true,
			sampled: true,
			storage: true,
			color_attachment: !is_depth,
			depth_stencil_attachment: is_depth,
			input_attachment: true,
			transient_attachment: false
		};

		StorageImage::with_usage(device, dimensions, format, usage, queue_families)
	}

	/// Same as `new`, but allows specifying the usage.
	pub fn with_usage<'a, I>(
		device: Arc<Device>, dimensions: ImageDimensions, format: F, usage: ImageUsage,
		queue_families: I
	) -> Result<Arc<StorageImage<F>>, ImageCreationError>
	where
		F: FormatDesc,
		I: IntoIterator<Item = QueueFamily<'a>>
	{
		let queue_families =
			queue_families.into_iter().map(|f| f.id()).collect::<SmallVec<[u32; 4]>>();

		let (image, mem_reqs) = unsafe {
			let sharing = if queue_families.len() >= 2 {
				Sharing::Concurrent(queue_families.iter().cloned())
			} else {
				Sharing::Exclusive
			};

			UnsafeImage::new(
				device.clone(),
				usage,
				format.format(),
				dimensions,
				1,
				1,
				sharing,
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

		let view = unsafe {
			UnsafeImageView::raw(
				&image,
				dimensions.into(),
				0 .. image.mipmap_levels(),
				0 .. image.dimensions().array_layers()
			)?
		};

		Ok(Arc::new(StorageImage {
			image,
			view,
			memory: mem,
			dimensions,
			format,
			queue_families,
			gpu_lock: AtomicUsize::new(0)
		}))
	}
}

impl<F, A> StorageImage<F, A>
where
	A: MemoryPool
{
	/// Returns the dimensions of the image.
	pub fn dimensions(&self) -> ImageDimensions { self.dimensions }
}

unsafe impl<F, A> ImageAccess for StorageImage<F, A>
where
	F: 'static + Send + Sync,
	A: MemoryPool
{
	fn inner(&self) -> ImageInner {
		ImageInner {
			image: &self.image,
			first_layer: 0,
			num_layers: self.dimensions.array_layers() as usize,
			first_mipmap_level: 0,
			num_mipmap_levels: 1
		}
	}

	fn initial_layout_requirement(&self) -> ImageLayout { ImageLayout::General }

	fn final_layout_requirement(&self) -> ImageLayout { ImageLayout::General }

	fn conflicts_buffer(&self, other: &BufferAccess) -> bool { false }

	fn conflicts_image(&self, other: &ImageAccess) -> bool {
		self.conflict_key() == other.conflict_key() // TODO:
	}

	fn conflict_key(&self) -> u64 { self.image.key() }

	fn try_gpu_lock(&self, _: bool, expected_layout: ImageLayout) -> Result<(), AccessError> {
		// TODO: handle initial layout transition
		if expected_layout != ImageLayout::General && expected_layout != ImageLayout::Undefined {
			return Err(AccessError::UnexpectedImageLayout {
				requested: expected_layout,
				expected: ImageLayout::General
			})
		}

		let val = self.gpu_lock.compare_and_swap(0, 1, Ordering::SeqCst);
		if val == 0 {
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
		assert!(new_layout.is_none() || new_layout == Some(ImageLayout::General));
		self.gpu_lock.fetch_sub(1, Ordering::SeqCst);
	}
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for StorageImage<F, A>
where
	F: FormatDesc + 'static + Send + Sync,
	A: MemoryPool
{
	fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
		Some(self.format.decode_clear_value(value))
	}
}

unsafe impl<P, F, A> ImageContent<P> for StorageImage<F, A>
where
	F: 'static + Send + Sync,
	A: MemoryPool
{
	fn matches_format(&self) -> bool {
		true // FIXME:
	}
}

unsafe impl<F, A> ImageViewAccess for StorageImage<F, A>
where
	F: 'static + Send + Sync,
	A: MemoryPool
{
	fn parent(&self) -> &ImageAccess { self }

	fn dimensions(&self) -> ImageDimensions { self.dimensions }

	fn inner(&self) -> &UnsafeImageView { &self.view }

	fn descriptor_set_storage_image_layout(&self) -> ImageLayout { ImageLayout::General }

	fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout { ImageLayout::General }

	fn descriptor_set_sampled_image_layout(&self) -> ImageLayout { ImageLayout::General }

	fn descriptor_set_input_attachment_layout(&self) -> ImageLayout { ImageLayout::General }

	fn identity_swizzle(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
	use super::StorageImage;
	use crate::{format::Format, image::ImageDimensions};

	#[test]
	fn create() {
		let (device, queue) = gfx_dev_and_queue!();
		let _img = StorageImage::new(
			device,
			ImageDimensions::Dim2D { width: 32, height: 32 },
			Format::R8G8B8A8Unorm,
			Some(queue.family())
		)
		.unwrap();
	}
}
