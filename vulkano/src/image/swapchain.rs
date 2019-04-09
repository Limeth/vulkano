// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use crate::{
	buffer::BufferAccess,
	format::{ClearValue, Format, FormatDesc},
	image::{
		sys::UnsafeImageView,
		traits::{ImageAccess, ImageClearValue, ImageContent, ImageViewAccess},
		ImageDimensions,
		ImageInner,
		ImageLayout,
		ImageViewType
	},
	swapchain::Swapchain,
	sync::AccessError,
	OomError
};

/// An image that is part of a swapchain.
///
/// Creating a `SwapchainImage` is automatically done when creating a swapchain.
///
/// A swapchain image is special in the sense that it can only be used after being acquired by
/// calling the `acquire` method on the swapchain. You have no way to know in advance which
/// swapchain image is going to be acquired, so you should keep all of them alive.
///
/// After a swapchain image has been acquired, you are free to perform all the usual operations
/// on it. When you are done you can then *present* the image (by calling the corresponding
/// method on the swapchain), which will have the effect of showing the content of the image to
/// the screen. Once an image has been presented, it can no longer be used unless it is acquired
/// again.
// TODO: #[derive(Debug)]
pub struct SwapchainImage<W> {
	swapchain: Arc<Swapchain<W>>,
	image_offset: usize,
	view: UnsafeImageView
}

impl<W> SwapchainImage<W> {
	/// Builds a `SwapchainImage` from raw components.
	///
	/// This is an internal method that you shouldn't call.
	pub unsafe fn from_raw(
		swapchain: Arc<Swapchain<W>>, id: usize
	) -> Result<Arc<SwapchainImage<W>>, OomError> {
		let image = swapchain.raw_image(id).unwrap();
		let view = UnsafeImageView::raw(&image.image, ImageViewType::Dim2D, 0 .. 1, 0 .. 1)?;

		Ok(Arc::new(SwapchainImage { swapchain: swapchain.clone(), image_offset: id, view }))
	}

	/// Returns the dimensions of the image.
	///
	/// A `SwapchainImage` is always two-dimensional.
	pub fn dimensions(&self) -> [u32; 2] {
		let dims = self.my_image().image.dimensions();
		[dims.width(), dims.height()]
	}

	/// Returns the swapchain this image belongs to.
	pub fn swapchain(&self) -> &Arc<Swapchain<W>> { &self.swapchain }

	fn my_image(&self) -> ImageInner { self.swapchain.raw_image(self.image_offset).unwrap() }
}

unsafe impl<W> ImageAccess for SwapchainImage<W> {
	fn inner(&self) -> ImageInner { self.my_image() }

	fn initial_layout_requirement(&self) -> ImageLayout { ImageLayout::PresentSrc }

	fn final_layout_requirement(&self) -> ImageLayout { ImageLayout::PresentSrc }

	fn conflicts_buffer(&self, other: &BufferAccess) -> bool { false }

	fn conflicts_image(&self, other: &ImageAccess) -> bool {
		self.my_image().image.key() == other.conflict_key() // TODO:
	}

	fn conflict_key(&self) -> u64 { self.my_image().image.key() }

	fn try_gpu_lock(&self, _: bool, _: ImageLayout) -> Result<(), AccessError> {
		// Swapchain image are only accessible after being acquired.
		Err(AccessError::SwapchainImageAcquireOnly)
	}

	unsafe fn increase_gpu_lock(&self) {}

	unsafe fn unlock(&self, _: Option<ImageLayout>) {
		// TODO: store that the image was initialized
	}
}

unsafe impl<W> ImageClearValue<<Format as FormatDesc>::ClearValue> for SwapchainImage<W> {
	fn decode(&self, value: <Format as FormatDesc>::ClearValue) -> Option<ClearValue> {
		Some(self.swapchain.format().decode_clear_value(value))
	}
}

unsafe impl<P, W> ImageContent<P> for SwapchainImage<W> {
	fn matches_format(&self) -> bool {
		true // FIXME:
	}
}

unsafe impl<W> ImageViewAccess for SwapchainImage<W> {
	fn parent(&self) -> &ImageAccess { self }

	fn dimensions(&self) -> ImageDimensions {
		let dims = self.swapchain.dimensions();
		ImageDimensions::Dim2D { width: dims[0], height: dims[1] }
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
