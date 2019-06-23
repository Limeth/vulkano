// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{num::NonZeroU32, sync::Arc};

use crate::{
	buffer::BufferAccess,
	format::{ClearValue, Format, FormatDesc},
	image::{
		layout::{ImageLayout, ImageLayoutEnd, RequiredLayouts},
		sys::{UnsafeImage, UnsafeImageView, UnsafeImageViewCreationError},
		traits::{ImageAccess, ImageClearValue, ImageContent, ImageViewAccess},
		ImageDimensions,
		ImageSubresourceLayoutError,
		ImageSubresourceRange,
		ImageViewType
	},
	swapchain::Swapchain,
	sync::AccessError,
	OomError
};

pub trait SwapchainImage: Send + Sync + std::fmt::Debug + ImageAccess + ImageViewAccess {
	// type Swapchain;

	/// Returns the dimensions of the image.
	///
	/// A `SwapchainImage` is always two-dimensional.
	fn inner_dimensions(&self) -> [NonZeroU32; 2];

	// /// Returns the swapchain this image belongs to.
	// fn swapchain(&self) -> &Arc<Self::Swapchain>;

	fn inner_image(&self) -> &UnsafeImage;

	fn index(&self) -> usize;
}

/// An image and view combination that is part of a swapchain.
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
pub struct VulkanSwapchainImage<W> {
	swapchain: Arc<Swapchain<W>>,
	image_offset: usize,
	view: UnsafeImageView
}

impl<'a, W: 'a + Send + Sync> VulkanSwapchainImage<W> {
	const REQUIRED_LAYOUTS: RequiredLayouts =
		RequiredLayouts { global: Some(ImageLayoutEnd::PresentSrc), ..RequiredLayouts::none() };

	/// Builds a `VulkanSwapchainImage` from raw components.
	///
	/// This is an internal method that you shouldn't call.
	pub unsafe fn from_raw(
		swapchain: Arc<Swapchain<W>>, id: usize
	) -> Result<Arc<dyn SwapchainImage + 'a>, OomError> {
		let image = swapchain.raw_image(id).unwrap();
		let view = match UnsafeImageView::new(
			&image,
			Some(ImageViewType::Dim2D),
			None,
			Default::default(),
			ImageSubresourceRange {
				array_layers: crate::NONZERO_ONE,
				array_layers_offset: 0,

				mipmap_levels: crate::NONZERO_ONE,
				mipmap_levels_offset: 0
			}
		) {
			Ok(v) => v,
			Err(UnsafeImageViewCreationError::OomError(e)) => return Err(e),
			e => panic!("Could not create swapchain view: {:?}", e)
		};

		Ok(Arc::new(VulkanSwapchainImage { swapchain: swapchain.clone(), image_offset: id, view }))
	}
}

impl<W: Send + Sync> SwapchainImage for VulkanSwapchainImage<W> {
	// type Swapchain = Swapchain<W>;

	/// Returns the dimensions of the image.
	///
	/// A `SwapchainImage` is always two-dimensional.
	fn inner_dimensions(&self) -> [NonZeroU32; 2] { self.inner_image().dimensions().width_height() }

	// /// Returns the swapchain this image belongs to.
	// fn swapchain(&self) -> &Arc<Self::Swapchain> { &self.swapchain }

	fn inner_image(&self) -> &UnsafeImage { self.swapchain.raw_image(self.image_offset).unwrap() }

        fn index(&self) -> usize { self.image_offset }
}

unsafe impl<W: Send + Sync> ImageAccess for VulkanSwapchainImage<W> {
	fn inner(&self) -> &UnsafeImage { self.inner_image() }

	fn conflicts_buffer(&self, other: &BufferAccess) -> bool { false }

	fn conflicts_image(
		&self, subresource_range: ImageSubresourceRange, other: &dyn ImageAccess,
		other_subresource_range: ImageSubresourceRange
	) -> bool {
		if ImageAccess::inner(self).key() == other.conflict_key() {
			subresource_range.overlaps_with(&other_subresource_range)
		} else {
			false
		}
	}

	fn conflict_key(&self) -> u64 { ImageAccess::inner(self).key() }

	fn current_layout(
		&self, _: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		// TODO: Is this okay?
		Ok(ImageLayout::PresentSrc)
	}

	fn initiate_gpu_lock(
		&self, _: ImageSubresourceRange, _: bool, _: ImageLayout
	) -> Result<(), AccessError> {
		// Swapchain image are only accessible after being acquired.
		// This is handled by the swapchain itself.
		Err(AccessError::SwapchainImageAcquireOnly)
	}

	unsafe fn increase_gpu_lock(&self, _: ImageSubresourceRange) {}

	unsafe fn decrease_gpu_lock(
		&self, _: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
	) {
		// TODO: store that the image was initialized?
	}
}

// unsafe impl<W: Send + Sync> ImageClearValue<<Format as FormatDesc>::ClearValue> for VulkanSwapchainImage<W> {
// 	fn decode(&self, value: <Format as FormatDesc>::ClearValue) -> Option<ClearValue> {
// 		Some(self.swapchain.format().decode_clear_value(value))
// 	}
// }
// unsafe impl<P, W: Send + Sync> ImageContent<P> for VulkanSwapchainImage<W> {
// 	fn matches_format(&self) -> bool {
// 		true // FIXME:
// 	}
// }
unsafe impl<W: Send + Sync> ImageViewAccess for VulkanSwapchainImage<W> {
	fn parent(&self) -> &ImageAccess { self }

	fn inner(&self) -> &UnsafeImageView { &self.view }

	fn dimensions(&self) -> ImageDimensions { self.view.dimensions() }

	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool { false }

	fn required_layouts(&self) -> &RequiredLayouts { &Self::REQUIRED_LAYOUTS }
}
impl<W> std::fmt::Debug for VulkanSwapchainImage<W> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(
			f,
			"VulkanSwapchainImage {{ swapchain: {:?}, image_offset: {}, view: {:?} }}",
			self.swapchain, self.image_offset, self.view
		)
	}
}
