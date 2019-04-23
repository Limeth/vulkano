// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt};

use vk_sys as vk;

use crate::image::ImageUsage;

pub mod matrix;
pub mod typesafety;

pub use matrix::{
	ImageLayoutMatrix,
	ImageLayoutMatrixEntry,
	ImageLayoutMatrixIter,
	ImageLayoutMatrixIterMut
};
pub use typesafety::*;

/// Layout of an image.
///
/// In the Vulkan API, each mipmap level of each array layer is in one of the layouts of this enum.
///
/// Unless you use some sort of high-level shortcut function, an image always starts in either
/// the `Undefined` or the `Preinitialized` layout.
/// Before you can use an image for a given purpose, you must ensure that the image in question is
/// in the layout required for that purpose. For example if you want to write data to an image, you
/// must first transition the image to the `TransferDstOptimal` layout. The `General` layout can
/// also be used as a general-purpose fit-all layout, but using it will result in slower operations.
///
/// Transitioning between layouts can only be done through a GPU-side operation that is part of
/// a command buffer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageLayout {
	Undefined = vk::IMAGE_LAYOUT_UNDEFINED,

	General = vk::IMAGE_LAYOUT_GENERAL,

	ColorAttachmentOptimal = vk::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	DepthStencilAttachmentOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	DepthStencilReadOnlyOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
	ShaderReadOnlyOptimal = vk::IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,

	TransferSrcOptimal = vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
	TransferDstOptimal = vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,

	Preinitialized = vk::IMAGE_LAYOUT_PREINITIALIZED,

	PresentSrc = vk::IMAGE_LAYOUT_PRESENT_SRC_KHR,

	DepthReadOnlyStencilAttachmentOptimal =
		vk::IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
	DepthAttachmentStencilReadOnlyOptimal =
		vk::IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL /* SharedPresent = vk::IMAGE_LAYOUT_SHARED_PRESENT_KHR? What is this? What is it good for? */
}
impl ImageLayout {
	/// Returns Ok(()) if layout is valid for given usage.
	pub fn valid_for_usage(&self, usage: ImageUsage) -> Result<(), InvalidLayoutUsageError> {
		let result = match self {
			ImageLayout::Undefined => true,
			ImageLayout::General => true,

			ImageLayout::ColorAttachmentOptimal => usage.color_attachment,
			ImageLayout::DepthStencilAttachmentOptimal => usage.depth_stencil_attachment,
			ImageLayout::DepthStencilReadOnlyOptimal => usage.depth_stencil_attachment,
			ImageLayout::ShaderReadOnlyOptimal => usage.sampled || usage.input_attachment,

			ImageLayout::TransferSrcOptimal => usage.transfer_source,
			ImageLayout::TransferDstOptimal => usage.transfer_destination,

			ImageLayout::Preinitialized => true,

			ImageLayout::PresentSrc => true,

			ImageLayout::DepthReadOnlyStencilAttachmentOptimal => usage.depth_stencil_attachment,
			ImageLayout::DepthAttachmentStencilReadOnlyOptimal => usage.depth_stencil_attachment
		};

		if result {
			Ok(())
		} else {
			Err(InvalidLayoutUsageError { layout: *self, usage })
		}
	}
}

/// Describes how the view behaves in respect to requesting layouts.
#[derive(Debug, Default)]
pub struct RequiredLayouts {
	/// The layout required at the end of a command buffer.
	///
	/// Is `None`, the view won't request a specific layout at the end of the
	/// auto command buffer.
	pub global: Option<ImageLayoutEnd>,

	/// Layout in descriptor sets as storage image.
	///
	/// None means that this view cannot be used in such descriptors.
	pub storage: Option<ImageLayoutStorageImage>,
	/// Layout in descriptor sets as sampled image.
	///
	/// None means that this view cannot be used in such descriptors.
	pub sampled: Option<ImageLayoutSampledImage>,
	/// Layout in descriptor sets as combined image and sampler.
	///
	/// None means that this view cannot be used in such descriptors.
	pub combined: Option<ImageLayoutCombinedImage>,
	/// Layout in descriptor sets as input attachment.
	///
	/// None means that this view cannot be used in such descriptors.
	/// TODO: currently is ignored in descriptor set logic
	pub input_attachment: Option<ImageLayoutInputAttachment>
}
impl RequiredLayouts {
	/// Same as default but const.
	pub const fn none() -> Self {
		RequiredLayouts {
			global: None,

			storage: None,
			sampled: None,
			combined: None,
			input_attachment: None
		}
	}

	/// Calls `valid_for_usage` on each field.
	pub fn valid_for_usage(&self, usage: ImageUsage) -> Result<(), InvalidLayoutUsageError> {
		if let Some(layout) = self.global {
			layout.valid_for_usage(usage)?;
		}
		if let Some(layout) = self.storage {
			layout.valid_for_usage(usage)?;
		}
		if let Some(layout) = self.sampled {
			layout.valid_for_usage(usage)?;
		}
		if let Some(layout) = self.combined {
			layout.valid_for_usage(usage)?;
		}
		if let Some(layout) = self.input_attachment {
			layout.valid_for_usage(usage)?;
		}

		Ok(())
	}
}

/// The layout is invalid with this usage.
#[derive(Debug)]
pub struct InvalidLayoutUsageError {
	pub layout: ImageLayout,
	pub usage: ImageUsage
}
impl fmt::Display for InvalidLayoutUsageError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "The layout {:?} is invalid with usage {:?}", self.layout, self.usage)
	}
}
impl error::Error for InvalidLayoutUsageError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
