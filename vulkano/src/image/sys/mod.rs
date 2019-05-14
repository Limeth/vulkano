// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level implementation of images and images views.
//!
//! This module contains low-level wrappers around the Vulkan image and image view types. All
//! other image or image view types of this library, and all custom image or image view types
//! that you create must wrap around the types in this module.

mod unsafe_image;
mod unsafe_image_view;

pub use self::{
	unsafe_image::{UnsafeImage, UnsafeImageCreationError},
	unsafe_image_view::{UnsafeImageView, UnsafeImageViewCreationError}
};

/// Describes the memory layout of an image with linear tiling.
///
/// Obtained by calling `*_linear_layout` on the image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LinearLayout {
	/// Number of bytes from the start of the memory and the start of the queried subresource.
	pub offset: usize,
	/// Total number of bytes for the queried subresource. Can be used for a safety check.
	pub size: usize,
	/// Number of bytes between two texels or two blocks in adjacent rows.
	pub row_pitch: usize,
	/// Number of bytes between two texels or two blocks in adjacent array layers. This value is
	/// undefined for images with only one array layer.
	pub array_pitch: usize,
	/// Number of bytes between two texels or two blocks in adjacent depth layers. This value is
	/// undefined for images that are not three-dimensional.
	pub depth_pitch: usize
}

#[cfg(test)]
mod tests {
	use std::{iter::Empty, num::NonZeroU32, u32};

	use super::{UnsafeImage, UnsafeImageCreationError};

	use crate::{
		format::Format,
		image::{ImageDimensions, ImageUsage},
		sync::Sharing
	};

	const NONZERO_1: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(1) };
	const NONZERO_32: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(32) };

	#[test]
	fn create_sampled() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let (_img, _) = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NONZERO_1,
				NONZERO_1,
				false,
				false
			)
		}
		.unwrap();
	}

	#[test]
	fn create_transient() {
		let (device, _) = gfx_dev_and_queue!();

		let usage =
			ImageUsage { transient_attachment: true, color_attachment: true, ..ImageUsage::none() };

		let (_img, _) = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NONZERO_1,
				NONZERO_1,
				false,
				false
			)
		}
		.unwrap();
	}

	#[test]
	fn non_po2_sample() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NonZeroU32::new_unchecked(5),
				NONZERO_1,
				false,
				false
			)
		};

		match res {
			Err(UnsafeImageCreationError::UnsupportedSamplesCount { .. }) => (),
			_ => panic!()
		};
	}

	#[test]
	#[ignore] // TODO: AMD card seems to support a u32::MAX number of mipmaps
	fn mipmaps_too_high() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NONZERO_1,
				NonZeroU32::new_unchecked(u32::MAX),
				false,
				false
			)
		};

		match res {
			Err(UnsafeImageCreationError::InvalidMipmapsCount { requested, valid_range }) => {
				assert_eq!(requested, u32::MAX);
				assert_eq!(valid_range.start, 1);
			}
			_ => panic!()
		};
	}

	#[test]
	fn shader_storage_image_multisample() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { storage: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NonZeroU32::new_unchecked(2),
				NONZERO_1,
				false,
				false
			)
		};

		match res {
			Err(UnsafeImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled) => (),
			Err(UnsafeImageCreationError::UnsupportedSamplesCount { .. }) => (), /* unlikely but possible */
			_ => panic!()
		};
	}

	#[test]
	fn compressed_not_color_attachment() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { color_attachment: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::ASTC_5x4UnormBlock,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NONZERO_1,
				NonZeroU32::new_unchecked(u32::MAX),
				false,
				false
			)
		};

		match res {
			Err(UnsafeImageCreationError::FormatNotSupported) => (),
			Err(UnsafeImageCreationError::UnsupportedUsage) => (),
			_ => panic!()
		};
	}

	#[test]
	fn transient_forbidden_with_some_usages() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { transient_attachment: true, sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				Sharing::Exclusive::<Empty<_>>,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: NONZERO_32, height: NONZERO_32 },
				NONZERO_1,
				NONZERO_1,
				false,
				false
			)
		};

		match res {
			Err(UnsafeImageCreationError::UnsupportedUsage) => (),
			_ => panic!()
		};
	}
}
