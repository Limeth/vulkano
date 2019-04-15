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

use std::{error, fmt, ops::Range};

use crate::{image::ImageDimensions, memory::DeviceMemoryAllocError, Error, OomError};

mod unsafe_image;
mod unsafe_image_view;

pub use self::{
	unsafe_image::{LinearLayout, UnsafeImage},
	unsafe_image_view::UnsafeImageView
};

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageCreationError {
	/// Allocating memory failed.
	AllocError(DeviceMemoryAllocError),
	/// The dimensions are too large, or one of the dimensions is 0.
	UnsupportedDimensions(ImageDimensions),
	/// A wrong number of mipmaps was provided.
	InvalidMipmapsCount { requested: u32, valid_range: Range<u32> },
	/// The requested number of samples is not supported, or is 0.
	UnsupportedSamplesCount(u32),
	/// The requested format is not supported by the Vulkan implementation.
	FormatNotSupported,
	/// The format is supported, but at least one of the requested usages is not supported.
	UnsupportedUsage,
	/// The `shader_storage_image_multisample` feature must be enabled to create such an image.
	ShaderStorageImageMultisampleFeatureNotEnabled
}
impl fmt::Display for ImageCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ImageCreationError::AllocError(e) => write!(f, "Memory allocation failed: {}", e),
			ImageCreationError::InvalidMipmapsCount { requested, valid_range } => write!(
				f,
				"A wrong number of mipmaps provided: {} valid range: {:?}",
				requested, valid_range
			),
			ImageCreationError::UnsupportedSamplesCount(samples) => {
				write!(f, "The requested number of sampler is not supported: {}", samples)
			}
			ImageCreationError::UnsupportedDimensions(dims) => {
				write!(f, "The requested dimensions are not supported: {:?}", dims)
			}
			ImageCreationError::FormatNotSupported => {
				write!(f, "The requested format is not supported")
			}
			ImageCreationError::UnsupportedUsage => {
				write!(f, "The requested usage is not supported for requested format")
			}
			ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
				write!(f, "The `shader_storage_image_multisample` feature must be enabled")
			}
		}
	}
}
impl error::Error for ImageCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			ImageCreationError::AllocError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for ImageCreationError {
	fn from(err: OomError) -> ImageCreationError {
		ImageCreationError::AllocError(DeviceMemoryAllocError::OomError(err))
	}
}
impl From<DeviceMemoryAllocError> for ImageCreationError {
	fn from(err: DeviceMemoryAllocError) -> ImageCreationError {
		ImageCreationError::AllocError(err)
	}
}
impl From<Error> for ImageCreationError {
	fn from(err: Error) -> ImageCreationError {
		match err {
			err @ Error::OutOfHostMemory => ImageCreationError::AllocError(err.into()),
			err @ Error::OutOfDeviceMemory => ImageCreationError::AllocError(err.into()),
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{iter::Empty, num::NonZeroU32, u32};

	use super::{ImageCreationError, UnsafeImage};

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
			Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (),
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
			Err(ImageCreationError::InvalidMipmapsCount { requested, valid_range }) => {
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
			Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled) => (),
			Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (), // unlikely but possible
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
			Err(ImageCreationError::FormatNotSupported) => (),
			Err(ImageCreationError::UnsupportedUsage) => (),
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
			Err(ImageCreationError::UnsupportedUsage) => (),
			_ => panic!()
		};
	}
}
