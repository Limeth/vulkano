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
	unsafe_image::{ImageCreationError, LinearLayout, UnsafeImage},
	unsafe_image_view::UnsafeImageView
};

#[cfg(test)]
mod tests {
	use std::{iter::Empty, u32};

	use super::{ImageCreationError, UnsafeImage};

	use crate::{
		format::Format,
		image::{ImageDimensions, ImageUsage},
		sync::Sharing
	};

	#[test]
	fn create_sampled() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let (_img, _) = unsafe {
			UnsafeImage::new(
				device,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				1,
				Sharing::Exclusive::<Empty<_>>,
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
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				1,
				Sharing::Exclusive::<Empty<_>>,
				false,
				false
			)
		}
		.unwrap();
	}

	#[test]
	fn zero_sample() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				0,
				1,
				Sharing::Exclusive::<Empty<_>>,
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
	fn non_po2_sample() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				5,
				1,
				Sharing::Exclusive::<Empty<_>>,
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
	fn zero_mipmap() {
		let (device, _) = gfx_dev_and_queue!();

		let usage = ImageUsage { sampled: true, ..ImageUsage::none() };

		let res = unsafe {
			UnsafeImage::new(
				device,
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				0,
				Sharing::Exclusive::<Empty<_>>,
				false,
				false
			)
		};

		match res {
			Err(ImageCreationError::InvalidMipmapsCount { .. }) => (),
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
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				u32::MAX,
				Sharing::Exclusive::<Empty<_>>,
				false,
				false
			)
		};

		match res {
			Err(ImageCreationError::InvalidMipmapsCount { obtained, valid_range }) => {
				assert_eq!(obtained, u32::MAX);
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
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				2,
				1,
				Sharing::Exclusive::<Empty<_>>,
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
				usage,
				Format::ASTC_5x4UnormBlock,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				u32::MAX,
				Sharing::Exclusive::<Empty<_>>,
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
				usage,
				Format::R8G8B8A8Unorm,
				ImageDimensions::Dim2D { width: 32, height: 32 },
				1,
				1,
				Sharing::Exclusive::<Empty<_>>,
				false,
				false
			)
		};

		match res {
			Err(ImageCreationError::UnsupportedUsage) => (),
			_ => panic!()
		};
	}

	// #[test]
	// fn cubecompatible_dims_mismatch() {
	// let (device, _) = gfx_dev_and_queue!();
	//
	// let usage = ImageUsage { sampled: true, ..ImageUsage::none() };
	//
	// let res = unsafe {
	// UnsafeImage::new(
	// device,
	// usage,
	// Format::R8G8B8A8Unorm,
	// ImageDimensions::Dim2D {
	// width: 32,
	// height: 64,
	// array_layers: 1,
	// cubemap_compatible: true
	// },
	// 1,
	// 1,
	// Sharing::Exclusive::<Empty<_>>,
	// false,
	// false
	// )
	// };
	//
	// match res {
	// Err(ImageCreationError::UnsupportedDimensions { .. }) => (),
	// _ => panic!()
	// };
	// }
}
