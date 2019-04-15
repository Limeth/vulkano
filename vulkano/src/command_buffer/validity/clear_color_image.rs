// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt};

use crate::{device::Device, image::ImageViewAccess, VulkanObject};

/// Checks whether a clear color image command is valid.
///
/// # Panic
///
/// - Panics if the destination was not created with `device`.
pub fn check_clear_color_image<I>(
	device: &Device, image: &I, first_layer: u32, num_layers: u32, first_mipmap: u32,
	num_mipmaps: u32
) -> Result<(), CheckClearColorImageError>
where
	I: ?Sized + ImageViewAccess
{
	assert_eq!(image.parent().device().internal_object(), device.internal_object());

	if !image.usage().transfer_destination {
		return Err(CheckClearColorImageError::MissingTransferUsage)
	}

	if first_layer + num_layers > image.subresource_range().array_layers.get() {
		return Err(CheckClearColorImageError::OutOfRange)
	}

	if first_mipmap + num_mipmaps > image.subresource_range().mipmap_levels.get() {
		return Err(CheckClearColorImageError::OutOfRange)
	}

	Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckClearColorImageError {
	/// The image is missing the transfer destination usage.
	MissingTransferUsage,
	/// The array layers and mipmap levels are out of range.
	OutOfRange
}
impl fmt::Display for CheckClearColorImageError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckClearColorImageError::MissingTransferUsage => {
				write!(f, "The image is missing the transfer destination usage")
			}
			CheckClearColorImageError::OutOfRange => {
				write!(f, "The array layers and mipmap levels are out of range")
			}
		}
	}
}
impl error::Error for CheckClearColorImageError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
