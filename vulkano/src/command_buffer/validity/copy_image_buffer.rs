// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt};

use crate::{
	buffer::TypedBufferAccess,
	device::{Device, DeviceOwned},
	format::{AcceptsPixels, Format, IncompatiblePixelsType},
	image::ImageViewAccess,
	VulkanObject
};

/// Type of operation to check.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CheckCopyBufferImageTy {
	BufferToImage,
	ImageToBuffer
}

/// Checks whether a copy buffer-image command is valid. Can check both buffer-to-image copies and
/// image-to-buffer copies.
///
/// # Panic
///
/// - Panics if the buffer and image were not created with `device`.
// TODO: handle compressed image formats
pub fn check_copy_buffer_image<B, I, P>(
	device: &Device, buffer: &B, image: &I, ty: CheckCopyBufferImageTy, image_offset: [u32; 3],
	image_size: [u32; 3], image_first_layer: u32, image_num_layers: u32, image_mipmap: u32
) -> Result<(), CheckCopyBufferImageError>
where
	I: ?Sized + ImageViewAccess,
	B: ?Sized + TypedBufferAccess<Content = [P]>,
	Format: AcceptsPixels<P> // TODO: use a trait on the image itself instead
{
	let buffer_inner = buffer.inner();
	assert_eq!(buffer_inner.buffer.device().internal_object(), device.internal_object());
	assert_eq!(image.parent().device().internal_object(), device.internal_object());

	match ty {
		CheckCopyBufferImageTy::BufferToImage => {
			if !buffer_inner.buffer.usage_transfer_source() {
				return Err(CheckCopyBufferImageError::SourceMissingTransferUsage)
			}
			if !image.inner().usage_transfer_destination() {
				return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage)
			}
		}
		CheckCopyBufferImageTy::ImageToBuffer => {
			if !image.inner().usage_transfer_source() {
				return Err(CheckCopyBufferImageError::SourceMissingTransferUsage)
			}
			if !buffer_inner.buffer.usage_transfer_destination() {
				return Err(CheckCopyBufferImageError::DestinationMissingTransferUsage)
			}
		}
	}

	if image.parent().samples().get() != 1 {
		return Err(CheckCopyBufferImageError::UnexpectedMultisampled)
	}

	let image_dimensions = match image.dimensions().mipmap_dimensions(image_mipmap) {
		Some(d) => d,
		None => return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange)
	};

	if image_first_layer + image_num_layers > image_dimensions.array_layers().get() {
		return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange)
	}

	if image_offset[0] + image_size[0] > image_dimensions.width().get() {
		return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange)
	}

	if image_offset[1] + image_size[1] > image_dimensions.height().get() {
		return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange)
	}

	if image_offset[2] + image_size[2] > image_dimensions.depth().get() {
		return Err(CheckCopyBufferImageError::ImageCoordinatesOutOfRange)
	}

	image.format().ensure_accepts()?;

	{
		let num_texels = image_size[0] * image_size[1] * image_size[2] * image_num_layers;
		let required_len = num_texels as usize * image.format().rate() as usize;
		if required_len > buffer.len() {
			return Err(CheckCopyBufferImageError::BufferTooSmall {
				required_len,
				actual_len: buffer.len()
			})
		}
	}

	// TODO: check memory overlap?

	Ok(())
}

/// Error that can happen from `check_copy_buffer_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyBufferImageError {
	/// The source buffer or image is missing the transfer source usage.
	SourceMissingTransferUsage,
	/// The destination buffer or image is missing the transfer destination usage.
	DestinationMissingTransferUsage,
	/// The source and destination are overlapping.
	OverlappingRanges,
	/// The image must not be multisampled.
	UnexpectedMultisampled,
	/// The image coordinates are out of range.
	ImageCoordinatesOutOfRange,
	/// The type of pixels in the buffer isn't compatible with the image format.
	WrongPixelType(IncompatiblePixelsType),
	/// The buffer is too small for the copy operation.
	BufferTooSmall {
		/// Required number of elements in the buffer.
		required_len: usize,
		/// Actual number of elements in the buffer.
		actual_len: usize
	}
}
impl fmt::Display for CheckCopyBufferImageError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckCopyBufferImageError::SourceMissingTransferUsage => {
				write!(f, "The source buffer or image is missing the transfer source usage")
			}
			CheckCopyBufferImageError::DestinationMissingTransferUsage => write!(
				f,
				"The destination buffer or image is missing the transfer destination usage"
			),
			CheckCopyBufferImageError::OverlappingRanges => {
				write!(f, "The source and destination are overlapping")
			}
			CheckCopyBufferImageError::UnexpectedMultisampled => {
				write!(f, "The image must not be multisampled")
			}
			CheckCopyBufferImageError::ImageCoordinatesOutOfRange => {
				write!(f, "The image coordinates are out of range")
			}
			CheckCopyBufferImageError::WrongPixelType(e) => e.fmt(f),
			CheckCopyBufferImageError::BufferTooSmall { required_len, actual_len } => write!(
				f,
				"The buffer is too small ({}) for the copy operation ({})",
				actual_len, required_len
			)
		}
	}
}
impl error::Error for CheckCopyBufferImageError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
impl From<IncompatiblePixelsType> for CheckCopyBufferImageError {
	fn from(err: IncompatiblePixelsType) -> CheckCopyBufferImageError {
		CheckCopyBufferImageError::WrongPixelType(err)
	}
}
