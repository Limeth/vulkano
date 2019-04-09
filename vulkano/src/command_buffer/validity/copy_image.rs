// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt};

use crate::{
	device::Device,
	format::{FormatTy, PossibleCompressedFormatDesc},
	image::{ImageAccess, ImageDimensionType},
	VulkanObject
};

/// Checks whether a copy image command is valid.
///
/// Note that this doesn't check whether `layer_count` is equal to 0. TODO: change that?
///
/// # Panic
///
/// - Panics if the source or the destination was not created with `device`.
pub fn check_copy_image<S, D>(
	device: &Device, source: &S, source_offset: [i32; 3], source_base_array_layer: u32,
	source_mip_level: u32, destination: &D, destination_offset: [i32; 3],
	destination_base_array_layer: u32, destination_mip_level: u32, extent: [u32; 3],
	layer_count: u32
) -> Result<(), CheckCopyImageError>
where
	S: ?Sized + ImageAccess,
	D: ?Sized + ImageAccess
{
	let source_inner = source.inner();
	let destination_inner = destination.inner();

	assert_eq!(source_inner.image.device().internal_object(), device.internal_object());
	assert_eq!(destination_inner.image.device().internal_object(), device.internal_object());

	if !source_inner.image.usage_transfer_source() {
		return Err(CheckCopyImageError::MissingTransferSourceUsage)
	}

	if !destination_inner.image.usage_transfer_destination() {
		return Err(CheckCopyImageError::MissingTransferDestinationUsage)
	}

	if source.samples() != destination.samples() {
		return Err(CheckCopyImageError::SampleCountMismatch)
	}

	let source_format_ty = source.format().ty();
	let destination_format_ty = destination.format().ty();

	if source_format_ty.is_depth_and_or_stencil() {
		if source.format() != destination.format() {
			return Err(CheckCopyImageError::DepthStencilFormatMismatch)
		}
	}

	// TODO: The correct check here is that the uncompressed element size of the source is
	// equal to the compressed element size of the destination.  However, format doesn't
	// currently expose this information, so to be safe, we simply disallow compressed formats.
	if source.format().is_compressed()
		|| destination.format().is_compressed()
		|| (source.format().size() != destination.format().size())
	{
		return Err(CheckCopyImageError::SizeIncompatibleFormatsTypes {
			source_format_ty: source.format().ty(),
			destination_format_ty: destination.format().ty()
		})
	}

	let source_dimensions = match source.dimensions().mipmap_dimensions(source_mip_level) {
		Some(d) => d,
		None => return Err(CheckCopyImageError::SourceCoordinatesOutOfRange)
	};

	let destination_dimensions =
		match destination.dimensions().mipmap_dimensions(destination_mip_level) {
			Some(d) => d,
			None => return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange)
		};

	if source_base_array_layer + layer_count > source_dimensions.array_layers() {
		return Err(CheckCopyImageError::SourceCoordinatesOutOfRange)
	}

	if destination_base_array_layer + layer_count > destination_dimensions.array_layers() {
		return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange)
	}

	if source_offset[0] < 0 || source_offset[0] as u32 + extent[0] > source_dimensions.width() {
		return Err(CheckCopyImageError::SourceCoordinatesOutOfRange)
	}

	if source_offset[1] < 0 || source_offset[1] as u32 + extent[1] > source_dimensions.height() {
		return Err(CheckCopyImageError::SourceCoordinatesOutOfRange)
	}

	if source_offset[2] < 0 || source_offset[2] as u32 + extent[2] > source_dimensions.depth() {
		return Err(CheckCopyImageError::SourceCoordinatesOutOfRange)
	}

	if destination_offset[0] < 0
		|| destination_offset[0] as u32 + extent[0] > destination_dimensions.width()
	{
		return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_offset[1] < 0
		|| destination_offset[1] as u32 + extent[1] > destination_dimensions.height()
	{
		return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_offset[2] < 0
		|| destination_offset[2] as u32 + extent[2] > destination_dimensions.depth()
	{
		return Err(CheckCopyImageError::DestinationCoordinatesOutOfRange)
	}

	match source_dimensions.dimension_type() {
		ImageDimensionType::D1 => {
			if source_offset[1] != 0 || extent[1] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
			if source_offset[2] != 0 || extent[2] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionType::D2 | ImageDimensionType::Cube => {
			if source_offset[2] != 0 || extent[2] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionType::D3 => {}
	}

	match destination_dimensions.dimension_type() {
		ImageDimensionType::D1 => {
			if destination_offset[1] != 0 || extent[1] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
			if destination_offset[2] != 0 || extent[2] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionType::D2 | ImageDimensionType::Cube => {
			if destination_offset[2] != 0 || extent[2] != 1 {
				return Err(CheckCopyImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionType::D3 => {}
	}

	Ok(())
}

/// Error that can happen from `check_copy_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckCopyImageError {
	/// The source is missing the transfer source usage.
	MissingTransferSourceUsage,
	/// The destination is missing the transfer destination usage.
	MissingTransferDestinationUsage,
	/// The number of samples in the source and destination do not match.
	SampleCountMismatch,
	/// The format of the source and destination must be equal when copying depth/stencil images.
	DepthStencilFormatMismatch,
	/// The types of the source format and the destination format aren't size-compatible.
	SizeIncompatibleFormatsTypes { source_format_ty: FormatTy, destination_format_ty: FormatTy },
	/// The offsets, array layers and/or mipmap levels are out of range in the source image.
	SourceCoordinatesOutOfRange,
	/// The offsets, array layers and/or mipmap levels are out of range in the destination image.
	DestinationCoordinatesOutOfRange,
	/// The offsets or extent are incompatible with the image type.
	IncompatibleRangeForImageType
}
impl fmt::Display for CheckCopyImageError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckCopyImageError::MissingTransferSourceUsage => write!(f, "The source is missing the transfer source usage"),
			CheckCopyImageError::MissingTransferDestinationUsage => write!(f, "The destination is missing the transfer destination usage"),
			CheckCopyImageError::SampleCountMismatch => write!(f, "The number of samples in the source and destination do not match"),
			CheckCopyImageError::DepthStencilFormatMismatch => write!(f, "The format of the source and destination must be equal when blitting depth/stencil images"),
			CheckCopyImageError::SizeIncompatibleFormatsTypes { source_format_ty, destination_format_ty }
				=> write!(f, "The types of the source format ({:?}) and the destination format ({:?}) aren't size-compatible", source_format_ty, destination_format_ty),
			CheckCopyImageError::SourceCoordinatesOutOfRange => write!(f, "The offsets, array layers and/or mipmap levels are out of range in the source image"),
			CheckCopyImageError::DestinationCoordinatesOutOfRange => write!(f, "The offsets, array layers and/or mipmap levels are out of range in the destination image"),
			CheckCopyImageError::IncompatibleRangeForImageType => write!(f, "The top-left and/or bottom-right coordinates are incompatible with the image type")
		}
	}
}
impl error::Error for CheckCopyImageError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
