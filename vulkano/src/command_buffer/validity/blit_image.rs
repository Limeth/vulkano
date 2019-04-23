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
	format::FormatTy,
	image::{ImageDimensionsType, ImageViewAccess},
	sampler::Filter,
	VulkanObject
};

/// Checks whether a blit image command is valid.
///
/// Note that this doesn't check whether `layer_count` is equal to 0. TODO: change that?
///
/// # Panic
///
/// - Panics if the source or the destination was not created with `device`.
pub fn check_blit_image<S, D>(
	device: &Device, source: &S, source_top_left: [i32; 3], source_bottom_right: [i32; 3],
	source_base_array_layer: u32, source_mip_level: u32, destination: &D,
	destination_top_left: [i32; 3], destination_bottom_right: [i32; 3],
	destination_base_array_layer: u32, destination_mip_level: u32, layer_count: u32,
	filter: Filter
) -> Result<(), CheckBlitImageError>
where
	S: ?Sized + ImageViewAccess,
	D: ?Sized + ImageViewAccess
{
	assert_eq!(source.parent().device().internal_object(), device.internal_object());
	assert_eq!(destination.parent().device().internal_object(), device.internal_object());

	if !source.usage().transfer_source {
		return Err(CheckBlitImageError::MissingTransferSourceUsage)
	}

	if !destination.usage().transfer_destination {
		return Err(CheckBlitImageError::MissingTransferDestinationUsage)
	}

	if !source.parent().inner().supports_blit_source() {
		return Err(CheckBlitImageError::SourceFormatNotSupported)
	}

	if !destination.parent().inner().supports_blit_destination() {
		return Err(CheckBlitImageError::DestinationFormatNotSupported)
	}

	if source.parent().samples().get() != 1 || destination.parent().samples().get() != 1 {
		return Err(CheckBlitImageError::UnexpectedMultisampled)
	}

	let source_format_ty = source.format().ty();
	let destination_format_ty = destination.format().ty();

	if source_format_ty.is_depth_and_or_stencil() {
		if source.format() != destination.format() {
			return Err(CheckBlitImageError::DepthStencilFormatMismatch)
		}

		if filter != Filter::Nearest {
			return Err(CheckBlitImageError::DepthStencilNearestMandatory)
		}
	}

	let types_should_be_same = source_format_ty == FormatTy::Uint
		|| destination_format_ty == FormatTy::Uint
		|| source_format_ty == FormatTy::Sint
		|| destination_format_ty == FormatTy::Sint;
	if types_should_be_same && (source_format_ty != destination_format_ty) {
		return Err(CheckBlitImageError::IncompatibleFormatsTypes {
			source_format_ty: source.format().ty(),
			destination_format_ty: destination.format().ty()
		})
	}

	let source_dimensions = match source.dimensions().mipmap_dimensions(source_mip_level) {
		Some(d) => d,
		None => return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	};

	let destination_dimensions =
		match destination.dimensions().mipmap_dimensions(destination_mip_level) {
			Some(d) => d,
			None => return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
		};

	if source_base_array_layer + layer_count > source_dimensions.array_layers().get() {
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if destination_base_array_layer + layer_count > destination_dimensions.array_layers().get() {
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if source_top_left[0] < 0 || source_top_left[0] > source_dimensions.width().get() as i32 {
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if source_top_left[1] < 0 || source_top_left[1] > source_dimensions.height().get() as i32 {
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if source_top_left[2] < 0 || source_top_left[2] > source_dimensions.depth().get() as i32 {
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if source_bottom_right[0] < 0 || source_bottom_right[0] > source_dimensions.width().get() as i32
	{
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if source_bottom_right[1] < 0
		|| source_bottom_right[1] > source_dimensions.height().get() as i32
	{
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if source_bottom_right[2] < 0 || source_bottom_right[2] > source_dimensions.depth().get() as i32
	{
		return Err(CheckBlitImageError::SourceCoordinatesOutOfRange)
	}

	if destination_top_left[0] < 0
		|| destination_top_left[0] > destination_dimensions.width().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_top_left[1] < 0
		|| destination_top_left[1] > destination_dimensions.height().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_top_left[2] < 0
		|| destination_top_left[2] > destination_dimensions.depth().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_bottom_right[0] < 0
		|| destination_bottom_right[0] > destination_dimensions.width().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_bottom_right[1] < 0
		|| destination_bottom_right[1] > destination_dimensions.height().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	if destination_bottom_right[2] < 0
		|| destination_bottom_right[2] > destination_dimensions.depth().get() as i32
	{
		return Err(CheckBlitImageError::DestinationCoordinatesOutOfRange)
	}

	match source_dimensions.dimensions_type() {
		ImageDimensionsType::D1 => {
			if source_top_left[1] != 0 || source_bottom_right[1] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
			if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionsType::D2 | ImageDimensionsType::Cube => {
			if source_top_left[2] != 0 || source_bottom_right[2] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionsType::D3 => {}
	}

	match destination_dimensions.dimensions_type() {
		ImageDimensionsType::D1 => {
			if destination_top_left[1] != 0 || destination_bottom_right[1] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
			if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionsType::D2 | ImageDimensionsType::Cube => {
			if destination_top_left[2] != 0 || destination_bottom_right[2] != 1 {
				return Err(CheckBlitImageError::IncompatibleRangeForImageType)
			}
		}
		ImageDimensionsType::D3 => {}
	}

	Ok(())
}

/// Error that can happen from `check_clear_color_image`.
#[derive(Debug, Copy, Clone)]
pub enum CheckBlitImageError {
	/// The source is missing the transfer source usage.
	MissingTransferSourceUsage,
	/// The destination is missing the transfer destination usage.
	MissingTransferDestinationUsage,
	/// The format of the source image doesn't support blit operations.
	SourceFormatNotSupported,
	/// The format of the destination image doesn't support blit operations.
	DestinationFormatNotSupported,
	/// You must use the nearest filter when blitting depth/stencil images.
	DepthStencilNearestMandatory,
	/// The format of the source and destination must be equal when blitting depth/stencil images.
	DepthStencilFormatMismatch,
	/// The types of the source format and the destination format aren't compatible.
	IncompatibleFormatsTypes { source_format_ty: FormatTy, destination_format_ty: FormatTy },
	/// Blitting between multisampled images is forbidden.
	UnexpectedMultisampled,
	/// The offsets, array layers and/or mipmap levels are out of range in the source image.
	SourceCoordinatesOutOfRange,
	/// The offsets, array layers and/or mipmap levels are out of range in the destination image.
	DestinationCoordinatesOutOfRange,
	/// The top-left and/or bottom-right coordinates are incompatible with the image type.
	IncompatibleRangeForImageType
}
impl fmt::Display for CheckBlitImageError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckBlitImageError::MissingTransferSourceUsage => write!(f, "The source is missing the transfer source usage"),
			CheckBlitImageError::MissingTransferDestinationUsage => write!(f, "The destination is missing the transfer destination usage"),
			CheckBlitImageError::SourceFormatNotSupported => write!(f, "The format of the source image doesn't support blit operations"),
			CheckBlitImageError::DestinationFormatNotSupported => write!(f, "The format of the destination image doesn't support blit operations"),
			CheckBlitImageError::DepthStencilNearestMandatory => write!(f, "You must use the nearest filter when blitting depth/stencil images"),
			CheckBlitImageError::DepthStencilFormatMismatch => write!(f, "The format of the source and destination must be equal when blitting depth/stencil images"),
			CheckBlitImageError::IncompatibleFormatsTypes { source_format_ty, destination_format_ty }
				=> write!(f, "The types of the source format ({:?}) and the destination format ({:?}) aren't compatible", source_format_ty, destination_format_ty),
			CheckBlitImageError::UnexpectedMultisampled => write!(f, "Blitting between multisampled images is forbidden"),
			CheckBlitImageError::SourceCoordinatesOutOfRange => write!(f, "The offsets, array layers and/or mipmap levels are out of range in the source image"),
			CheckBlitImageError::DestinationCoordinatesOutOfRange => write!(f, "The offsets, array layers and/or mipmap levels are out of range in the destination image"),
			CheckBlitImageError::IncompatibleRangeForImageType => write!(f, "The top-left and/or bottom-right coordinates are incompatible with the image type")
		}
	}
}
impl error::Error for CheckBlitImageError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
