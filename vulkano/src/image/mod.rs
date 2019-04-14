// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image storage (1D, 2D, 3D, arrays, etc.).
//!
//! An *image* is a region of memory whose purpose is to store multi-dimensional data. Its
//! most common use is to store a 2D array of color pixels (in other words an *image* in
//! everyday language), but it can also be used to store arbitrary data.
//!
//! The advantage of using an image compared to a buffer is that the memory layout is optimized
//! for locality. When reading a specific pixel of an image, reading the nearby pixels is really
//! fast. Most implementations have hardware dedicated to reading from images if you access them
//! through a sampler.
//!
//! # Properties of an image
//!
//! # Images and image views
//!
//! There is a distinction between *images* and *image views*. As its name suggests, an image
//! view describes how the GPU must interpret the image.
//!
//! Transfer and memory operations operate on images themselves, while reading/writing an image
//! operates on image views. You can create multiple image views from the same image.
//!
//! # High-level wrappers
//!
//! In the vulkano library, an image is any object that implements the `Image` trait and an image
//! view is any object that implements the `ImageView` trait.
//!
//! Since these traits are low-level, you are encouraged to not implement them yourself but instead
//! use one of the provided implementations that are specialized depending on the way you are going
//! to use the image:
//!
//! - An `AttachmentImage` can be used when you want to draw to an image.
//! - An `ImmutableImage` stores data which never need be changed after the initial upload,
//!   like a texture.
//!
//! # Low-level information
//!
//! To be written.
//!

use std::{cmp, num::NonZeroU32, ops::Range};

use vk_sys as vk;

use crate::instance::Limits;

mod layout;
mod usage;

// mod attachment;
// mod immutable;
// mod storage;
mod swapchain;

mod view;

pub mod sync;
pub mod sys;
pub mod traits;

// pub use attachment::AttachmentImage;
// pub use immutable::ImmutableImage;
// pub use storage::StorageImage;
pub use swapchain::SwapchainImage;

pub use layout::ImageLayout;
pub use sys::ImageCreationError;
pub use traits::{ImageAccess, ImageViewAccess};

pub use usage::ImageUsage;

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Debug, Copy, Clone)]
pub enum MipmapsCount {
	/// Allocates the number of mipmaps required to store all the mipmaps of the image where each
	/// mipmap is half the dimensions of the previous level. Guaranteed to be always supported.
	///
	/// Note that this is not necessarily the maximum number of mipmaps, as the Vulkan
	/// implementation may report that it supports a greater value.
	Log2,

	/// Allocate one mipmap (ie. just the main level). Always supported.
	One,

	/// Allocate the given number of mipmaps. May result in an error if the value is out of range
	/// of what the implementation supports.
	Specific(u32)
}
impl From<u32> for MipmapsCount {
	fn from(num: u32) -> MipmapsCount { MipmapsCount::Specific(num) }
}

/// Specifies how the components of an image must be swizzled.
///
/// When creating an image view, it is possible to ask the implementation to modify the value
/// returned when accessing a given component from within a shader.
///
/// If all the members are `Identity`, then the view is said to have identity swizzling. This is
/// what the `Default` trait implementation of this struct returns.
/// Views that don't have identity swizzling may not be supported for some operations. For example
/// attaching a view to a framebuffer is only possible if the view is identity-swizzled.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Swizzle {
	/// First component.
	pub r: ComponentSwizzle,
	/// Second component.
	pub g: ComponentSwizzle,
	/// Third component.
	pub b: ComponentSwizzle,
	/// Fourth component.
	pub a: ComponentSwizzle
}
impl Swizzle {
	/// Returns true if this is an identity swizzle.
	pub fn identity(&self) -> bool {
		if self.r == ComponentSwizzle::Identity
			&& self.g == ComponentSwizzle::Identity
			&& self.b == ComponentSwizzle::Identity
			&& self.a == ComponentSwizzle::Identity
		{
			true
		} else {
			false
		}
	}
}
impl Into<vk::ComponentMapping> for Swizzle {
	fn into(self) -> vk::ComponentMapping {
		vk::ComponentMapping {
			r: self.r as u32,
			g: self.g as u32,
			b: self.b as u32,
			a: self.a as u32
		}
	}
}

/// Describes the value that an individual component must return when being accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ComponentSwizzle {
	/// Returns the value that this component should normally have.
	Identity = vk::COMPONENT_SWIZZLE_IDENTITY,
	/// Always return zero.
	Zero = vk::COMPONENT_SWIZZLE_ZERO,
	/// Always return one.
	One = vk::COMPONENT_SWIZZLE_ONE,
	/// Returns the value of the first component.
	Red = vk::COMPONENT_SWIZZLE_R,
	/// Returns the value of the second component.
	Green = vk::COMPONENT_SWIZZLE_G,
	/// Returns the value of the third component.
	Blue = vk::COMPONENT_SWIZZLE_B,
	/// Returns the value of the fourth component.
	Alpha = vk::COMPONENT_SWIZZLE_A
}
impl Default for ComponentSwizzle {
	fn default() -> ComponentSwizzle { ComponentSwizzle::Identity }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensions {
	Dim1D { width: NonZeroU32 },
	Dim1DArray { width: NonZeroU32, array_layers: NonZeroU32 },

	Dim2D { width: NonZeroU32, height: NonZeroU32 },
	Dim2DArray { width: NonZeroU32, height: NonZeroU32, array_layers: NonZeroU32 },

	Cubemap { size: NonZeroU32 },
	CubemapArray { size: NonZeroU32, array_layers: NonZeroU32 },

	Dim3D { width: NonZeroU32, height: NonZeroU32, depth: NonZeroU32 }
}
impl ImageDimensions {
	pub fn width(&self) -> NonZeroU32 {
		match self {
			ImageDimensions::Dim1D { width } => *width,
			ImageDimensions::Dim1DArray { width, .. } => *width,

			ImageDimensions::Dim2D { width, .. } => *width,
			ImageDimensions::Dim2DArray { width, .. } => *width,

			ImageDimensions::Cubemap { size } => *size,
			ImageDimensions::CubemapArray { size, .. } => *size,

			ImageDimensions::Dim3D { width, .. } => *width
		}
	}

	pub fn height(&self) -> NonZeroU32 {
		match self {
			ImageDimensions::Dim1D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim1DArray { .. } => crate::NONZERO_ONE,

			ImageDimensions::Dim2D { height, .. } => *height,
			ImageDimensions::Dim2DArray { height, .. } => *height,

			ImageDimensions::Cubemap { size } => *size,
			ImageDimensions::CubemapArray { size, .. } => *size,

			ImageDimensions::Dim3D { height, .. } => *height
		}
	}

	pub fn depth(&self) -> NonZeroU32 {
		match self {
			ImageDimensions::Dim1D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim1DArray { .. } => crate::NONZERO_ONE,

			ImageDimensions::Dim2D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim2DArray { .. } => crate::NONZERO_ONE,

			ImageDimensions::Cubemap { .. } => crate::NONZERO_ONE,
			ImageDimensions::CubemapArray { .. } => crate::NONZERO_ONE,

			ImageDimensions::Dim3D { depth, .. } => *depth
		}
	}

	/// Internally stored as NonZeroU32.
	pub fn width_height(&self) -> [NonZeroU32; 2] { [self.width(), self.height()] }

	/// Internally stored as NonZeroU32.
	pub fn width_height_depth(&self) -> [NonZeroU32; 3] { [self.width(), self.height(), self.depth()] }

	pub fn array_layers(&self) -> NonZeroU32 {
		match self {
			ImageDimensions::Dim1D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim1DArray { array_layers, .. } => *array_layers,

			ImageDimensions::Dim2D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim2DArray { array_layers, .. } => *array_layers,

			ImageDimensions::Cubemap { .. } => crate::NONZERO_ONE,
			ImageDimensions::CubemapArray { array_layers, .. } => *array_layers,

			ImageDimensions::Dim3D { .. } => crate::NONZERO_ONE
		}
	}

	pub fn array_layers_with_cube(&self) -> NonZeroU32 {
		match self {
			ImageDimensions::Dim1D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim1DArray { array_layers, .. } => *array_layers,

			ImageDimensions::Dim2D { .. } => crate::NONZERO_ONE,
			ImageDimensions::Dim2DArray { array_layers, .. } => *array_layers,

			ImageDimensions::Cubemap { .. } => unsafe { NonZeroU32::new_unchecked(6) },
			ImageDimensions::CubemapArray { array_layers, .. } => unsafe { NonZeroU32::new_unchecked(array_layers.get() * 6) },

			ImageDimensions::Dim3D { .. } => crate::NONZERO_ONE
		}
	}

	/// Returns the total number of texels for an image of these dimensions.
	pub fn num_texels(&self) -> NonZeroU32 {
		unsafe {
			NonZeroU32::new_unchecked(
				self.width().get() * self.height().get() * self.depth().get() * self.array_layers_with_cube().get()
			)
		}
	}

	/// Returns the maximum number of mipmaps for these image dimensions.
	///
	/// The returned value is always at greater than or equal to 1.
	///
	/// # Example
	///
	/// ```
	/// use vulkano::image::ImageDimensions;
	///
	/// let dims = ImageDimensions::Dim2D {
	/// 	width: 32,
	/// 	height: 50,
	/// 	cubemap_compatible: false,
	/// 	array_layers: 1
	/// 	};
	///
	/// assert_eq!(dims.max_mipmaps(), 7);
	/// ```
	pub fn max_mipmaps(&self) -> NonZeroU32 {
		let max_dim = cmp::max(cmp::max(self.width().get(), self.height().get()), self.depth().get());
		let num_zeroes = 32 - (max_dim - 1).leading_zeros();
		unsafe {
			NonZeroU32::new_unchecked(
				num_zeroes + 1
			)
		}
	}

	/// Returns the dimensions of the `level`th mipmap level. If `level` is 0, then the dimensions
	/// are left unchanged.
	///
	/// Returns `None` if `level` is superior or equal to `max_mipmaps()`.
	///
	/// # Example
	///
	/// ```
	/// use vulkano::image::ImageDimensions;
	///
	/// let dims = ImageDimensions::Dim2D {
	/// 	width: 963,
	/// 	height: 256,
	/// 	cubemap_compatible: false,
	/// 	array_layers: 1
	/// 	};
	///
	/// assert_eq!(dims.mipmap_dimensions(0), Some(dims));
	/// assert_eq!(
	/// 	dims.mipmap_dimensions(1),
	/// 	Some(ImageDimensions::Dim2D {
	/// 		width: 512,
	/// 		height: 128,
	/// 		cubemap_compatible: false,
	/// 		array_layers: 1
	/// 		})
	/// 	);
	/// assert_eq!(
	/// 	dims.mipmap_dimensions(6),
	/// 	Some(ImageDimensions::Dim2D {
	/// 		width: 16,
	/// 		height: 4,
	/// 		cubemap_compatible: false,
	/// 		array_layers: 1
	/// 		})
	/// 	);
	/// assert_eq!(
	/// 	dims.mipmap_dimensions(9),
	/// 	Some(ImageDimensions::Dim2D {
	/// 		width: 2,
	/// 		height: 1,
	/// 		cubemap_compatible: false,
	/// 		array_layers: 1
	/// 		})
	/// 	);
	/// assert_eq!(dims.mipmap_dimensions(11), None);
	/// ```
	///
	/// # Panic
	///
	/// In debug mode, Panics if `width`, `height` or `depth` is equal to 0. In release, returns
	/// an unspecified value.
	pub fn mipmap_dimensions(&self, level: u32) -> Option<ImageDimensions> {
		if level == 0 {
			return Some(*self)
		}
		if level >= self.max_mipmaps().get() {
			return None
		}

		let mlvl = |n: u32| {
			unsafe {
				NonZeroU32::new_unchecked(
					(((n - 1) >> level) + 1).next_power_of_two()
				)
			}
		};

		Some(match *self {
			ImageDimensions::Dim1D { width } => {
				ImageDimensions::Dim1D { width: mlvl(width.get()) }
			}
			ImageDimensions::Dim1DArray { width, array_layers } => {
				ImageDimensions::Dim1DArray {
					width: mlvl(width.get()),
					array_layers
				}
			}

			ImageDimensions::Dim2D { width, height } => {
				ImageDimensions::Dim2D {
					width: mlvl(width.get()),
					height: mlvl(height.get())
				}
			}
			ImageDimensions::Dim2DArray { width, height, array_layers } => {
				ImageDimensions::Dim2DArray {
					width: mlvl(width.get()),
					height: mlvl(height.get()),
					array_layers
				}
			}

			ImageDimensions::Cubemap { size } => {
				ImageDimensions::Cubemap { size: mlvl(size.get()) }
			}
			ImageDimensions::CubemapArray { size, array_layers } => {
				ImageDimensions::CubemapArray {
					size: mlvl(size.get()),
					array_layers
				}
			}

			ImageDimensions::Dim3D { width, height, depth } => {
				ImageDimensions::Dim3D {
					width: mlvl(width.get()),
					height: mlvl(height.get()),
					depth: mlvl(depth.get())
				}
			}
		})
	}

	/// Returns true if these are array dimensions.
	pub fn is_array(&self) -> bool { ImageViewType::from(*self).is_array() }

	/// Returns the number of dimensions these dimensions have.
	pub fn dimension_type(&self) -> ImageDimensionType {
		ImageViewType::from(*self).dimension_type()
	}

	/// Returns true if these dimensions are not over
	/// the device limits.
	pub fn check_limits(&self, limits: Limits) -> bool {
		match self {
			ImageDimensions::Dim1D { width } => width.get() <= limits.max_image_dimension_1d(),
			ImageDimensions::Dim1DArray { width, array_layers } => {
				width.get() <= limits.max_image_dimension_1d()
					&& array_layers.get() <= limits.max_image_array_layers()
			}

			ImageDimensions::Dim2D { width, height } => {
				width.get() <= limits.max_image_dimension_2d()
					&& height.get() <= limits.max_image_dimension_2d()
			}
			ImageDimensions::Dim2DArray { width, height, array_layers } => {
				width.get() <= limits.max_image_dimension_2d()
					&& height.get() <= limits.max_image_dimension_2d()
					&& array_layers.get() <= limits.max_image_array_layers()
			}

			ImageDimensions::Cubemap { size } => size.get() <= limits.max_image_dimension_cube(),
			ImageDimensions::CubemapArray { size, array_layers } => {
				size.get() <= limits.max_image_dimension_cube()
					&& array_layers.get() * 6 <= limits.max_image_array_layers()
			}

			ImageDimensions::Dim3D { width, height, depth } => {
				width.get() <= limits.max_image_dimension_3d()
					&& height.get() <= limits.max_image_dimension_3d()
					&& depth.get() <= limits.max_image_dimension_3d()
			}
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageViewType {
	Dim1D,
	Dim1DArray,

	Dim2D,
	Dim2DArray,

	Cubemap,
	CubemapArray,

	Dim3D
}
impl From<ImageDimensions> for ImageViewType {
	fn from(dims: ImageDimensions) -> Self {
		match dims {
			ImageDimensions::Dim1D { .. } => ImageViewType::Dim1D,
			ImageDimensions::Dim1DArray { .. } => ImageViewType::Dim1DArray,

			ImageDimensions::Dim2D { .. } => ImageViewType::Dim2D,
			ImageDimensions::Dim2DArray { .. } => ImageViewType::Dim2DArray,

			ImageDimensions::Cubemap { .. } => ImageViewType::Cubemap,
			ImageDimensions::CubemapArray { .. } => ImageViewType::CubemapArray,

			ImageDimensions::Dim3D { .. } => ImageViewType::Dim3D
		}
	}
}
impl ImageViewType {
	/// Returns true if an image view of type `self` can
	/// be created for an image with dimensions of type `other`.
	pub fn compatible_with(&self, other: ImageViewType) -> bool {
		if *self == other {
			return true
		}

		match (self, other) {
			(ImageViewType::Dim1D, ImageViewType::Dim1DArray) => true,
			(ImageViewType::Dim1DArray, ImageViewType::Dim1DArray) => true,

			(ImageViewType::Dim2D, ImageViewType::Dim2DArray)
			| (ImageViewType::Dim2D, ImageViewType::Cubemap)
			| (ImageViewType::Dim2D, ImageViewType::CubemapArray) => true,
			(ImageViewType::Dim2DArray, ImageViewType::Cubemap)
			| (ImageViewType::Dim1DArray, ImageViewType::CubemapArray) => true,

			(ImageViewType::Cubemap, ImageViewType::CubemapArray) => true,
			(ImageViewType::CubemapArray, ImageViewType::CubemapArray) => true,

			(ImageViewType::Dim3D, ImageViewType::Dim3D) => true,

			_ => false
		}
	}

	/// Returns true if this view type can be an array.
	pub fn is_array(&self) -> bool {
		match self {
			ImageViewType::Dim1DArray
			| ImageViewType::Dim2DArray
			| ImageViewType::Cubemap
			| ImageViewType::CubemapArray => true,
			_ => false
		}
	}

	/// Returns the ImageDimensionType this type has.
	pub fn dimension_type(&self) -> ImageDimensionType { ImageDimensionType::from(*self) }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensionType {
	D1,
	D2,
	Cube,
	D3
}
impl ImageDimensionType {
	/// Returns the number of dimensions.
	pub fn number(&self) -> u32 {
		match self {
			ImageDimensionType::D1 => 1,
			ImageDimensionType::D2 | ImageDimensionType::Cube => 2,
			ImageDimensionType::D3 => 3
		}
	}
}
impl From<ImageViewType> for ImageDimensionType {
	fn from(view_type: ImageViewType) -> Self {
		match view_type {
			ImageViewType::Dim1D | ImageViewType::Dim1DArray => ImageDimensionType::D1,
			ImageViewType::Dim2D | ImageViewType::Dim2DArray => ImageDimensionType::D2,
			ImageViewType::Cubemap | ImageViewType::CubemapArray => ImageDimensionType::Cube,
			ImageViewType::Dim3D => ImageDimensionType::D3
		}
	}
}
impl From<ImageDimensions> for ImageDimensionType {
	fn from(dims: ImageDimensions) -> Self { ImageDimensionType::from(ImageViewType::from(dims)) }
}
impl Into<u32> for ImageDimensionType {
	fn into(self) -> u32 { self.number() }
}

// TODO: Aspects flags? Do we need them here? vk_sys doesn't even define most of them.
// For that small subset, it's probably not worth it.
/// Describes an image subresource (mipmap levels and array layers) range.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)] // derive Hash because `SubresourceImageResourceLocker`
pub struct ImageSubresourceRange {
	/// Number of array layers.
	pub array_layers: NonZeroU32,
	/// Offset of the first array layer.
	pub array_layers_offset: u32,

	/// Number of mipmaps levels.
	pub mipmap_levels: NonZeroU32,
	/// Offset of the first mipmap level.
	pub mipmap_levels_offset: u32
}
impl ImageSubresourceRange {
	pub fn array_layers_end(&self) -> NonZeroU32 {
		unsafe {
			// Safe because NonZeroU32 + anything non-negative > 0
			NonZeroU32::new_unchecked(self.array_layers_offset + self.array_layers.get())
		}
	}

	pub fn mipmap_levels_end(&self) -> NonZeroU32 {
		unsafe {
			// Safe because NonZeroU32 + anything non-negative > 0
			NonZeroU32::new_unchecked(self.mipmap_levels_offset + self.mipmap_levels.get())
		}
	}

	/// Returns a `Range<u32>` of the subresource array layers.
	pub fn array_layers_range(&self) -> Range<u32> {
		self.array_layers_offset .. self.array_layers_end().get()
	}

	/// Returns a `Range<u32>` of the subresource mipmap levels.
	pub fn mipmap_levels_range(&self) -> Range<u32> {
		self.mipmap_levels_offset .. self.mipmap_levels_end().get()
	}

	/// Returns true if the two `ImageSubresourceRange` overlap with each other.
	///
	/// This means that they share at least one common mipmap level at one common array layer.
	pub fn overlaps_with(&self, other: &ImageSubresourceRange) -> bool {
		fn range_overlaps(a: Range<u32>, b: Range<u32>) -> bool {
			!(a.end <= b.start || a.start >= b.end)
		}

		range_overlaps(self.array_layers_range(), other.array_layers_range())
			&& range_overlaps(self.mipmap_levels_range(), other.mipmap_levels_range())
	}
}

#[cfg(test)]
mod tests {
	use crate::image::ImageDimensions;

	#[test]
	fn max_mipmaps() {
		let dims = ImageDimensions::Dim2D { width: 2, height: 1 };
		assert_eq!(dims.max_mipmaps(), 2);

		let dims = ImageDimensions::Dim2D { width: 2, height: 3 };
		assert_eq!(dims.max_mipmaps(), 3);

		let dims = ImageDimensions::Dim2D { width: 512, height: 512 };
		assert_eq!(dims.max_mipmaps(), 10);
	}

	#[test]
	fn mipmap_dimensions() {
		let dims = ImageDimensions::Dim2D { width: 283, height: 175 };
		assert_eq!(dims.mipmap_dimensions(0), Some(dims));
		assert_eq!(
			dims.mipmap_dimensions(1),
			Some(ImageDimensions::Dim2D { width: 256, height: 128 })
		);
		assert_eq!(
			dims.mipmap_dimensions(2),
			Some(ImageDimensions::Dim2D { width: 128, height: 64 })
		);
		assert_eq!(
			dims.mipmap_dimensions(3),
			Some(ImageDimensions::Dim2D { width: 64, height: 32 })
		);
		assert_eq!(
			dims.mipmap_dimensions(4),
			Some(ImageDimensions::Dim2D { width: 32, height: 16 })
		);
		assert_eq!(
			dims.mipmap_dimensions(5),
			Some(ImageDimensions::Dim2D { width: 16, height: 8 })
		);
		assert_eq!(dims.mipmap_dimensions(6), Some(ImageDimensions::Dim2D { width: 8, height: 4 }));
		assert_eq!(dims.mipmap_dimensions(7), Some(ImageDimensions::Dim2D { width: 4, height: 2 }));
		assert_eq!(dims.mipmap_dimensions(8), Some(ImageDimensions::Dim2D { width: 2, height: 1 }));
		assert_eq!(dims.mipmap_dimensions(9), Some(ImageDimensions::Dim2D { width: 1, height: 1 }));
		assert_eq!(dims.mipmap_dimensions(10), None);
	}
}
