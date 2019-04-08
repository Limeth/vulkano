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

use std::cmp;

mod attachment;
mod immutable;
mod layout;
mod storage;
mod swapchain;
pub mod sys;
pub mod traits;
mod usage;

pub use attachment::AttachmentImage;
pub use immutable::ImmutableImage;
pub use storage::StorageImage;
pub use swapchain::SwapchainImage;

pub use layout::ImageLayout;
pub use sys::ImageCreationError;
pub use traits::{ImageAccess, ImageInner, ImageViewAccess};

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

/// Describes the value that an individual component must return when being accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComponentSwizzle {
	/// Returns the value that this component should normally have.
	Identity,
	/// Always return zero.
	Zero,
	/// Always return one.
	One,
	/// Returns the value of the first component.
	Red,
	/// Returns the value of the second component.
	Green,
	/// Returns the value of the third component.
	Blue,
	/// Returns the value of the fourth component.
	Alpha
}
impl Default for ComponentSwizzle {
	fn default() -> ComponentSwizzle { ComponentSwizzle::Identity }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensions {
	Dim1D { width: u32 },
	Dim1DArray { width: u32, array_layers: u32 },

	Dim2D { width: u32, height: u32 },
	Dim2DArray { width: u32, height: u32, array_layers: u32 },

	Cubemap { size: u32 },
	CubemapArray { size: u32, array_layers: u32 },

	Dim3D { width: u32, height: u32, depth: u32 }
}
impl ImageDimensions {
	pub fn width(&self) -> u32 {
		match *self {
			ImageDimensions::Dim1D { width } => width,
			ImageDimensions::Dim1DArray { width, .. } => width,

			ImageDimensions::Dim2D { width, .. } => width,
			ImageDimensions::Dim2DArray { width, .. } => width,

			ImageDimensions::Cubemap { size } => size,
			ImageDimensions::CubemapArray { size, .. } => size,

			ImageDimensions::Dim3D { width, .. } => width
		}
	}

	pub fn height(&self) -> u32 {
		match *self {
			ImageDimensions::Dim1D { .. } => 1,
			ImageDimensions::Dim1DArray { .. } => 1,

			ImageDimensions::Dim2D { height, .. } => height,
			ImageDimensions::Dim2DArray { height, .. } => height,

			ImageDimensions::Cubemap { size } => size,
			ImageDimensions::CubemapArray { size, .. } => size,

			ImageDimensions::Dim3D { height, .. } => height
		}
	}

	pub fn depth(&self) -> u32 {
		match *self {
			ImageDimensions::Dim1D { .. } => 1,
			ImageDimensions::Dim1DArray { .. } => 1,

			ImageDimensions::Dim2D { .. } => 1,
			ImageDimensions::Dim2DArray { .. } => 1,

			ImageDimensions::Cubemap { .. } => 1,
			ImageDimensions::CubemapArray { .. } => 1,

			ImageDimensions::Dim3D { depth, .. } => depth
		}
	}

	pub fn width_height(&self) -> [u32; 2] { [self.width(), self.height()] }

	pub fn width_height_depth(&self) -> [u32; 3] { [self.width(), self.height(), self.depth()] }

	pub fn array_layers(&self) -> u32 {
		match *self {
			ImageDimensions::Dim1D { .. } => 1,
			ImageDimensions::Dim1DArray { array_layers, .. } => array_layers,

			ImageDimensions::Dim2D { .. } => 1,
			ImageDimensions::Dim2DArray { array_layers, .. } => array_layers,

			ImageDimensions::Cubemap { .. } => 1,
			ImageDimensions::CubemapArray { array_layers, .. } => array_layers,

			ImageDimensions::Dim3D { .. } => 1
		}
	}

	pub fn array_layers_with_cube(&self) -> u32 {
		match *self {
			ImageDimensions::Dim1D { .. } => 1,
			ImageDimensions::Dim1DArray { array_layers, .. } => array_layers,

			ImageDimensions::Dim2D { .. } => 1,
			ImageDimensions::Dim2DArray { array_layers, .. } => array_layers,

			ImageDimensions::Cubemap { .. } => 6,
			ImageDimensions::CubemapArray { array_layers, .. } => array_layers * 6,

			ImageDimensions::Dim3D { .. } => 1
		}
	}

	/// Returns the total number of texels for an image of these dimensions.
	pub fn num_texels(&self) -> u32 {
		self.width() * self.height() * self.depth() * self.array_layers_with_cube()
	}

	/// Returns the maximum number of mipmaps for these image dimensions.
	///
	/// The returned value is always at least superior or equal to 1.
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
	///
	/// # Panic
	///
	/// May panic if the dimensions are 0.
	pub fn max_mipmaps(&self) -> u32 {
		let max_dim = cmp::max(cmp::max(self.width(), self.height()), self.depth());
		let num_zeroes = 32 - (max_dim - 1).leading_zeros();
		num_zeroes + 1
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
		if level >= self.max_mipmaps() {
			return None
		}

		Some(match *self {
			ImageDimensions::Dim1D { width } => {
				debug_assert_ne!(width, 0);
				ImageDimensions::Dim1D { width: (((width - 1) >> level) + 1).next_power_of_two() }
			}
			ImageDimensions::Dim1DArray { width, array_layers } => {
				debug_assert_ne!(width, 0);
				ImageDimensions::Dim1DArray {
					width: (((width - 1) >> level) + 1).next_power_of_two(),
					array_layers
				}
			}

			ImageDimensions::Dim2D { width, height } => {
				debug_assert_ne!(width, 0);
				debug_assert_ne!(height, 0);

				ImageDimensions::Dim2D {
					width: (((width - 1) >> level) + 1).next_power_of_two(),
					height: (((height - 1) >> level) + 1).next_power_of_two()
				}
			}
			ImageDimensions::Dim2DArray { width, height, array_layers } => {
				debug_assert_ne!(width, 0);
				debug_assert_ne!(height, 0);
				ImageDimensions::Dim2DArray {
					width: (((width - 1) >> level) + 1).next_power_of_two(),
					height: (((height - 1) >> level) + 1).next_power_of_two(),
					array_layers
				}
			}

			ImageDimensions::Cubemap { size } => {
				debug_assert_ne!(size, 0);
				ImageDimensions::Cubemap { size: (((size - 1) >> level) + 1).next_power_of_two() }
			}
			ImageDimensions::CubemapArray { size, array_layers } => {
				debug_assert_ne!(size, 0);
				ImageDimensions::CubemapArray {
					size: (((size - 1) >> level) + 1).next_power_of_two(),
					array_layers
				}
			}

			ImageDimensions::Dim3D { width, height, depth } => {
				debug_assert_ne!(width, 0);
				debug_assert_ne!(height, 0);
				debug_assert_ne!(depth, 0);

				ImageDimensions::Dim3D {
					width: (((width - 1) >> level) + 1).next_power_of_two(),
					height: (((height - 1) >> level) + 1).next_power_of_two(),
					depth: (((depth - 1) >> level) + 1).next_power_of_two()
				}
			}
		})
	}

	/// Returns true if these are array dimensions.
	pub fn is_array(&self) -> bool {
		match self {
			ImageDimensions::Dim1DArray { .. }
			| ImageDimensions::Dim2DArray { .. }
			| ImageDimensions::Cubemap { .. }
			| ImageDimensions::CubemapArray { .. } => true,
			_ => false
		}
	}

	/// Returns the number of dimensions these dimensions have.
	pub fn dimensions(&self) -> u32 {
		match self {
			ImageDimensions::Dim1D { .. } | ImageDimensions::Dim1DArray { .. } => 1,
			ImageDimensions::Dim2D { .. }
			| ImageDimensions::Dim2DArray { .. }
			| ImageDimensions::Cubemap { .. }
			| ImageDimensions::CubemapArray { .. } => 2,
			ImageDimensions::Dim3D { .. } => 3
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewType {
	Dim1D,
	Dim1DArray,

	Dim2D,
	Dim2DArray,

	Cubemap,
	CubemapArray,

	Dim3D
}
impl From<ImageDimensions> for ViewType {
	fn from(dims: ImageDimensions) -> Self {
		match dims {
			ImageDimensions::Dim1D { .. } => ViewType::Dim1D,
			ImageDimensions::Dim1DArray { .. } => ViewType::Dim1DArray,

			ImageDimensions::Dim2D { .. } => ViewType::Dim2D,
			ImageDimensions::Dim2DArray { .. } => ViewType::Dim2DArray,

			ImageDimensions::Cubemap { .. } => ViewType::Cubemap,
			ImageDimensions::CubemapArray { .. } => ViewType::CubemapArray,

			ImageDimensions::Dim3D { .. } => ViewType::Dim3D
		}
	}
}
impl ViewType {
	/// Returns true if an image view of type `self` can
	/// be created for an image with dimensions of type `other`.
	pub fn compatible_with(&self, other: ViewType) -> bool {
		if *self == other {
			return true
		}

		match (self, other) {
			(ViewType::Dim1D, ViewType::Dim1DArray) => true,
			(ViewType::Dim1DArray, ViewType::Dim1DArray) => true,

			(ViewType::Dim2D, ViewType::Dim2DArray)
			| (ViewType::Dim2D, ViewType::Cubemap)
			| (ViewType::Dim2D, ViewType::CubemapArray) => true,
			(ViewType::Dim2DArray, ViewType::Cubemap)
			| (ViewType::Dim1DArray, ViewType::CubemapArray) => true,

			(ViewType::Cubemap, ViewType::CubemapArray) => true,
			(ViewType::CubemapArray, ViewType::CubemapArray) => true,

			(ViewType::Dim3D, ViewType::Dim3D) => true,

			_ => false
		}
	}

	/// Returns true if this view type can be an array.
	pub fn is_array(&self) -> bool {
		match self {
			ViewType::Dim1DArray
			| ViewType::Dim2DArray
			| ViewType::Cubemap
			| ViewType::CubemapArray => true,
			_ => false
		}
	}

	/// Returns the number of dimensions this type has.
	pub fn dimensions(&self) -> u32 {
		match self {
			ViewType::Dim1D | ViewType::Dim1DArray => 1,
			ViewType::Dim2D | ViewType::Dim2DArray | ViewType::Cubemap | ViewType::CubemapArray => {
				2
			}
			ViewType::Dim3D => 3
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::image::ImageDimensions;

	#[test]
	fn max_mipmaps() {
		let dims = ImageDimensions::Dim2D {
			width: 2,
			height: 1,
			cubemap_compatible: false,
			array_layers: 1
		};
		assert_eq!(dims.max_mipmaps(), 2);

		let dims = ImageDimensions::Dim2D {
			width: 2,
			height: 3,
			cubemap_compatible: false,
			array_layers: 1
		};
		assert_eq!(dims.max_mipmaps(), 3);

		let dims = ImageDimensions::Dim2D {
			width: 512,
			height: 512,
			cubemap_compatible: false,
			array_layers: 1
		};
		assert_eq!(dims.max_mipmaps(), 10);
	}

	#[test]
	fn mipmap_dimensions() {
		let dims = ImageDimensions::Dim2D {
			width: 283,
			height: 175,
			cubemap_compatible: false,
			array_layers: 1
		};
		assert_eq!(dims.mipmap_dimensions(0), Some(dims));
		assert_eq!(
			dims.mipmap_dimensions(1),
			Some(ImageDimensions::Dim2D {
				width: 256,
				height: 128,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(2),
			Some(ImageDimensions::Dim2D {
				width: 128,
				height: 64,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(3),
			Some(ImageDimensions::Dim2D {
				width: 64,
				height: 32,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(4),
			Some(ImageDimensions::Dim2D {
				width: 32,
				height: 16,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(5),
			Some(ImageDimensions::Dim2D {
				width: 16,
				height: 8,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(6),
			Some(ImageDimensions::Dim2D {
				width: 8,
				height: 4,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(7),
			Some(ImageDimensions::Dim2D {
				width: 4,
				height: 2,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(8),
			Some(ImageDimensions::Dim2D {
				width: 2,
				height: 1,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(9),
			Some(ImageDimensions::Dim2D {
				width: 1,
				height: 1,
				cubemap_compatible: false,
				array_layers: 1
			})
		);
		assert_eq!(dims.mipmap_dimensions(10), None);
	}
}
