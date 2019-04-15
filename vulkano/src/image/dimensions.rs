use std::{cmp, num::NonZeroU32, ops::Range};

use vk_sys as vk;

use crate::instance::Limits;

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
	pub fn width_height_depth(&self) -> [NonZeroU32; 3] {
		[self.width(), self.height(), self.depth()]
	}

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
			ImageDimensions::CubemapArray { array_layers, .. } => unsafe {
				NonZeroU32::new_unchecked(array_layers.get() * 6)
			},

			ImageDimensions::Dim3D { .. } => crate::NONZERO_ONE
		}
	}

	/// Returns the total number of texels for an image of these dimensions.
	pub fn num_texels(&self) -> NonZeroU32 {
		unsafe {
			NonZeroU32::new_unchecked(
				self.width().get()
					* self.height().get()
					* self.depth().get() * self.array_layers_with_cube().get()
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
		let max_dim =
			cmp::max(cmp::max(self.width().get(), self.height().get()), self.depth().get());
		let num_zeroes = 32 - (max_dim - 1).leading_zeros();
		unsafe { NonZeroU32::new_unchecked(num_zeroes + 1) }
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

		let mlvl = |n: u32| unsafe {
			NonZeroU32::new_unchecked((((n - 1) >> level) + 1).next_power_of_two())
		};

		Some(match *self {
			ImageDimensions::Dim1D { width } => ImageDimensions::Dim1D { width: mlvl(width.get()) },
			ImageDimensions::Dim1DArray { width, array_layers } => {
				ImageDimensions::Dim1DArray { width: mlvl(width.get()), array_layers }
			}

			ImageDimensions::Dim2D { width, height } => {
				ImageDimensions::Dim2D { width: mlvl(width.get()), height: mlvl(height.get()) }
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
				ImageDimensions::CubemapArray { size: mlvl(size.get()), array_layers }
			}

			ImageDimensions::Dim3D { width, height, depth } => ImageDimensions::Dim3D {
				width: mlvl(width.get()),
				height: mlvl(height.get()),
				depth: mlvl(depth.get())
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

	/// Transforms these dimensions into vulkan types
	pub fn vk_type(&self) -> (vk::ImageType, vk::Extent3D, u32, vk::ImageCreateFlagBits) {
		match self {
			ImageDimensions::Dim1D { .. } | ImageDimensions::Dim1DArray { .. } => {
				let extent = vk::Extent3D { width: self.width().get(), height: 1, depth: 1 };
				(vk::IMAGE_TYPE_1D, extent, self.array_layers().get(), 0)
			}
			ImageDimensions::Dim2D { .. } | ImageDimensions::Dim2DArray { .. } => {
				let extent = vk::Extent3D {
					width: self.width().get(),
					height: self.height().get(),
					depth: 1
				};
				(vk::IMAGE_TYPE_2D, extent, self.array_layers().get(), 0)
			}
			ImageDimensions::Cubemap { .. } | ImageDimensions::CubemapArray { .. } => {
				let extent = vk::Extent3D {
					width: self.width().get(),
					height: self.width().get(),
					depth: 1
				};
				(
					vk::IMAGE_TYPE_2D,
					extent,
					self.array_layers().get(),
					vk::IMAGE_CREATE_CUBE_COMPATIBLE_BIT
				)
			}
			ImageDimensions::Dim3D { .. } => {
				let extent = vk::Extent3D {
					width: self.width().get(),
					height: self.height().get(),
					depth: self.depth().get()
				};
				(vk::IMAGE_TYPE_3D, extent, 1, 0)
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
	use std::num::NonZeroU32;

	#[test]
	fn max_mipmaps() {
		let dims = ImageDimensions::Dim2D {
			width: unsafe { NonZeroU32::new_unchecked(2) },
			height: unsafe { NonZeroU32::new_unchecked(1) }
		};
		assert_eq!(dims.max_mipmaps().get(), 2);

		let dims = ImageDimensions::Dim2D {
			width: unsafe { NonZeroU32::new_unchecked(2) },
			height: unsafe { NonZeroU32::new_unchecked(3) }
		};
		assert_eq!(dims.max_mipmaps().get(), 3);

		let dims = ImageDimensions::Dim2D {
			width: unsafe { NonZeroU32::new_unchecked(512) },
			height: unsafe { NonZeroU32::new_unchecked(512) }
		};
		assert_eq!(dims.max_mipmaps().get(), 10);
	}

	#[test]
	fn mipmap_dimensions() {
		let dims = ImageDimensions::Dim2D {
			width: unsafe { NonZeroU32::new_unchecked(283) },
			height: unsafe { NonZeroU32::new_unchecked(175) }
		};
		assert_eq!(dims.mipmap_dimensions(0), Some(dims));
		assert_eq!(
			dims.mipmap_dimensions(1),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(256) },
				height: unsafe { NonZeroU32::new_unchecked(128) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(2),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(128) },
				height: unsafe { NonZeroU32::new_unchecked(64) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(3),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(64) },
				height: unsafe { NonZeroU32::new_unchecked(32) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(4),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(32) },
				height: unsafe { NonZeroU32::new_unchecked(16) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(5),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(16) },
				height: unsafe { NonZeroU32::new_unchecked(8) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(6),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(8) },
				height: unsafe { NonZeroU32::new_unchecked(4) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(7),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(4) },
				height: unsafe { NonZeroU32::new_unchecked(2) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(8),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(2) },
				height: unsafe { NonZeroU32::new_unchecked(1) }
			})
		);
		assert_eq!(
			dims.mipmap_dimensions(9),
			Some(ImageDimensions::Dim2D {
				width: unsafe { NonZeroU32::new_unchecked(1) },
				height: unsafe { NonZeroU32::new_unchecked(1) }
			})
		);
		assert_eq!(dims.mipmap_dimensions(10), None);
	}
}
