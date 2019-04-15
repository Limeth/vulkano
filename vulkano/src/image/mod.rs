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

use std::{error, fmt, num::NonZeroU32};

use vk_sys as vk;

mod dimensions;
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

pub use layout::{ImageLayout, ImageEndLayout};
pub use sys::ImageCreationError;
pub use traits::{ImageAccess, ImageViewAccess};

pub use dimensions::{ImageDimensionType, ImageDimensions, ImageSubresourceRange, ImageViewType};
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
	Specific(NonZeroU32)
}
impl MipmapsCount {
	/// Returns number of mipmaps for an image with given dimensions.
	///
	/// Returns Err(number) if self is Specific but the provided number of mipmap
	/// levels is more than the maximum number of mipmaps for the dimensions.
	pub fn for_image(&self, dimensions: ImageDimensions) -> Result<NonZeroU32, NonZeroU32> {
		Ok(match self {
			MipmapsCount::Specific(number) => {
				let max_mipmaps = dimensions.max_mipmaps();
				if number.get() > max_mipmaps.get() {
					return Err(*number)
				}

				*number
			}
			MipmapsCount::Log2 => dimensions.max_mipmaps(),
			MipmapsCount::One => crate::NONZERO_ONE
		})
	}
}
impl From<NonZeroU32> for MipmapsCount {
	fn from(num: NonZeroU32) -> MipmapsCount { MipmapsCount::Specific(num) }
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

/// Describes an error that can happen when requesting the current
/// layout for an image subresource.
#[derive(Debug)]
pub enum ImageSubresourceLayoutError {
	/// The subresource has multiple different layouts.
	MultipleLayouts
}
impl fmt::Display for ImageSubresourceLayoutError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ImageSubresourceLayoutError::MultipleLayouts => {
				write!(f, "The subresource has multiple different layouts")
			}
		}
	}
}
impl error::Error for ImageSubresourceLayoutError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
