// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! View of a buffer, in order to use it as a uniform texel buffer or storage texel buffer.
//!
//! In order to use a buffer as a uniform texel buffer or a storage texel buffer, you have to
//! create a `BufferView`, which indicates which format the data is in.
//!
//! In order to create a view from a buffer, the buffer must have been created with either the
//! `uniform_texel_buffer` or the `storage_texel_buffer` usage.
//!
//! # Example
//!
//! ```
//! # use std::sync::Arc;
//! use vulkano::{
//! 	buffer::{immutable::ImmutableBuffer, BufferUsage, BufferView},
//! 	format
//! 	};
//!
//! # let device: Arc<vulkano::device::Device> = return;
//! # let queue: Arc<vulkano::device::Queue> = return;
//! let usage = BufferUsage { storage_texel_buffer: true, ..BufferUsage::none() };
//!
//! let (buffer, _future) =
//! 	ImmutableBuffer::<[u32]>::from_iter((0 .. 128).map(|n| n), usage, queue.clone()).unwrap();
//! let _view = BufferView::new(buffer, format::R32Uint).unwrap();
//! ```

use std::{error, fmt, marker::PhantomData, mem, ptr, sync::Arc};

use crate::{
	buffer::{BufferAccess, BufferInner, TypedBufferAccess},
	device::{Device, DeviceOwned},
	format::{FormatDesc, StrongStorage}
};

use crate::{check_errors, Error, OomError, SafeDeref, VulkanObject};
use vk_sys as vk;

/// Represents a way for the GPU to interpret buffer data. See the documentation of the
/// `view` module.
pub struct BufferView<F, B>
where
	B: BufferAccess
{
	view: vk::BufferView,
	buffer: B,
	marker: PhantomData<F>,
	atomic_accesses: bool
}

impl<F, B> BufferView<F, B>
where
	B: BufferAccess
{
	/// Builds a new buffer view.
	pub fn new(buffer: B, format: F) -> Result<BufferView<F, B>, BufferViewCreationError>
	where
		B: TypedBufferAccess<Content = [F::Pixel]>,
		F: StrongStorage + 'static
	{
		unsafe { BufferView::unchecked(buffer, format) }
	}

	/// Builds a new buffer view without checking that the format is correct.
	pub unsafe fn unchecked(
		org_buffer: B, format: F
	) -> Result<BufferView<F, B>, BufferViewCreationError>
	where
		B: BufferAccess,
		F: FormatDesc + 'static
	{
		let (view, format_props) = {
			let size = org_buffer.size();
			let BufferInner { buffer, offset } = org_buffer.inner();

			let device = buffer.device();
			let format = format.format();

			if (offset
				% device.physical_device().limits().min_texel_buffer_offset_alignment() as usize)
				!= 0
			{
				return Err(BufferViewCreationError::WrongBufferAlignment)
			}

			if !buffer.usage_uniform_texel_buffer() && !buffer.usage_storage_texel_buffer() {
				return Err(BufferViewCreationError::WrongBufferUsage)
			}

			{
				let nb =
					size / format.size().expect("Can't use a compressed format for buffer views");
				let l = device.physical_device().limits().max_texel_buffer_elements();
				if nb > l as usize {
					return Err(BufferViewCreationError::MaxTexelBufferElementsExceeded)
				}
			}

			let format_props = {
				let vk_i = device.instance().pointers();
				let mut output = mem::uninitialized();
				vk_i.GetPhysicalDeviceFormatProperties(
					device.physical_device().internal_object(),
					format as u32,
					&mut output
				);
				output.bufferFeatures
			};

			if buffer.usage_uniform_texel_buffer() {
				if (format_props & vk::FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) == 0 {
					return Err(BufferViewCreationError::UnsupportedFormat)
				}
			}

			if buffer.usage_storage_texel_buffer() {
				if (format_props & vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT) == 0 {
					return Err(BufferViewCreationError::UnsupportedFormat)
				}
			}

			let infos = vk::BufferViewCreateInfo {
				sType: vk::STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
				pNext: ptr::null(),
				flags: 0, // reserved,
				buffer: buffer.internal_object(),
				format: format as u32,
				offset: offset as u64,
				range: size as u64
			};

			let vk = device.pointers();
			let mut output = mem::uninitialized();
			check_errors(vk.CreateBufferView(
				device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			(output, format_props)
		};

		Ok(BufferView {
			view,
			buffer: org_buffer,
			marker: PhantomData,
			atomic_accesses: (format_props & vk::FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT)
				!= 0
		})
	}

	/// Returns the buffer associated to this view.
	pub fn buffer(&self) -> &B { &self.buffer }

	/// Returns true if the buffer view can be used as a uniform texel buffer.
	pub fn uniform_texel_buffer(&self) -> bool {
		self.buffer.inner().buffer.usage_uniform_texel_buffer()
	}

	/// Returns true if the buffer view can be used as a storage texel buffer.
	pub fn storage_texel_buffer(&self) -> bool {
		self.buffer.inner().buffer.usage_storage_texel_buffer()
	}

	/// Returns true if the buffer view can be used as a storage texel buffer with atomic accesses.
	pub fn storage_texel_buffer_atomic(&self) -> bool {
		self.atomic_accesses && self.storage_texel_buffer()
	}
}

unsafe impl<F, B> VulkanObject for BufferView<F, B>
where
	B: BufferAccess
{
	type Object = vk::BufferView;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT;

	fn internal_object(&self) -> vk::BufferView { self.view }
}

unsafe impl<F, B> DeviceOwned for BufferView<F, B>
where
	B: BufferAccess
{
	fn device(&self) -> &Arc<Device> { self.buffer.device() }
}

impl<F, B> fmt::Debug for BufferView<F, B>
where
	B: BufferAccess + fmt::Debug
{
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt.debug_struct("BufferView")
			.field("raw", &self.view)
			.field("buffer", &self.buffer)
			.finish()
	}
}

impl<F, B> Drop for BufferView<F, B>
where
	B: BufferAccess
{
	fn drop(&mut self) {
		unsafe {
			let vk = self.buffer.inner().buffer.device().pointers();
			vk.DestroyBufferView(
				self.buffer.inner().buffer.device().internal_object(),
				self.view,
				ptr::null()
			);
		}
	}
}

pub unsafe trait BufferViewRef {
	type BufferAccess: BufferAccess;
	type Format;

	fn view(&self) -> &BufferView<Self::Format, Self::BufferAccess>;
}

unsafe impl<F, B> BufferViewRef for BufferView<F, B>
where
	B: BufferAccess
{
	type BufferAccess = B;
	type Format = F;

	fn view(&self) -> &BufferView<F, B> { self }
}

unsafe impl<T, F, B> BufferViewRef for T
where
	T: SafeDeref<Target = BufferView<F, B>>,
	B: BufferAccess
{
	type BufferAccess = B;
	type Format = F;

	fn view(&self) -> &BufferView<F, B> { &**self }
}

/// Error that can happen when creating a buffer view.
#[derive(Debug, Copy, Clone)]
pub enum BufferViewCreationError {
	/// Out of memory.
	OomError(OomError),

	/// The buffer was not creating with one of the `storage_texel_buffer` or
	/// `uniform_texel_buffer` usages.
	WrongBufferUsage,

	/// The offset within the buffer is not a multiple of the `min_texel_buffer_offset_alignment`
	/// limit.
	WrongBufferAlignment,

	/// The requested format is not supported for this usage.
	UnsupportedFormat,

	/// The maximum number of elements in the buffer view has been exceeded.
	MaxTexelBufferElementsExceeded
}
impl fmt::Display for BufferViewCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			BufferViewCreationError::OomError(e) => e.fmt(f),
			BufferViewCreationError::WrongBufferUsage
				=> write!(f, "The buffer is missing the required usage flags"),
			BufferViewCreationError::WrongBufferAlignment
				=> write!(f, "The offset within the buffer is not a multiple of the `min_texel_buffer_offset_alignment` limit"),
			BufferViewCreationError::UnsupportedFormat
				=> write!(f, "The requested format is not supported for this usage"),
				BufferViewCreationError::MaxTexelBufferElementsExceeded
				=> write!(f, "The maximum number of texel elements is exceeded"),
		}
	}
}
impl error::Error for BufferViewCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			BufferViewCreationError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for BufferViewCreationError {
	fn from(err: OomError) -> BufferViewCreationError { BufferViewCreationError::OomError(err) }
}

impl From<Error> for BufferViewCreationError {
	fn from(err: Error) -> BufferViewCreationError { OomError::from(err).into() }
}

#[cfg(test)]
mod tests {
	use crate::buffer::{
		immutable::ImmutableBuffer,
		view::BufferViewCreationError,
		BufferUsage,
		BufferView
	};
	use format;

	#[test]
	fn create_uniform() {
		// `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
		let (device, queue) = gfx_dev_and_queue!();

		let usage = BufferUsage { uniform_texel_buffer: true, ..BufferUsage::none() };

		let (buffer, _) = ImmutableBuffer::<[[u8; 4]]>::from_iter(
			(0 .. 128).map(|_| [0; 4]),
			usage,
			queue.clone()
		)
		.unwrap();
		let view = BufferView::new(buffer, format::R8G8B8A8Unorm).unwrap();

		assert!(view.uniform_texel_buffer());
	}

	#[test]
	fn create_storage() {
		// `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
		let (device, queue) = gfx_dev_and_queue!();

		let usage = BufferUsage { storage_texel_buffer: true, ..BufferUsage::none() };

		let (buffer, _) = ImmutableBuffer::<[[u8; 4]]>::from_iter(
			(0 .. 128).map(|_| [0; 4]),
			usage,
			queue.clone()
		)
		.unwrap();
		let view = BufferView::new(buffer, format::R8G8B8A8Unorm).unwrap();

		assert!(view.storage_texel_buffer());
	}

	#[test]
	fn create_storage_atomic() {
		// `VK_FORMAT_R32_UINT` guaranteed to be a supported format for atomics
		let (device, queue) = gfx_dev_and_queue!();

		let usage = BufferUsage { storage_texel_buffer: true, ..BufferUsage::none() };

		let (buffer, _) =
			ImmutableBuffer::<[u32]>::from_iter((0 .. 128).map(|_| 0), usage, queue.clone())
				.unwrap();
		let view = BufferView::new(buffer, format::R32Uint).unwrap();

		assert!(view.storage_texel_buffer());
		assert!(view.storage_texel_buffer_atomic());
	}

	#[test]
	fn wrong_usage() {
		// `VK_FORMAT_R8G8B8A8_UNORM` guaranteed to be a supported format
		let (device, queue) = gfx_dev_and_queue!();

		let (buffer, _) = ImmutableBuffer::<[[u8; 4]]>::from_iter(
			(0 .. 128).map(|_| [0; 4]),
			BufferUsage::none(),
			queue.clone()
		)
		.unwrap();

		match BufferView::new(buffer, format::R8G8B8A8Unorm) {
			Err(BufferViewCreationError::WrongBufferUsage) => (),
			_ => panic!()
		}
	}

	#[test]
	fn unsupported_format() {
		let (device, queue) = gfx_dev_and_queue!();

		let usage = BufferUsage {
			uniform_texel_buffer: true,
			storage_texel_buffer: true,
			..BufferUsage::none()
		};

		let (buffer, _) = ImmutableBuffer::<[[f64; 4]]>::from_iter(
			(0 .. 128).map(|_| [0.0; 4]),
			usage,
			queue.clone()
		)
		.unwrap();

		// TODO: what if R64G64B64A64Sfloat is supported?
		match BufferView::new(buffer, format::R64G64B64A64Sfloat) {
			Err(BufferViewCreationError::UnsupportedFormat) => (),
			_ => panic!()
		}
	}
}
