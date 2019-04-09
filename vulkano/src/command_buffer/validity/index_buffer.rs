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
	buffer::{BufferAccess, TypedBufferAccess},
	device::{Device, DeviceOwned},
	pipeline::input_assembly::Index,
	VulkanObject
};

/// Checks whether an index buffer can be bound.
///
/// # Panic
///
/// - Panics if the buffer was not created with `device`.
pub fn check_index_buffer<B, I>(
	device: &Device, buffer: &B
) -> Result<CheckIndexBuffer, CheckIndexBufferError>
where
	B: ?Sized + BufferAccess + TypedBufferAccess<Content = [I]>,
	I: Index
{
	assert_eq!(buffer.inner().buffer.device().internal_object(), device.internal_object());

	if !buffer.inner().buffer.usage_index_buffer() {
		return Err(CheckIndexBufferError::BufferMissingUsage)
	}

	// TODO: The sum of offset and the address of the range of VkDeviceMemory object that is
	//       backing buffer, must be a multiple of the type indicated by indexType

	// TODO: fullDrawIndexUint32 feature

	Ok(CheckIndexBuffer { num_indices: buffer.len() })
}

/// Information returned if `check_index_buffer` succeeds.
pub struct CheckIndexBuffer {
	/// Number of indices in the index buffer.
	pub num_indices: usize
}

/// Error that can happen when checking whether binding an index buffer is valid.
#[derive(Debug, Copy, Clone)]
pub enum CheckIndexBufferError {
	/// The buffer is missing the index buffer usage.
	BufferMissingUsage,
	/// The data or size must be 4-bytes aligned.
	WrongAlignment,
	/// The type of the indices is not supported by the device.
	UnsupportIndexType
}
impl fmt::Display for CheckIndexBufferError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckIndexBufferError::BufferMissingUsage => {
				write!(f, "The buffer is missing the index buffer usage")
			}
			CheckIndexBufferError::WrongAlignment => {
				write!(f, "The data or size must be 4-bytes aligned")
			}
			CheckIndexBufferError::UnsupportIndexType => {
				write!(f, "The type of the indices is not supported by the device")
			}
		}
	}
}
impl error::Error for CheckIndexBufferError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::buffer::{BufferUsage, CpuAccessibleBuffer};

	#[test]
	fn num_indices() {
		let (device, queue) = gfx_dev_and_queue!();
		let buffer = CpuAccessibleBuffer::from_iter(
			device.clone(),
			BufferUsage::index_buffer(),
			0 .. 500u32
		)
		.unwrap();

		match check_index_buffer(&device, &buffer) {
			Ok(CheckIndexBuffer { num_indices }) => {
				assert_eq!(num_indices, 500);
			}
			_ => panic!()
		}
	}

	#[test]
	fn missing_usage() {
		let (device, queue) = gfx_dev_and_queue!();
		let buffer = CpuAccessibleBuffer::from_iter(
			device.clone(),
			BufferUsage::vertex_buffer(),
			0 .. 500u32
		)
		.unwrap();

		match check_index_buffer(&device, &buffer) {
			Err(CheckIndexBufferError::BufferMissingUsage) => (),
			_ => panic!()
		}
	}

	#[test]
	fn wrong_device() {
		let (dev1, queue) = gfx_dev_and_queue!();
		let (dev2, _) = gfx_dev_and_queue!();

		let buffer = CpuAccessibleBuffer::from_iter(dev1, BufferUsage::all(), 0 .. 500u32).unwrap();

		assert_should_panic!({
			let _ = check_index_buffer(&dev2, &buffer);
		});
	}
}
