// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use crate::{
	buffer::BufferAccess,
	command_buffer::submit::SubmitAnyBuilder,
	device::{Device, DeviceOwned, Queue},
	image::{ImageLayout, ImageViewAccess},
	sync::{AccessCheckError, AccessFlagBits, FlushError, GpuFuture, PipelineStages}
};

/// Builds a future that represents "now".
pub fn now(device: Arc<Device>) -> NowFuture { NowFuture { device } }

/// A dummy future that represents "now".
#[derive(Debug)]
pub struct NowFuture {
	device: Arc<Device>
}
unsafe impl GpuFuture for NowFuture {
	fn cleanup_finished(&mut self) {}

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		Ok(SubmitAnyBuilder::Empty)
	}

	fn flush(&self) -> Result<(), FlushError> { Ok(()) }

	unsafe fn signal_finished(&self) {}

	fn queue_change_allowed(&self) -> bool { true }

	fn queue(&self) -> Option<Arc<Queue>> { None }

	fn check_buffer_access(
		&self, buffer: &BufferAccess, _: bool, _: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		Err(AccessCheckError::Unknown)
	}

	fn check_image_access(
		&self, _: &dyn ImageViewAccess, _: ImageLayout, _: bool, _: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		Err(AccessCheckError::Unknown)
	}
}
unsafe impl DeviceOwned for NowFuture {
	fn device(&self) -> &Arc<Device> { &self.device }
}
