// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::{
	atomic::{AtomicBool, Ordering},
	Arc,
	Mutex
};

use crate::{
	buffer::BufferAccess,
	command_buffer::submit::{
		SubmitAnyBuilder,
		SubmitCommandBufferBuilder,
		SubmitSemaphoresWaitBuilder
	},
	device::{Device, DeviceOwned, Queue},
	image::{ImageLayout, ImageViewAccess},
	sync::{AccessCheckError, AccessFlagBits, FlushError, GpuFuture, PipelineStages, Semaphore}
};

/// Builds a new semaphore signal future.
pub fn then_signal_semaphore<F>(future: F) -> SemaphoreSignalFuture<F>
where
	F: GpuFuture
{
	let device = future.device().clone();

	assert!(future.queue().is_some()); // TODO: document

	SemaphoreSignalFuture {
		previous: future,
		semaphore: Semaphore::from_pool(device).unwrap(),
		wait_submitted: Mutex::new(false),
		finished: AtomicBool::new(false)
	}
}

/// Represents a semaphore being signaled after a previous event.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished \
              processing the submission"]
#[derive(Debug)]
pub struct SemaphoreSignalFuture<F>
where
	F: GpuFuture
{
	previous: F,
	semaphore: Semaphore,
	// True if the signaling command has already been submitted.
	// If flush is called multiple times, we want to block so that only one flushing is executed.
	// Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
	wait_submitted: Mutex<bool>,
	finished: AtomicBool
}
unsafe impl<F> GpuFuture for SemaphoreSignalFuture<F>
where
	F: GpuFuture
{
	fn cleanup_finished(&mut self) { self.previous.cleanup_finished(); }

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		// Flushing the signaling part, since it must always be submitted before the waiting part.
		self.flush()?;

		let mut sem = SubmitSemaphoresWaitBuilder::new();
		sem.add_wait_semaphore(&self.semaphore);
		Ok(SubmitAnyBuilder::SemaphoresWait(sem))
	}

	fn flush(&self) -> Result<(), FlushError> {
		unsafe {
			let mut wait_submitted = self.wait_submitted.lock().unwrap();

			if *wait_submitted {
				return Ok(())
			}

			let queue = self.previous.queue().unwrap().clone();

			match self.previous.build_submission()? {
				SubmitAnyBuilder::Empty => {
					let mut builder = SubmitCommandBufferBuilder::new();
					builder.add_signal_semaphore(&self.semaphore);
					builder.submit(&queue)?;
				}
				SubmitAnyBuilder::SemaphoresWait(sem) => {
					let mut builder: SubmitCommandBufferBuilder = sem.into();
					builder.add_signal_semaphore(&self.semaphore);
					builder.submit(&queue)?;
				}
				SubmitAnyBuilder::CommandBuffer(mut builder) => {
					debug_assert_eq!(builder.num_signal_semaphores(), 0);
					builder.add_signal_semaphore(&self.semaphore);
					builder.submit(&queue)?;
				}
				SubmitAnyBuilder::BindSparse(_) => {
					unimplemented!() // TODO: how to do that?
				 // debug_assert_eq!(builder.num_signal_semaphores(), 0);
				 // builder.add_signal_semaphore(&self.semaphore);
				 // builder.submit(&queue)?;
				}
				SubmitAnyBuilder::QueuePresent(present) => {
					present.submit(&queue)?;
					let mut builder = SubmitCommandBufferBuilder::new();
					builder.add_signal_semaphore(&self.semaphore);
					builder.submit(&queue)?; // FIXME: problematic because if we return an error and flush() is called again, then we'll submit the present twice
				}
			};

			// Only write `true` here in order to try again next time if an error occurs.
			*wait_submitted = true;
			Ok(())
		}
	}

	unsafe fn signal_finished(&self) {
		debug_assert!(*self.wait_submitted.lock().unwrap());
		self.finished.store(true, Ordering::SeqCst);
		self.previous.signal_finished();
	}

	fn queue_change_allowed(&self) -> bool { true }

	fn queue(&self) -> Option<Arc<Queue>> { self.previous.queue() }

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		self.previous.check_buffer_access(buffer, exclusive, queue).map(|_| None)
	}

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		self.previous.check_image_access(image, layout, exclusive, queue).map(|_| None)
	}
}
unsafe impl<F> DeviceOwned for SemaphoreSignalFuture<F>
where
	F: GpuFuture
{
	fn device(&self) -> &Arc<Device> { self.semaphore.device() }
}
impl<F> Drop for SemaphoreSignalFuture<F>
where
	F: GpuFuture
{
	fn drop(&mut self) {
		unsafe {
			if !*self.finished.get_mut() {
				// TODO: handle errors?
				self.flush().unwrap();
				// Block until the queue finished.
				self.queue().unwrap().wait().unwrap();
				self.previous.signal_finished();
			}
		}
	}
}
