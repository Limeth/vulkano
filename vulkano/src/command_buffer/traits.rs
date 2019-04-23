// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{
	borrow::Cow,
	error,
	fmt,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
		Mutex
	}
};

use crate::{
	buffer::BufferAccess,
	command_buffer::{
		submit::{SubmitAnyBuilder, SubmitCommandBufferBuilder},
		sys::UnsafeCommandBuffer
	},
	device::{Device, DeviceOwned, Queue},
	image::{ImageLayout, ImageViewAccess},
	sync::{
		now,
		AccessCheckError,
		AccessError,
		AccessFlagBits,
		FlushError,
		GpuFuture,
		NowFuture,
		PipelineStages
	},
	SafeDeref,
	VulkanObject
};

pub unsafe trait CommandBuffer: DeviceOwned {
	/// The command pool of the command buffer.
	type PoolAlloc;

	/// Returns the underlying `UnsafeCommandBuffer` of this command buffer.
	fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc>;

	// /// Returns the queue family of the command buffer.
	// fn queue_family(&self) -> QueueFamily
	// where Self::PoolAlloc: CommandPoolAlloc
	// {
	// self.inner().queue_family()
	// }

	/// Checks whether this command buffer is allowed to be submitted after the `future` and on
	/// the given queue, and if so locks it.
	///
	/// If you call this function, then you should call `unlock` afterwards.
	fn lock_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError>;

	/// Unlocks the command buffer. Should be called once for each call to `lock_submit`.
	///
	/// # Safety
	///
	/// Must not be called if you haven't called `lock_submit` before.
	unsafe fn unlock(&self);

	/// Executes this command buffer on a queue.
	///
	/// This function returns an object that implements the `GpuFuture` trait. See the
	/// documentation of the `sync` module for more information.
	///
	/// The command buffer is not actually executed until you call `flush()` on the object.
	/// You are encouraged to chain together as many futures as possible before calling `flush()`,
	/// and call `.then_signal_future()` before doing so. Note however that once you called
	/// `execute()` there is no way to cancel the execution, even if you didn't flush yet.
	///
	/// > **Note**: In the future this function may return `-> impl GpuFuture` instead of a
	/// > concrete type.
	///
	/// > **Note**: This is just a shortcut for `execute_after(vulkano::sync::now(), queue)`.
	///
	/// # Panic
	///
	/// Panics if the device of the command buffer is not the same as the device of the future.
	fn execute(
		self, queue: Arc<Queue>
	) -> Result<CommandBufferExecFuture<NowFuture, Self>, CommandBufferExecError>
	where
		Self: Sized + 'static
	{
		let device = queue.device().clone();
		self.execute_after(now(device), queue)
	}

	/// Executes the command buffer after an existing future.
	///
	/// This function returns an object that implements the `GpuFuture` trait. See the
	/// documentation of the `sync` module for more information.
	///
	/// The command buffer is not actually executed until you call `flush()` on the object.
	/// You are encouraged to chain together as many futures as possible before calling `flush()`,
	/// and call `.then_signal_future()` before doing so. Note however that once you called
	/// `execute()` there is no way to cancel the execution, even if you didn't flush yet.
	///
	/// > **Note**: In the future this function may return `-> impl GpuFuture` instead of a
	/// > concrete type.
	///
	/// This function requires the `'static` lifetime to be on the command buffer. This is because
	/// this function returns a `CommandBufferExecFuture` whose job is to lock resources and keep
	/// them alive while they are in use by the GPU. If `'static` wasn't required, you could call
	/// `std::mem::forget` on that object and "unlock" these resources. For more information about
	/// this problem, search the web for "rust thread scoped leakpocalypse".
	///
	/// # Panic
	///
	/// Panics if the device of the command buffer is not the same as the device of the future.
	fn execute_after<F>(
		self, future: F, queue: Arc<Queue>
	) -> Result<CommandBufferExecFuture<F, Self>, CommandBufferExecError>
	where
		Self: Sized + 'static,
		F: GpuFuture
	{
		assert_eq!(self.device().internal_object(), future.device().internal_object());

		if !future.queue_change_allowed() {
			assert!(future.queue().unwrap().is_same(&queue));
		}

		self.lock_submit(&future, &queue)?;

		Ok(CommandBufferExecFuture {
			previous: future,
			command_buffer: self,
			queue,
			submitted: Mutex::new(false),
			finished: AtomicBool::new(false)
		})
	}

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

	// FIXME: lots of other methods
}

unsafe impl<T> CommandBuffer for T
where
	T: SafeDeref,
	T::Target: CommandBuffer
{
	type PoolAlloc = <T::Target as CommandBuffer>::PoolAlloc;

	fn inner(&self) -> &UnsafeCommandBuffer<Self::PoolAlloc> { (**self).inner() }

	fn lock_submit(&self, future: &GpuFuture, queue: &Queue) -> Result<(), CommandBufferExecError> {
		(**self).lock_submit(future, queue)
	}

	unsafe fn unlock(&self) { (**self).unlock(); }

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		(**self).check_buffer_access(buffer, exclusive, queue)
	}

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		(**self).check_image_access(image, layout, exclusive, queue)
	}
}

/// Represents a command buffer being executed by the GPU and the moment when the execution
/// finishes.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct CommandBufferExecFuture<F, Cb>
where
	F: GpuFuture,
	Cb: CommandBuffer
{
	previous: F,
	command_buffer: Cb,
	queue: Arc<Queue>,
	// True if the command buffer has already been submitted.
	// If flush is called multiple times, we want to block so that only one flushing is executed.
	// Therefore we use a `Mutex<bool>` and not an `AtomicBool`.
	submitted: Mutex<bool>,
	finished: AtomicBool
}
unsafe impl<F, Cb> GpuFuture for CommandBufferExecFuture<F, Cb>
where
	F: GpuFuture,
	Cb: CommandBuffer
{
	fn cleanup_finished(&mut self) { self.previous.cleanup_finished(); }

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		Ok(match self.previous.build_submission()? {
			SubmitAnyBuilder::Empty => {
				let mut builder = SubmitCommandBufferBuilder::new();
				builder.add_command_buffer(self.command_buffer.inner());
				SubmitAnyBuilder::CommandBuffer(builder)
			}
			SubmitAnyBuilder::SemaphoresWait(sem) => {
				let mut builder: SubmitCommandBufferBuilder = sem.into();
				builder.add_command_buffer(self.command_buffer.inner());
				SubmitAnyBuilder::CommandBuffer(builder)
			}
			SubmitAnyBuilder::CommandBuffer(mut builder) => {
				// FIXME: add pipeline barrier
				builder.add_command_buffer(self.command_buffer.inner());
				SubmitAnyBuilder::CommandBuffer(builder)
			}
			SubmitAnyBuilder::QueuePresent(_) | SubmitAnyBuilder::BindSparse(_) => {
				unimplemented!() // TODO:
				 // present.submit();     // TODO: wrong
				 // let mut builder = SubmitCommandBufferBuilder::new();
				 // builder.add_command_buffer(self.command_buffer.inner());
				 // SubmitAnyBuilder::CommandBuffer(builder)
			}
		})
	}

	fn flush(&self) -> Result<(), FlushError> {
		unsafe {
			let mut submitted = self.submitted.lock().unwrap();
			if *submitted {
				return Ok(())
			}

			let queue = self.queue.clone();

			match self.build_submission()? {
				SubmitAnyBuilder::Empty => {}
				SubmitAnyBuilder::CommandBuffer(builder) => {
					builder.submit(&queue)?;
				}
				_ => unreachable!()
			};

			// Only write `true` here in order to try again next time if we failed to submit.
			*submitted = true;
			Ok(())
		}
	}

	unsafe fn signal_finished(&self) {
		if self.finished.swap(true, Ordering::SeqCst) == false {
			self.command_buffer.unlock();
		}

		self.previous.signal_finished();
	}

	fn queue_change_allowed(&self) -> bool { false }

	fn queue(&self) -> Option<Arc<Queue>> { Some(self.queue.clone()) }

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		match self.command_buffer.check_buffer_access(buffer, exclusive, queue) {
			Ok(v) => Ok(v),
			Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
			Err(AccessCheckError::Unknown) => {
				self.previous.check_buffer_access(buffer, exclusive, queue)
			}
		}
	}

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		match self.command_buffer.check_image_access(image, layout, exclusive, queue) {
			Ok(v) => Ok(v),
			Err(AccessCheckError::Denied(err)) => Err(AccessCheckError::Denied(err)),
			Err(AccessCheckError::Unknown) => {
				self.previous.check_image_access(image, layout, exclusive, queue)
			}
		}
	}
}
unsafe impl<F, Cb> DeviceOwned for CommandBufferExecFuture<F, Cb>
where
	F: GpuFuture,
	Cb: CommandBuffer
{
	fn device(&self) -> &Arc<Device> { self.command_buffer.device() }
}
impl<F, Cb> Drop for CommandBufferExecFuture<F, Cb>
where
	F: GpuFuture,
	Cb: CommandBuffer
{
	fn drop(&mut self) {
		unsafe {
			if !*self.finished.get_mut() {
				// TODO: handle errors?
				self.flush().unwrap();
				// Block until the queue finished.
				self.queue.wait().unwrap();
				self.command_buffer.unlock();
				self.previous.signal_finished();
			}
		}
	}
}
impl<F: GpuFuture, Cb: CommandBuffer> fmt::Debug for CommandBufferExecFuture<F, Cb> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		// TODO: CommandBuffer debug
		// TODO: Stack overflow if the previous chain is longish
		write!(f,
			"CommandBufferExecFuture {{ previous: <GpuFuture>, command_buffer: CommandBuffer, queue: {:?}, \
			submitted: {:?}, finished: {:?} }}",
			self.queue, self.submitted, self.finished
		)
	}
}

/// Error that can happen when attempting to execute a command buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandBufferExecError {
	/// Access to a resource aws denied.
	AccessError {
		error: AccessError,
		command_name: Cow<'static, str>,
		command_param: Cow<'static, str>,
		command_offset: usize
	},

	/// One of the one time command buffers has already been executed
	///
	/// The command buffer or one of the secondary command buffers it executes was created with the
	/// "one time submit" flag, but has already been submitted it the past.
	OneTimeSubmitAlreadySubmitted,

	/// One of the non-concurrent command buffers is already in use
	///
	/// The command buffer or one of the secondary command buffers it executes is already in use by
	/// the GPU and was not created with the "concurrent" flag.
	ExclusiveAlreadyInUse /* TODO: missing entries (eg. wrong queue family, secondary command buffer) */
}
impl fmt::Display for CommandBufferExecError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CommandBufferExecError::AccessError {
				error,
				command_name,
				command_param,
				command_offset
			} => write!(
				f,
				"Access to a resource was denied: [{}] {}({}): {}",
				command_offset, command_name, command_param, error
			),
			CommandBufferExecError::OneTimeSubmitAlreadySubmitted => {
				write!(f, "One of the one time command buffers has already been executed")
			}
			CommandBufferExecError::ExclusiveAlreadyInUse => {
				write!(f, "One of the non-concurrent command buffers is already in use")
			}
		}
	}
}
impl error::Error for CommandBufferExecError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			CommandBufferExecError::AccessError { error, .. } => Some(error),
			_ => None
		}
	}
}
