// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt, sync::Arc};

use crate::{
	buffer::BufferAccess,
	command_buffer::{
		submit::{
			SubmitAnyBuilder,
			SubmitBindSparseError,
			SubmitCommandBufferError,
			SubmitPresentError
		},
		CommandBuffer,
		CommandBufferExecError,
		CommandBufferExecFuture
	},
	device::{DeviceOwned, Queue},
	image::{ImageAccess, ImageLayout},
	swapchain::{self, PresentFuture, PresentRegion, Swapchain},
	sync::{AccessFlagBits, FenceWaitError, PipelineStages},
	OomError
};

pub use self::{
	fence_signal::{FenceSignalFuture, FenceSignalFutureBehavior},
	join::JoinFuture,
	now::{now, NowFuture},
	semaphore_signal::SemaphoreSignalFuture
};

mod fence_signal;
mod join;
mod now;
mod semaphore_signal;

/// Represents an event that will happen on the GPU in the future.
///
/// See the documentation of the `sync` module for explanations about futures.
// TODO: consider switching all methods to take `&mut self` for optimization purposes
pub unsafe trait GpuFuture: DeviceOwned {
	/// If possible, checks whether the submission has finished. If so, gives up ownership of the
	/// resources used by these submissions.
	///
	/// It is highly recommended to call `cleanup_finished` from time to time. Doing so will
	/// prevent memory usage from increasing over time, and will also destroy the locks on
	/// resources used by the GPU.
	fn cleanup_finished(&mut self);

	/// Builds a submission that, if submitted, makes sure that the event represented by this
	/// `GpuFuture` will happen, and possibly contains extra elements (eg. a semaphore wait or an
	/// event wait) that makes the dependency with subsequent operations work.
	///
	/// It is the responsibility of the caller to ensure that the submission is going to be
	/// submitted only once. However keep in mind that this function can perfectly be called
	/// multiple times (as long as the returned object is only submitted once).
	/// Also note that calling `flush()` on the future  may change the value returned by
	/// `build_submission()`.
	///
	/// It is however the responsibility of the implementation to not return the same submission
	/// from multiple different future objects. For example if you implement `GpuFuture` on
	/// `Arc<Foo>` then `build_submission()` must always return `SubmitAnyBuilder::Empty`,
	/// otherwise it would be possible for the user to clone the `Arc` and make the same
	/// submission be submitted multiple times.
	///
	/// It is also the responsibility of the implementation to ensure that it works if you call
	/// `build_submission()` and submits the returned value without calling `flush()` first. In
	/// other words, `build_submission()` should perform an implicit flush if necessary.
	///
	/// Once the caller has submitted the submission and has determined that the GPU has finished
	/// executing it, it should call `signal_finished`. Failure to do so will incur a large runtime
	/// overhead, as the future will have to block to make sure that it is finished.
	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError>;

	/// Flushes the future and submits to the GPU the actions that will permit this future to
	/// occur.
	///
	/// The implementation must remember that it was flushed. If the function is called multiple
	/// times, only the first time must result in a flush.
	fn flush(&self) -> Result<(), FlushError>;

	/// Sets the future to its "complete" state, meaning that it can safely be destroyed.
	///
	/// This must only be done if you called `build_submission()`, submitted the returned
	/// submission, and determined that it was finished.
	///
	/// The implementation must be aware that this function can be called multiple times on the
	/// same future.
	unsafe fn signal_finished(&self);

	/// Returns the queue that triggers the event. Returns `None` if unknown or irrelevant.
	///
	/// If this function returns `None` and `queue_change_allowed` returns `false`, then a panic
	/// is likely to occur if you use this future. This is only a problem if you implement
	/// the `GpuFuture` trait yourself for a type outside of vulkano.
	fn queue(&self) -> Option<Arc<Queue>>;

	/// Returns `true` if elements submitted after this future can be submitted to a different
	/// queue than the other returned by `queue()`.
	fn queue_change_allowed(&self) -> bool;

	/// Checks whether submitting something after this future grants access (exclusive or shared,
	/// depending on the parameter) to the given buffer on the given queue.
	///
	/// If the access is granted, returns the pipeline stage and access flags of the latest usage
	/// of this resource, or `None` if irrelevant.
	///
	/// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
	/// > "don't know". Therefore returning `Err` is never unsafe.
	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

	/// Checks whether submitting something after this future grants access (exclusive or shared,
	/// depending on the parameter) to the given image on the given queue.
	///
	/// If the access is granted, returns the pipeline stage and access flags of the latest usage
	/// of this resource, or `None` if irrelevant.
	///
	/// Implementations must ensure that the image is in the given layout. However if the `layout`
	/// is `Undefined` then the implementation should accept any actual layout.
	///
	/// > **Note**: Returning `Ok` means "access granted", while returning `Err` means
	/// > "don't know". Therefore returning `Err` is never unsafe.
	///
	/// > **Note**: Keep in mind that changing the layout of an image also requires exclusive
	/// > access.
	fn check_image_access(
		&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError>;

	/// Joins this future with another one, representing the moment when both events have happened.
	// TODO: handle errors
	fn join<F>(self, other: F) -> JoinFuture<Self, F>
	where
		Self: Sized,
		F: GpuFuture
	{
		join::join(self, other)
	}

	/// Executes a command buffer after this future.
	///
	/// > **Note**: This is just a shortcut function. The actual implementation is in the
	/// > `CommandBuffer` trait.
	fn then_execute<Cb>(
		self, queue: Arc<Queue>, command_buffer: Cb
	) -> Result<CommandBufferExecFuture<Self, Cb>, CommandBufferExecError>
	where
		Self: Sized,
		Cb: CommandBuffer + 'static
	{
		command_buffer.execute_after(self, queue)
	}

	/// Executes a command buffer after this future, on the same queue as the future.
	///
	/// > **Note**: This is just a shortcut function. The actual implementation is in the
	/// > `CommandBuffer` trait.
	fn then_execute_same_queue<Cb>(
		self, command_buffer: Cb
	) -> Result<CommandBufferExecFuture<Self, Cb>, CommandBufferExecError>
	where
		Self: Sized,
		Cb: CommandBuffer + 'static
	{
		let queue = self.queue().unwrap().clone();
		command_buffer.execute_after(self, queue)
	}

	/// Signals a semaphore after this future. Returns another future that represents the signal.
	///
	/// Call this function when you want to execute some operations on a queue and want to see the
	/// result on another queue.
	fn then_signal_semaphore(self) -> SemaphoreSignalFuture<Self>
	where
		Self: Sized
	{
		semaphore_signal::then_signal_semaphore(self)
	}

	/// Signals a semaphore after this future and flushes it. Returns another future that
	/// represents the moment when the semaphore is signalled.
	///
	/// This is a just a shortcut for `then_signal_semaphore()` followed with `flush()`.
	///
	/// When you want to execute some operations A on a queue and some operations B on another
	/// queue that need to see the results of A, it can be a good idea to submit A as soon as
	/// possible while you're preparing B.
	///
	/// If you ran A and B on the same queue, you would have to decide between submitting A then
	/// B, or A and B simultaneously. Both approaches have their trade-offs. But if A and B are
	/// on two different queues, then you would need two submits anyway and it is always
	/// advantageous to submit A as soon as possible.
	fn then_signal_semaphore_and_flush(self) -> Result<SemaphoreSignalFuture<Self>, FlushError>
	where
		Self: Sized
	{
		let f = self.then_signal_semaphore();
		f.flush()?;
		Ok(f)
	}

	/// Signals a fence after this future. Returns another future that represents the signal.
	///
	/// > **Note**: More often than not you want to immediately flush the future after calling this
	/// > function. If so, consider using `then_signal_fence_and_flush`.
	fn then_signal_fence(self) -> FenceSignalFuture<Self>
	where
		Self: Sized
	{
		fence_signal::then_signal_fence(self, FenceSignalFutureBehavior::Continue)
	}

	/// Signals a fence after this future. Returns another future that represents the signal.
	///
	/// This is a just a shortcut for `then_signal_fence()` followed with `flush()`.
	fn then_signal_fence_and_flush(self) -> Result<FenceSignalFuture<Self>, FlushError>
	where
		Self: Sized
	{
		let f = self.then_signal_fence();
		f.flush()?;
		Ok(f)
	}

	/// Presents a swapchain image after this future.
	///
	/// You should only ever do this indirectly after a `SwapchainAcquireFuture` of the same image,
	/// otherwise an error will occur when flushing.
	///
	/// > **Note**: This is just a shortcut for the `Swapchain::present()` function.
	fn then_swapchain_present<W>(
		self, queue: Arc<Queue>, swapchain: Arc<Swapchain<W>>, image_index: usize
	) -> PresentFuture<Self, W>
	where
		Self: Sized
	{
		swapchain::present(swapchain, self, queue, image_index)
	}

	/// Same as `then_swapchain_present`, except it allows specifying a present region.
	///
	/// > **Note**: This is just a shortcut for the `Swapchain::present_incremental()` function.
	fn then_swapchain_present_incremental<W>(
		self, queue: Arc<Queue>, swapchain: Arc<Swapchain<W>>, image_index: usize,
		present_region: PresentRegion
	) -> PresentFuture<Self, W>
	where
		Self: Sized
	{
		swapchain::present_incremental(swapchain, self, queue, image_index, present_region)
	}
}

unsafe impl<F: ?Sized> GpuFuture for Box<F>
where
	F: GpuFuture
{
	fn cleanup_finished(&mut self) { (**self).cleanup_finished() }

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		(**self).build_submission()
	}

	fn flush(&self) -> Result<(), FlushError> { (**self).flush() }

	unsafe fn signal_finished(&self) { (**self).signal_finished() }

	fn queue_change_allowed(&self) -> bool { (**self).queue_change_allowed() }

	fn queue(&self) -> Option<Arc<Queue>> { (**self).queue() }

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		(**self).check_buffer_access(buffer, exclusive, queue)
	}

	fn check_image_access(
		&self, image: &ImageAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		(**self).check_image_access(image, layout, exclusive, queue)
	}
}

/// Access to a resource was denied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessError {
	/// Exclusive access is denied.
	ExclusiveDenied,

	/// The resource is already in use, and there is no tracking of concurrent usages.
	AlreadyInUse,

	/// The image layout was unexpected.
	UnexpectedImageLayout { expected: ImageLayout, requested: ImageLayout },

	/// The image was in uninitialized or preinitialized layout.
	ImageNotInitialized {
		/// The layout that was requested for the image.
		requested: ImageLayout
	},

	/// Buffer was not initialized.
	BufferNotInitialized,

	/// Swapchain image can only be used after it has been acquired.
	SwapchainImageAcquireOnly
}
impl fmt::Display for AccessError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			AccessError::ExclusiveDenied => write!(f, "Exclusive access was denied"),
			AccessError::AlreadyInUse => write!(f, "The resource is already in use"),

			AccessError::UnexpectedImageLayout { expected, requested } => {
				write!(f, "The image layout ({:?}) was unexpected ({:?})", requested, expected)
			}
			AccessError::ImageNotInitialized { requested } => write!(
				f,
				"The image was in uninitialized or preinitialized layout, requested: {:?}",
				requested
			),

			AccessError::BufferNotInitialized => write!(f, "Buffer was not initialized"),
			AccessError::SwapchainImageAcquireOnly => {
				write!(f, "Swapchain image can only be used after it has been acquired")
			}
		}
	}
}
impl error::Error for AccessError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}

/// Error that can happen when checking whether we have access to a resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessCheckError {
	/// Access to the resource has been denied.
	Denied(AccessError),
	/// The resource is unknown, therefore we cannot possibly answer whether we have access or not.
	Unknown
}
impl fmt::Display for AccessCheckError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			AccessCheckError::Denied(e) => e.fmt(f),
			AccessCheckError::Unknown => write!(f, "The resource is unknown")
		}
	}
}
impl error::Error for AccessCheckError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			AccessCheckError::Denied(e) => e.source(),
			_ => None
		}
	}
}
impl From<AccessError> for AccessCheckError {
	fn from(err: AccessError) -> AccessCheckError { AccessCheckError::Denied(err) }
}

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlushError {
	/// Access to a resource has been denied.
	AccessError(AccessError),

	/// Not enough memory.
	OomError(OomError),

	/// The device was lost.
	DeviceLost,

	/// The surface was lost.
	SurfaceLost,

	/// The swapchain is out of date.
	OutOfDate,

	/// The flush operation didn't happen in given time.
	Timeout
}
impl fmt::Display for FlushError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			FlushError::AccessError(e) => e.fmt(f),
			FlushError::OomError(e) => e.fmt(f),

			FlushError::DeviceLost => write!(f, "The connection to the device device was lost"),
			FlushError::SurfaceLost => write!(f, "The surface was lost"),
			FlushError::OutOfDate => write!(f, "The swapchain is out of date"),
			FlushError::Timeout => write!(f, "The flush operation didn't happen in given time")
		}
	}
}
impl error::Error for FlushError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			FlushError::AccessError(e) => e.source(),
			FlushError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<AccessError> for FlushError {
	fn from(err: AccessError) -> FlushError { FlushError::AccessError(err) }
}
impl From<SubmitPresentError> for FlushError {
	fn from(err: SubmitPresentError) -> FlushError {
		match err {
			SubmitPresentError::OomError(err) => FlushError::OomError(err),
			SubmitPresentError::DeviceLost => FlushError::DeviceLost,
			SubmitPresentError::SurfaceLost => FlushError::SurfaceLost,
			SubmitPresentError::OutOfDate => FlushError::OutOfDate
		}
	}
}
impl From<SubmitCommandBufferError> for FlushError {
	fn from(err: SubmitCommandBufferError) -> FlushError {
		match err {
			SubmitCommandBufferError::OomError(err) => FlushError::OomError(err),
			SubmitCommandBufferError::DeviceLost => FlushError::DeviceLost
		}
	}
}
impl From<SubmitBindSparseError> for FlushError {
	fn from(err: SubmitBindSparseError) -> FlushError {
		match err {
			SubmitBindSparseError::OomError(err) => FlushError::OomError(err),
			SubmitBindSparseError::DeviceLost => FlushError::DeviceLost
		}
	}
}
impl From<FenceWaitError> for FlushError {
	fn from(err: FenceWaitError) -> FlushError {
		match err {
			FenceWaitError::OomError(err) => FlushError::OomError(err),
			FenceWaitError::Timeout => FlushError::Timeout,
			FenceWaitError::DeviceLostError => FlushError::DeviceLost
		}
	}
}
