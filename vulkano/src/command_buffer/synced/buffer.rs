use fnv::FnvHashMap;
use std::sync::{Arc, Mutex};

use crate::{
	buffer::BufferAccess,
	command_buffer::{sys::UnsafeCommandBuffer, CommandBufferExecError},
	device::{Device, DeviceOwned, Queue},
	image::{ImageLayout, ImageViewAccess},
	sync::{AccessCheckError, AccessError, AccessFlagBits, GpuFuture, PipelineStages}
};

use super::state::{
	builder::KeyTy,
	buffer::{CbKey, ResourceFinalState, FinalCommand}
};

/// Command buffer built from a `SyncCommandBufferBuilder` that provides utilities to handle
/// synchronization.
pub struct SyncCommandBuffer<P> {
	// The actual Vulkan command buffer.
	pub(super) inner: UnsafeCommandBuffer<P>,

	// State of all the resources used by this command buffer.
	pub(super) resources: FnvHashMap<CbKey<'static>, ResourceFinalState>,

	// List of commands used by the command buffer. Used to hold the various resources that are
	// being used. Each element of `resources` has a copy of this `Arc`, but we need to keep one
	// here in case `resources` is empty.
	pub(super) commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>
}
impl<P> AsRef<UnsafeCommandBuffer<P>> for SyncCommandBuffer<P> {
	fn as_ref(&self) -> &UnsafeCommandBuffer<P> { &self.inner }
}
impl<P> SyncCommandBuffer<P> {
	/// Tries to lock the resources used by the command buffer.
	///
	/// > **Note**: You should call this in the implementation of the `CommandBuffer` trait.
	pub fn lock_submit(
		&self, future: &GpuFuture, queue: &Queue
	) -> Result<(), CommandBufferExecError> {
		let commands_lock = self.commands.lock().unwrap();

		// Number of resources in `self.resources` that have been successfully locked.
		let mut locked_resources = 0;
		// Final return value of this function.
		let mut ret_value = Ok(());

		// Try locking resources. Updates `locked_resources` and `ret_value`, and break if an error
		// happens.
		for (key, entry) in self.resources.iter() {
			let (command_id, resource_ty, resource_index) = match *key {
				CbKey::Command { command_id, resource_ty, resource_index, .. } => {
					(command_id, resource_ty, resource_index)
				}
				_ => unreachable!()
			};

			match resource_ty {
				KeyTy::Buffer => {
					let cmd = &commands_lock[command_id];
					let buf = cmd.buffer(resource_index);

					// Because try_gpu_lock needs to be called first,
					// this should never return Ok without first returning Err
					let prev_err = match future.check_buffer_access(&buf, entry.exclusive, queue) {
						Ok(_) => {
							unsafe {
								buf.increase_gpu_lock();
							}
							locked_resources += 1;
							continue
						}
						Err(err) => err
					};

					match (buf.try_gpu_lock(entry.exclusive, queue), prev_err) {
						(Ok(_), _) => (),
						(Err(err), AccessCheckError::Unknown)
						| (_, AccessCheckError::Denied(err)) => {
							ret_value = Err(CommandBufferExecError::AccessError {
								error: err,
								command_name: cmd.name().into(),
								command_param: cmd.buffer_name(resource_index),
								command_offset: command_id
							});
							break
						}
					};

					locked_resources += 1;
				}

				KeyTy::Image => {
					let cmd = &commands_lock[command_id];
					let img = cmd.image(resource_index);

					let prev_err = match future.check_image_access(
						img,
						entry.initial_layout,
						entry.exclusive,
						queue
					) {
						Ok(_) => {
							unsafe {
								img.increase_gpu_lock();
							}
							locked_resources += 1;
							continue
						}
						Err(err) => err
					};

					match (img.initiate_gpu_lock(entry.exclusive, entry.initial_layout), prev_err) {
						(Ok(_), _) => (),
						(Err(err), AccessCheckError::Unknown)
						| (_, AccessCheckError::Denied(err)) => {
							ret_value = Err(CommandBufferExecError::AccessError {
								error: err,
								command_name: cmd.name().into(),
								command_param: cmd.image_name(resource_index),
								command_offset: command_id
							});
							break
						}
					};

					locked_resources += 1;
				}
			}
		}

		// If we are going to return an error, we have to unlock all the resources we locked above.
		if let Err(_) = ret_value {
			for key in self.resources.keys().take(locked_resources) {
				let (command_id, resource_ty, resource_index) = match *key {
					CbKey::Command { command_id, resource_ty, resource_index, .. } => {
						(command_id, resource_ty, resource_index)
					}
					_ => unreachable!()
				};

				match resource_ty {
					KeyTy::Buffer => {
						let cmd = &commands_lock[command_id];
						let buf = cmd.buffer(resource_index);
						unsafe {
							buf.unlock();
						}
					}

					KeyTy::Image => {
						let cmd = &commands_lock[command_id];
						let img = cmd.image(resource_index);
						unsafe {
							img.decrease_gpu_lock(None);
						}
					}
				}
			}
		}

		// TODO: pipeline barriers if necessary?

		ret_value
	}

	/// Unlocks the resources used by the command buffer.
	///
	/// > **Note**: You should call this in the implementation of the `CommandBuffer` trait.
	///
	/// # Safety
	///
	/// The command buffer must have been successfully locked with `lock_submit()`.
	pub unsafe fn unlock(&self) {
		let commands_lock = self.commands.lock().unwrap();

		for (key, value) in self.resources.iter() {
			let (command_id, resource_ty, resource_index) = match *key {
				CbKey::Command { command_id, resource_ty, resource_index, .. } => {
					(command_id, resource_ty, resource_index)
				}
				_ => unreachable!()
			};

			match resource_ty {
				KeyTy::Buffer => {
					let cmd = &commands_lock[command_id];
					let buf = cmd.buffer(resource_index);
					buf.unlock();
				}
				KeyTy::Image => {
					let cmd = &commands_lock[command_id];
					let img = cmd.image(resource_index);

					img.decrease_gpu_lock(value.final_layout);
				}
			}
		}
	}

	/// Checks whether this command buffer has access to a buffer.
	///
	/// > **Note**: Suitable when implementing the `CommandBuffer` trait.
	pub fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		// TODO: check the queue family

		if let Some(value) = self.resources.get(&CbKey::BufferRef(buffer)) {
			if !value.exclusive && exclusive {
				return Err(AccessCheckError::Unknown)
			}

			return Ok(Some((value.final_stages, value.final_access)))
		}

		Err(AccessCheckError::Unknown)
	}

	/// Checks whether this command buffer has access to an image.
	///
	/// > **Note**: Suitable when implementing the `CommandBuffer` trait.
	pub fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		// TODO: check the queue family

		if let Some(value) = self.resources.get(&CbKey::ImageRef(image)) {
			if
				layout != ImageLayout::Undefined
				&&
				value.current_layout() != layout
			{
				return Err(AccessCheckError::Denied(AccessError::ImageLayoutMismatch {
					actual: value.current_layout(),
					expected: layout
				}))
			}

			if !value.exclusive && exclusive {
				return Err(AccessCheckError::Unknown)
			}

			return Ok(Some((value.final_stages, value.final_access)))
		}

		Err(AccessCheckError::Unknown)
	}
}
unsafe impl<P> DeviceOwned for SyncCommandBuffer<P> {
	fn device(&self) -> &Arc<Device> { self.inner.device() }
}
