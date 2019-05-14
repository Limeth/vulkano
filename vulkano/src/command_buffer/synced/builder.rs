use fnv::FnvHashMap;
use std::{
	collections::hash_map::Entry,
	fmt,
	sync::{Arc, Mutex}
};

use crate::{
	command_buffer::{
		pool::{CommandPool, CommandPoolAlloc, CommandPoolBuilderAlloc},
		sys::{Flags, Kind, UnsafeCommandBufferBuilder, UnsafeCommandBufferBuilderPipelineBarrier}
	},
	device::{Device, DeviceOwned},
	framebuffer::{FramebufferAbstract, RenderPassAbstract},
	image::layout::{ImageLayout, ImageLayoutEnd},
	sync::{AccessFlagBits, PipelineStages},
	OomError
};

use super::{
	buffer::SyncCommandBuffer,
	state::{
		buffer::SyncCommandBufferBuilderError,
		builder::{BuilderKey, Command, Commands, KeyTy, ResourceState, ResourceTypeInfo}
	}
};

// # How pipeline stages work in Vulkan
//
// Imagine you create a command buffer that contains 10 dispatch commands, and submit that command
// buffer. According to the Vulkan specs, the implementation is free to execute the 10 commands
// simultaneously.
//
// Now imagine that the command buffer contains 10 draw commands instead. Contrary to the dispatch
// commands, the draw pipeline contains multiple stages: draw indirect, vertex input, vertex shader,
// ..., fragment shader, late fragment test, color output. When there are multiple stages, the
// implementations must start and end the stages in order. In other words it can start the draw
// indirect stage of all 10 commands, then start the vertex input stage of all 10 commands, and so
// on. But it can't for example start the fragment shader stage of a command before starting the
// vertex shader stage of another command. Same thing for ending the stages in the right order.
//
// Depending on the type of the command, the pipeline stages are different. Compute shaders use the
// compute stage, while transfer commands use the transfer stage. The compute and transfer stages
// aren't ordered.
//
// When you submit multiple command buffers to a queue, the implementation doesn't do anything in
// particular and behaves as if the command buffers were appended to one another. Therefore if you
// submit a command buffer with 10 dispatch commands, followed with another command buffer with 5
// dispatch commands, then the implementation can perform the 15 commands simultaneously.
//
// ## Introducing barriers
//
// In some situations this is not the desired behaviour. If you add a command that writes to a
// buffer followed with another command that reads that buffer, you don't want them to execute
// simultaneously. Instead you want the second one to wait until the first one is finished. This
// is done by adding a pipeline barrier between the two commands.
//
// A pipeline barriers has a source stage and a destination stage (plus various other things).
// A barrier represents a split in the list of commands. When you add it, the stages of the commands
// before the barrier corresponding to the source stage of the barrier, must finish before the
// stages of the commands after the barrier corresponding to the destination stage of the barrier
// can start.
//
// For example if you add a barrier that transitions from the compute stage to the compute stage,
// then the compute stage of all the commands before the barrier must end before the compute stage
// of all the commands after the barrier can start. This is appropriate for the example about
// writing then reading the same buffer.
//
// ## Batching barriers
//
// Since barriers are "expensive" (as the queue must block), vulkano attempts to group as many
// pipeline barriers as possible into one.
//
// Adding a command to a sync command buffer builder does not immediately add it to the underlying
// command buffer builder. Instead the command is added to a queue, and the builder keeps a
// prototype of a barrier that must be added before the commands in the queue are flushed.
//
// Whenever you add a command, the builder will find out whether a barrier is needed before the
// command. If so, it will try to merge this barrier with the prototype and add the command to the
// queue. If not possible, the queue will be entirely flushed and the command added to a fresh new
// queue with a fresh new barrier prototype.

/// Wrapper around `UnsafeCommandBufferBuilder` that handles synchronization for you.
///
/// Each method of the `UnsafeCommandBufferBuilder` has an equivalent in this wrapper, except
/// for `pipeline_layout` which is automatically handled. This wrapper automatically builds
/// pipeline barriers, keeps used resources alive and implements the `CommandBuffer` trait.
///
/// Since the implementation needs to cache commands in a `Vec`, most methods have additional
/// `Send + Sync + 'static` trait requirements on their generics.
///
/// If this builder finds out that a command isn't valid because of synchronization reasons (eg.
/// trying to copy from a buffer to an image which share the same memory), then an error is
/// returned.
/// Note that all methods are still unsafe, because this builder doesn't check the validity of
/// the commands except for synchronization purposes. The builder may panic if you pass invalid
/// commands.
///
/// The `P` generic is the same as `UnsafeCommandBufferBuilder`.
pub struct SyncCommandBufferBuilder<P> {
	// The actual Vulkan command buffer builder.
	inner: UnsafeCommandBufferBuilder<P>,

	// Stores the current state of all resources (buffers and images) that are in use by the
	// command buffer.
	resources: FnvHashMap<BuilderKey<P>, ResourceState>,

	// Prototype for the pipeline barrier that must be submitted before flushing the commands
	// in `commands`.
	pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier,

	// Stores all the commands that were added to the sync builder. Some of them are maybe not
	// submitted to the inner builder yet. A copy of this `Arc` is stored in each `BuilderKey`.
	commands: Arc<Mutex<Commands<P>>>,

	// True if we're a secondary command buffer.
	is_secondary: bool
}
impl<P> fmt::Debug for SyncCommandBufferBuilder<P> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { fmt::Debug::fmt(&self.inner, f) }
}
impl<P> SyncCommandBufferBuilder<P> {
	/// Builds a new `SyncCommandBufferBuilder`. The parameters are the same as the
	/// `UnsafeCommandBufferBuilder::new` function.
	///
	/// # Safety
	///
	/// See `UnsafeCommandBufferBuilder::new()` and `SyncCommandBufferBuilder`.
	pub unsafe fn new<Pool, R, F, A>(
		pool: &Pool, kind: Kind<R, F>, flags: Flags
	) -> Result<SyncCommandBufferBuilder<P>, OomError>
	where
		Pool: CommandPool<Builder = P, Alloc = A>,
		P: CommandPoolBuilderAlloc<Alloc = A>,
		A: CommandPoolAlloc,
		R: RenderPassAbstract,
		F: FramebufferAbstract
	{
		let (is_secondary, inside_render_pass) = match kind {
			Kind::Primary => (false, false),
			Kind::Secondary { ref render_pass, .. } => (true, render_pass.is_some())
		};

		let cmd = UnsafeCommandBufferBuilder::new(pool, kind, flags)?;
		Ok(SyncCommandBufferBuilder::from_unsafe_cmd(cmd, is_secondary, inside_render_pass))
	}

	/// Builds a `SyncCommandBufferBuilder` from an existing `UnsafeCommandBufferBuilder`.
	///
	/// # Safety
	///
	/// See `UnsafeCommandBufferBuilder::new()` and `SyncCommandBufferBuilder`.
	///
	/// In addition to this, the `UnsafeCommandBufferBuilder` should be empty. If it isn't, then
	/// you must take into account the fact that the `SyncCommandBufferBuilder` won't be aware of
	/// any existing resource usage.
	pub unsafe fn from_unsafe_cmd(
		cmd: UnsafeCommandBufferBuilder<P>, is_secondary: bool, inside_render_pass: bool
	) -> SyncCommandBufferBuilder<P> {
		let latest_render_pass_enter = if inside_render_pass { Some(0) } else { None };

		SyncCommandBufferBuilder {
			inner: cmd,
			resources: FnvHashMap::default(),
			pending_barrier: UnsafeCommandBufferBuilderPipelineBarrier::new(),
			commands: Arc::new(Mutex::new(Commands {
				first_unflushed: 0,
				latest_render_pass_enter,
				commands: Vec::new()
			})),
			is_secondary
		}
	}

	// Adds a command to be processed by the builder.
	//
	// After this method has been called, call `prev_cmd_resource` for each buffer or image used
	// by the command.
	pub(super) fn append_command<C>(&mut self, command: C)
	where
		C: Command<P> + Send + Sync + 'static
	{
		// Note that we don't submit the command to the inner command buffer yet.
		self.commands.lock().unwrap().commands.push(Box::new(command));
	}

	// Call this when the previous command entered a render pass.
	pub(super) fn prev_cmd_entered_render_pass(&mut self) {
		let mut cmd_lock = self.commands.lock().unwrap();
		cmd_lock.latest_render_pass_enter = Some(cmd_lock.commands.len() - 1);
	}

	// Call this when the previous command left a render pass.
	pub(super) fn prev_cmd_left_render_pass(&mut self) {
		let mut cmd_lock = self.commands.lock().unwrap();
		debug_assert!(cmd_lock.latest_render_pass_enter.is_some());
		cmd_lock.latest_render_pass_enter = None;
	}

	// After a command is added to the list of pending commands, this function must be called for
	// each resource used by the command that has just been added.
	// The function will take care of handling the pipeline barrier or flushing.
	//
	// `resource_ty` and `resource_index` designate the resource in the previous command (accessed
	// through `Command::buffer(..)` or `Command::image(..)`.
	//
	// `exclusive`, `stages` and `access` must match the way the resource has been used.
	//
	// `start_layout` and `end_layout` designate the image layout that the image is expected to be
	// in when the command starts, and the image layout that the image will be transitioned to
	// during the command. When it comes to buffers, you should pass `Undefined` for both.
	pub(super) fn prev_cmd_resource(
		&mut self, resource_index: usize, exclusive: bool, stages: PipelineStages,
		access: AccessFlagBits, resource_type_info: ResourceTypeInfo
	) -> Result<(), SyncCommandBufferBuilderError> {
		// Sanity checks.
		debug_assert!(access.is_compatible_with(&stages));

		let (first_unflushed_cmd_id, latest_command_id) = {
			let commands_lock = self.commands.lock().unwrap();
			debug_assert!(commands_lock.commands.len() >= 1);
			(commands_lock.first_unflushed, commands_lock.commands.len() - 1)
		};

		let resource_type = resource_type_info.resource_type();
		let key = BuilderKey {
			commands: self.commands.clone(),
			command_id: latest_command_id,
			resource_type,
			resource_index
		};

		match self.resources.entry(key) {
			// Situation where this is the first time we use this resource in this command buffer.
			Entry::Vacant(entry) => {
				if let ResourceTypeInfo::Buffer = resource_type_info {
					entry.insert(ResourceState {
						stages,
						access,

						exclusive_any: exclusive,
						exclusive,

						// We don't care about these values for buffers
						initial_layout: ImageLayout::Undefined,
						current_layout: ImageLayout::Undefined
					});
				} else {
					let (start_layout, current_layout) = match resource_type_info {
						ResourceTypeInfo::Image(start_layout) => (start_layout, start_layout),
						ResourceTypeInfo::ImageTransitioning(start_layout, end_layout) => {
							debug_assert!(
								start_layout != ImageLayout::from(end_layout),
								"ResourceTypeInfo::ImageTransitioning(start_layout, end_layout) start_layout must not be equal to end_layout"
							);
							debug_assert!(exclusive, "Access must be exclusive when ResourceTypeInfo::ImageTransitioning");

							(start_layout, end_layout.into())
						}
						_ => unsafe { std::hint::unreachable_unchecked() } /* checked in the if branch */
					};

					let initial_layout;
					let layout_change;
					if start_layout == ImageLayout::Undefined
						|| start_layout == ImageLayout::Preinitialized
						|| self.is_secondary
					{
						initial_layout = start_layout;
						layout_change = false;
					} else {
						let commands_lock = self.commands.lock().unwrap();
						let img = commands_lock.commands[latest_command_id].image(resource_index);
						let current_layout = img.current_layout().expect(
							"Cannot process an image view which has multiple different layouts"
						);

						if current_layout != start_layout {
							// Note that we transition from `bottom_of_pipe`, which means that we
							// wait for all the previous commands to be entirely finished. This is
							// suboptimal, but:
							//
							// - If we're at the start of the command buffer we have no choice anyway,
							//   because we have no knowledge about what comes before.
							// - If we're in the middle of the command buffer, this pipeline is going
							//   to be merged with an existing barrier. While it may still be
							//   suboptimal in some cases, in the general situation it will be ok.
							//
							unsafe {
								let b = &mut self.pending_barrier;
								b.add_image_memory_barrier(
									img,
									PipelineStages {
										bottom_of_pipe: true,
										..PipelineStages::none()
									},
									AccessFlagBits::none(),
									stages,
									access,
									true,
									None,
									current_layout,
									ImageLayoutEnd::try_from_image_layout(start_layout).unwrap() /* already checked above */
								);
							}

							initial_layout = current_layout;
							layout_change = true;
						} else {
							initial_layout = start_layout;
							layout_change = false;
						}
					}

					entry.insert(ResourceState {
						exclusive_any: exclusive || layout_change,

						stages,
						access,

						exclusive,

						initial_layout,
						current_layout
					});
				}
			}

			// Situation where this resource was used before in this command buffer.
			Entry::Occupied(entry) => {
				// `collision_cmd_id` contains the ID of the command that we are potentially colliding with.
				let collision_cmd_id = entry.key().command_id;
				debug_assert!(collision_cmd_id <= latest_command_id);

				let entry_key_resource_index = entry.key().resource_index;
				let entry_key_resource_type = entry.key().resource_type;
				let entry = entry.into_mut();

				if let ResourceTypeInfo::Buffer = resource_type_info {
					if exclusive || entry.exclusive {
						// We might need to insert a new barrier between the `collision_cmd_id` and `latest_command_id`.
						// Here we detect a case where the `pending_barrier` is still before `collision_cmd_id`.
						// That is, we detect a state where:
						//
						// |---B---C---L|
						//
						// because we want a state where:
						//
						// |---C---BL|
						//
						// where C = collision_cmd_id, L = latest_command_id and B = pending_barrier.
						if collision_cmd_id >= first_unflushed_cmd_id {
							unsafe {
								SyncCommandBufferBuilder::flush_pending_barrier(
									&mut self.inner,
									&self.pending_barrier,
									&self.commands,
									collision_cmd_id,
									entry_key_resource_index,
									entry_key_resource_type,
									latest_command_id,
									resource_index,
									resource_type
								)?;
								self.pending_barrier =
									UnsafeCommandBufferBuilderPipelineBarrier::new();
							}
						}

						// We add a buffer barrier.
						unsafe {
							let commands_lock = self.commands.lock().unwrap();
							let buf =
								commands_lock.commands[latest_command_id].buffer(resource_index);

							let b = &mut self.pending_barrier;
							b.add_buffer_memory_barrier(
								buf,
								entry.stages,
								entry.access,
								stages,
								access,
								true,
								None,
								0,
								buf.size()
							);
						}

						// Update state.
						entry.stages = stages;
						entry.access = access;
						entry.exclusive_any = true;
						entry.exclusive = exclusive;
					} else {
						// There is no collision. Simply merge the stages and accesses.
						// TODO: what about simplifying the newly-constructed stages/accesses?
						//       this would simplify the job of the driver, but is it worth it?
						entry.stages = entry.stages | stages;
						entry.access = entry.access | access;
					}
				} else {
					let (start_layout, current_layout) = match resource_type_info {
						ResourceTypeInfo::Image(start_layout) => {
							let start_layout = match start_layout {
								ImageLayout::Undefined | ImageLayout::Preinitialized => {
									entry.current_layout
								}
								_ => start_layout
							};

							(start_layout, start_layout)
						}
						ResourceTypeInfo::ImageTransitioning(start_layout, end_layout) => {
							debug_assert!(
								start_layout != ImageLayout::from(end_layout),
								"ResourceTypeInfo::ImageTransitioning(start_layout, end_layout) start_layout must not be equal to end_layout"
							);
							debug_assert!(exclusive, "Access must be exclusive when ResourceTypeInfo::ImageTransitioning");

							let start_layout = match start_layout {
								ImageLayout::Undefined | ImageLayout::Preinitialized => {
									entry.current_layout
								}
								_ => start_layout
							};

							(start_layout, end_layout.into())
						}
						_ => unsafe { std::hint::unreachable_unchecked() } /* checked in the if branch */
					};

					if exclusive || entry.exclusive || start_layout != entry.current_layout {
						// More info in `ResourceTypeInfo::Buffer` arm above.
						if collision_cmd_id >= first_unflushed_cmd_id {
							unsafe {
								SyncCommandBufferBuilder::flush_pending_barrier(
									&mut self.inner,
									&self.pending_barrier,
									&self.commands,
									collision_cmd_id,
									entry_key_resource_index,
									entry_key_resource_type,
									latest_command_id,
									resource_index,
									resource_type
								)?;
								self.pending_barrier =
									UnsafeCommandBufferBuilderPipelineBarrier::new();
							}
						}

						// We add an image barrier, possibly transitioning layouts.
						unsafe {
							let commands_lock = self.commands.lock().unwrap();
							let img =
								commands_lock.commands[latest_command_id].image(resource_index);

							let b = &mut self.pending_barrier;
							b.add_image_memory_barrier(
								img,
								entry.stages,
								entry.access,
								stages,
								access,
								true,
								None,
								entry.current_layout.into(),
								ImageLayoutEnd::try_from_image_layout(start_layout).unwrap() /* checked above */
							);
						}

						// Update state.
						entry.stages = stages;
						entry.access = access;
						entry.exclusive_any = true;
						entry.exclusive = exclusive;
						entry.current_layout = current_layout;
					} else {
						// There is no collision. Simply merge the stages and accesses.
						// TODO: what about simplifying the newly-constructed stages/accesses?
						// This would simplify the job of the driver, but is it worth it?
						entry.stages = entry.stages | stages;
						entry.access = entry.access | access;
					}
				}
			}
		}


		Ok(())
	}

	// Part of the `prev_cmd_resource` function extracted so it's reusable
	// and more readable.
	#[inline(always)]
	unsafe fn flush_pending_barrier(
		inner: &mut UnsafeCommandBufferBuilder<P>,
		pending_barrier: &UnsafeCommandBufferBuilderPipelineBarrier,
		commands: &Arc<Mutex<Commands<P>>>, collision_cmd_id: usize,
		entry_key_resource_index: usize, entry_key_resource_type: KeyTy, latest_command_id: usize,
		resource_index: usize, resource_type: KeyTy
	) -> Result<(), SyncCommandBufferBuilderError> {
		// Flush the pending barrier.
		inner.pipeline_barrier(pending_barrier);

		let mut commands_lock = commands.lock().unwrap();
		let start = commands_lock.first_unflushed;
		let end = if let Some(rp_enter) = commands_lock.latest_render_pass_enter {
			rp_enter
		} else {
			latest_command_id
		};

		if collision_cmd_id >= end {
			let cmd1 = &commands_lock.commands[collision_cmd_id];
			let cmd2 = &commands_lock.commands[latest_command_id];

			return Err(SyncCommandBufferBuilderError::Conflict {
				command1_name: cmd1.name(),
				command1_param: match entry_key_resource_type {
					KeyTy::Buffer => cmd1.buffer_name(entry_key_resource_index),
					KeyTy::Image => cmd1.image_name(entry_key_resource_index)
				},
				command1_offset: collision_cmd_id,

				command2_name: cmd2.name(),
				command2_param: match resource_type {
					KeyTy::Buffer => cmd2.buffer_name(resource_index),
					KeyTy::Image => cmd2.image_name(resource_index)
				},
				command2_offset: latest_command_id
			})
		}
		for command in &mut commands_lock.commands[start .. end] {
			command.send(inner);
		}
		commands_lock.first_unflushed = end;

		Ok(())
	}

	/// Builds the command buffer and turns it into a `SyncCommandBuffer`.
	pub fn build(mut self) -> Result<SyncCommandBuffer<P::Alloc>, OomError>
	where
		P: CommandPoolBuilderAlloc
	{
		let mut commands_lock = self.commands.lock().unwrap();
		debug_assert!(
			commands_lock.latest_render_pass_enter.is_none() || self.pending_barrier.is_empty()
		);

		// The commands that haven't been sent to the inner command buffer yet need to be sent.
		unsafe {
			self.inner.pipeline_barrier(&self.pending_barrier);
			let f = commands_lock.first_unflushed;
			for command in &mut commands_lock.commands[f ..] {
				command.send(&mut self.inner);
			}
		}

		// Transition images to their desired final layout.
		if !self.is_secondary {
			unsafe {
				// TODO: this could be optimized by merging the barrier with the barrier above?
				let mut barrier = UnsafeCommandBufferBuilderPipelineBarrier::new();

				for (key, state) in &mut self.resources {
					if key.resource_type != KeyTy::Image {
						continue
					}

					let img = commands_lock.commands[key.command_id].image(key.resource_index);
					let requested_layout = match img.required_layouts().global {
						None => continue,
						Some(l) => l
					};;
					if ImageLayout::from(requested_layout) == state.current_layout {
						continue
					}

					barrier.add_image_memory_barrier(
						img,
						state.stages,
						state.access,
						PipelineStages { top_of_pipe: true, ..PipelineStages::none() },
						AccessFlagBits::none(),
						true,
						None, // TODO: queue transfers?
						state.current_layout.into(),
						requested_layout
					);

					state.exclusive_any = true;
					state.current_layout = ImageLayout::from(requested_layout);
				}

				self.inner.pipeline_barrier(&barrier);
			}
		}

		// Turns the commands into a list of "final commands" that are slimmer.
		let final_commands = {
			let mut final_commands = Vec::with_capacity(commands_lock.commands.len());
			for command in commands_lock.commands.drain(..) {
				final_commands.push(command.into_final_command());
			}
			Arc::new(Mutex::new(final_commands))
		};

		// Build the final resources states.
		let final_resources_states: FnvHashMap<_, _> = {
			self.resources
				.into_iter()
				.map(|(resource, state)| {
					(resource.into_cb_key(final_commands.clone()), state.finalize())
				})
				.collect()
		};

		Ok(SyncCommandBuffer {
			inner: self.inner.build()?,
			resources: final_resources_states,
			commands: final_commands
		})
	}
}
unsafe impl<P> DeviceOwned for SyncCommandBufferBuilder<P> {
	fn device(&self) -> &Arc<Device> { self.inner.device() }
}
