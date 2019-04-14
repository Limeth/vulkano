use std::{
	borrow::Cow,
	error,
	fmt,
	hash::{Hash, Hasher},
	sync::{Arc, Mutex}
};

use crate::{
	buffer::BufferAccess,
	command_buffer::sys::UnsafeCommandBufferBuilder,
	image::{ImageLayout, ImageViewAccess},
	sync::{AccessFlagBits, PipelineStages}
};

// Usage of a resource in a finished command buffer.
#[derive(Debug, Clone)]
pub(super) struct ResourceFinalState {
	// Stages of the last command that uses the resource.
	pub final_stages: PipelineStages,
	// Access for the last command that uses the resource.
	pub final_access: AccessFlagBits,

	// True if the resource is used in exclusive mode.
	pub exclusive: bool,

	// Layout that an image must be in at the start of the command buffer. Can be `Undefined` if we
	// don't care.
	pub initial_layout: ImageLayout,

	// Layout the image will be in at the end of the command buffer.
	pub final_layout: ImageLayout /* TODO: maybe wrap in an Option to mean that the layout doesn't change? because of buffers? */
}

/// Equivalent to `Command`, but with less methods. Typically contains less things than the
/// `Command` it comes from.
pub trait FinalCommand {
	// Returns a user-friendly name for the command, for error reporting purposes.
	fn name(&self) -> &'static str;

	// Gives access to the `num`th buffer used by the command.
	fn buffer(&self, _num: usize) -> &BufferAccess { panic!() }

	// Gives access to the `num`th image used by the command.
	fn image(&self, _num: usize) -> &dyn ImageViewAccess { panic!() }

	// Returns a user-friendly name for the `num`th buffer used by the command, for error
	// reporting purposes.
	fn buffer_name(&self, _num: usize) -> Cow<'static, str> { panic!() }

	// Returns a user-friendly name for the `num`th image used by the command, for error
	// reporting purposes.
	fn image_name(&self, _num: usize) -> Cow<'static, str> { panic!() }
}

impl FinalCommand for &'static str {
	fn name(&self) -> &'static str { *self }
}

// Equivalent of `BuilderKey` for a finished command buffer.
//
// In addition to this, it also add two other variants which are `BufferRef` and `ImageRef`. These
// variants are used in order to make it possible to compare a `CbKey` stored in the
// `SyncCommandBuffer` with a temporarily-created `CbKey`. The Rust HashMap doesn't allow us to do
// that otherwise.
//
// You should never store a `BufferRef` or a `ImageRef` inside the `SyncCommandBuffer`.
pub(super) enum CbKey<'a> {
	// The resource is held in the list of commands.
	Command {
		// Same `Arc` as in the `SyncCommandBufferBuilder`.
		commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>,
		// Index of the command that holds the resource within `commands`.
		command_id: usize,
		// Type of the resource.
		resource_ty: KeyTy,
		// Index of the resource within the command.
		resource_index: usize
	},

	// Temporary key that holds a reference to a buffer. Should never be stored in the list of
	// resources of `SyncCommandBuffer`.
	BufferRef(&'a BufferAccess),

	// Temporary key that holds a reference to an image. Should never be stored in the list of
	// resources of `SyncCommandBuffer`.
	ImageRef(&'a ImageViewAccess)
}

// The `CbKey::Command` variants implements `Send` and `Sync`, but the other two variants don't
// because it would be too constraining.
//
// Since only `CbKey::Command` must be stored in the resources hashmap, we force-implement `Send`
// and `Sync` so that the hashmap itself implements `Send` and `Sync`.
unsafe impl<'a> Send for CbKey<'a> {}
unsafe impl<'a> Sync for CbKey<'a> {}

impl<'a> CbKey<'a> {
	fn conflicts_buffer(
		&self, commands_lock: Option<&Vec<Box<FinalCommand + Send + Sync>>>, buf: &BufferAccess
	) -> bool {
		match *self {
			CbKey::Command { ref commands, command_id, resource_ty, resource_index } => {
				let lock =
					if commands_lock.is_none() { Some(commands.lock().unwrap()) } else { None };
				let commands_lock = commands_lock.unwrap_or_else(|| lock.as_ref().unwrap());

				// TODO: put the conflicts_* methods directly on the FinalCommand trait to avoid an indirect call?
				match resource_ty {
					KeyTy::Buffer => {
						let c = &commands_lock[command_id];
						c.buffer(resource_index).conflicts_buffer(buf)
					}
					KeyTy::Image => {
						let c = &commands_lock[command_id];
						c.image(resource_index).conflicts_buffer(buf)
					}
				}
			}

			CbKey::BufferRef(b) => b.conflicts_buffer(buf),
			CbKey::ImageRef(i) => i.conflicts_buffer(buf)
		}
	}

	fn conflicts_image(
		&self, commands_lock: Option<&Vec<Box<FinalCommand + Send + Sync>>>, img: &ImageViewAccess
	) -> bool {
		match *self {
			CbKey::Command { ref commands, command_id, resource_ty, resource_index } => {
				let lock =
					if commands_lock.is_none() { Some(commands.lock().unwrap()) } else { None };
				let commands_lock = commands_lock.unwrap_or_else(|| lock.as_ref().unwrap());

				// TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
				match resource_ty {
					KeyTy::Buffer => {
						let c = &commands_lock[command_id];
						c.buffer(resource_index).conflicts_image(img)
					}
					KeyTy::Image => {
						let c = &commands_lock[command_id];
						c.image(resource_index).conflicts_image(img)
					}
				}
			}

			CbKey::BufferRef(b) => b.conflicts_image(img),
			CbKey::ImageRef(i) => i.conflicts_image(img)
		}
	}
}

impl<'a> PartialEq for CbKey<'a> {
	fn eq(&self, other: &CbKey) -> bool {
		match *self {
			CbKey::BufferRef(a) => other.conflicts_buffer(None, a),
			CbKey::ImageRef(a) => other.conflicts_image(None, a),
			CbKey::Command { ref commands, command_id, resource_ty, resource_index } => {
				let commands_lock = commands.lock().unwrap();

				match resource_ty {
					KeyTy::Buffer => {
						let c = &commands_lock[command_id];
						other.conflicts_buffer(Some(&commands_lock), c.buffer(resource_index))
					}
					KeyTy::Image => {
						let c = &commands_lock[command_id];
						other.conflicts_image(Some(&commands_lock), c.image(resource_index))
					}
				}
			}
		}
	}
}

impl<'a> Eq for CbKey<'a> {}

impl<'a> Hash for CbKey<'a> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		match *self {
			CbKey::Command { ref commands, command_id, resource_ty, resource_index } => {
				let commands_lock = commands.lock().unwrap();

				match resource_ty {
					KeyTy::Buffer => {
						let c = &commands_lock[command_id];
						c.buffer(resource_index).conflict_key().hash(state)
					}
					KeyTy::Image => {
						let c = &commands_lock[command_id];
						c.image(resource_index).conflict_key().hash(state)
					}
				}
			}

			CbKey::BufferRef(buf) => buf.conflict_key().hash(state),
			CbKey::ImageRef(img) => img.conflict_key().hash(state)
		}
	}
}

/// Error returned if the builder detects that there's an unsolvable conflict.
#[derive(Debug, Clone)]
pub enum SyncCommandBufferBuilderError {
	/// Unsolvable conflict.
	Conflict {
		command1_name: &'static str,
		command1_param: Cow<'static, str>,
		command1_offset: usize,

		command2_name: &'static str,
		command2_param: Cow<'static, str>,
		command2_offset: usize
	}
}
impl fmt::Display for SyncCommandBufferBuilderError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			SyncCommandBufferBuilderError::Conflict {
				command1_name,
				command1_param,
				command1_offset,
				command2_name,
				command2_param,
				command2_offset
			} => write!(
				f,
				"Unsolvable conflict: [{}] {}({}) vs [{}] {}({})",
				command1_offset,
				command1_name,
				command1_param,
				command2_offset,
				command2_name,
				command2_param,
			)
		}
	}
}
impl error::Error for SyncCommandBufferBuilderError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}

// List of commands stored inside a `SyncCommandBufferBuilder`.
pub(super) struct Commands<P> {
	// Only the commands before `first_unflushed` have already been sent to the inner
	// `UnsafeCommandBufferBuilder`.
	pub first_unflushed: usize,

	// If we're currently inside a render pass, contains the index of the `CmdBeginRenderPass`
	// command.
	pub latest_render_pass_enter: Option<usize>,

	// The actual list.
	pub commands: Vec<Box<Command<P> + Send + Sync>>
}

// Trait for single commands within the list of commands.
pub trait Command<P> {
	// Returns a user-friendly name for the command, for error reporting purposes.
	fn name(&self) -> &'static str;

	// Sends the command to the `UnsafeCommandBufferBuilder`. Calling this method twice on the same
	// object will likely lead to a panic.
	unsafe fn send(&mut self, out: &mut UnsafeCommandBufferBuilder<P>);

	// Turns this command into a `FinalCommand`.
	fn into_final_command(self: Box<Self>) -> Box<FinalCommand + Send + Sync>;

	// Gives access to the `num`th buffer used by the command.
	fn buffer(&self, _num: usize) -> &BufferAccess { panic!() }

	// Gives access to the `num`th image used by the command.
	fn image(&self, _num: usize) -> &dyn ImageViewAccess { panic!() }

	// Returns a user-friendly name for the `num`th buffer used by the command, for error
	// reporting purposes.
	fn buffer_name(&self, _num: usize) -> Cow<'static, str> { panic!() }

	// Returns a user-friendly name for the `num`th image used by the command, for error
	// reporting purposes.
	fn image_name(&self, _num: usize) -> Cow<'static, str> { panic!() }
}

/// Type of resource whose state is to be tracked.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum KeyTy {
	Buffer,
	Image
}

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
//
// This works by holding an Arc to the list of commands and the index of the command that holds
// the resource.
pub(super) struct BuilderKey<P> {
	// Same `Arc` as in the `SyncCommandBufferBuilder`.
	pub commands: Arc<Mutex<Commands<P>>>,
	// Index of the command that holds the resource within `commands`.
	pub command_id: usize,
	// Type of the resource.
	pub resource_ty: KeyTy,
	// Index of the resource within the command.
	pub resource_index: usize
}

impl<P> BuilderKey<P> {
	// Turns this key used by the builder into a key used by the final command buffer.
	// Called when the command buffer is being built.
	pub fn into_cb_key(
		self, final_commands: Arc<Mutex<Vec<Box<FinalCommand + Send + Sync>>>>
	) -> CbKey<'static> {
		CbKey::Command {
			commands: final_commands,
			command_id: self.command_id,
			resource_ty: self.resource_ty,
			resource_index: self.resource_index
		}
	}

	fn conflicts_buffer(&self, commands_lock: &Commands<P>, buf: &BufferAccess) -> bool {
		// TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
		match self.resource_ty {
			KeyTy::Buffer => {
				let c = &commands_lock.commands[self.command_id];
				c.buffer(self.resource_index).conflicts_buffer(buf)
			}
			KeyTy::Image => {
				let c = &commands_lock.commands[self.command_id];
				c.image(self.resource_index).parent().conflicts_buffer(buf)
			}
		}
	}

	fn conflicts_image(&self, commands_lock: &Commands<P>, img: &ImageViewAccess) -> bool {
		// TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
		match self.resource_ty {
			KeyTy::Buffer => {
				let c = &commands_lock.commands[self.command_id];
				c.buffer(self.resource_index).conflicts_image(img)
			}
			KeyTy::Image => {
				let c = &commands_lock.commands[self.command_id];
				c.image(self.resource_index).conflicts_image(img)
			}
		}
	}
}

impl<P> PartialEq for BuilderKey<P> {
	fn eq(&self, other: &BuilderKey<P>) -> bool {
		debug_assert!(Arc::ptr_eq(&self.commands, &other.commands));
		let commands_lock = self.commands.lock().unwrap();

		match other.resource_ty {
			KeyTy::Buffer => {
				let c = &commands_lock.commands[other.command_id];
				self.conflicts_buffer(&commands_lock, c.buffer(other.resource_index))
			}
			KeyTy::Image => {
				let c = &commands_lock.commands[other.command_id];
				self.conflicts_image(&commands_lock, c.image(other.resource_index))
			}
		}
	}
}

impl<P> Eq for BuilderKey<P> {}

impl<P> Hash for BuilderKey<P> {
	fn hash<H: Hasher>(&self, state: &mut H) {
		let commands_lock = self.commands.lock().unwrap();

		match self.resource_ty {
			KeyTy::Buffer => {
				let c = &commands_lock.commands[self.command_id];
				c.buffer(self.resource_index).conflict_key().hash(state)
			}
			KeyTy::Image => {
				let c = &commands_lock.commands[self.command_id];
				c.image(self.resource_index).parent().conflict_key().hash(state)
			}
		}
	}
}

// State of a resource during the building of the command buffer.
#[derive(Debug, Clone)]
pub(super) struct ResourceState {
	// Stage of the command that last used this resource.
	pub stages: PipelineStages,
	// Access for the command that last used this resource.
	pub access: AccessFlagBits,

	// True if the resource was used in exclusive mode at any point during the building of the
	// command buffer. Also true if an image layout transition or queue transfer has been performed.
	pub exclusive_any: bool,

	// True if the last command that used this resource used it in exclusive mode.
	pub exclusive: bool,

	// Layout at the first use of the resource by the command buffer. Can be `Undefined` if we
	// don't care.
	pub initial_layout: ImageLayout,

	// Current layout at this stage of the building.
	pub current_layout: ImageLayout
}

impl ResourceState {
	// Turns this `ResourceState` into a `ResourceFinalState`. Called when the command buffer is
	// being built.
	pub fn finalize(self) -> ResourceFinalState {
		ResourceFinalState {
			final_stages: self.stages,
			final_access: self.access,
			exclusive: self.exclusive_any,
			initial_layout: self.initial_layout,
			final_layout: self.current_layout
		}
	}
}
