use std::{
	borrow::Cow,
	hash::{Hash, Hasher},
	sync::{Arc, Mutex}
};

use crate::{
	buffer::BufferAccess,
	command_buffer::sys::UnsafeCommandBufferBuilder,
	image::{ImageLayout, ImageLayoutEnd, ImageViewAccess},
	sync::{AccessFlagBits, PipelineStages}
};

use super::buffer::{CbKey, FinalCommand, ResourceFinalState};

// List of commands stored inside a `SyncCommandBufferBuilder`.
pub struct Commands<P> {
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

/// Describes resource type and its info.
///
/// This enum is used as a parameter for the `prev_cmd_resource` function.
pub enum ResourceTypeInfo {
	/// Resource is a buffer.
	///
	/// Buffers don't care about image layouts.
	Buffer,

	/// Resource is an image.
	///
	/// The image will be transitioned to the specified layout if necessary by
	/// inserting an image barrier before the command.
	Image(ImageLayout),

	/// Resource is an image and it will possibly change layouts.
	///
	/// The image will be transitioned to the specified layout if necessary by
	/// inserting an image barrier before the command.
	///
	/// The image will be in the second layout after the command ends.
	ImageTransitioning(ImageLayout, ImageLayoutEnd)
}
impl ResourceTypeInfo {
	pub fn resource_type(&self) -> KeyTy {
		match self {
			ResourceTypeInfo::Buffer => KeyTy::Buffer,
			ResourceTypeInfo::Image(_) | ResourceTypeInfo::ImageTransitioning(_, _) => KeyTy::Image
		}
	}
}

// Key that identifies a resource. Implements `PartialEq`, `Eq` and `Hash` so that two resources
// that conflict with each other compare equal.
//
// This works by holding an Arc to the list of commands and the index of the command that holds
// the resource.
pub struct BuilderKey<P> {
	// Same `Arc` as in the `SyncCommandBufferBuilder`.
	pub commands: Arc<Mutex<Commands<P>>>,
	// Index of the command that holds the resource within `commands`.
	pub command_id: usize,
	// Type of the resource.
	pub resource_type: KeyTy,
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
			resource_ty: self.resource_type,
			resource_index: self.resource_index
		}
	}

	fn conflicts_buffer(&self, commands_lock: &Commands<P>, buf: &BufferAccess) -> bool {
		// TODO: put the conflicts_* methods directly on the Command trait to avoid an indirect call?
		match self.resource_type {
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
		match self.resource_type {
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

		match other.resource_type {
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

		match self.resource_type {
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
pub struct ResourceState {
	// True if any commands had exclusive set to true.
	pub exclusive_any: bool,

	// Stage of the command that last used this resource.
	pub stages: PipelineStages,
	// Access for the command that last used this resource.
	pub access: AccessFlagBits,

	// True if the last command that uses this resource required write access.
	pub exclusive: bool,

	// Layout at the first use of the resource by the command buffer. Can be `Undefined` if we don't care.
	pub initial_layout: ImageLayout,
	// Current layout at this stage of the building.
	pub current_layout: ImageLayout
}
impl ResourceState {
	/// Turns this `ResourceState` into a `ResourceFinalState`.
	///
	/// Called when the command buffer is being built.
	pub fn finalize(self) -> ResourceFinalState {
		let final_layout = {
			if self.current_layout == self.initial_layout {
				None
			} else {
				Some(ImageLayoutEnd::try_from_image_layout(self.current_layout).unwrap())
			}
		};

		ResourceFinalState {
			final_stages: self.stages,
			final_access: self.access,

			exclusive: self.exclusive_any,

			initial_layout: self.initial_layout,
			final_layout
		}
	}
}
