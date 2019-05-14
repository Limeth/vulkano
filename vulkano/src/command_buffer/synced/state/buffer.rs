use std::{
	borrow::Cow,
	error,
	fmt,
	hash::{Hash, Hasher},
	sync::{Arc, Mutex}
};

use crate::{
	buffer::BufferAccess,
	image::{ImageLayout, ImageLayoutEnd, ImageViewAccess},
	sync::{AccessFlagBits, PipelineStages}
};

use super::builder::KeyTy;

// Equivalent of `BuilderKey` for a finished command buffer.
//
// In addition to this, it also add two other variants which are `BufferRef` and `ImageRef`. These
// variants are used in order to make it possible to compare a `CbKey` stored in the
// `SyncCommandBuffer` with a temporarily-created `CbKey`. The Rust HashMap doesn't allow us to do
// that otherwise.
//
// You should never store a `BufferRef` or a `ImageRef` inside the `SyncCommandBuffer`.
pub enum CbKey<'a> {
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

// Usage of a resource in a finished command buffer.
#[derive(Debug, Clone)]
pub struct ResourceFinalState {
	// Stages of the last command that uses the resource.
	pub final_stages: PipelineStages,
	// Access for the last command that uses the resource.
	pub final_access: AccessFlagBits,

	// True if the resource is used in exclusive mode.
	pub exclusive: bool,

	// Layout at the first use of the resource by the command buffer. Can be `Undefined` if we don't care.
	pub initial_layout: ImageLayout,
	// Layout the image will be in at the end of the command buffer.
	pub final_layout: Option<ImageLayoutEnd>
}
impl ResourceFinalState {
	pub fn current_layout(&self) -> ImageLayout {
		match self.final_layout {
			None => self.initial_layout,
			Some(l) => ImageLayout::from(l)
		}
	}
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