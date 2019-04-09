// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::{cmp, error, fmt, marker::PhantomData, mem, ptr, sync::Arc};

use crate::{
	device::{Device, DeviceOwned},
	format::ClearValue,
	framebuffer::{
		ensure_image_view_compatible,
		AttachmentDescription,
		AttachmentsList,
		FramebufferAbstract,
		IncompatibleRenderPassAttachmentError,
		PassDependencyDescription,
		PassDescription,
		RenderPassAbstract,
		RenderPassDesc,
		RenderPassDescClearValues,
		RenderPassSys
	},
	image::ImageViewAccess
};

use crate::{check_errors, Error, OomError, VulkanObject};
use vk_sys as vk;

/// Contains a render pass and the image views that are attached to it.
///
/// Creating a framebuffer is done by calling `Framebuffer::start`, which returns a
/// `FramebufferBuilder` object. You can then add the framebuffer attachments one by one by
/// calling `add(image)`. When you are done, call `build()`.
///
/// Both the `add` and the `build` functions perform various checks to make sure that the number
/// of images is correct and that each image is compatible with the attachment definition in the
/// render pass.
///
/// ```
/// # use std::sync::Arc;
/// # use vulkano::framebuffer::RenderPassAbstract;
/// use vulkano::framebuffer::Framebuffer;
///
/// # let render_pass: Arc<RenderPassAbstract + Send + Sync> = return;
/// # let my_image: Arc<vulkano::image::AttachmentImage<vulkano::format::Format>> = return;
/// // let render_pass: Arc<_> = ...;
/// let framebuffer =
/// 	Framebuffer::start(render_pass.clone()).add(my_image).unwrap().build().unwrap();
/// ```
///
/// Just like render pass objects implement the `RenderPassAbstract` trait, all framebuffer
/// objects implement the `FramebufferAbstract` trait. This means that you can cast any
/// `Arc<Framebuffer<..>>` into an `Arc<FramebufferAbstract + Send + Sync>` for easier storage.
///
/// ## Framebuffer dimensions
///
/// If you use `Framebuffer::start()` to create a framebuffer then vulkano will automatically
/// make sure that all the attachments have the same dimensions, as this is the most common
/// situation.
///
/// Alternatively you can also use `with_intersecting_dimensions`, in which case the dimensions of
/// the framebuffer will be the intersection of the dimensions of all attachments, or
/// `with_dimensions` if you want to specify exact dimensions. If you use `with_dimensions`, you
/// are allowed to attach images that are larger than these dimensions.
///
/// If the dimensions of the framebuffer don't match the dimensions of one of its attachment, then
/// only the top-left hand corner of the image will be drawn to.
#[derive(Debug)]
pub struct Framebuffer<Rp, A> {
	device: Arc<Device>,
	render_pass: Rp,
	framebuffer: vk::Framebuffer,
	dimensions: [u32; 3],
	resources: A
}

impl<Rp> Framebuffer<Rp, ()> {
	/// Starts building a framebuffer.
	pub fn start(render_pass: Rp) -> FramebufferBuilder<Rp, ()> {
		FramebufferBuilder {
			render_pass,
			raw_ids: SmallVec::new(),
			dimensions: FramebufferBuildeImageDimensions::AutoIdentical(None),
			attachments: ()
		}
	}

	/// Starts building a framebuffer. The dimensions of the framebuffer will automatically be
	/// the intersection of the dimensions of all the attachments.
	pub fn with_intersecting_dimensions(render_pass: Rp) -> FramebufferBuilder<Rp, ()> {
		FramebufferBuilder {
			render_pass,
			raw_ids: SmallVec::new(),
			dimensions: FramebufferBuildeImageDimensions::AutoSmaller(None),
			attachments: ()
		}
	}

	/// Starts building a framebuffer.
	pub fn with_dimensions(render_pass: Rp, dimensions: [u32; 3]) -> FramebufferBuilder<Rp, ()> {
		FramebufferBuilder {
			render_pass,
			raw_ids: SmallVec::new(),
			dimensions: FramebufferBuildeImageDimensions::Specific(dimensions),
			attachments: ()
		}
	}
}

/// Prototype of a framebuffer.
pub struct FramebufferBuilder<Rp, A> {
	render_pass: Rp,
	raw_ids: SmallVec<[vk::ImageView; 8]>,
	dimensions: FramebufferBuildeImageDimensions,
	attachments: A
}

impl<Rp, A> fmt::Debug for FramebufferBuilder<Rp, A>
where
	Rp: fmt::Debug,
	A: fmt::Debug
{
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt.debug_struct("FramebufferBuilder")
			.field("render_pass", &self.render_pass)
			.field("dimensions", &self.dimensions)
			.field("attachments", &self.attachments)
			.finish()
	}
}

#[derive(Debug)]
enum FramebufferBuildeImageDimensions {
	AutoIdentical(Option<[u32; 3]>),
	AutoSmaller(Option<[u32; 3]>),
	Specific([u32; 3])
}

impl<Rp, A> FramebufferBuilder<Rp, A>
where
	Rp: RenderPassAbstract,
	A: AttachmentsList
{
	/// Appends an attachment to the prototype of the framebuffer.
	///
	/// Attachments must be added in the same order as the one defined in the render pass.
	pub fn add<T>(
		self, attachment: T
	) -> Result<FramebufferBuilder<Rp, (A, T)>, FramebufferCreationError>
	where
		T: ImageViewAccess
	{
		if self.raw_ids.len() >= self.render_pass.num_attachments() {
			return Err(FramebufferCreationError::AttachmentsCountMismatch {
				expected: self.render_pass.num_attachments(),
				obtained: self.raw_ids.len() + 1
			})
		}

		match ensure_image_view_compatible(&self.render_pass, self.raw_ids.len(), &attachment) {
			Ok(()) => (),
			Err(err) => return Err(FramebufferCreationError::IncompatibleAttachment(err))
		};

		let img_dims = attachment.dimensions();
		debug_assert_eq!(img_dims.depth(), 1);

		let dimensions = match self.dimensions {
			FramebufferBuildeImageDimensions::AutoIdentical(None) => {
				let dims = [img_dims.width(), img_dims.height(), img_dims.array_layers()];
				FramebufferBuildeImageDimensions::AutoIdentical(Some(dims))
			}
			FramebufferBuildeImageDimensions::AutoIdentical(Some(current)) => {
				if img_dims.width() != current[0]
					|| img_dims.height() != current[1]
					|| img_dims.array_layers() != current[2]
				{
					return Err(FramebufferCreationError::AttachmenImageDimensionsIncompatible {
						expected: current,
						obtained: [img_dims.width(), img_dims.height(), img_dims.array_layers()]
					})
				}

				FramebufferBuildeImageDimensions::AutoIdentical(Some(current))
			}
			FramebufferBuildeImageDimensions::AutoSmaller(None) => {
				let dims = [img_dims.width(), img_dims.height(), img_dims.array_layers()];
				FramebufferBuildeImageDimensions::AutoSmaller(Some(dims))
			}
			FramebufferBuildeImageDimensions::AutoSmaller(Some(current)) => {
				let new_dims = [
					cmp::min(current[0], img_dims.width()),
					cmp::min(current[1], img_dims.height()),
					cmp::min(current[2], img_dims.array_layers())
				];

				FramebufferBuildeImageDimensions::AutoSmaller(Some(new_dims))
			}
			FramebufferBuildeImageDimensions::Specific(current) => {
				if img_dims.width() < current[0]
					|| img_dims.height() < current[1]
					|| img_dims.array_layers() < current[2]
				{
					return Err(FramebufferCreationError::AttachmenImageDimensionsIncompatible {
						expected: current,
						obtained: [img_dims.width(), img_dims.height(), img_dims.array_layers()]
					})
				}

				FramebufferBuildeImageDimensions::Specific([
					img_dims.width(),
					img_dims.height(),
					img_dims.array_layers()
				])
			}
		};

		let mut raw_ids = self.raw_ids;
		raw_ids.push(attachment.inner().internal_object());

		Ok(FramebufferBuilder {
			render_pass: self.render_pass,
			raw_ids,
			dimensions,
			attachments: (self.attachments, attachment)
		})
	}

	/// Turns this builder into a `FramebufferBuilder<Rp, Box<AttachmentsList>>`.
	///
	/// This allows you to store the builder in situations where you don't know in advance the
	/// number of attachments.
	///
	/// > **Note**: This is a very rare corner case and you shouldn't have to use this function
	/// > in most situations.
	pub fn boxed(self) -> FramebufferBuilder<Rp, Box<AttachmentsList>>
	where
		A: 'static
	{
		FramebufferBuilder {
			render_pass: self.render_pass,
			raw_ids: self.raw_ids,
			dimensions: self.dimensions,
			attachments: Box::new(self.attachments) as Box<_>
		}
	}

	/// Builds the framebuffer.
	pub fn build(self) -> Result<Framebuffer<Rp, A>, FramebufferCreationError> {
		let device = self.render_pass.device().clone();

		// Check the number of attachments.
		if self.raw_ids.len() != self.render_pass.num_attachments() {
			return Err(FramebufferCreationError::AttachmentsCountMismatch {
				expected: self.render_pass.num_attachments(),
				obtained: self.raw_ids.len()
			})
		}

		// Compute the dimensions.
		let dimensions = match self.dimensions {
			FramebufferBuildeImageDimensions::Specific(dims)
			| FramebufferBuildeImageDimensions::AutoIdentical(Some(dims))
			| FramebufferBuildeImageDimensions::AutoSmaller(Some(dims)) => dims,
			FramebufferBuildeImageDimensions::AutoIdentical(None)
			| FramebufferBuildeImageDimensions::AutoSmaller(None) => {
				return Err(FramebufferCreationError::CantDetermineDimensions)
			}
		};

		// Checking the dimensions against the limits.
		{
			let limits = device.physical_device().limits();
			let limits = [
				limits.max_framebuffer_width(),
				limits.max_framebuffer_height(),
				limits.max_framebuffer_layers()
			];
			if dimensions[0] > limits[0] || dimensions[1] > limits[1] || dimensions[2] > limits[2] {
				return Err(FramebufferCreationError::ImageDimensionsTooLarge)
			}
		}

		let framebuffer = unsafe {
			let vk = device.pointers();

			let infos = vk::FramebufferCreateInfo {
				sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				pNext: ptr::null(),
				flags: 0, // reserved
				renderPass: self.render_pass.inner().internal_object(),
				attachmentCount: self.raw_ids.len() as u32,
				pAttachments: self.raw_ids.as_ptr(),
				width: dimensions[0],
				height: dimensions[1],
				layers: dimensions[2]
			};

			let mut output = mem::uninitialized();
			check_errors(vk.CreateFramebuffer(
				device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			output
		};

		Ok(Framebuffer {
			device,
			render_pass: self.render_pass,
			framebuffer,
			dimensions,
			resources: self.attachments
		})
	}
}

impl<Rp, A> Framebuffer<Rp, A> {
	/// Returns the width, height and layers of this framebuffer.
	pub fn dimensions(&self) -> [u32; 3] { self.dimensions }

	/// Returns the width of the framebuffer in pixels.
	pub fn width(&self) -> u32 { self.dimensions[0] }

	/// Returns the height of the framebuffer in pixels.
	pub fn height(&self) -> u32 { self.dimensions[1] }

	/// Returns the number of layers (or depth) of the framebuffer.
	pub fn layers(&self) -> u32 { self.dimensions[2] }

	/// Returns the device that was used to create this framebuffer.
	pub fn device(&self) -> &Arc<Device> { &self.device }

	/// Returns the renderpass that was used to create this framebuffer.
	pub fn render_pass(&self) -> &Rp { &self.render_pass }
}

unsafe impl<Rp, A> FramebufferAbstract for Framebuffer<Rp, A>
where
	Rp: RenderPassAbstract,
	A: AttachmentsList
{
	fn inner(&self) -> FramebufferSys { FramebufferSys(self.framebuffer, PhantomData) }

	fn dimensions(&self) -> [u32; 3] { self.dimensions }

	fn attached_image_view(&self, index: usize) -> Option<&ImageViewAccess> {
		self.resources.as_image_view_access(index)
	}
}

unsafe impl<Rp, A> RenderPassDesc for Framebuffer<Rp, A>
where
	Rp: RenderPassDesc
{
	fn num_attachments(&self) -> usize { self.render_pass.num_attachments() }

	fn attachment_desc(&self, num: usize) -> Option<AttachmentDescription> {
		self.render_pass.attachment_desc(num)
	}

	fn num_subpasses(&self) -> usize { self.render_pass.num_subpasses() }

	fn subpass_desc(&self, num: usize) -> Option<PassDescription> {
		self.render_pass.subpass_desc(num)
	}

	fn num_dependencies(&self) -> usize { self.render_pass.num_dependencies() }

	fn dependency_desc(&self, num: usize) -> Option<PassDependencyDescription> {
		self.render_pass.dependency_desc(num)
	}
}

unsafe impl<C, Rp, A> RenderPassDescClearValues<C> for Framebuffer<Rp, A>
where
	Rp: RenderPassDescClearValues<C>
{
	fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
		self.render_pass.convert_clear_values(vals)
	}
}

unsafe impl<Rp, A> RenderPassAbstract for Framebuffer<Rp, A>
where
	Rp: RenderPassAbstract
{
	fn inner(&self) -> RenderPassSys { self.render_pass.inner() }
}

unsafe impl<Rp, A> DeviceOwned for Framebuffer<Rp, A> {
	fn device(&self) -> &Arc<Device> { &self.device }
}

impl<Rp, A> Drop for Framebuffer<Rp, A> {
	fn drop(&mut self) {
		unsafe {
			let vk = self.device.pointers();
			vk.DestroyFramebuffer(self.device.internal_object(), self.framebuffer, ptr::null());
		}
	}
}

/// Opaque object that represents the internals of a framebuffer.
#[derive(Debug, Copy, Clone)]
pub struct FramebufferSys<'a>(vk::Framebuffer, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for FramebufferSys<'a> {
	type Object = vk::Framebuffer;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT;

	fn internal_object(&self) -> vk::Framebuffer { self.0 }
}

/// Error that can happen when creating a framebuffer object.
#[derive(Copy, Clone, Debug)]
pub enum FramebufferCreationError {
	/// Out of memory.
	OomError(OomError),
	/// The requested dimensions exceed the device's limits.
	ImageDimensionsTooLarge,
	/// The attachment has a size that isn't compatible with the requested framebuffer dimensions.
	AttachmenImageDimensionsIncompatible {
		/// Expected dimensions.
		expected: [u32; 3],
		/// Attachment dimensions.
		obtained: [u32; 3]
	},
	/// The number of attachments doesn't match the number expected.
	AttachmentsCountMismatch {
		/// Expected number of attachments.
		expected: usize,
		/// Number of attachments that were given.
		obtained: usize
	},
	/// One of the images cannot be used as the requested attachment.
	IncompatibleAttachment(IncompatibleRenderPassAttachmentError),
	/// Cannot determing the dimensions because the framebuffer has
	/// no attachments and no dimensions were specified.
	CantDetermineDimensions
}
impl fmt::Display for FramebufferCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			FramebufferCreationError::OomError(e) => e.fmt(f),
			FramebufferCreationError::ImageDimensionsTooLarge
			=> write!(f, "The requested dimensions exceed the device's limits"),
			FramebufferCreationError::AttachmenImageDimensionsIncompatible { expected, obtained }
			=> write!(f, "An attachment has dimensions ({:?}) that aren't compatible with the framebuffer dimensions ({:?})",
				expected, obtained
			),
			FramebufferCreationError::AttachmentsCountMismatch { expected, obtained }
			=> write!(f, "The number of attachments ({}) doesn't match the number expected ({})", obtained, expected),
			FramebufferCreationError::IncompatibleAttachment(e) => e.fmt(f),
			FramebufferCreationError::CantDetermineDimensions
			=> write!(f, "Cannot determing the dimensions because the framebuffer has no attachments and no dimensions were specified"),
		}
	}
}
impl error::Error for FramebufferCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			FramebufferCreationError::OomError(e) => e.source(),
			FramebufferCreationError::IncompatibleAttachment(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for FramebufferCreationError {
	fn from(err: OomError) -> FramebufferCreationError { FramebufferCreationError::OomError(err) }
}
impl From<Error> for FramebufferCreationError {
	fn from(err: Error) -> FramebufferCreationError {
		FramebufferCreationError::from(OomError::from(err))
	}
}

#[cfg(test)]
mod tests {
	use crate::{
		format::Format,
		framebuffer::{
			EmptySinglePassRenderPassDesc,
			Framebuffer,
			FramebufferCreationError,
			RenderPassDesc
		},
		image::attachment::AttachmentImage
	};
	use std::sync::Arc;

	#[test]
	fn simple_create() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					color: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let image =
			AttachmentImage::new(device.clone(), [1024, 768], Format::R8G8B8A8Unorm).unwrap();
		let _ = Framebuffer::start(render_pass).add(image.clone()).unwrap().build().unwrap();
	}

	#[test]
	fn check_device_limits() {
		let (device, _) = gfx_dev_and_queue!();

		let rp = EmptySinglePassRenderPassDesc.build_render_pass(device).unwrap();
		let res = Framebuffer::with_dimensions(rp, [0xffffffff, 0xffffffff, 0xffffffff]).build();
		match res {
			Err(FramebufferCreationError::ImageDimensionsTooLarge) => (),
			_ => panic!()
		}
	}

	#[test]
	fn attachment_format_mismatch() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					color: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let image = AttachmentImage::new(device.clone(), [1024, 768], Format::R8Unorm).unwrap();

		match Framebuffer::start(render_pass).add(image.clone()) {
			Err(FramebufferCreationError::IncompatibleAttachment(_)) => (),
			_ => panic!()
		}
	}

	// TODO: check samples mismatch

	#[test]
	fn attachment_dims_larger_than_specified_valid() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					color: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let img = AttachmentImage::new(device.clone(), [600, 600], Format::R8G8B8A8Unorm).unwrap();

		let _ = Framebuffer::with_dimensions(render_pass, [512, 512, 1])
			.add(img)
			.unwrap()
			.build()
			.unwrap();
	}

	#[test]
	fn attachment_dims_smaller_than_specified() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					color: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [color],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let img = AttachmentImage::new(device.clone(), [512, 700], Format::R8G8B8A8Unorm).unwrap();

		match Framebuffer::with_dimensions(render_pass, [600, 600, 1]).add(img) {
			Err(FramebufferCreationError::AttachmenImageDimensionsIncompatible {
				expected,
				obtained
			}) => {
				assert_eq!(expected, [600, 600, 1]);
				assert_eq!(obtained, [512, 700, 1]);
			}
			_ => panic!()
		}
	}

	#[test]
	fn multi_attachments_dims_not_identical() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					a: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					},
					b: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [a, b],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let a = AttachmentImage::new(device.clone(), [512, 512], Format::R8G8B8A8Unorm).unwrap();
		let b = AttachmentImage::new(device.clone(), [512, 513], Format::R8G8B8A8Unorm).unwrap();

		match Framebuffer::start(render_pass).add(a).unwrap().add(b) {
			Err(FramebufferCreationError::AttachmenImageDimensionsIncompatible {
				expected,
				obtained
			}) => {
				assert_eq!(expected, [512, 512, 1]);
				assert_eq!(obtained, [512, 513, 1]);
			}
			_ => panic!()
		}
	}

	#[test]
	fn multi_attachments_auto_smaller() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					a: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					},
					b: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [a, b],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let a = AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8Unorm).unwrap();
		let b = AttachmentImage::new(device.clone(), [512, 128], Format::R8G8B8A8Unorm).unwrap();

		let fb = Framebuffer::with_intersecting_dimensions(render_pass)
			.add(a)
			.unwrap()
			.add(b)
			.unwrap()
			.build()
			.unwrap();

		match (fb.width(), fb.height(), fb.layers()) {
			(256, 128, 1) => (),
			_ => panic!()
		}
	}

	#[test]
	fn not_enough_attachments() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					a: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					},
					b: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [a, b],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let img = AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8Unorm).unwrap();

		let res = Framebuffer::with_intersecting_dimensions(render_pass).add(img).unwrap().build();

		match res {
			Err(FramebufferCreationError::AttachmentsCountMismatch {
				expected: 2,
				obtained: 1
			}) => (),
			_ => panic!()
		}
	}

	#[test]
	fn too_many_attachments() {
		let (device, _) = gfx_dev_and_queue!();

		let render_pass = Arc::new(
			single_pass_renderpass!(device.clone(),
				attachments: {
					a: {
						load: Clear,
						store: DontCare,
						format: Format::R8G8B8A8Unorm,
						samples: 1,
					}
				},
				pass: {
					color: [a],
					depth_stencil: {}
				}
			)
			.unwrap()
		);

		let a = AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8Unorm).unwrap();
		let b = AttachmentImage::new(device.clone(), [256, 512], Format::R8G8B8A8Unorm).unwrap();

		let res = Framebuffer::with_intersecting_dimensions(render_pass).add(a).unwrap().add(b);

		match res {
			Err(FramebufferCreationError::AttachmentsCountMismatch {
				expected: 1,
				obtained: 2
			}) => (),
			_ => panic!()
		}
	}

	#[test]
	fn empty_working() {
		let (device, _) = gfx_dev_and_queue!();

		let rp = EmptySinglePassRenderPassDesc.build_render_pass(device).unwrap();
		let _ = Framebuffer::with_dimensions(rp, [512, 512, 1]).build().unwrap();
	}

	#[test]
	fn cant_determine_dimensions_auto() {
		let (device, _) = gfx_dev_and_queue!();

		let rp = EmptySinglePassRenderPassDesc.build_render_pass(device).unwrap();
		let res = Framebuffer::start(rp).build();
		match res {
			Err(FramebufferCreationError::CantDetermineDimensions) => (),
			_ => panic!()
		}
	}

	#[test]
	fn cant_determine_dimensions_intersect() {
		let (device, _) = gfx_dev_and_queue!();

		let rp = EmptySinglePassRenderPassDesc.build_render_pass(device).unwrap();
		let res = Framebuffer::with_intersecting_dimensions(rp).build();
		match res {
			Err(FramebufferCreationError::CantDetermineDimensions) => (),
			_ => panic!()
		}
	}
}
