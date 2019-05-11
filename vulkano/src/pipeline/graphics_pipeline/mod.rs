// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{fmt, marker::PhantomData, ptr, sync::Arc, u32};

use crate::{
	buffer::BufferAccess,
	descriptor::{
		descriptor::DescriptorDesc,
		descriptor_set::UnsafeDescriptorSetLayout,
		pipeline_layout::{PipelineLayout, PipelineLayoutDesc, PipelineLayoutDescPcRange, PipelineLayoutSys}
	},
	device::{Device, DeviceOwned},
	format::ClearValue,
	framebuffer::{
		AttachmentDescription,
		PassDependencyDescription,
		PassDescription,
		RenderPassAbstract,
		RenderPassDesc,
		RenderPassDescClearValues,
		RenderPassSys,
		Subpass
	},
	pipeline::{
		shader::EmptyEntryPointDummy,
		vertex::{
			BufferlessDefinition,
			IncompatibleVertexDefinitionError,
			VertexDefinition,
			VertexSource
		}
	},
	SafeDeref,
	VulkanObject
};
use vk_sys as vk;

pub use self::{builder::GraphicsPipelineBuilder, creation_error::GraphicsPipelineCreationError};

mod builder;
mod creation_error;
// FIXME: restore
// mod tests;

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct GraphicsPipeline<VertexDefinition, RenderP> {
	inner: Inner,
	layout: Arc<PipelineLayout>,

	render_pass: RenderP,
	render_pass_subpass: u32,

	vertex_definition: VertexDefinition,

	dynamic_line_width: bool,
	dynamic_viewport: bool,
	dynamic_scissor: bool,
	dynamic_depth_bias: bool,
	dynamic_depth_bounds: bool,
	dynamic_stencil_compare_mask: bool,
	dynamic_stencil_write_mask: bool,
	dynamic_stencil_reference: bool,
	dynamic_blend_constants: bool,

	num_viewports: u32
}

struct Inner {
	pipeline: vk::Pipeline,
	device: Arc<Device>
}

impl GraphicsPipeline<(), ()> {
	/// Starts the building process of a graphics pipeline. Returns a builder object that you can
	/// fill with the various parameters.
	pub fn start<'a>() -> GraphicsPipelineBuilder<
		BufferlessDefinition,
		EmptyEntryPointDummy,
		(),
		EmptyEntryPointDummy,
		(),
		EmptyEntryPointDummy,
		(),
		EmptyEntryPointDummy,
		(),
		EmptyEntryPointDummy,
		(),
		()
	> {
		GraphicsPipelineBuilder::new()
	}
}

impl<Mv, Rp> GraphicsPipeline<Mv, Rp> {
	/// Returns the vertex definition used in the constructor.
	pub fn vertex_definition(&self) -> &Mv { &self.vertex_definition }

	/// Returns the device used to create this pipeline.
	pub fn device(&self) -> &Arc<Device> { &self.inner.device }

	/// Returns the pipeline layout used in the constructor.
	pub fn layout(&self) -> &Arc<PipelineLayout> { &self.layout }
}

impl<Mv, Rp> GraphicsPipeline<Mv, Rp>
where
	Rp: RenderPassDesc
{
	/// Returns the pass used in the constructor.
	pub fn subpass(&self) -> Subpass<&Rp> {
		Subpass::from(&self.render_pass, self.render_pass_subpass).unwrap()
	}
}

impl<Mv, Rp> GraphicsPipeline<Mv, Rp> {
	/// Returns the render pass used in the constructor.
	pub fn render_pass(&self) -> &Rp { &self.render_pass }

	/// Returns true if the line width used by this pipeline is dynamic.
	pub fn has_dynamic_line_width(&self) -> bool { self.dynamic_line_width }

	/// Returns the number of viewports and scissors of this pipeline.
	pub fn num_viewports(&self) -> u32 { self.num_viewports }

	/// Returns true if the viewports used by this pipeline are dynamic.
	pub fn has_dynamic_viewports(&self) -> bool { self.dynamic_viewport }

	/// Returns true if the scissors used by this pipeline are dynamic.
	pub fn has_dynamic_scissors(&self) -> bool { self.dynamic_scissor }

	/// Returns true if the depth bounds used by this pipeline are dynamic.
	pub fn has_dynamic_depth_bounds(&self) -> bool { self.dynamic_depth_bounds }

	/// Returns true if the stencil compare masks used by this pipeline are dynamic.
	pub fn has_dynamic_stencil_compare_mask(&self) -> bool { self.dynamic_stencil_compare_mask }

	/// Returns true if the stencil write masks used by this pipeline are dynamic.
	pub fn has_dynamic_stencil_write_mask(&self) -> bool { self.dynamic_stencil_write_mask }

	/// Returns true if the stencil references used by this pipeline are dynamic.
	pub fn has_dynamic_stencil_reference(&self) -> bool { self.dynamic_stencil_reference }
}

unsafe impl<Mv, Rp> DeviceOwned for GraphicsPipeline<Mv, Rp> {
	fn device(&self) -> &Arc<Device> { &self.inner.device }
}

impl<Mv, Rp> fmt::Debug for GraphicsPipeline<Mv, Rp> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(fmt, "<Vulkan graphics pipeline {:?}>", self.inner.pipeline)
	}
}

unsafe impl<Mv, Rp> RenderPassAbstract for GraphicsPipeline<Mv, Rp>
where
	Rp: RenderPassAbstract
{
	fn inner(&self) -> RenderPassSys { self.render_pass.inner() }
}

unsafe impl<Mv, Rp> RenderPassDesc for GraphicsPipeline<Mv, Rp>
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

unsafe impl<Mv, Rp, C> RenderPassDescClearValues<C> for GraphicsPipeline<Mv, Rp>
where
	Rp: RenderPassDescClearValues<C>
{
	fn convert_clear_values(&self, vals: C) -> Box<Iterator<Item = ClearValue>> {
		self.render_pass.convert_clear_values(vals)
	}
}

unsafe impl<Mv, Rp> VulkanObject for GraphicsPipeline<Mv, Rp> {
	type Object = vk::Pipeline;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT;

	fn internal_object(&self) -> vk::Pipeline { self.inner.pipeline }
}

impl Drop for Inner {
	fn drop(&mut self) {
		unsafe {
			let vk = self.device.pointers();
			vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
		}
	}
}

/// Trait implemented on objects that reference a graphics pipeline. Can be made into a trait
/// object.
/// When using this trait `AutoCommandBufferBuilder::draw*` calls will need the buffers to be
/// wrapped in a `vec!()`.
pub unsafe trait GraphicsPipelineAbstract:
	DeviceOwned + RenderPassAbstract + VertexSource<Vec<Arc<BufferAccess + Send + Sync>>>
{
	fn layout(&self) -> &Arc<PipelineLayout>;

	/// Returns an opaque object that represents the inside of the graphics pipeline.
	fn inner(&self) -> GraphicsPipelineSys;

	/// Returns the index of the subpass this graphics pipeline is rendering to.
	fn subpass_index(&self) -> u32;

	/// Returns the subpass this graphics pipeline is rendering to.
	fn subpass(self) -> Subpass<Self>
	where
		Self: Sized
	{
		let index = self.subpass_index();
		Subpass::from(self, index)
			.expect("Wrong subpass index in GraphicsPipelineAbstract::subpass")
	}

	/// Returns true if the line width used by this pipeline is dynamic.
	fn has_dynamic_line_width(&self) -> bool;

	/// Returns the number of viewports and scissors of this pipeline.
	fn num_viewports(&self) -> u32;

	/// Returns true if the viewports used by this pipeline are dynamic.
	fn has_dynamic_viewports(&self) -> bool;

	/// Returns true if the scissors used by this pipeline are dynamic.
	fn has_dynamic_scissors(&self) -> bool;

	/// Returns true if the depth bounds used by this pipeline are dynamic.
	fn has_dynamic_depth_bounds(&self) -> bool;

	/// Returns true if the stencil compare masks used by this pipeline are dynamic.
	fn has_dynamic_stencil_compare_mask(&self) -> bool;

	/// Returns true if the stencil write masks used by this pipeline are dynamic.
	fn has_dynamic_stencil_write_mask(&self) -> bool;

	/// Returns true if the stencil references used by this pipeline are dynamic.
	fn has_dynamic_stencil_reference(&self) -> bool;
}

unsafe impl<Mv, Rp> GraphicsPipelineAbstract for GraphicsPipeline<Mv, Rp>
where
	Rp: RenderPassAbstract,
	Mv: VertexSource<Vec<Arc<BufferAccess + Send + Sync>>>
{
	fn layout(&self) -> &Arc<PipelineLayout> { &self.layout }

	fn inner(&self) -> GraphicsPipelineSys { GraphicsPipelineSys(self.inner.pipeline, PhantomData) }

	fn subpass_index(&self) -> u32 { self.render_pass_subpass }

	fn has_dynamic_line_width(&self) -> bool { self.dynamic_line_width }

	fn num_viewports(&self) -> u32 { self.num_viewports }

	fn has_dynamic_viewports(&self) -> bool { self.dynamic_viewport }

	fn has_dynamic_scissors(&self) -> bool { self.dynamic_scissor }

	fn has_dynamic_depth_bounds(&self) -> bool { self.dynamic_depth_bounds }

	fn has_dynamic_stencil_compare_mask(&self) -> bool { self.dynamic_stencil_compare_mask }

	fn has_dynamic_stencil_write_mask(&self) -> bool { self.dynamic_stencil_write_mask }

	fn has_dynamic_stencil_reference(&self) -> bool { self.dynamic_stencil_reference }
}

unsafe impl<T> GraphicsPipelineAbstract for T
where
	T: SafeDeref,
	T::Target: GraphicsPipelineAbstract
{
	fn layout(&self) -> &Arc<PipelineLayout> { (**self).layout() }

	fn inner(&self) -> GraphicsPipelineSys { GraphicsPipelineAbstract::inner(&**self) }

	fn subpass_index(&self) -> u32 { (**self).subpass_index() }

	fn has_dynamic_line_width(&self) -> bool { (**self).has_dynamic_line_width() }

	fn num_viewports(&self) -> u32 { (**self).num_viewports() }

	fn has_dynamic_viewports(&self) -> bool { (**self).has_dynamic_viewports() }

	fn has_dynamic_scissors(&self) -> bool { (**self).has_dynamic_scissors() }

	fn has_dynamic_depth_bounds(&self) -> bool { (**self).has_dynamic_depth_bounds() }

	fn has_dynamic_stencil_compare_mask(&self) -> bool {
		(**self).has_dynamic_stencil_compare_mask()
	}

	fn has_dynamic_stencil_write_mask(&self) -> bool { (**self).has_dynamic_stencil_write_mask() }

	fn has_dynamic_stencil_reference(&self) -> bool { (**self).has_dynamic_stencil_reference() }
}

/// Opaque object that represents the inside of the graphics pipeline.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for GraphicsPipelineSys<'a> {
	type Object = vk::Pipeline;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT;

	fn internal_object(&self) -> vk::Pipeline { self.0 }
}

unsafe impl<Mv, Rp, I> VertexDefinition<I> for GraphicsPipeline<Mv, Rp>
where
	Mv: VertexDefinition<I>
{
	type AttribsIter = <Mv as VertexDefinition<I>>::AttribsIter;
	type BuffersIter = <Mv as VertexDefinition<I>>::BuffersIter;

	fn definition(
		&self, interface: &I
	) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
		self.vertex_definition.definition(interface)
	}
}

unsafe impl<Mv, Rp, S> VertexSource<S> for GraphicsPipeline<Mv, Rp>
where
	Mv: VertexSource<S>
{
	fn decode(&self, s: S) -> (Vec<Box<BufferAccess + Send + Sync>>, usize, usize) {
		self.vertex_definition.decode(s)
	}
}
