// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{ffi::CStr, fmt, mem, ptr, sync::Arc};

use smallvec::SmallVec;

use vk_sys as vk;

use crate::{
	buffer::{BufferAccess, BufferInner},
	check_errors,
	command_buffer::{
		pool::{CommandPool, CommandPoolAlloc, CommandPoolBuilderAlloc},
	},
	descriptor::{
		descriptor::ShaderStages,
		descriptor_set::UnsafeDescriptorSet,
		pipeline_layout::PipelineLayoutAbstract
	},
	device::{Device, DeviceOwned},
	format::{ClearValue, FormatTy},
	format::{
		FormatDesc
	},
	framebuffer::{
		EmptySinglePassRenderPassDesc,
		Framebuffer,
		FramebufferAbstract,
		RenderPass,
		RenderPassAbstract,
		Subpass,
		SubpassContents
	},
	image::{
		layout::{ImageLayoutImageDst, ImageLayoutImageSrc},
		ImageViewAccess
	},
	instance::QueueFamily,
	pipeline::{
		input_assembly::IndexType,
		viewport::{Scissor, Viewport},
		ComputePipelineAbstract,
		GraphicsPipelineAbstract
	},
	query::{QueryPipelineStatisticFlags, UnsafeQueriesRange, UnsafeQuery},
	sampler::Filter,
	sync::{Event, PipelineStages},
	OomError,
	VulkanObject
};

mod misc;
pub use misc::*;

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<R, F> {
	/// A primary command buffer can execute all commands and can call secondary command buffers.
	Primary,

	/// A secondary command buffer.
	Secondary {
		/// If `Some`, can only call draw operations that can be executed from within a specific
		/// subpass. Otherwise it can execute all dispatch and transfer operations, but not drawing
		/// operations.
		render_pass: Option<KindSecondaryRenderPass<R, F>>,

		/// Whether it is allowed to have an active occlusion query in the primary command buffer
		/// when executing this secondary command buffer.
		occlusion_query: KindOcclusionQuery,

		/// Which pipeline statistics queries are allowed to be active when this secondary command
		/// buffer starts.
		///
		/// Note that the `pipeline_statistics_query` feature must be enabled if any of the flags
		/// of this value are set.
		query_statistics_flags: QueryPipelineStatisticFlags
	}
}

/// Additional information for `Kind::Secondary`.
#[derive(Debug, Clone)]
pub struct KindSecondaryRenderPass<R, F> {
	/// Which subpass this secondary command buffer can be called from.
	pub subpass: Subpass<R>,

	/// The framebuffer object that will be used when calling the command buffer.
	/// This parameter is optional and is an optimization hint for the implementation.
	pub framebuffer: Option<F>
}

/// Additional information for `Kind::Secondary`.
#[derive(Debug, Copy, Clone)]
pub enum KindOcclusionQuery {
	/// It is allowed to have an active occlusion query in the primary command buffer when
	/// executing this secondary command buffer.
	///
	/// The `inherited_queries` feature must be enabled on the device for this to be a valid option.
	Allowed {
		/// The occlusion query can have the `control_precise` flag.
		control_precise_allowed: bool
	},

	/// It is forbidden to have an active occlusion query.
	Forbidden
}

impl
	Kind<
		RenderPass<EmptySinglePassRenderPassDesc>,
		Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>
	>
{
	/// Equivalent to `Kind::Primary`.
	///
	/// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
	/// > compilation error because the Rust compiler couldn't determine the template parameters
	/// > of `Kind`. To solve that problem in an easy way you can use this function instead.
	pub fn primary() -> Kind<
		Arc<RenderPass<EmptySinglePassRenderPassDesc>>,
		Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>
	> {
		Kind::Primary
	}

	/// Equivalent to `Kind::Secondary`.
	///
	/// > **Note**: If you use `let kind = Kind::Secondary;` in your code, you will probably get a
	/// > compilation error because the Rust compiler couldn't determine the template parameters
	/// > of `Kind`. To solve that problem in an easy way you can use this function instead.
	pub fn secondary(
		occlusion_query: KindOcclusionQuery, query_statistics_flags: QueryPipelineStatisticFlags
	) -> Kind<
		Arc<RenderPass<EmptySinglePassRenderPassDesc>>,
		Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>
	> {
		Kind::Secondary { render_pass: None, occlusion_query, query_statistics_flags }
	}
}

/// Flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Flags {
	/// The command buffer can be used multiple times, but must not execute more than once
	/// simultaneously.
	None,

	/// The command buffer can be executed multiple times in parallel.
	SimultaneousUse,

	/// The command buffer can only be submitted once. Any further submit is forbidden.
	OneTimeSubmit
}

/// Command buffer being built.
///
/// You can add commands to an `UnsafeCommandBufferBuilder` by using the `AddCommand` trait.
/// The `AddCommand<&Cmd>` trait is implemented on the `UnsafeCommandBufferBuilder` for any `Cmd`
/// that is a raw Vulkan command.
///
/// When you are finished adding commands, you can use the `CommandBufferBuild` trait to turn this
/// builder into an `UnsafeCommandBuffer`.
pub struct UnsafeCommandBufferBuilder<P> {
	// The command buffer obtained from the pool. Contains `None` if `build()` has been called.
	cmd: Option<P>,

	// The raw `cmd`. Avoids having to specify a trait bound on `P`.
	cmd_raw: vk::CommandBuffer,

	// Device that owns the command buffer.
	// TODO: necessary?
	device: Arc<Device>
}
impl<P> fmt::Debug for UnsafeCommandBufferBuilder<P> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "<Vulkan command buffer builder #{}>", self.cmd_raw)
	}
}
impl<P> UnsafeCommandBufferBuilder<P> {
	/// Creates a new builder.
	///
	/// # Safety
	///
	/// Creating and destroying an unsafe command buffer is not unsafe per se, but the commands
	/// that you add to it are unchecked, do not have any synchronization, and are not kept alive.
	///
	/// In other words, it is your job to make sure that the commands you add are valid, that they
	/// don't use resources that have been destroyed, and that they do not introduce any race
	/// condition.
	///
	/// > **Note**: Some checks are still made with `debug_assert!`. Do not expect to be able to
	/// > submit invalid commands.
	pub unsafe fn new<Pool, R, F, A>(
		pool: &Pool, kind: Kind<R, F>, flags: Flags
	) -> Result<UnsafeCommandBufferBuilder<P>, OomError>
	where
		Pool: CommandPool<Builder = P, Alloc = A>,
		P: CommandPoolBuilderAlloc<Alloc = A>,
		A: CommandPoolAlloc,
		R: RenderPassAbstract,
		F: FramebufferAbstract
	{
		let secondary = match kind {
			Kind::Primary => false,
			Kind::Secondary { .. } => true
		};

		let cmd = pool
			.alloc(secondary, 1)?
			.next()
			.expect("Requested one command buffer from the command pool, but got zero.");
		UnsafeCommandBufferBuilder::already_allocated(cmd, kind, flags)
	}

	/// Creates a new command buffer builder from an already-allocated command buffer.
	///
	/// # Safety
	///
	/// See the `new` method.
	///
	/// The kind must match how the command buffer was allocated.
	pub unsafe fn already_allocated<R, F>(
		alloc: P, kind: Kind<R, F>, flags: Flags
	) -> Result<UnsafeCommandBufferBuilder<P>, OomError>
	where
		R: RenderPassAbstract,
		F: FramebufferAbstract,
		P: CommandPoolBuilderAlloc
	{
		let device = alloc.device().clone();
		let vk = device.pointers();
		let cmd = alloc.inner().internal_object();

		let vk_flags = {
			let a = match flags {
				Flags::None => 0,
				Flags::SimultaneousUse => vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				Flags::OneTimeSubmit => vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
			};

			let b = match kind {
				Kind::Secondary { ref render_pass, .. } if render_pass.is_some() => {
					vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
				}
				_ => 0
			};

			a | b
		};

		let (rp, sp, fb) = match kind {
			Kind::Secondary { render_pass: Some(ref render_pass), .. } => {
				let rp = render_pass.subpass.render_pass().inner().internal_object();
				let sp = render_pass.subpass.index();
				let fb = match render_pass.framebuffer {
					Some(ref fb) => {
						// TODO: debug assert that the framebuffer is compatible with
						//       the render pass?
						FramebufferAbstract::inner(fb).internal_object()
					}
					None => 0
				};
				(rp, sp, fb)
			}
			_ => (0, 0, 0)
		};

		let (oqe, qf, ps) = match kind {
			Kind::Secondary { occlusion_query, query_statistics_flags, .. } => {
				let ps: vk::QueryPipelineStatisticFlagBits = query_statistics_flags.into();
				debug_assert!(
					ps == 0 || alloc.device().enabled_features().pipeline_statistics_query
				);

				let (oqe, qf) = match occlusion_query {
					KindOcclusionQuery::Allowed { control_precise_allowed } => {
						debug_assert!(alloc.device().enabled_features().inherited_queries);
						let qf =
							if control_precise_allowed { vk::QUERY_CONTROL_PRECISE_BIT } else { 0 };
						(vk::TRUE, qf)
					}
					KindOcclusionQuery::Forbidden => (0, 0)
				};

				(oqe, qf, ps)
			}
			_ => (0, 0, 0)
		};

		let inheritance = vk::CommandBufferInheritanceInfo {
			sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
			pNext: ptr::null(),
			renderPass: rp,
			subpass: sp,
			framebuffer: fb,
			occlusionQueryEnable: oqe,
			queryFlags: qf,
			pipelineStatistics: ps
		};

		let infos = vk::CommandBufferBeginInfo {
			sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			pNext: ptr::null(),
			flags: vk_flags,
			pInheritanceInfo: &inheritance
		};

		check_errors(vk.BeginCommandBuffer(cmd, &infos))?;

		Ok(UnsafeCommandBufferBuilder { cmd: Some(alloc), cmd_raw: cmd, device: device.clone() })
	}

	/// Returns the queue family of the builder.
	pub fn queue_family(&self) -> QueueFamily
	where
		P: CommandPoolBuilderAlloc
	{
		self.cmd.as_ref().unwrap().queue_family()
	}

	/// Turns the builder into an actual command buffer.
	pub fn build(mut self) -> Result<UnsafeCommandBuffer<P::Alloc>, OomError>
	where
		P: CommandPoolBuilderAlloc
	{
		unsafe {
			let cmd = self.cmd.take().unwrap();
			let vk = self.device.pointers();
			check_errors(vk.EndCommandBuffer(cmd.inner().internal_object()))?;
			let cmd_raw = cmd.inner().internal_object();

			Ok(UnsafeCommandBuffer { cmd: cmd.into_alloc(), cmd_raw, device: self.device.clone() })
		}
	}

	/// Calls `vkCmdBeginQuery` on the builder.
	pub unsafe fn begin_query(&mut self, query: UnsafeQuery, precise: bool) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		let flags = if precise { vk::QUERY_CONTROL_PRECISE_BIT } else { 0 };
		vk.CmdBeginQuery(cmd, query.pool().internal_object(), query.index(), flags);
	}

	/// Calls `vkCmdBeginRenderPass` on the builder.
	pub unsafe fn begin_render_pass<F, I>(
		&mut self, framebuffer: &F, subpass_contents: SubpassContents, clear_values: I
	) where
		F: ?Sized + FramebufferAbstract,
		I: Iterator<Item = ClearValue>
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		// TODO: allow passing a different render pass
		let raw_render_pass = RenderPassAbstract::inner(&framebuffer).internal_object();
		let raw_framebuffer = FramebufferAbstract::inner(&framebuffer).internal_object();

		let raw_clear_values: SmallVec<[_; 12]> = clear_values
			.map(|clear_value| match clear_value {
				ClearValue::None => {
					vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }
				}
				ClearValue::Float(val) => {
					vk::ClearValue { color: vk::ClearColorValue { float32: val } }
				}
				ClearValue::Int(val) => {
					vk::ClearValue { color: vk::ClearColorValue { int32: val } }
				}
				ClearValue::Uint(val) => {
					vk::ClearValue { color: vk::ClearColorValue { uint32: val } }
				}
				ClearValue::Depth(val) => vk::ClearValue {
					depthStencil: vk::ClearDepthStencilValue { depth: val, stencil: 0 }
				},
				ClearValue::Stencil(val) => vk::ClearValue {
					depthStencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: val }
				},
				ClearValue::DepthStencil((depth, stencil)) => {
					vk::ClearValue { depthStencil: vk::ClearDepthStencilValue { depth, stencil } }
				}
			})
			.collect();

		// TODO: allow customizing
		let rect = [0 .. framebuffer.dimensions()[0], 0 .. framebuffer.dimensions()[1]];

		let begin = vk::RenderPassBeginInfo {
			sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			pNext: ptr::null(),
			renderPass: raw_render_pass,
			framebuffer: raw_framebuffer,
			renderArea: vk::Rect2D {
				offset: vk::Offset2D { x: rect[0].start as i32, y: rect[1].start as i32 },
				extent: vk::Extent2D {
					width: rect[0].end - rect[0].start,
					height: rect[1].end - rect[1].start
				}
			},
			clearValueCount: raw_clear_values.len() as u32,
			pClearValues: raw_clear_values.as_ptr()
		};

		vk.CmdBeginRenderPass(cmd, &begin, subpass_contents as u32);
	}

	/// Calls `vkCmdBindDescriptorSets` on the builder.
	///
	/// Does nothing if the list of descriptor sets is empty, as it would be a no-op and isn't a
	/// valid usage of the command anyway.
	pub unsafe fn bind_descriptor_sets<'s, Pl, S, I>(
		&mut self, graphics: bool, pipeline_layout: &Pl, first_binding: u32, sets: S,
		dynamic_offsets: I
	) where
		Pl: ?Sized + PipelineLayoutAbstract,
		S: Iterator<Item = &'s UnsafeDescriptorSet>,
		I: Iterator<Item = u32>
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let sets: SmallVec<[_; 12]> = sets.map(|s| s.internal_object()).collect();
		if sets.is_empty() {
			return
		}
		let dynamic_offsets: SmallVec<[u32; 32]> = dynamic_offsets.collect();

		let num_bindings = sets.len() as u32;
		debug_assert!(first_binding + num_bindings <= pipeline_layout.num_sets() as u32);

		let bind_point = if graphics {
			vk::PIPELINE_BIND_POINT_GRAPHICS
		} else {
			vk::PIPELINE_BIND_POINT_COMPUTE
		};

		vk.CmdBindDescriptorSets(
			cmd,
			bind_point,
			pipeline_layout.sys().internal_object(),
			first_binding,
			num_bindings,
			sets.as_ptr(),
			dynamic_offsets.len() as u32,
			dynamic_offsets.as_ptr()
		);
	}

	/// Calls `vkCmdBindIndexBuffer` on the builder.
	pub unsafe fn bind_index_buffer<B>(&mut self, buffer: &B, index_ty: IndexType)
	where
		B: ?Sized + BufferAccess
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let inner = buffer.inner();
		debug_assert!(inner.offset < inner.buffer.size());
		debug_assert!(inner.buffer.usage_index_buffer());

		vk.CmdBindIndexBuffer(
			cmd,
			inner.buffer.internal_object(),
			inner.offset as vk::DeviceSize,
			index_ty as vk::IndexType
		);
	}

	/// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
	pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: &Cp)
	where
		Cp: ?Sized + ComputePipelineAbstract
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdBindPipeline(
			cmd,
			vk::PIPELINE_BIND_POINT_COMPUTE,
			pipeline.inner().internal_object()
		);
	}

	/// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
	pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: &Gp)
	where
		Gp: ?Sized + GraphicsPipelineAbstract
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		let inner = GraphicsPipelineAbstract::inner(pipeline).internal_object();
		vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, inner);
	}

	/// Calls `vkCmdBindVertexBuffers` on the builder.
	///
	/// Does nothing if the list of buffers is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn bind_vertex_buffers(
		&mut self, first_binding: u32, params: UnsafeCommandBufferBuilderBindVertexBuffer
	) {
		debug_assert_eq!(params.raw_buffers.len(), params.offsets.len());

		if params.raw_buffers.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let num_bindings = params.raw_buffers.len() as u32;

		debug_assert!({
			let max_bindings = self.device().physical_device().limits().max_vertex_input_bindings();
			first_binding + num_bindings <= max_bindings
		});

		vk.CmdBindVertexBuffers(
			cmd,
			first_binding,
			num_bindings,
			params.raw_buffers.as_ptr(),
			params.offsets.as_ptr()
		);
	}

	/// Calls `vkCmdCopyImage` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn copy_image<S, D, R>(
		&mut self, source: &S, source_layout: ImageLayoutImageSrc, destination: &D,
		destination_layout: ImageLayoutImageDst, regions: R
	) where
		S: ?Sized + ImageViewAccess,
		D: ?Sized + ImageViewAccess,
		R: Iterator<Item = UnsafeCommandBufferBuilderImageCopy>
	{
		// TODO: The correct check here is that the uncompressed element size of the source is
		// equal to the compressed element size of the destination.
		debug_assert!(
			source.format().is_compressed()
				|| destination.format().is_compressed()
				|| source.format().size() == destination.format().size()
		);

		// Depth/Stencil formats are required to match exactly.
		debug_assert!(
			!source.format().ty().is_depth_and_or_stencil()
				|| source.format() == destination.format()
		);

		debug_assert_eq!(source.parent().samples(), destination.parent().samples());
		debug_assert!(source.usage().transfer_source);

		debug_assert!(destination.usage().transfer_destination);

		let regions: SmallVec<[_; 8]> = regions
			.filter_map(|copy| {
				// TODO: not everything is checked here
				debug_assert!(
					copy.source_base_array_layer + copy.layer_count
						<= source.subresource_range().array_layers.get()
				);
				debug_assert!(
					copy.destination_base_array_layer + copy.layer_count
						<= destination.subresource_range().array_layers.get()
				);
				debug_assert!(
					copy.source_mip_level < destination.subresource_range().mipmap_levels.get()
				);
				debug_assert!(
					copy.destination_mip_level
						< destination.subresource_range().mipmap_levels.get()
				);

				if copy.layer_count == 0 {
					return None
				}

				Some(vk::ImageCopy {
					srcSubresource: vk::ImageSubresourceLayers {
						aspectMask: copy.aspect.to_vk_bits(),
						mipLevel: copy.source_mip_level,
						baseArrayLayer: copy.source_base_array_layer
							+ source.subresource_range().array_layers_offset,
						layerCount: copy.layer_count
					},
					srcOffset: vk::Offset3D {
						x: copy.source_offset[0],
						y: copy.source_offset[1],
						z: copy.source_offset[2]
					},
					dstSubresource: vk::ImageSubresourceLayers {
						aspectMask: copy.aspect.to_vk_bits(),
						mipLevel: copy.destination_mip_level,
						baseArrayLayer: copy.destination_base_array_layer
							+ destination.subresource_range().array_layers_offset,
						layerCount: copy.layer_count
					},
					dstOffset: vk::Offset3D {
						x: copy.destination_offset[0],
						y: copy.destination_offset[1],
						z: copy.destination_offset[2]
					},
					extent: vk::Extent3D {
						width: copy.extent[0],
						height: copy.extent[1],
						depth: copy.extent[2]
					}
				})
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdCopyImage(
			cmd,
			source.parent().inner().internal_object(),
			source_layout as u32,
			destination.parent().inner().internal_object(),
			destination_layout as u32,
			regions.len() as u32,
			regions.as_ptr()
		);
	}

	/// Calls `vkCmdBlitImage` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn blit_image<S, D, R>(
		&mut self, source: &S, source_layout: ImageLayoutImageSrc, destination: &D,
		destination_layout: ImageLayoutImageDst, regions: R, filter: Filter
	) where
		S: ?Sized + ImageViewAccess,
		D: ?Sized + ImageViewAccess,
		R: Iterator<Item = UnsafeCommandBufferBuilderImageBlit>
	{
		debug_assert!(filter == Filter::Nearest || !source.format().ty().is_depth_and_or_stencil());
		debug_assert!(
			(source.format().ty() == FormatTy::Uint)
				== (destination.format().ty() == FormatTy::Uint)
		);
		debug_assert!(
			(source.format().ty() == FormatTy::Sint)
				== (destination.format().ty() == FormatTy::Sint)
		);
		debug_assert!(
			source.format() == destination.format()
				|| !source.format().ty().is_depth_and_or_stencil()
		);

		debug_assert_eq!(source.parent().samples().get(), 1);
		debug_assert!(source.parent().supports_blit_source());
		debug_assert!(source.usage().transfer_source);

		debug_assert_eq!(destination.parent().samples().get(), 1);
		debug_assert!(destination.parent().supports_blit_destination());
		debug_assert!(destination.usage().transfer_destination);

		let regions: SmallVec<[_; 8]> = regions
			.filter_map(|blit| {
				// TODO: not everything is checked here
				debug_assert!(
					blit.source_base_array_layer + blit.layer_count
						<= source.subresource_range().array_layers.get()
				);
				debug_assert!(
					blit.destination_base_array_layer + blit.layer_count
						<= destination.subresource_range().array_layers.get()
				);
				debug_assert!(
					blit.source_mip_level < destination.subresource_range().mipmap_levels.get()
				);
				debug_assert!(
					blit.destination_mip_level
						< destination.subresource_range().mipmap_levels.get()
				);

				if blit.layer_count == 0 {
					return None
				}

				Some(vk::ImageBlit {
					srcSubresource: vk::ImageSubresourceLayers {
						aspectMask: blit.aspect.to_vk_bits(),
						mipLevel: blit.source_mip_level,
						baseArrayLayer: blit.source_base_array_layer
							+ source.subresource_range().array_layers_offset,
						layerCount: blit.layer_count
					},
					srcOffsets: [
						vk::Offset3D {
							x: blit.source_top_left[0],
							y: blit.source_top_left[1],
							z: blit.source_top_left[2]
						},
						vk::Offset3D {
							x: blit.source_bottom_right[0],
							y: blit.source_bottom_right[1],
							z: blit.source_bottom_right[2]
						}
					],
					dstSubresource: vk::ImageSubresourceLayers {
						aspectMask: blit.aspect.to_vk_bits(),
						mipLevel: blit.destination_mip_level,
						baseArrayLayer: blit.destination_base_array_layer
							+ destination.subresource_range().array_layers_offset,
						layerCount: blit.layer_count
					},
					dstOffsets: [
						vk::Offset3D {
							x: blit.destination_top_left[0],
							y: blit.destination_top_left[1],
							z: blit.destination_top_left[2]
						},
						vk::Offset3D {
							x: blit.destination_bottom_right[0],
							y: blit.destination_bottom_right[1],
							z: blit.destination_bottom_right[2]
						}
					]
				})
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdBlitImage(
			cmd,
			source.parent().inner().internal_object(),
			source_layout as u32,
			destination.parent().inner().internal_object(),
			destination_layout as u32,
			regions.len() as u32,
			regions.as_ptr(),
			filter as u32
		);
	}

	// TODO: missing structs
	// /// Calls `vkCmdClearAttachments` on the builder.
	//
	// Does nothing if the list of attachments or the list of rects is empty, as it would be a
	// no-op and isn't a valid usage of the command anyway.
	// pub unsafe fn clear_attachments<A, R>(&mut self, attachments: A, rects: R)
	// where A: Iterator<Item = >,
	// R: Iterator<Item = >
	// {
	// let attachments: SmallVec<[_; 16]> = attachments.map().collect();
	// let rects: SmallVec<[_; 4]> = rects.map().collect();
	//
	// if attachments.is_empty() || rects.is_empty() {
	// return;
	// }
	//
	// let vk = self.device().pointers();
	// let cmd = self.internal_object();
	// vk.CmdClearAttachments(cmd, attachments.len() as u32, attachments.as_ptr(),
	// rects.len() as u32, rects.as_ptr());
	// }

	/// Calls `vkCmdClearColorImage` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	// TODO: ClearValue could be more precise
	pub unsafe fn clear_color_image<I, R>(
		&mut self, image: &I, layout: ImageLayoutImageDst, color: ClearValue, regions: R
	) where
		I: ?Sized + ImageViewAccess,
		R: Iterator<Item = UnsafeCommandBufferBuilderColorImageClear>
	{
		debug_assert!(
			image.format().ty() == FormatTy::Float
				|| image.format().ty() == FormatTy::Uint
				|| image.format().ty() == FormatTy::Sint
		);

		debug_assert!(image.usage().transfer_destination);

		let color = match color {
			ClearValue::Float(val) => vk::ClearColorValue { float32: val },
			ClearValue::Int(val) => vk::ClearColorValue { int32: val },
			ClearValue::Uint(val) => vk::ClearColorValue { uint32: val },
			_ => vk::ClearColorValue { float32: [0.0; 4] }
		};

		let regions: SmallVec<[_; 8]> = regions
			.filter_map(|region| {
				debug_assert!(
					region.layer_count + region.base_array_layer
						<= image.subresource_range().array_layers.get()
				);
				debug_assert!(
					region.level_count + region.base_mip_level
						<= image.subresource_range().mipmap_levels.get()
				);

				if region.layer_count == 0 || region.level_count == 0 {
					return None
				}

				Some(vk::ImageSubresourceRange {
					aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
					baseMipLevel: region.base_mip_level
						+ image.subresource_range().mipmap_levels_offset,
					levelCount: region.level_count,
					baseArrayLayer: region.base_array_layer
						+ image.subresource_range().array_layers_offset,
					layerCount: region.layer_count
				})
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdClearColorImage(
			cmd,
			image.parent().inner().internal_object(),
			layout as u32,
			&color,
			regions.len() as u32,
			regions.as_ptr()
		);
	}

	/// Calls `vkCmdCopyBuffer` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn copy_buffer<S, D, R>(&mut self, source: &S, destination: &D, regions: R)
	where
		S: ?Sized + BufferAccess,
		D: ?Sized + BufferAccess,
		R: Iterator<Item = (usize, usize, usize)>
	{
		// TODO: debug assert that there's no overlap in the destinations?

		let source = source.inner();
		debug_assert!(source.offset < source.buffer.size());
		debug_assert!(source.buffer.usage_transfer_source());

		let destination = destination.inner();
		debug_assert!(destination.offset < destination.buffer.size());
		debug_assert!(destination.buffer.usage_transfer_destination());

		let regions: SmallVec<[_; 8]> = regions
			.map(|(sr, de, sz)| vk::BufferCopy {
				srcOffset: (sr + source.offset) as vk::DeviceSize,
				dstOffset: (de + destination.offset) as vk::DeviceSize,
				size: sz as vk::DeviceSize
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdCopyBuffer(
			cmd,
			source.buffer.internal_object(),
			destination.buffer.internal_object(),
			regions.len() as u32,
			regions.as_ptr()
		);
	}

	/// Calls `vkCmdCopyBufferToImage` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn copy_buffer_to_image<S, D, R>(
		&mut self, source: &S, destination: &D, destination_layout: ImageLayoutImageDst, regions: R
	) where
		S: ?Sized + BufferAccess,
		D: ?Sized + ImageViewAccess,
		R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
	{
		let source = source.inner();
		debug_assert!(source.offset < source.buffer.size());
		debug_assert!(source.buffer.usage_transfer_source());

		debug_assert_eq!(destination.parent().samples().get(), 1);
		debug_assert!(destination.usage().transfer_destination);

		let regions: SmallVec<[_; 8]> = regions
			.map(|copy| {
				debug_assert!(
					copy.image_layer_count <= destination.subresource_range().array_layers.get()
				);
				debug_assert!(
					copy.image_mip_level < destination.subresource_range().mipmap_levels.get()
				);

				vk::BufferImageCopy {
					bufferOffset: (source.offset + copy.buffer_offset) as vk::DeviceSize,
					bufferRowLength: copy.buffer_row_length,
					bufferImageHeight: copy.buffer_image_height,
					imageSubresource: vk::ImageSubresourceLayers {
						aspectMask: copy.image_aspect.to_vk_bits(),
						mipLevel: copy.image_mip_level
							+ destination.subresource_range().mipmap_levels_offset,
						baseArrayLayer: copy.image_base_array_layer
							+ destination.subresource_range().array_layers_offset,
						layerCount: copy.image_layer_count
					},
					imageOffset: vk::Offset3D {
						x: copy.image_offset[0],
						y: copy.image_offset[1],
						z: copy.image_offset[2]
					},
					imageExtent: vk::Extent3D {
						width: copy.image_extent[0],
						height: copy.image_extent[1],
						depth: copy.image_extent[2]
					}
				}
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdCopyBufferToImage(
			cmd,
			source.buffer.internal_object(),
			destination.parent().inner().internal_object(),
			destination_layout as u32,
			regions.len() as u32,
			regions.as_ptr()
		);
	}

	/// Calls `vkCmdCopyImageToBuffer` on the builder.
	///
	/// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
	/// usage of the command anyway.
	pub unsafe fn copy_image_to_buffer<S, D, R>(
		&mut self, source: &S, source_layout: ImageLayoutImageSrc, destination: &D, regions: R
	) where
		S: ?Sized + ImageViewAccess,
		D: ?Sized + BufferAccess,
		R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
	{
		debug_assert_eq!(source.parent().samples().get(), 1);
		debug_assert!(source.usage().transfer_source);

		let destination = destination.inner();
		debug_assert!(destination.offset < destination.buffer.size());
		debug_assert!(destination.buffer.usage_transfer_destination());

		let regions: SmallVec<[_; 8]> = regions
			.map(|copy| {
				debug_assert!(
					copy.image_layer_count <= source.subresource_range().array_layers.get()
				);
				debug_assert!(
					copy.image_mip_level < source.subresource_range().mipmap_levels.get()
				);

				vk::BufferImageCopy {
					bufferOffset: (destination.offset + copy.buffer_offset) as vk::DeviceSize,
					bufferRowLength: copy.buffer_row_length,
					bufferImageHeight: copy.buffer_image_height,
					imageSubresource: vk::ImageSubresourceLayers {
						aspectMask: copy.image_aspect.to_vk_bits(),
						mipLevel: copy.image_mip_level
							+ source.subresource_range().mipmap_levels_offset,
						baseArrayLayer: copy.image_base_array_layer
							+ source.subresource_range().array_layers_offset,
						layerCount: copy.image_layer_count
					},
					imageOffset: vk::Offset3D {
						x: copy.image_offset[0],
						y: copy.image_offset[1],
						z: copy.image_offset[2]
					},
					imageExtent: vk::Extent3D {
						width: copy.image_extent[0],
						height: copy.image_extent[1],
						depth: copy.image_extent[2]
					}
				}
			})
			.collect();

		if regions.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdCopyImageToBuffer(
			cmd,
			source.parent().inner().internal_object(),
			source_layout as u32,
			destination.buffer.internal_object(),
			regions.len() as u32,
			regions.as_ptr()
		);
	}

	/// Calls `vkCmdCopyQueryPoolResults` on the builder.
	pub unsafe fn copy_query_pool_results(
		&mut self, queries: UnsafeQueriesRange, destination: &BufferAccess, stride: usize
	) {
		let destination = destination.inner();
		debug_assert!(destination.offset < destination.buffer.size());
		debug_assert!(destination.buffer.usage_transfer_destination());

		let flags = 0; // FIXME:

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdCopyQueryPoolResults(
			cmd,
			queries.pool().internal_object(),
			queries.first_index(),
			queries.count(),
			destination.buffer.internal_object(),
			destination.offset as vk::DeviceSize,
			stride as vk::DeviceSize,
			flags
		);
	}

	/// Calls `vkCmdDispatch` on the builder.
	pub unsafe fn dispatch(&mut self, dimensions: [u32; 3]) {
		debug_assert!({
			let max_dims = self.device().physical_device().limits().max_compute_work_group_count();
			dimensions[0] <= max_dims[0]
				&& dimensions[1] <= max_dims[1]
				&& dimensions[2] <= max_dims[2]
		});

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdDispatch(cmd, dimensions[0], dimensions[1], dimensions[2]);
	}

	/// Calls `vkCmdDispatchIndirect` on the builder.
	pub unsafe fn dispatch_indirect<B>(&mut self, buffer: &B)
	where
		B: ?Sized + BufferAccess
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let inner = buffer.inner();
		debug_assert!(inner.offset < buffer.size());
		debug_assert!(inner.buffer.usage_indirect_buffer());
		debug_assert_eq!(inner.offset % 4, 0);

		vk.CmdDispatchIndirect(cmd, inner.buffer.internal_object(), inner.offset as vk::DeviceSize);
	}

	/// Calls `vkCmdDraw` on the builder.
	pub unsafe fn draw(
		&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32
	) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdDraw(cmd, vertex_count, instance_count, first_vertex, first_instance);
	}

	/// Calls `vkCmdDrawIndexed` on the builder.
	pub unsafe fn draw_indexed(
		&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32,
		first_instance: u32
	) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdDrawIndexed(
			cmd,
			index_count,
			instance_count,
			first_index,
			vertex_offset,
			first_instance
		);
	}

	/// Calls `vkCmdDrawIndirect` on the builder.
	pub unsafe fn draw_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
	where
		B: ?Sized + BufferAccess
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		debug_assert!(
			draw_count == 0
				|| ((stride % 4) == 0)
					&& stride as usize >= mem::size_of::<vk::DrawIndirectCommand>()
		);

		let inner = buffer.inner();
		debug_assert!(inner.offset < inner.buffer.size());
		debug_assert!(inner.buffer.usage_indirect_buffer());

		vk.CmdDrawIndirect(
			cmd,
			inner.buffer.internal_object(),
			inner.offset as vk::DeviceSize,
			draw_count,
			stride
		);
	}

	/// Calls `vkCmdDrawIndexedIndirect` on the builder.
	pub unsafe fn draw_indexed_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
	where
		B: ?Sized + BufferAccess
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let inner = buffer.inner();
		debug_assert!(inner.offset < inner.buffer.size());
		debug_assert!(inner.buffer.usage_indirect_buffer());

		vk.CmdDrawIndexedIndirect(
			cmd,
			inner.buffer.internal_object(),
			inner.offset as vk::DeviceSize,
			draw_count,
			stride
		);
	}

	/// Calls `vkCmdEndQuery` on the builder.
	pub unsafe fn end_query(&mut self, query: UnsafeQuery) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdEndQuery(cmd, query.pool().internal_object(), query.index());
	}

	/// Calls `vkCmdEndRenderPass` on the builder.
	pub unsafe fn end_render_pass(&mut self) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdEndRenderPass(cmd);
	}

	/// Calls `vkCmdExecuteCommands` on the builder.
	///
	/// Does nothing if the list of command buffers is empty, as it would be a no-op and isn't a
	/// valid usage of the command anyway.
	pub unsafe fn execute_commands(&mut self, cbs: UnsafeCommandBufferBuilderExecuteCommands) {
		if cbs.raw_cbs.is_empty() {
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdExecuteCommands(cmd, cbs.raw_cbs.len() as u32, cbs.raw_cbs.as_ptr());
	}

	/// Calls `vkCmdFillBuffer` on the builder.
	pub unsafe fn fill_buffer<B>(&mut self, buffer: &B, data: u32)
	where
		B: ?Sized + BufferAccess
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let size = buffer.size();

		let (buffer_handle, offset) = {
			let BufferInner { buffer: buffer_inner, offset } = buffer.inner();
			debug_assert!(buffer_inner.usage_transfer_destination());
			debug_assert_eq!(offset % 4, 0);
			(buffer_inner.internal_object(), offset)
		};

		vk.CmdFillBuffer(
			cmd,
			buffer_handle,
			offset as vk::DeviceSize,
			size as vk::DeviceSize,
			data
		);
	}

	/// Calls `vkCmdNextSubpass` on the builder.
	pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdNextSubpass(cmd, subpass_contents as u32);
	}

	pub unsafe fn pipeline_barrier(&mut self, command: &UnsafeCommandBufferBuilderPipelineBarrier) {
		// If barrier is empty, don't do anything.
		if command.src_stage_mask == 0 || command.dst_stage_mask == 0 {
			debug_assert!(command.src_stage_mask == 0 && command.dst_stage_mask == 0);
			debug_assert!(command.memory_barriers.is_empty());
			debug_assert!(command.buffer_barriers.is_empty());
			debug_assert!(command.image_barriers.is_empty());
			return
		}

		let vk = self.device().pointers();
		let cmd = self.internal_object();

		debug_assert_ne!(command.src_stage_mask, 0);
		debug_assert_ne!(command.dst_stage_mask, 0);

		vk.CmdPipelineBarrier(
			cmd,
			command.src_stage_mask,
			command.dst_stage_mask,
			command.dependency_flags,
			command.memory_barriers.len() as u32,
			command.memory_barriers.as_ptr(),
			command.buffer_barriers.len() as u32,
			command.buffer_barriers.as_ptr(),
			command.image_barriers.len() as u32,
			command.image_barriers.as_ptr()
		);
	}

	/// Calls `vkCmdPushConstants` on the builder.
	pub unsafe fn push_constants<Pl, D>(
		&mut self, pipeline_layout: &Pl, stages: ShaderStages, offset: u32, size: u32, data: &D
	) where
		Pl: ?Sized + PipelineLayoutAbstract,
		D: ?Sized
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		debug_assert!(stages != ShaderStages::none());
		debug_assert!(size > 0);
		debug_assert_eq!(size % 4, 0);
		debug_assert_eq!(offset % 4, 0);
		debug_assert!(mem::size_of_val(data) >= size as usize);

		vk.CmdPushConstants(
			cmd,
			pipeline_layout.sys().internal_object(),
			stages.into_vulkan_bits(),
			offset as u32,
			size as u32,
			data as *const D as *const _
		);
	}

	/// Calls `vkCmdResetEvent` on the builder.
	pub unsafe fn reset_event(&mut self, event: &Event, stages: PipelineStages) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		debug_assert!(!stages.host);
		debug_assert_ne!(stages, PipelineStages::none());

		vk.CmdResetEvent(cmd, event.internal_object(), stages.into_vulkan_bits());
	}

	/// Calls `vkCmdResetQueryPool` on the builder.
	pub unsafe fn reset_query_pool(&mut self, queries: UnsafeQueriesRange) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdResetQueryPool(
			cmd,
			queries.pool().internal_object(),
			queries.first_index(),
			queries.count()
		);
	}

	/// Calls `vkCmdSetBlendConstants` on the builder.
	pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdSetBlendConstants(cmd, constants); // TODO: correct to pass array?
	}

	/// Calls `vkCmdSetDepthBias` on the builder.
	pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		debug_assert!(clamp == 0.0 || self.device().enabled_features().depth_bias_clamp);
		vk.CmdSetDepthBias(cmd, constant_factor, clamp, slope_factor);
	}

	/// Calls `vkCmdSetDepthBounds` on the builder.
	pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		debug_assert!(min >= 0.0 && min <= 1.0);
		debug_assert!(max >= 0.0 && max <= 1.0);
		vk.CmdSetDepthBounds(cmd, min, max);
	}

	/// Calls `vkCmdSetEvent` on the builder.
	pub unsafe fn set_event(&mut self, event: &Event, stages: PipelineStages) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		debug_assert!(!stages.host);
		debug_assert_ne!(stages, PipelineStages::none());

		vk.CmdSetEvent(cmd, event.internal_object(), stages.into_vulkan_bits());
	}

	/// Calls `vkCmdSetLineWidth` on the builder.
	pub unsafe fn set_line_width(&mut self, line_width: f32) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		debug_assert!(line_width == 1.0 || self.device().enabled_features().wide_lines);
		vk.CmdSetLineWidth(cmd, line_width);
	}

	// TODO: missing enum for face_mask
	// /// Calls `vkCmdSetStencilCompareMask` on the builder.
	// pub unsafe fn set_stencil_compare_mask(&mut self, face_mask: , compare_mask: u32) {
	// let vk = self.device().pointers();
	// let cmd = self.internal_object();
	// vk.CmdSetStencilCompareMask(cmd, face_mask, compare_mask);
	// }

	// TODO: missing enum for face_mask
	// /// Calls `vkCmdSetStencilReference` on the builder.
	// pub unsafe fn set_stencil_reference(&mut self, face_mask: , reference: u32) {
	// let vk = self.device().pointers();
	// let cmd = self.internal_object();
	// vk.CmdSetStencilReference(cmd, face_mask, reference);
	// }

	// TODO: missing enum for face_mask
	// /// Calls `vkCmdSetStencilWriteMask` on the builder.
	// pub unsafe fn set_stencil_write_mask(&mut self, face_mask: , write_mask: u32) {
	// let vk = self.device().pointers();
	// let cmd = self.internal_object();
	// vk.CmdSetStencilWriteMask(cmd, face_mask, write_mask);
	// }

	/// Calls `vkCmdSetScissor` on the builder.
	///
	/// If the list is empty then the command is automatically ignored.
	pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
	where
		I: Iterator<Item = Scissor>
	{
		let scissors =
			scissors.map(|v| v.clone().into_vulkan_rect()).collect::<SmallVec<[_; 16]>>();
		if scissors.is_empty() {
			return
		}

		debug_assert!(scissors.iter().all(|s| s.offset.x >= 0 && s.offset.y >= 0));
		debug_assert!(scissors.iter().all(|s| {
			s.extent.width < i32::max_value() as u32
				&& s.extent.height < i32::max_value() as u32
				&& s.offset.x.checked_add(s.extent.width as i32).is_some()
				&& s.offset.y.checked_add(s.extent.height as i32).is_some()
		}));
		debug_assert!(
			(first_scissor == 0 && scissors.len() == 1)
				|| self.device().enabled_features().multi_viewport
		);
		debug_assert!({
			let max = self.device().physical_device().limits().max_viewports();
			first_scissor + scissors.len() as u32 <= max
		});

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdSetScissor(cmd, first_scissor, scissors.len() as u32, scissors.as_ptr());
	}

	/// Calls `vkCmdSetViewport` on the builder.
	///
	/// If the list is empty then the command is automatically ignored.
	pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
	where
		I: Iterator<Item = Viewport>
	{
		let viewports =
			viewports.map(|v| v.clone().into_vulkan_viewport()).collect::<SmallVec<[_; 16]>>();
		if viewports.is_empty() {
			return
		}

		debug_assert!(
			(first_viewport == 0 && viewports.len() == 1)
				|| self.device().enabled_features().multi_viewport
		);
		debug_assert!({
			let max = self.device().physical_device().limits().max_viewports();
			first_viewport + viewports.len() as u32 <= max
		});

		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdSetViewport(cmd, first_viewport, viewports.len() as u32, viewports.as_ptr());
	}

	/// Calls `vkCmdUpdateBuffer` on the builder.
	pub unsafe fn update_buffer<B, D>(&mut self, buffer: &B, data: &D)
	where
		B: ?Sized + BufferAccess,
		D: ?Sized
	{
		let vk = self.device().pointers();
		let cmd = self.internal_object();

		let size = buffer.size();
		debug_assert_eq!(size % 4, 0);
		debug_assert!(size <= 65536);
		debug_assert!(size <= mem::size_of_val(data));

		let (buffer_handle, offset) = {
			let BufferInner { buffer: buffer_inner, offset } = buffer.inner();
			debug_assert!(buffer_inner.usage_transfer_destination());
			debug_assert_eq!(offset % 4, 0);
			(buffer_inner.internal_object(), offset)
		};

		vk.CmdUpdateBuffer(
			cmd,
			buffer_handle,
			offset as vk::DeviceSize,
			size as vk::DeviceSize,
			data as *const D as *const _
		);
	}

	/// Calls `vkCmdWriteTimestamp` on the builder.
	pub unsafe fn write_timestamp(&mut self, query: UnsafeQuery, stages: PipelineStages) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdWriteTimestamp(
			cmd,
			stages.into_vulkan_bits(),
			query.pool().internal_object(),
			query.index()
		);
	}

	/// Calls `vkCmdDebugMarkerBeginEXT` on the builder.
	///
	/// # Panics
	/// Requires the `VK_EXT_debug_marker` device extension to be loaded.
	///
	/// # Safety
	/// The command pool that this command buffer was allocated from must support graphics or
	/// compute operations
	pub unsafe fn debug_marker_begin(&mut self, name: &CStr, color: [f32; 4]) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		let info = vk::DebugMarkerMarkerInfoEXT {
			sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT,
			pNext: ptr::null(),
			pMarkerName: name.as_ptr(),
			color
		};
		vk.CmdDebugMarkerBeginEXT(cmd, &info);
	}

	/// Calls `vkCmdDebugMarkerEndEXT` on the builder.
	///
	/// # Panics
	/// Requires the `VK_EXT_debug_marker` device extension to be loaded.
	///
	/// # Safety
	/// There must be an outstanding `vkCmdDebugMarkerBeginEXT` command prior to the
	/// `vkCmdDebugMarkerEndEXT` on the queue that this command buffer is submitted to. If the
	/// matching `vkCmdDebugMarkerBeginEXT` command was in a secondary command buffer, the
	/// `vkCmdDebugMarkerEndEXT` must be in the same command buffer.
	pub unsafe fn debug_marker_end(&mut self) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		vk.CmdDebugMarkerEndEXT(cmd);
	}

	/// Calls `vkCmdDebugMarkerInsertEXT` on the builder.
	///
	/// # Panics
	/// Requires the `VK_EXT_debug_marker` device extension to be loaded.
	///
	/// # Safety
	/// The command pool that this command buffer was allocated from must support graphics or
	/// compute operations
	pub unsafe fn debug_marker_insert(&mut self, name: &CStr, color: [f32; 4]) {
		let vk = self.device().pointers();
		let cmd = self.internal_object();
		let info = vk::DebugMarkerMarkerInfoEXT {
			sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT,
			pNext: ptr::null(),
			pMarkerName: name.as_ptr(),
			color
		};
		vk.CmdDebugMarkerInsertEXT(cmd, &info);
	}
}
unsafe impl<P> DeviceOwned for UnsafeCommandBufferBuilder<P> {
	fn device(&self) -> &Arc<Device> { &self.device }
}
unsafe impl<P> VulkanObject for UnsafeCommandBufferBuilder<P> {
	type Object = vk::CommandBuffer;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT;

	fn internal_object(&self) -> vk::CommandBuffer {
		debug_assert!(self.cmd.is_some());
		self.cmd_raw
	}
}


/// Command buffer that has been built.
///
/// Doesn't perform any synchronization and doesn't keep the object it uses alive.
pub struct UnsafeCommandBuffer<P> {
	// The Vulkan command buffer.
	cmd: P,

	// The raw version of `cmd`. Avoids having to require a trait on `P`.
	cmd_raw: vk::CommandBuffer,

	// Device that owns the command buffer.
	// TODO: necessary?
	device: Arc<Device>
}
unsafe impl<P> DeviceOwned for UnsafeCommandBuffer<P> {
	fn device(&self) -> &Arc<Device> { &self.device }
}
unsafe impl<P> VulkanObject for UnsafeCommandBuffer<P> {
	type Object = vk::CommandBuffer;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT;

	fn internal_object(&self) -> vk::CommandBuffer { self.cmd_raw }
}
