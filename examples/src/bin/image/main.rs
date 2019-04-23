// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{num::NonZeroU32, sync::Arc};

use vulkano_win::VkSurfaceBuild;
use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use vulkano::{
	buffer::{BufferUsage, CpuAccessibleBuffer},
	command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState},
	descriptor::descriptor_set::PersistentDescriptorSet,
	device::{Device, DeviceExtensions},
	format::Format,
	framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
	image::{
		ImageDimensions,
		ImageLayoutCombinedImage,
		ImageLayoutSampledImage,
		ImageSubresourceRange,
		ImageUsage,
		ImageView,
		MipmapsCount,
		RequiredLayouts,
		SwapchainImage,
		Swizzle,
		SyncImage
	},
	instance::{Instance, PhysicalDevice},
	pipeline::{viewport::Viewport, GraphicsPipeline},
	sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
	swapchain::{
		self,
		AcquireError,
		PresentMode,
		SurfaceTransform,
		Swapchain,
		SwapchainCreationError
	},
	sync::{self, FlushError, GpuFuture}
};

fn main() {
	// The start of this example is exactly the same as `triangle`. You should read the
	// `triangle` example if you haven't done so yet.

	let extensions = vulkano_win::required_extensions();
	let instance = Instance::new(None, &extensions, None).unwrap();

	let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
	println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

	let mut events_loop = EventsLoop::new();
	let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
	let window = surface.window();

	let queue_family = physical
		.queue_families()
		.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
		.unwrap();

	let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
	let (device, mut queues) = Device::new(
		physical,
		physical.supported_features(),
		&device_ext,
		[(queue_family, 0.5)].iter().cloned()
	)
	.unwrap();
	let queue = queues.next().unwrap();

	let (mut swapchain, images) = {
		let caps = surface.capabilities(physical).unwrap();

		let usage = caps.supported_usage_flags;
		let alpha = caps.supported_composite_alpha.iter().next().unwrap();
		let (format, color_space) = caps.supported_formats[0];

		let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
			// convert to physical pixels
			let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
			[NonZeroU32::new(dimensions.0).unwrap(), NonZeroU32::new(dimensions.1).unwrap()]
		} else {
			// The window no longer exists so exit the application.
			return
		};

		Swapchain::new(
			device.clone(),
			surface.clone(),
			&queue,
			initial_dimensions,
			NonZeroU32::new(1).unwrap(),
			NonZeroU32::new(caps.min_image_count).unwrap(),
			format,
			color_space,
			usage,
			SurfaceTransform::Identity,
			alpha,
			PresentMode::Fifo,
			true,
			None
		)
		.unwrap()
	};


	#[derive(Debug, Clone)]
	struct Vertex {
		position: [f32; 2]
	}
	vulkano::impl_vertex!(Vertex, position);

	let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
		device.clone(),
		BufferUsage::all(),
		[
			Vertex { position: [-0.5, -0.5] },
			Vertex { position: [-0.5, 0.5] },
			Vertex { position: [0.5, -0.5] },
			Vertex { position: [0.5, 0.5] }
		]
		.iter()
		.cloned()
	)
	.unwrap();

	let vs = vs::Shader::load(device.clone()).unwrap();
	let fs = fs::Shader::load(device.clone()).unwrap();

	let render_pass = Arc::new(
		vulkano::single_pass_renderpass!(device.clone(),
			attachments: {
				color: {
					load: Clear,
					store: Store,
					format: swapchain.format(),
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

	let (texture, tex_future) = {
		let texture_dimensions = ImageDimensions::Dim2D {
			width: NonZeroU32::new(93).unwrap(),
			height: NonZeroU32::new(93).unwrap()
		};

		let image: Arc<SyncImage> = Arc::new(
			SyncImage::new(
				device.clone(),
				ImageUsage { transfer_destination: true, sampled: true, ..ImageUsage::default() },
				Format::R8G8B8A8Srgb,
				texture_dimensions,
				NonZeroU32::new(1).unwrap(),
				MipmapsCount::One
			)
			.unwrap()
		);
		let view = Arc::new(
			ImageView::new(
				image.clone(),
				None,
				None::<Format>,
				Swizzle::default(),
				ImageSubresourceRange::whole_image(&image),
				RequiredLayouts {
					sampled: Some(ImageLayoutSampledImage::ShaderReadOnlyOptimal),
					combined: Some(ImageLayoutCombinedImage::ShaderReadOnlyOptimal),
					..Default::default()
				}
			)
			.unwrap()
		);

		let image_bytes = image::load_from_memory_with_format(
			include_bytes!("image_img.png"),
			image::ImageFormat::PNG
		)
		.unwrap()
		.to_rgba();
		let image_data = image_bytes.into_raw().clone();
		let staging_buffer = CpuAccessibleBuffer::from_iter(
			device.clone(),
			BufferUsage::transfer_source(),
			image_data.iter().cloned()
		)
		.unwrap();

		let cb = AutoCommandBufferBuilder::new(device.clone(), queue.family())
			.unwrap()
			.copy_buffer_to_image_dimensions(
				staging_buffer,
				view.clone(),
				[0, 0, 0],
				[
					texture_dimensions.width().get(),
					texture_dimensions.height().get(),
					texture_dimensions.depth().get()
				],
				0,
				texture_dimensions.array_layers_with_cube().get(),
				0
			)
			.unwrap()
			.build()
			.unwrap();

		// We wait because why not.
		let future = cb.execute(queue.clone()).unwrap().then_signal_fence_and_flush().unwrap();

		(view, future)
	};

	let sampler = Sampler::new(
		device.clone(),
		Filter::Linear,
		Filter::Linear,
		MipmapMode::Nearest,
		SamplerAddressMode::Repeat,
		SamplerAddressMode::Repeat,
		SamplerAddressMode::Repeat,
		0.0,
		1.0,
		0.0,
		0.0
	)
	.unwrap();

	let pipeline = Arc::new(
		GraphicsPipeline::start()
			.vertex_input_single_buffer::<Vertex>()
			.vertex_shader(vs.main_entry_point(), ())
			.triangle_strip()
			.viewports_dynamic_scissors_irrelevant(1)
			.fragment_shader(fs.main_entry_point(), ())
			.blend_alpha_blending()
			.render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
			.build(device.clone())
			.unwrap()
	);

	let set = Arc::new(
		PersistentDescriptorSet::start(pipeline.clone(), 0)
			.add_sampled_image(texture.clone(), sampler.clone())
			.unwrap()
			.build()
			.unwrap()
	);

	let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };
	let mut framebuffers =
		window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

	let mut recreate_swapchain = false;
	let mut previous_frame_end = Box::new(tex_future) as Box<GpuFuture>;

	loop {
		previous_frame_end.cleanup_finished();
		if recreate_swapchain {
			let dimensions = if let Some(dimensions) = window.get_inner_size() {
				let dimensions: (u32, u32) =
					dimensions.to_physical(window.get_hidpi_factor()).into();
				[NonZeroU32::new(dimensions.0).unwrap(), NonZeroU32::new(dimensions.1).unwrap()]
			} else {
				return
			};

			let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
				Ok(r) => r,
				Err(SwapchainCreationError::UnsupportedDimensions) => continue,
				Err(err) => panic!("{:?}", err)
			};

			swapchain = new_swapchain;
			framebuffers =
				window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);

			recreate_swapchain = false;
		}

		let (image_num, future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
			Ok(r) => r,
			Err(AcquireError::OutOfDate) => {
				recreate_swapchain = true;
				continue
			}
			Err(err) => panic!("{:?}", err)
		};

		let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
		let cb = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
			.unwrap()
			.begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
			.unwrap()
			.draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), set.clone(), ())
			.unwrap()
			.end_render_pass()
			.unwrap()
			.build()
			.unwrap();

		let future = previous_frame_end
			.join(future)
			.then_execute(queue.clone(), cb)
			.unwrap()
			.then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
			.then_signal_fence_and_flush();

		match future {
			Ok(future) => {
				previous_frame_end = Box::new(future) as Box<_>;
			}
			Err(FlushError::OutOfDate) => {
				recreate_swapchain = true;
				previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
			}
			Err(e) => {
				println!("{:?}", e);
				previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
			}
		}

		let mut done = false;
		events_loop.poll_events(|ev| match ev {
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
			Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
			_ => ()
		});
		if done {
			return
		}
	}
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
	images: &[Arc<SwapchainImage<Window>>], render_pass: Arc<RenderPassAbstract + Send + Sync>,
	dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
	let dimensions = images[0].dimensions();

	let viewport = Viewport {
		origin: [0.0, 0.0],
		dimensions: [dimensions[0].get() as f32, dimensions[1].get() as f32],
		depth_range: 0.0 .. 1.0
	};
	dynamic_state.viewports = Some(vec![viewport]);

	images
		.iter()
		.map(|image| {
			Arc::new(
				Framebuffer::start(render_pass.clone())
					.add(image.clone())
					.unwrap()
					.build()
					.unwrap()
			) as Arc<FramebufferAbstract + Send + Sync>
		})
		.collect::<Vec<_>>()
}

mod vs {
	vulkano_shaders::shader! {
		ty: "vertex",
		src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}"
	}
}

mod fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		src: "
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = texture(tex, tex_coords);
}"
	}
}
