// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{num::NonZeroU32, sync::Arc};

use vulkano::{
	buffer::{BufferUsage, CpuAccessibleBuffer},
	command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
	device::{Device, DeviceExtensions},
	format::Format,
	image::{
		ImageDimensions,
		ImageUsage,
		ImageView,
		MipmapsCount,
		RequiredLayouts,
		Swizzle,
		SyncImage
	},
	instance::{
		self,
		debug::{DebugCallback, MessageTypes},
		Instance,
		InstanceExtensions,
		PhysicalDevice
	},
	sync::GpuFuture
};

fn main() {
	// Vulkano Debugging Example Code
	//
	// This example code will demonstrate using the debug functions of the Vulkano API.
	//
	// There is documentation here about how to setup debugging:
	// https://vulkan.lunarg.com/doc/view/1.0.13.0/windows/layers.html
	//
	// .. but if you just want a template of code that has everything ready to go then follow
	// this example. First, enable debugging using this extension: VK_EXT_debug_report
	let extensions = InstanceExtensions { ext_debug_report: true, ..InstanceExtensions::none() };

	// You also need to specify (unless you've used the methods linked above) which debugging layers
	// your code should use. Each layer is a bunch of checks or messages that provide information of
	// some sort.
	//
	// The main layer you might want is: VK_LAYER_LUNARG_standard_validation
	// This includes a number of the other layers for you and is quite detailed.
	//
	// Additional layers can be installed (gpu vendor provided, something you found on GitHub, etc)
	// and you should verify that list for safety - Vulkano will return an error if you specify
	// any layers that are not installed on this system. That code to do could look like this:
	println!("List of Vulkan debugging layers available to use:");
	let mut layers = instance::layers_list().unwrap();
	while let Some(l) = layers.next() {
		println!("\t{}", l.name());
	}

	// NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
	let layer = "VK_LAYER_LUNARG_standard_validation";
	let layers = vec![layer];

	// Important: pass the extension(s) and layer(s) when creating the vulkano instance
	let instance =
		Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// After creating the instance we must register the debugging callback.                                      //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Note: If you let this debug_callback binding fall out of scope then the callback will stop providing events
	// Note: There is a helper method too: DebugCallback::errors_and_warnings(&instance, |msg| {...

	let all = MessageTypes {
		error: true,
		warning: true,
		performance_warning: true,
		information: true,
		debug: true
	};

	let _debug_callback = DebugCallback::new(&instance, all, |msg| {
		let ty = if msg.ty.error {
			"error"
		} else if msg.ty.warning {
			"warning"
		} else if msg.ty.performance_warning {
			"performance_warning"
		} else if msg.ty.information {
			"information"
		} else if msg.ty.debug {
			"debug"
		} else {
			panic!("no-impl");
		};
		println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
	})
	.ok();

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Create Vulkan objects in the same way as the other examples                                               //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
	let queue_family = physical.queue_families().next().expect("couldn't find a queue family");
	let (device, mut queues) = Device::new(
		physical,
		physical.supported_features(),
		&DeviceExtensions::none(),
		vec![(queue_family, 0.5)]
	)
	.expect("failed to create device");
	let queue = queues.next().unwrap();

	// Create an image and upload some data in order to generate some additional logging:
	let pixel_format = Format::R8G8B8A8Uint;
	let dimensions = ImageDimensions::Dim2D {
		width: NonZeroU32::new(4096).unwrap(),
		height: NonZeroU32::new(4096).unwrap()
	};
	const DATA: [[u8; 4]; 4096 * 4096] = [[0; 4]; 4096 * 4096];

	let image: Arc<SyncImage> = Arc::new(
		SyncImage::new(
			device.clone(),
			ImageUsage { transfer_destination: true, ..ImageUsage::default() },
			pixel_format,
			dimensions,
			NonZeroU32::new(1).unwrap(),
			MipmapsCount::One
		)
		.unwrap()
	);
	let view = Arc::new(ImageView::whole_image(image.clone()).unwrap());

	let staging_buffer = CpuAccessibleBuffer::from_iter(
		device.clone(),
		BufferUsage::transfer_source(),
		DATA.iter().cloned()
	)
	.unwrap();

	let cb = AutoCommandBufferBuilder::new(device.clone(), queue.family())
		.unwrap()
		.copy_buffer_to_image_dimensions(
			staging_buffer,
			view,
			[0, 0, 0],
			[dimensions.width().get(), dimensions.height().get(), dimensions.depth().get()],
			0,
			dimensions.array_layers_with_cube().get(),
			0
		)
		.unwrap()
		.build()
		.unwrap();

	// We wait because why not.
	let future = cb.execute(queue).unwrap().then_signal_fence_and_flush().unwrap();
	future.wait(None).unwrap();

	// (At this point you should see a bunch of messages printed to the terminal window - have fun debugging!)
	eprintln!("Done");
}
