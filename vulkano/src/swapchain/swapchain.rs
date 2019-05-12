// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{
	error,
	fmt,
	mem,
	num::NonZeroU32,
	ptr,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
		Mutex
	},
	time::Duration
};

use vk_sys as vk;

use crate::{
	buffer::BufferAccess,
	check_errors,
	command_buffer::submit::{SubmitAnyBuilder, SubmitPresentBuilder, SubmitSemaphoresWaitBuilder},
	device::{Device, DeviceOwned, Queue},
	format::{Format, FormatDesc},
	image::{
		sys::UnsafeImage,
		ImageDimensions,
		ImageLayout,
		ImageUsage,
		ImageViewAccess,
		SwapchainImage
	},
	swapchain::{
		CapabilitiesError,
		ColorSpace,
		CompositeAlpha,
		PresentMode,
		PresentRegion,
		Surface,
		SurfaceSwapchainLock,
		SurfaceTransform
	},
	sync::{
		AccessCheckError,
		AccessError,
		AccessFlagBits,
		Fence,
		FlushError,
		GpuFuture,
		PipelineStages,
		Semaphore,
		SharingMode
	},
	Error,
	OomError,
	Success,
	VulkanObject
};


/// Tries to take ownership of an image in order to draw on it.
///
/// The function returns the index of the image in the array of images that was returned
/// when creating the swapchain, plus a future that represents the moment when the image will
/// become available from the GPU (which may not be *immediately*).
///
/// If you try to draw on an image without acquiring it first, the execution will block return
/// an error.
pub fn acquire_next_image<W>(
	swapchain: Arc<Swapchain<W>>, timeout: Option<Duration>
) -> Result<(usize, SwapchainAcquireFuture<W>), AcquireError> {
	let semaphore = Semaphore::from_pool(swapchain.device.clone())?;
	let fence = Fence::from_pool(swapchain.device.clone())?;

	// TODO: propagate `suboptimal` to the user
	let AcquiredImage { id, suboptimal } = {
		// Check that this is not an old swapchain. From specs:
		// > swapchain must not have been replaced by being passed as the
		// > VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
		let stale = swapchain.stale.lock().unwrap();
		if *stale {
			return Err(AcquireError::OutOfDate)
		}

		unsafe { acquire_next_image_raw(&swapchain, timeout, Some(&semaphore), Some(&fence)) }?
	};

	Ok((
		id,
		SwapchainAcquireFuture {
			swapchain,
			semaphore: Some(semaphore),
			fence: Some(fence),
			image_id: id,
			finished: AtomicBool::new(false)
		}
	))
}

/// Presents an image on the screen.
///
/// The parameter is the same index as what `acquire_next_image` returned. The image must
/// have been acquired first.
///
/// The actual behavior depends on the present mode that you passed when creating the
/// swapchain.
pub fn present<F, W>(
	swapchain: Arc<Swapchain<W>>, before: F, queue: Arc<Queue>, index: usize
) -> PresentFuture<F, W>
where
	F: GpuFuture
{
	assert!(index < swapchain.images.len());

	// TODO: restore this check with a dummy ImageAccess implementation
	// let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
	// Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
	// function on the image instead. But since we know that this method on `SwapchainImage`
	// always returns false anyway (by design), we don't need to do it.
	// assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead

	PresentFuture {
		previous: before,
		queue,
		swapchain,
		image_id: index,
		present_region: None,
		flushed: AtomicBool::new(false),
		finished: AtomicBool::new(false)
	}
}

/// Same as `swapchain::present`, except it allows specifying a present region.
/// Areas outside the present region may be ignored by Vulkan in order to optimize presentation.
///
/// This is just an optimization hint, as the Vulkan driver is free to ignore the given present region.
///
/// If `VK_KHR_incremental_present` is not enabled on the device, the parameter will be ignored.
pub fn present_incremental<F, W>(
	swapchain: Arc<Swapchain<W>>, before: F, queue: Arc<Queue>, index: usize,
	present_region: PresentRegion
) -> PresentFuture<F, W>
where
	F: GpuFuture
{
	assert!(index < swapchain.images.len());

	// TODO: restore this check with a dummy ImageAccess implementation
	// let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
	// Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
	// function on the image instead. But since we know that this method on `SwapchainImage`
	// always returns false anyway (by design), we don't need to do it.
	// assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead

	PresentFuture {
		previous: before,
		queue,
		swapchain,
		image_id: index,
		present_region: Some(present_region),
		flushed: AtomicBool::new(false),
		finished: AtomicBool::new(false)
	}
}

#[derive(Debug)]
struct ImageEntry {
	// The actual image.
	image: UnsafeImage,
	// If true, then the image is still in the undefined layout and must be transitioned.
	undefined_layout: AtomicBool
}

/// Contains the swapping system and the images that can be shown on a surface.
pub struct Swapchain<W> {
	// The Vulkan device this swapchain was created with.
	device: Arc<Device>,
	// The surface, which we need to keep alive.
	surface: Arc<Surface<W>>,
	// The swapchain object.
	swapchain: vk::SwapchainKHR,

	// The images of this swapchain.
	images: Vec<ImageEntry>,

	// If true, that means we have tried to use this swapchain to recreate a new swapchain. The current
	// swapchain can no longer be used for anything except presenting already-acquired images.
	//
	// We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
	// we acquire the image.
	stale: Mutex<bool>,

	// Parameters passed to the constructor.
	// Some of these parameters are redundant because they are also stored by images themselves.
	// TODO: Not sure if to do anything about it?
	num_images: NonZeroU32,
	format: Format,
	color_space: ColorSpace,
	dimensions: [NonZeroU32; 2],
	layers: NonZeroU32,
	usage: ImageUsage,
	sharing: SharingMode,
	transform: SurfaceTransform,
	alpha: CompositeAlpha,
	mode: PresentMode,
	clipped: bool
}
impl<W> Swapchain<W> {
	/// Builds a new swapchain. Allocates images who content can be made visible on a surface.
	///
	/// See also the `Surface::get_capabilities` function which returns the values that are
	/// supported by the implementation. All the parameters that you pass to `Swapchain::new`
	/// must be supported.
	///
	/// The `clipped` parameter indicates whether the implementation is allowed to discard
	/// rendering operations that affect regions of the surface which aren't visible. This is
	/// important to take into account if your fragment shader has side-effects or if you want to
	/// read back the content of the image afterwards.
	///
	/// This function returns the swapchain plus a list of the images that belong to the
	/// swapchain. The order in which the images are returned is important for the
	/// `acquire_next_image` and `present` functions.
	///
	/// # Panic
	///
	/// - Panics if the device and the surface don't belong to the same instance.
	/// - Panics if `usage` is empty.
	// TODO: isn't it unsafe to take the surface through an Arc when it comes to vulkano-win?
	pub fn new<F: FormatDesc, S: Into<SharingMode>>(
		device: Arc<Device>, surface: Arc<Surface<W>>, sharing: S, dimensions: [NonZeroU32; 2],
		layers: NonZeroU32, num_images: NonZeroU32, format: F, color_space: ColorSpace,
		usage: ImageUsage, transform: SurfaceTransform, alpha: CompositeAlpha, mode: PresentMode,
		clipped: bool, old_swapchain: Option<&Swapchain<W>>
	) -> Result<(Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>), SwapchainCreationError> {
		assert_eq!(device.instance().internal_object(), surface.instance().internal_object());

		// Checking that the requested parameters match the capabilities.
		let capabilities = surface.capabilities(device.physical_device())?;
		let format = format.format();

		if num_images.get() < capabilities.min_image_count {
			return Err(SwapchainCreationError::UnsupportedMinImagesCount)
		}
		if let Some(c) = capabilities.max_image_count {
			if num_images.get() > c {
				return Err(SwapchainCreationError::UnsupportedMaxImagesCount)
			}
		}
		if !capabilities.supported_formats.iter().any(|&(f, c)| f == format && c == color_space) {
			return Err(SwapchainCreationError::UnsupportedFormat)
		}
		if dimensions[0].get() < capabilities.min_image_extent[0] {
			return Err(SwapchainCreationError::UnsupportedDimensions)
		}
		if dimensions[0].get() > capabilities.max_image_extent[0] {
			return Err(SwapchainCreationError::UnsupportedDimensions)
		}
		if dimensions[1].get() < capabilities.min_image_extent[1] {
			return Err(SwapchainCreationError::UnsupportedDimensions)
		}
		if dimensions[1].get() > capabilities.max_image_extent[1] {
			return Err(SwapchainCreationError::UnsupportedDimensions)
		}
		if layers.get() > capabilities.max_image_array_layers {
			return Err(SwapchainCreationError::UnsupportedArrayLayers)
		}
		if (usage.to_usage_bits() & capabilities.supported_usage_flags.to_usage_bits())
			!= usage.to_usage_bits()
		{
			return Err(SwapchainCreationError::UnsupportedUsageFlags)
		}
		if !capabilities.supported_transforms.supports(transform) {
			return Err(SwapchainCreationError::UnsupportedSurfaceTransform)
		}
		if !capabilities.supported_composite_alpha.supports(alpha) {
			return Err(SwapchainCreationError::UnsupportedCompositeAlpha)
		}
		if !capabilities.present_modes.supports(mode) {
			return Err(SwapchainCreationError::UnsupportedPresentMode)
		}

		// If we recreate a swapchain, make sure that the surface is the same.
		if let Some(sc) = old_swapchain {
			if surface.internal_object() != sc.surface.internal_object() {
				return Err(SwapchainCreationError::OldSwapchainSurfaceMismatch)
			}
		}

		// Checking that the surface doesn't already have a swapchain.
		if old_swapchain.is_none() {
			let has_already = surface.flag().swap(true, Ordering::AcqRel);
			if has_already {
				return Err(SwapchainCreationError::SurfaceInUse)
			}
		}

		if !device.loaded_extensions().khr_swapchain {
			return Err(SwapchainCreationError::MissingExtension)
		}

		// Required by the specs.
		assert_ne!(usage, ImageUsage::default());

		if let Some(ref old_swapchain) = old_swapchain {
			let mut stale = old_swapchain.stale.lock().unwrap();

			// The swapchain has already been used to create a new one.
			if *stale {
				return Err(SwapchainCreationError::OldSwapchainAlreadyUsed)
			} else {
				// According to the documentation of VkSwapchainCreateInfoKHR:
				//
				// > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
				// > any images not acquired by the application may be freed by the implementation,
				// > which may occur even if creation of the new swapchain fails.
				//
				// Therefore, we set stale to true and keep it to true even if the call to `vkCreateSwapchainKHR` below fails.
				*stale = true;
			}
		}

		let vk = device.pointers();

		let sharing = sharing.into();
		let swapchain = unsafe {
			let (sh_mode, sh_count, sh_indices) = match sharing {
				SharingMode::Exclusive(_) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
				SharingMode::Concurrent(ref ids) => {
					(vk::SHARING_MODE_CONCURRENT, ids.len() as u32, ids.as_ptr())
				}
			};

			let infos = vk::SwapchainCreateInfoKHR {
				sType: vk::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
				pNext: ptr::null(),
				flags: 0, // reserved
				surface: surface.internal_object(),
				minImageCount: num_images.get(),
				imageFormat: format as u32,
				imageColorSpace: color_space as u32,
				imageExtent: vk::Extent2D {
					width: dimensions[0].get(),
					height: dimensions[1].get()
				},
				imageArrayLayers: layers.get(),
				imageUsage: usage.to_usage_bits(),
				imageSharingMode: sh_mode,
				queueFamilyIndexCount: sh_count,
				pQueueFamilyIndices: sh_indices,
				preTransform: transform as u32,
				compositeAlpha: alpha as u32,
				presentMode: mode as u32,
				clipped: if clipped { vk::TRUE } else { vk::FALSE },
				oldSwapchain: if let Some(ref old_swapchain) = old_swapchain {
					old_swapchain.swapchain
				} else {
					0
				}
			};

			let mut output = mem::uninitialized();
			check_errors(vk.CreateSwapchainKHR(
				device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			output
		};

		let image_handles = unsafe {
			let mut num = 0;
			check_errors(vk.GetSwapchainImagesKHR(
				device.internal_object(),
				swapchain,
				&mut num,
				ptr::null_mut()
			))?;

			let mut images = Vec::with_capacity(num as usize);
			check_errors(vk.GetSwapchainImagesKHR(
				device.internal_object(),
				swapchain,
				&mut num,
				images.as_mut_ptr()
			))?;
			images.set_len(num as usize);
			images
		};

		let images = image_handles
			.into_iter()
			.map(|image| unsafe {
				let dims = if layers.get() == 1 {
					ImageDimensions::Dim2D { width: dimensions[0], height: dimensions[1] }
				} else {
					ImageDimensions::Dim2DArray {
						width: dimensions[0],
						height: dimensions[1],
						array_layers: layers
					}
				};

				let img = UnsafeImage::from_raw(
					device.clone(),
					image,
					usage,
					format,
					dims,
					crate::NONZERO_ONE,
					crate::NONZERO_ONE
				);

				ImageEntry { image: img, undefined_layout: AtomicBool::new(true) }
			})
			.collect::<Vec<_>>();

		let swapchain = Arc::new(Swapchain {
			device: device.clone(),
			surface: surface.clone(),
			swapchain,
			images,
			stale: Mutex::new(false),
			num_images,
			format,
			color_space,
			dimensions,
			layers,
			usage: usage.clone(),
			sharing,
			transform,
			alpha,
			mode,
			clipped
		});

		let swapchain_images = unsafe {
			let mut swapchain_images = Vec::with_capacity(swapchain.images.len());
			for n in 0 .. swapchain.images.len() {
				swapchain_images.push(SwapchainImage::from_raw(swapchain.clone(), n)?);
			}
			swapchain_images
		};

		Ok((swapchain, swapchain_images))
	}

	/// Recreates the swapchain with new dimensions.
	pub fn recreate_with_dimension(
		&self, dimensions: [NonZeroU32; 2]
	) -> Result<(Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>), SwapchainCreationError> {
		Swapchain::new(
			self.device.clone(),
			self.surface.clone(),
			self.sharing.clone(),
			dimensions,
			self.layers,
			self.num_images,
			self.format,
			self.color_space,
			self.usage,
			self.transform,
			self.alpha,
			self.mode,
			self.clipped,
			Some(self)
		)
	}

	/// Returns the inner image of this swapchain at index.
	pub fn raw_image(&self, index: usize) -> Option<&UnsafeImage> {
		self.images.get(index).map(|e| &e.image)
	}

	/// Returns the number of images of the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn num_images(&self) -> NonZeroU32 {
		unsafe { NonZeroU32::new_unchecked(self.images.len() as u32) }
	}

	/// Returns the format of the images of the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn format(&self) -> Format { self.format }

	/// Returns the color space of the images of the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn color_space(&self) -> ColorSpace { self.color_space }

	/// Returns the dimensions of the images of the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn dimensions(&self) -> [NonZeroU32; 2] { self.dimensions }

	/// Returns the number of layers of the images of the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn layers(&self) -> NonZeroU32 { self.layers }

	/// Returns the transform that was passed when creating the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn transform(&self) -> SurfaceTransform { self.transform }

	/// Returns the alpha mode that was passed when creating the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn composite_alpha(&self) -> CompositeAlpha { self.alpha }

	/// Returns the present mode that was passed when creating the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn present_mode(&self) -> PresentMode { self.mode }

	/// Returns the value of `clipped` that was passed when creating the swapchain.
	///
	/// See the documentation of `Swapchain::new`.
	pub fn clipped(&self) -> bool { self.clipped }
}
unsafe impl<W> VulkanObject for Swapchain<W> {
	type Object = vk::SwapchainKHR;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT;

	fn internal_object(&self) -> vk::SwapchainKHR { self.swapchain }
}
unsafe impl<W> DeviceOwned for Swapchain<W> {
	fn device(&self) -> &Arc<Device> { &self.device }
}
impl<W> fmt::Debug for Swapchain<W> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(fmt, "<Vulkan swapchain {:?}>", self.swapchain)
	}
}
impl<W> Drop for Swapchain<W> {
	fn drop(&mut self) {
		unsafe {
			let vk = self.device.pointers();
			vk.DestroySwapchainKHR(self.device.internal_object(), self.swapchain, ptr::null());
			self.surface.flag().store(false, Ordering::Release);
		}
	}
}

/// Error that can happen when creation a swapchain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapchainCreationError {
	/// Not enough memory.
	OomError(OomError),
	/// The device was lost.
	DeviceLost,
	/// The surface was lost.
	SurfaceLost,
	/// The surface is already used by another swapchain.
	SurfaceInUse,
	/// The window is already in use by another API.
	NativeWindowInUse,
	/// The `VK_KHR_swapchain` extension was not enabled.
	MissingExtension,
	/// Surface mismatch between old and new swapchain.
	OldSwapchainSurfaceMismatch,
	/// The old swapchain has already been used to recreate another one.
	OldSwapchainAlreadyUsed,
	/// The requested number of swapchain images is not supported by the surface.
	UnsupportedMinImagesCount,
	/// The requested number of swapchain images is not supported by the surface.
	UnsupportedMaxImagesCount,
	/// The requested image format is not supported by the surface.
	UnsupportedFormat,
	/// The requested dimensions are not supported by the surface.
	UnsupportedDimensions,
	/// The requested array layers count is not supported by the surface.
	UnsupportedArrayLayers,
	/// The requested image usage is not supported by the surface.
	UnsupportedUsageFlags,
	/// The requested surface transform is not supported by the surface.
	UnsupportedSurfaceTransform,
	/// The requested composite alpha is not supported by the surface.
	UnsupportedCompositeAlpha,
	/// The requested present mode is not supported by the surface.
	UnsupportedPresentMode
}
impl fmt::Display for SwapchainCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			SwapchainCreationError::OomError(e) => e.fmt(f),

			SwapchainCreationError::DeviceLost => {
				write!(f, "The connection to the device was lost")
			}
			SwapchainCreationError::SurfaceLost => write!(f, "The surface was lost"),
			SwapchainCreationError::SurfaceInUse => {
				write!(f, "The surface is already in use by another swapchain")
			}
			SwapchainCreationError::NativeWindowInUse => {
				write!(f, "The window is already in use by another API")
			}
			SwapchainCreationError::MissingExtension => {
				write!(f, "The `VK_KHR_swapchain` extension was not enabled")
			}

			SwapchainCreationError::OldSwapchainSurfaceMismatch => {
				write!(f, "Surface mismatch between the old and the new swapchain")
			}
			SwapchainCreationError::OldSwapchainAlreadyUsed => {
				write!(f, "Old swapchain has already been used to recreate a new one")
			}

			SwapchainCreationError::UnsupportedMinImagesCount => write!(
				f,
				"The requested number of swapchain images is less than the surface minimum"
			),
			SwapchainCreationError::UnsupportedMaxImagesCount => write!(
				f,
				"The requested number of swapchain image is more than the surface maximum"
			),
			SwapchainCreationError::UnsupportedFormat => {
				write!(f, "The requested format or colorspace is not supported by the surface")
			}
			SwapchainCreationError::UnsupportedDimensions => {
				write!(f, "The requested dimensions are not supported by the surface")
			}
			SwapchainCreationError::UnsupportedArrayLayers => {
				write!(f, "The requested layer count is not supported by the surface")
			}
			SwapchainCreationError::UnsupportedUsageFlags => {
				write!(f, "The requested usage is not supported by the surface")
			}
			SwapchainCreationError::UnsupportedSurfaceTransform => {
				write!(f, "The requested surface transform is not supported by the surface")
			}
			SwapchainCreationError::UnsupportedCompositeAlpha => {
				write!(f, "The requested composite alpha is not supported by the surface")
			}
			SwapchainCreationError::UnsupportedPresentMode => {
				write!(f, "The requested present mode is not supported by the surface")
			}
		}
	}
}
impl error::Error for SwapchainCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			SwapchainCreationError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<Error> for SwapchainCreationError {
	fn from(err: Error) -> SwapchainCreationError {
		match err {
			Error::OutOfHostMemory => SwapchainCreationError::OomError(OomError::from(err)),
			Error::OutOfDeviceMemory => SwapchainCreationError::OomError(OomError::from(err)),
			Error::DeviceLost => SwapchainCreationError::DeviceLost,
			Error::SurfaceLost => SwapchainCreationError::SurfaceLost,
			Error::NativeWindowInUse => SwapchainCreationError::NativeWindowInUse,
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}
impl From<OomError> for SwapchainCreationError {
	fn from(err: OomError) -> SwapchainCreationError { SwapchainCreationError::OomError(err) }
}
impl From<CapabilitiesError> for SwapchainCreationError {
	fn from(err: CapabilitiesError) -> SwapchainCreationError {
		match err {
			CapabilitiesError::OomError(err) => SwapchainCreationError::OomError(err),
			CapabilitiesError::SurfaceLost => SwapchainCreationError::SurfaceLost
		}
	}
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture<W> {
	swapchain: Arc<Swapchain<W>>,
	image_id: usize,
	// Semaphore that is signalled when the acquire is complete. Empty if the acquire has already
	// happened.
	semaphore: Option<Semaphore>,
	// Fence that is signalled when the acquire is complete. Empty if the acquire has already
	// happened.
	fence: Option<Fence>,
	finished: AtomicBool
}
impl<W> SwapchainAcquireFuture<W> {
	/// Returns the index of the image in the list of images returned when creating the swapchain.
	pub fn image_id(&self) -> usize { self.image_id }

	/// Returns the corresponding swapchain.
	pub fn swapchain(&self) -> &Arc<Swapchain<W>> { &self.swapchain }
}
unsafe impl<W> GpuFuture for SwapchainAcquireFuture<W> {
	fn cleanup_finished(&mut self) {}

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		if let Some(ref semaphore) = self.semaphore {
			let mut sem = SubmitSemaphoresWaitBuilder::new();
			sem.add_wait_semaphore(&semaphore);
			Ok(SubmitAnyBuilder::SemaphoresWait(sem))
		} else {
			Ok(SubmitAnyBuilder::Empty)
		}
	}

	fn flush(&self) -> Result<(), FlushError> { Ok(()) }

	unsafe fn signal_finished(&self) { self.finished.store(true, Ordering::SeqCst); }

	fn queue_change_allowed(&self) -> bool { true }

	fn queue(&self) -> Option<Arc<Queue>> { None }

	fn check_buffer_access(
		&self, _: &BufferAccess, _: bool, _: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		Err(AccessCheckError::Unknown)
	}

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, _: bool, _: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
		if swapchain_image.internal_object() != image.parent().inner().internal_object() {
			return Err(AccessCheckError::Unknown)
		}

		if self.swapchain.images[self.image_id].undefined_layout.load(Ordering::Relaxed)
			&& layout != ImageLayout::Undefined
		{
			return Err(AccessCheckError::Denied(AccessError::ImageLayoutMismatch {
				actual: ImageLayout::Undefined,
				expected: layout
			}))
		}

		if layout != ImageLayout::Undefined && layout != ImageLayout::PresentSrc {
			return Err(AccessCheckError::Denied(AccessError::ImageLayoutMismatch {
				actual: ImageLayout::PresentSrc,
				expected: layout
			}))
		}

		Ok(None)
	}
}
unsafe impl<W> DeviceOwned for SwapchainAcquireFuture<W> {
	fn device(&self) -> &Arc<Device> { &self.swapchain.device }
}
impl<W> Drop for SwapchainAcquireFuture<W> {
	fn drop(&mut self) {
		if !*self.finished.get_mut() {
			if let Some(ref fence) = self.fence {
				fence.wait(None).unwrap(); // TODO: handle error?
				self.semaphore = None;
			}
		} else {
			// We make sure that the fence is signalled. This also silences an error from the
			// validation layers about using a fence whose state hasn't been checked (even though
			// we know for sure that it must've been signalled).
			debug_assert!({
				let dur = Some(Duration::new(0, 0));
				self.fence.as_ref().map(|f| f.wait(dur).is_ok()).unwrap_or(true)
			});
		}

		// TODO: if this future is destroyed without being presented, then eventually acquiring
		// a new image will block forever ; difficulty: hard
	}
}
impl<W> fmt::Debug for SwapchainAcquireFuture<W> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(
			f,
			"SwapchainAcquireFuture {{ swapchain: {:?}, image_id: {}, \
			 semaphore: {:?}, fence: {:?}, finished: {:?} }}",
			self.swapchain, self.image_id, self.semaphore, self.fence, self.finished
		)
	}
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AcquireError {
	/// Not enough memory.
	OomError(OomError),

	/// The connection to the device has been lost.
	DeviceLost,

	/// The timeout of the function has been reached before an image was available.
	Timeout,

	/// The surface is no longer accessible and must be recreated.
	SurfaceLost,

	/// The surface has changed in a way that makes the swapchain unusable. You must query the
	/// surface's new properties and recreate a new swapchain if you want to continue drawing.
	OutOfDate
}
impl fmt::Display for AcquireError {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		match self {
			AcquireError::OomError(e) => e.fmt(f),

			AcquireError::DeviceLost => write!(f, "The connection to the device was lost"),
			AcquireError::Timeout => write!(f, "No image was available in given time"),
			AcquireError::SurfaceLost => write!(f, "The surface was lost"),
			AcquireError::OutOfDate => write!(f, "The swapchain is out of date")
		}
	}
}
impl error::Error for AcquireError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			AcquireError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for AcquireError {
	fn from(err: OomError) -> AcquireError { AcquireError::OomError(err) }
}
impl From<Error> for AcquireError {
	fn from(err: Error) -> AcquireError {
		match err {
			Error::OutOfHostMemory => AcquireError::OomError(OomError::from(err)),
			Error::OutOfDeviceMemory => AcquireError::OomError(OomError::from(err)),
			Error::DeviceLost => AcquireError::DeviceLost,
			Error::SurfaceLost => AcquireError::SurfaceLost,
			Error::OutOfDate => AcquireError::OutOfDate,
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P: GpuFuture, W> {
	previous: P,
	queue: Arc<Queue>,
	swapchain: Arc<Swapchain<W>>,
	image_id: usize,
	present_region: Option<PresentRegion>,
	// True if `flush()` has been called on the future, which means that the present command has
	// been submitted.
	flushed: AtomicBool,
	// True if `signal_finished()` has been called on the future, which means that the future has
	// been submitted and has already been processed by the GPU.
	finished: AtomicBool
}
impl<P: GpuFuture, W> PresentFuture<P, W> {
	/// Returns the index of the image in the list of images returned when creating the swapchain.
	pub fn image_id(&self) -> usize { self.image_id }

	/// Returns the corresponding swapchain.
	pub fn swapchain(&self) -> &Arc<Swapchain<W>> { &self.swapchain }
}
unsafe impl<P: GpuFuture, W> GpuFuture for PresentFuture<P, W> {
	fn cleanup_finished(&mut self) { self.previous.cleanup_finished(); }

	unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
		if self.flushed.load(Ordering::SeqCst) {
			return Ok(SubmitAnyBuilder::Empty)
		}

		let queue = self.previous.queue().map(|q| q.clone());

		// TODO: if the swapchain image layout is not PRESENT, should add a transition command
		// buffer

		Ok(match self.previous.build_submission()? {
			SubmitAnyBuilder::Empty => {
				let mut builder = SubmitPresentBuilder::new();
				builder.add_swapchain(
					&self.swapchain,
					self.image_id as u32,
					self.present_region.as_ref()
				);
				SubmitAnyBuilder::QueuePresent(builder)
			}
			SubmitAnyBuilder::SemaphoresWait(sem) => {
				let mut builder: SubmitPresentBuilder = sem.into();
				builder.add_swapchain(
					&self.swapchain,
					self.image_id as u32,
					self.present_region.as_ref()
				);
				SubmitAnyBuilder::QueuePresent(builder)
			}
			SubmitAnyBuilder::CommandBuffer(cb) => {
				// submit the command buffer by flushing previous.
				// Since the implementation should remember being flushed it's safe to call build_submission multiple times
				self.previous.flush()?;

				let mut builder = SubmitPresentBuilder::new();
				builder.add_swapchain(
					&self.swapchain,
					self.image_id as u32,
					self.present_region.as_ref()
				);
				SubmitAnyBuilder::QueuePresent(builder)
			}
			SubmitAnyBuilder::BindSparse(cb) => {
				// submit the command buffer by flushing previous.
				// Since the implementation should remember being flushed it's safe to call build_submission multiple times
				self.previous.flush()?;

				let mut builder = SubmitPresentBuilder::new();
				builder.add_swapchain(
					&self.swapchain,
					self.image_id as u32,
					self.present_region.as_ref()
				);
				SubmitAnyBuilder::QueuePresent(builder)
			}
			SubmitAnyBuilder::QueuePresent(present) => {
				unimplemented!() // TODO:
				 // present.submit();
				 // let mut builder = SubmitPresentBuilder::new();
				 // builder.add_swapchain(self.command_buffer.inner(), self.image_id);
				 // SubmitAnyBuilder::CommandBuffer(builder)
			}
		})
	}

	fn flush(&self) -> Result<(), FlushError> {
		unsafe {
			// If `flushed` already contains `true`, then `build_submission` will return `Empty`.
			match self.build_submission()? {
				SubmitAnyBuilder::Empty => (),
				SubmitAnyBuilder::QueuePresent(present) => {
					present.submit(&self.queue)?;
				}
				_ => unreachable!()
			}

			self.flushed.store(true, Ordering::SeqCst);
			Ok(())
		}
	}

	unsafe fn signal_finished(&self) {
		self.flushed.store(true, Ordering::SeqCst);
		self.finished.store(true, Ordering::SeqCst);
		self.previous.signal_finished();
	}

	fn queue_change_allowed(&self) -> bool { false }

	fn queue(&self) -> Option<Arc<Queue>> {
		debug_assert!(match self.previous.queue() {
			None => true,
			Some(q) => q.is_same(&self.queue)
		});

		Some(self.queue.clone())
	}

	fn check_buffer_access(
		&self, buffer: &BufferAccess, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		self.previous.check_buffer_access(buffer, exclusive, queue)
	}

	fn check_image_access(
		&self, image: &dyn ImageViewAccess, layout: ImageLayout, exclusive: bool, queue: &Queue
	) -> Result<Option<(PipelineStages, AccessFlagBits)>, AccessCheckError> {
		let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
		if swapchain_image.internal_object() == image.parent().inner().internal_object() {
			// This future presents the swapchain image, which "unlocks" it. Therefore any attempt
			// to use this swapchain image afterwards shouldn't get granted automatic access.
			// Instead any attempt to access the image afterwards should get an authorization from
			// a later swapchain acquire future. Hence why we return `Unknown` here.
			Err(AccessCheckError::Unknown)
		} else {
			self.previous.check_image_access(image, layout, exclusive, queue)
		}
	}
}
unsafe impl<P: GpuFuture, W> DeviceOwned for PresentFuture<P, W> {
	fn device(&self) -> &Arc<Device> { self.queue.device() }
}
impl<P: GpuFuture, W> Drop for PresentFuture<P, W> {
	fn drop(&mut self) {
		unsafe {
			if !*self.finished.get_mut() {
				match self.flush() {
					Ok(()) => {
						// Block until the queue finished.
						self.queue().unwrap().wait().unwrap();
						self.previous.signal_finished();
					}
					Err(_) => {
						// In case of error we simply do nothing, as there's nothing to do
						// anyway.
					}
				}
			}
		}
	}
}
impl<P: GpuFuture, W> fmt::Debug for PresentFuture<P, W> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		// TODO: Stack overflow if the previous chain is longish
		write!(
			f,
			"PresentFuture {{ previous: <GpuFuture>, queue: {:?}, swapchain: {:?}, \
			 image_id: {}, preset_region: {:?}, flushed: {:?}, finished: {:?} }}",
			self.queue, self.swapchain, self.image_id, self.present_region, self.flushed, self
		)
	}
}

pub struct AcquiredImage {
	pub id: usize,
	pub suboptimal: bool
}
/// Unsafe variant of `acquire_next_image`.
///
/// # Safety
///
/// - The semaphore and/or the fence must be kept alive until it is signaled.
/// - The swapchain must not have been replaced by being passed as the old swapchain when creating
///   a new one.
pub unsafe fn acquire_next_image_raw<W>(
	swapchain: &Swapchain<W>, timeout: Option<Duration>, semaphore: Option<&Semaphore>,
	fence: Option<&Fence>
) -> Result<AcquiredImage, AcquireError> {
	let vk = swapchain.device.pointers();

	let timeout_ns = if let Some(timeout) = timeout {
		timeout
			.as_secs()
			.saturating_mul(1_000_000_000)
			.saturating_add(timeout.subsec_nanos() as u64)
	} else {
		u64::max_value()
	};

	let mut out = mem::uninitialized();
	let r = check_errors(vk.AcquireNextImageKHR(
		swapchain.device.internal_object(),
		swapchain.swapchain,
		timeout_ns,
		semaphore.map(|s| s.internal_object()).unwrap_or(0),
		fence.map(|f| f.internal_object()).unwrap_or(0),
		&mut out
	))?;

	let (id, suboptimal) = match r {
		Success::Success => (out as usize, false),
		Success::Suboptimal => (out as usize, true),
		Success::NotReady => return Err(AcquireError::Timeout),
		Success::Timeout => return Err(AcquireError::Timeout),
		s => panic!("unexpected success value: {:?}", s)
	};

	Ok(AcquiredImage { id, suboptimal })
}
