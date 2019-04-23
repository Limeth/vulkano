use std::{error, fmt, mem, num::NonZeroU32, ops::Range, ptr, sync::Arc};

use smallvec::SmallVec;
use vk_sys as vk;

use crate::{
	check_errors,
	device::Device,
	format::{Format, FormatTy},
	image::{ImageDimensions, ImageUsage, MipmapsCount},
	instance::Limits,
	memory::{DeviceMemory, DeviceMemoryAllocError, MemoryRequirements},
	sync::Sharing,
	Error,
	OomError,
	VulkanObject
};

use super::LinearLayout;

/// A storage for pixels or arbitrary data.
///
/// # Safety
/// This type is not just unsafe but very unsafe. Don't use it directly.
/// - You must manually bind memory to the image with `bind_memory`. The memory must respect the
///   requirements returned by `new`.
/// - The memory that you bind to the image must be manually kept alive.
/// - The queue family ownership must be manually enforced.
/// - The usage must be manually enforced.
/// - The image layout must be manually enforced and transitioned.
pub struct UnsafeImage {
	device: Arc<Device>,

	image: vk::Image,
	usage: ImageUsage,

	format: Format,
	// Features that are supported for this particular format.
	format_features: vk::FormatFeatureFlagBits,

	dimensions: ImageDimensions,
	samples: NonZeroU32,
	mipmap_levels: NonZeroU32,

	// `vkDestroyImage` is called only if `needs_destruction` is true.
	needs_destruction: bool
}
impl UnsafeImage {
	/// Creates a new image and returns the memory allocation requirements.
	///
	/// The requirements should be used to allocate memory and then bind
	/// it to the image using `bind_memory(..)`.
	pub unsafe fn new<'a, M, I>(
		device: Arc<Device>, sharing: Sharing<I>, usage: ImageUsage, format: Format,
		dimensions: ImageDimensions, samples: NonZeroU32, mipmap_levels: M,
		preinitialized_layout: bool, linear_tiling: bool
	) -> Result<(UnsafeImage, MemoryRequirements), UnsafeImageCreationError>
	where
		M: Into<MipmapsCount>,
		I: Iterator<Item = u32>
	{
		// TODO: doesn't check that the proper features are enabled
		let vk = device.pointers();
		let vk_i = device.instance().pointers();

		let (sharing_mode, sharing_indices) = match sharing {
			Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
			Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect())
		};

		let device_limits = device.physical_device().limits();

		// Checking if image usage conforms to what is supported.
		let format_features = {
			let physical_device = device.physical_device().internal_object();

			let mut output = mem::uninitialized();
			vk_i.GetPhysicalDeviceFormatProperties(physical_device, format as u32, &mut output);

			let features = if linear_tiling {
				output.linearTilingFeatures
			} else {
				output.optimalTilingFeatures
			};
			if features == 0 {
				return Err(UnsafeImageCreationError::FormatNotSupported)
			}

			if usage.sampled && (features & vk::FORMAT_FEATURE_SAMPLED_IMAGE_BIT == 0) {
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
			if usage.storage && (features & vk::FORMAT_FEATURE_STORAGE_IMAGE_BIT == 0) {
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
			if usage.color_attachment && (features & vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT == 0) {
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
			if usage.depth_stencil_attachment
				&& (features & vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT == 0)
			{
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
			if usage.input_attachment
				&& (features
					& (vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
						| vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
					== 0)
			{
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
			if device.loaded_extensions().khr_maintenance1 {
				if usage.transfer_source
					&& (features & vk::FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR == 0)
				{
					return Err(UnsafeImageCreationError::UnsupportedUsage)
				}
				if usage.transfer_destination
					&& (features & vk::FORMAT_FEATURE_TRANSFER_DST_BIT_KHR == 0)
				{
					return Err(UnsafeImageCreationError::UnsupportedUsage)
				}
			}

			features
		};
		// If `transient_attachment` is true, then only `color_attachment`,
		// `depth_stencil_attachment` and `input_attachment` can be true as well.
		if usage.transient_attachment {
			let u = ImageUsage {
				transient_attachment: false,
				color_attachment: false,
				depth_stencil_attachment: false,
				input_attachment: false,
				..usage.clone()
			};

			if u != ImageUsage::default() {
				return Err(UnsafeImageCreationError::UnsupportedUsage)
			}
		}
		if !samples.get().is_power_of_two() {
			return Err(UnsafeImageCreationError::UnsupportedSamplesCount(samples.get()))
		}
		// If the `shaderStorageImageMultisample` feature is not enabled and we have
		// `usage_storage` set to true, then the number of samples must be 1.
		if usage.storage && samples.get() > 1 {
			if !device.enabled_features().shader_storage_image_multisample {
				return Err(UnsafeImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled)
			}
		}
		let (dimensions_type, dimensions_extent, array_layers, dimensions_flags) =
			dimensions.vk_type();

		// Check dimensions, samples and mipmaps againts device capabilities.
		let mipmap_levels = {
			let mut capabilities = mem::uninitialized();
			let physical_device = device.physical_device().internal_object();
			let r = vk_i.GetPhysicalDeviceImageFormatProperties(
				physical_device,
				format as u32,
				dimensions_type,
				if linear_tiling { vk::IMAGE_TILING_LINEAR } else { vk::IMAGE_TILING_OPTIMAL },
				usage.to_usage_bits(),
				dimensions_flags,
				&mut capabilities
			);
			match check_errors(r) {
				Ok(_) => (),
				Err(Error::FormatNotSupported) => {
					return Err(UnsafeImageCreationError::FormatNotSupported)
				}
				Err(err) => return Err(err.into())
			}

			UnsafeImage::check_capabilities(
				device_limits,
				capabilities,
				usage,
				format,
				dimensions,
				samples,
				mipmap_levels.into()
			)?
		};

		// Everything now ok. Creating the image.
		let image = {
			let infos = vk::ImageCreateInfo {
				sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				pNext: ptr::null(),
				flags: dimensions_flags,
				imageType: dimensions_type,
				format: format as u32,
				extent: dimensions_extent,
				mipLevels: mipmap_levels.get(),
				arrayLayers: array_layers,
				samples: samples.get(),
				tiling: if linear_tiling {
					vk::IMAGE_TILING_LINEAR
				} else {
					vk::IMAGE_TILING_OPTIMAL
				},
				usage: usage.to_usage_bits(),
				sharingMode: sharing_mode,
				queueFamilyIndexCount: sharing_indices.len() as u32,
				pQueueFamilyIndices: sharing_indices.as_ptr(),
				initialLayout: if preinitialized_layout {
					vk::IMAGE_LAYOUT_PREINITIALIZED
				} else {
					vk::IMAGE_LAYOUT_UNDEFINED
				}
			};

			let mut output = mem::uninitialized();
			check_errors(vk.CreateImage(
				device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			output
		};

		let mem_reqs = if device.loaded_extensions().khr_get_memory_requirements2 {
			let infos = vk::ImageMemoryRequirementsInfo2KHR {
				sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
				pNext: ptr::null_mut(),
				image
			};

			let mut output2 = if device.loaded_extensions().khr_dedicated_allocation {
				Some(vk::MemoryDedicatedRequirementsKHR {
					sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
					pNext: ptr::null(),
					prefersDedicatedAllocation: mem::uninitialized(),
					requiresDedicatedAllocation: mem::uninitialized()
				})
			} else {
				None
			};

			let mut output = vk::MemoryRequirements2KHR {
				sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
				pNext: output2
					.as_mut()
					.map(|o| o as *mut vk::MemoryDedicatedRequirementsKHR)
					.unwrap_or(ptr::null_mut()) as *mut _,
				memoryRequirements: mem::uninitialized()
			};

			vk.GetImageMemoryRequirements2KHR(device.internal_object(), &infos, &mut output);
			debug_assert!(output.memoryRequirements.memoryTypeBits != 0);

			let mut out = MemoryRequirements::from_vulkan_reqs(output.memoryRequirements);
			if let Some(output2) = output2 {
				debug_assert_eq!(output2.requiresDedicatedAllocation, 0);
				out.prefer_dedicated = output2.prefersDedicatedAllocation != 0;
			}
			out
		} else {
			let mut output: vk::MemoryRequirements = mem::uninitialized();
			vk.GetImageMemoryRequirements(device.internal_object(), image, &mut output);
			debug_assert!(output.memoryTypeBits != 0);
			MemoryRequirements::from_vulkan_reqs(output)
		};

		let image = UnsafeImage {
			device: device.clone(),
			image,
			usage,
			format,
			dimensions,
			samples,
			mipmap_levels,
			format_features,
			needs_destruction: true
		};

		Ok((image, mem_reqs))
	}

	/// Checks limits and capabilitied for the requested parameters.
	///
	/// Returns the computed number of mipmaps, as this is the only parameter
	/// passed that has a different output.
	// Part of the `new` function extracted so it's reusable
	// and more readable.
	#[inline(always)]
	fn check_capabilities(
		device_limits: Limits, capabilities: vk::ImageFormatProperties, usage: ImageUsage,
		format: Format, dimensions: ImageDimensions, samples: NonZeroU32,
		mipmap_levels: MipmapsCount
	) -> Result<NonZeroU32, UnsafeImageCreationError> {
		if !dimensions.check_limits(device_limits) {
			if dimensions.width().get() > capabilities.maxExtent.width
				|| dimensions.height().get() > capabilities.maxExtent.height
				|| dimensions.depth().get() > capabilities.maxExtent.depth
				|| dimensions.array_layers().get() > capabilities.maxArrayLayers
			{
				return Err(UnsafeImageCreationError::UnsupportedDimensions(dimensions))
			}
		}

		// Checking whether the number of samples is supported.
		{
			let mut supported_samples = 0x7f; // all bits up to VK_SAMPLE_COUNT_64_BIT

			if usage.sampled {
				match format.ty() {
					FormatTy::Float | FormatTy::Compressed => {
						supported_samples &= device_limits.sampled_image_color_sample_counts();
					}
					FormatTy::Uint | FormatTy::Sint => {
						supported_samples &= device_limits.sampled_image_integer_sample_counts();
					}
					FormatTy::Depth => {
						supported_samples &= device_limits.sampled_image_depth_sample_counts();
					}
					FormatTy::Stencil => {
						supported_samples &= device_limits.sampled_image_stencil_sample_counts();
					}
					FormatTy::DepthStencil => {
						supported_samples &= device_limits.sampled_image_depth_sample_counts();
						supported_samples &= device_limits.sampled_image_stencil_sample_counts();
					}
				}
			}

			if usage.storage {
				supported_samples &= device_limits.storage_image_sample_counts();
			}

			if usage.color_attachment
				|| usage.depth_stencil_attachment
				|| usage.input_attachment
				|| usage.transient_attachment
			{
				match format.ty() {
					FormatTy::Float | FormatTy::Compressed | FormatTy::Uint | FormatTy::Sint => {
						supported_samples &= device_limits.framebuffer_color_sample_counts();
					}
					FormatTy::Depth => {
						supported_samples &= device_limits.framebuffer_depth_sample_counts();
					}
					FormatTy::Stencil => {
						supported_samples &= device_limits.framebuffer_stencil_sample_counts();
					}
					FormatTy::DepthStencil => {
						supported_samples &= device_limits.framebuffer_depth_sample_counts();
						supported_samples &= device_limits.framebuffer_stencil_sample_counts();
					}
				}
			}

			if (samples.get() & supported_samples) == 0 {
				if (samples.get() & capabilities.sampleCounts) == 0 {
					return Err(UnsafeImageCreationError::UnsupportedSamplesCount(samples.get()))
				}
			}
		}

		let mipmap_levels = match mipmap_levels.for_image(dimensions) {
			Ok(number) => number,
			Err(number) => {
				if number.get() > capabilities.maxMipLevels {
					return Err(UnsafeImageCreationError::InvalidMipmapsCount {
						requested: number.get(),
						valid_range: 1 .. dimensions.max_mipmaps().get()
					})
				} else {
					number
				}
			}
		};

		Ok(mipmap_levels)
	}

	/// Creates an image from a raw handle.
	///
	/// This function expects the image to be already
	/// created and allocated. The image won't be destroyed on
	/// drop. This function is useful for wrapping the raw
	/// image objects obtained from Vulkan itself,
	/// such as when creating a swapchain.
	pub unsafe fn from_raw(
		device: Arc<Device>, handle: u64, usage: ImageUsage, format: Format,
		dimensions: ImageDimensions, samples: NonZeroU32, mipmap_levels: NonZeroU32
	) -> UnsafeImage {
		let vk_i = device.instance().pointers();
		let physical_device = device.physical_device().internal_object();

		let mut output = mem::uninitialized();
		vk_i.GetPhysicalDeviceFormatProperties(physical_device, format as u32, &mut output);

		// TODO: check that usage is correct in regard to `output`?

		UnsafeImage {
			device: device.clone(),
			image: handle,
			usage,
			format,
			dimensions,
			samples,
			mipmap_levels,
			format_features: output.optimalTilingFeatures,
			needs_destruction: false // TODO: pass as parameter
		}
	}

	pub unsafe fn bind_memory(&self, memory: &DeviceMemory, offset: usize) -> Result<(), OomError> {
		let vk = self.device.pointers();

		// We check for correctness in debug mode.
		debug_assert!({
			let mut mem_reqs = mem::uninitialized();
			vk.GetImageMemoryRequirements(self.device.internal_object(), self.image, &mut mem_reqs);
			mem_reqs.size <= (memory.size() - offset) as u64
				&& (offset as u64 % mem_reqs.alignment) == 0
				&& mem_reqs.memoryTypeBits & (1 << memory.memory_type().id()) != 0
		});

		check_errors(vk.BindImageMemory(
			self.device.internal_object(),
			self.image,
			memory.internal_object(),
			offset as vk::DeviceSize
		))?;
		Ok(())
	}

	/// Returns a key unique to each `UnsafeImage`. Can be used for the `conflicts_key` method.
	pub fn key(&self) -> u64 { self.image }

	/// Device getter.
	pub fn device(&self) -> &Arc<Device> { &self.device }

	/// Usage getter.
	pub fn usage(&self) -> ImageUsage { self.usage }

	/// Format getter.
	pub fn format(&self) -> Format { self.format }

	/// Dimensions getter.
	pub fn dimensions(&self) -> ImageDimensions { self.dimensions }

	/// Samples getter.
	pub fn samples(&self) -> NonZeroU32 { self.samples }

	/// Mipmap levels getter.
	pub fn mipmap_levels(&self) -> NonZeroU32 { self.mipmap_levels }

	/// Queries the layout of an image in memory. Only valid for images with linear tiling.
	///
	/// This function is only valid for images with a color format. See the other similar functions
	/// for the other aspects.
	///
	/// The layout is invariant for each image. However it is not cached, as this would waste
	/// memory in the case of non-linear-tiling images. You are encouraged to store the layout
	/// somewhere in order to avoid calling this semi-expensive function at every single memory
	/// access.
	///
	/// Note that while Vulkan allows querying the array layers other than 0, it is redundant as
	/// you can easily calculate the position of any layer.
	///
	/// # Panic
	///
	/// - Panics if the mipmap level is out of range.
	///
	/// # Safety
	///
	/// - The image must *not* have a depth, stencil or depth-stencil format.
	/// - The image must have been created with linear tiling.
	pub unsafe fn color_linear_layout(&self, mip_level: u32) -> LinearLayout {
		self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_COLOR_BIT)
	}

	/// Same as `color_linear_layout`, except that it retrieves the depth component of the image.
	///
	/// # Panic
	///
	/// - Panics if the mipmap level is out of range.
	///
	/// # Safety
	///
	/// - The image must have a depth or depth-stencil format.
	/// - The image must have been created with linear tiling.
	pub unsafe fn depth_linear_layout(&self, mip_level: u32) -> LinearLayout {
		self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_DEPTH_BIT)
	}

	/// Same as `color_linear_layout`, except that it retrieves the stencil component of the image.
	///
	/// # Panic
	///
	/// - Panics if the mipmap level is out of range.
	///
	/// # Safety
	///
	/// - The image must have a stencil or depth-stencil format.
	/// - The image must have been created with linear tiling.
	pub unsafe fn stencil_linear_layout(&self, mip_level: u32) -> LinearLayout {
		self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_STENCIL_BIT)
	}

	// Implementation of the `*_layout` functions.
	unsafe fn linear_layout_impl(&self, mip_level: u32, aspect: u32) -> LinearLayout {
		let vk = self.device.pointers();

		assert!(mip_level < self.mipmap_levels.get());

		let subresource =
			vk::ImageSubresource { aspectMask: aspect, mipLevel: mip_level, arrayLayer: 0 };

		let mut out = mem::uninitialized();
		vk.GetImageSubresourceLayout(
			self.device.internal_object(),
			self.image,
			&subresource,
			&mut out
		);

		LinearLayout {
			offset: out.offset as usize,
			size: out.size as usize,
			row_pitch: out.rowPitch as usize,
			array_pitch: out.arrayPitch as usize,
			depth_pitch: out.depthPitch as usize
		}
	}

	/// Returns true if the image can be used as a source for blits.
	pub fn supports_blit_source(&self) -> bool {
		(self.format_features & vk::FORMAT_FEATURE_BLIT_SRC_BIT) != 0
	}

	/// Returns true if the image can be used as a destination for blits.
	pub fn supports_blit_destination(&self) -> bool {
		(self.format_features & vk::FORMAT_FEATURE_BLIT_DST_BIT) != 0
	}

	/// Returns true if the image can be sampled with a linear filtering.
	pub fn supports_linear_filtering(&self) -> bool {
		(self.format_features & vk::FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0
	}
}

unsafe impl VulkanObject for UnsafeImage {
	type Object = vk::Image;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT;

	fn internal_object(&self) -> vk::Image { self.image }
}

impl fmt::Debug for UnsafeImage {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(fmt, "<Vulkan image {:?}>", self.image)
	}
}

impl Drop for UnsafeImage {
	fn drop(&mut self) {
		if self.needs_destruction {
			unsafe {
				let vk = self.device.pointers();
				vk.DestroyImage(self.device.internal_object(), self.image, ptr::null());
			}
		}
	}
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsafeImageCreationError {
	/// Allocating memory failed.
	AllocError(DeviceMemoryAllocError),
	/// The dimensions are too large, or one of the dimensions is 0.
	UnsupportedDimensions(ImageDimensions),
	/// A wrong number of mipmaps was provided.
	InvalidMipmapsCount { requested: u32, valid_range: Range<u32> },
	/// The requested number of samples is not supported, or is 0.
	UnsupportedSamplesCount(u32),
	/// The requested format is not supported by the Vulkan implementation.
	FormatNotSupported,
	/// The format is supported, but at least one of the requested usages is not supported.
	UnsupportedUsage,
	/// The `shader_storage_image_multisample` feature must be enabled to create such an image.
	ShaderStorageImageMultisampleFeatureNotEnabled
}
impl fmt::Display for UnsafeImageCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			UnsafeImageCreationError::AllocError(e) => write!(f, "Memory allocation failed: {}", e),
			UnsafeImageCreationError::InvalidMipmapsCount { requested, valid_range } => write!(
				f,
				"A wrong number of mipmaps provided: {} valid range: {:?}",
				requested, valid_range
			),
			UnsafeImageCreationError::UnsupportedSamplesCount(samples) => {
				write!(f, "The requested number of sampler is not supported: {}", samples)
			}
			UnsafeImageCreationError::UnsupportedDimensions(dims) => {
				write!(f, "The requested dimensions are not supported: {:?}", dims)
			}
			UnsafeImageCreationError::FormatNotSupported => {
				write!(f, "The requested format is not supported")
			}
			UnsafeImageCreationError::UnsupportedUsage => {
				write!(f, "The requested usage is not supported for requested format")
			}
			UnsafeImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
				write!(f, "The `shader_storage_image_multisample` feature must be enabled")
			}
		}
	}
}
impl error::Error for UnsafeImageCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			UnsafeImageCreationError::AllocError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for UnsafeImageCreationError {
	fn from(err: OomError) -> UnsafeImageCreationError {
		UnsafeImageCreationError::AllocError(DeviceMemoryAllocError::OomError(err))
	}
}
impl From<DeviceMemoryAllocError> for UnsafeImageCreationError {
	fn from(err: DeviceMemoryAllocError) -> UnsafeImageCreationError {
		UnsafeImageCreationError::AllocError(err)
	}
}
impl From<Error> for UnsafeImageCreationError {
	fn from(err: Error) -> UnsafeImageCreationError {
		match err {
			err @ Error::OutOfHostMemory => UnsafeImageCreationError::AllocError(err.into()),
			err @ Error::OutOfDeviceMemory => UnsafeImageCreationError::AllocError(err.into()),
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}
