use smallvec::SmallVec;
use std::{error, fmt, mem, ops::Range, ptr, sync::Arc};

use vk_sys as vk;

use crate::{
	check_errors,
	device::Device,
	format::{Format, FormatTy},
	image::{ImageDimensions, ImageUsage, MipmapsCount},
	memory::{DeviceMemory, DeviceMemoryAllocError, MemoryRequirements},
	sync::Sharing,
	Error,
	OomError,
	VulkanObject
};

/// A storage for pixels or arbitrary data.
///
/// # Safety
///
/// This type is not just unsafe but very unsafe. Don't use it directly.
///
/// - You must manually bind memory to the image with `bind_memory`. The memory must respect the
///   requirements returned by `new`.
/// - The memory that you bind to the image must be manually kept alive.
/// - The queue family ownership must be manually enforced.
/// - The usage must be manually enforced.
/// - The image layout must be manually enforced and transitioned.
pub struct UnsafeImage {
	pub(in crate::image) device: Arc<Device>,

	image: vk::Image,
	pub(in crate::image) usage: vk::ImageUsageFlagBits,

	pub(in crate::image) format: Format,
	// Features that are supported for this particular format.
	format_features: vk::FormatFeatureFlagBits,

	pub(in crate::image) dimensions: ImageDimensions,
	pub(in crate::image) samples: u32,
	pub(in crate::image) mipmap_levels: u32,

	// `vkDestroyImage` is called only if `needs_destruction` is true.
	needs_destruction: bool
}

impl UnsafeImage {
	/// Creates a new image and allocates memory for it.
	///
	/// # Panic
	///
	/// - Panics if one of the dimensions is 0.
	/// - Panics if the number of mipmaps is 0.
	/// - Panics if the number of samples is 0.
	pub unsafe fn new<'a, Mi, I>(
		device: Arc<Device>, usage: ImageUsage, format: Format, dimensions: ImageDimensions,
		num_samples: u32, mipmaps: Mi, sharing: Sharing<I>, linear_tiling: bool,
		preinitialized_layout: bool
	) -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError>
	where
		Mi: Into<MipmapsCount>,
		I: Iterator<Item = u32>
	{
		let sharing = match sharing {
			Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
			Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect())
		};

		UnsafeImage::new_impl(
			device,
			usage,
			format,
			dimensions,
			num_samples,
			mipmaps.into(),
			sharing,
			linear_tiling,
			preinitialized_layout
		)
	}

	// Non-templated version to avoid inlining and improve compile times.
	// TODO: Does it really?
	unsafe fn new_impl(
		device: Arc<Device>, usage: ImageUsage, format: Format, dimensions: ImageDimensions,
		num_samples: u32, mipmaps: MipmapsCount,
		(sh_mode, sh_indices): (vk::SharingMode, SmallVec<[u32; 8]>), linear_tiling: bool,
		preinitialized_layout: bool
	) -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError> {
		// TODO: doesn't check that the proper features are enabled

		let vk = device.pointers();
		let vk_i = device.instance().pointers();

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
				return Err(ImageCreationError::FormatNotSupported)
			}

			if usage.sampled && (features & vk::FORMAT_FEATURE_SAMPLED_IMAGE_BIT == 0) {
				return Err(ImageCreationError::UnsupportedUsage)
			}
			if usage.storage && (features & vk::FORMAT_FEATURE_STORAGE_IMAGE_BIT == 0) {
				return Err(ImageCreationError::UnsupportedUsage)
			}
			if usage.color_attachment && (features & vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT == 0) {
				return Err(ImageCreationError::UnsupportedUsage)
			}
			if usage.depth_stencil_attachment
				&& (features & vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT == 0)
			{
				return Err(ImageCreationError::UnsupportedUsage)
			}
			if usage.input_attachment
				&& (features
					& (vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
						| vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
					== 0)
			{
				return Err(ImageCreationError::UnsupportedUsage)
			}
			if device.loaded_extensions().khr_maintenance1 {
				if usage.transfer_source
					&& (features & vk::FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR == 0)
				{
					return Err(ImageCreationError::UnsupportedUsage)
				}
				if usage.transfer_destination
					&& (features & vk::FORMAT_FEATURE_TRANSFER_DST_BIT_KHR == 0)
				{
					return Err(ImageCreationError::UnsupportedUsage)
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

			if u != ImageUsage::none() {
				return Err(ImageCreationError::UnsupportedUsage)
			}
		}

		// This function is going to perform various checks and write to `capabilities_error` in
		// case of error.
		//
		// If `capabilities_error` is not `None` after the checks are finished, the function will
		// check for additional image capabilities (section 31.4 of the specs).
		let mut capabilities_error = None;

		// Compute the number of mipmaps.
		let mipmap_levels = match mipmaps.into() {
			MipmapsCount::Specific(num) => {
				let max_mipmaps = dimensions.max_mipmaps();
				debug_assert!(max_mipmaps >= 1);
				if num < 1 {
					return Err(ImageCreationError::InvalidMipmapsCount {
						obtained: num,
						valid_range: 1 .. max_mipmaps + 1
					})
				} else if num > max_mipmaps {
					capabilities_error = Some(ImageCreationError::InvalidMipmapsCount {
						obtained: num,
						valid_range: 1 .. max_mipmaps + 1
					});
				}

				num
			}
			MipmapsCount::Log2 => dimensions.max_mipmaps(),
			MipmapsCount::One => 1
		};

		// Checking whether the number of samples is supported.
		if num_samples == 0 {
			return Err(ImageCreationError::UnsupportedSamplesCount(num_samples))
		} else if !num_samples.is_power_of_two() {
			return Err(ImageCreationError::UnsupportedSamplesCount(num_samples))
		} else {
			let mut supported_samples = 0x7f; // all bits up to VK_SAMPLE_COUNT_64_BIT

			if usage.sampled {
				match format.ty() {
					FormatTy::Float | FormatTy::Compressed => {
						supported_samples &=
							device.physical_device().limits().sampled_image_color_sample_counts();
					}
					FormatTy::Uint | FormatTy::Sint => {
						supported_samples &=
							device.physical_device().limits().sampled_image_integer_sample_counts();
					}
					FormatTy::Depth => {
						supported_samples &=
							device.physical_device().limits().sampled_image_depth_sample_counts();
					}
					FormatTy::Stencil => {
						supported_samples &=
							device.physical_device().limits().sampled_image_stencil_sample_counts();
					}
					FormatTy::DepthStencil => {
						supported_samples &=
							device.physical_device().limits().sampled_image_depth_sample_counts();
						supported_samples &=
							device.physical_device().limits().sampled_image_stencil_sample_counts();
					}
				}
			}

			if usage.storage {
				supported_samples &=
					device.physical_device().limits().storage_image_sample_counts();
			}

			if usage.color_attachment
				|| usage.depth_stencil_attachment
				|| usage.input_attachment
				|| usage.transient_attachment
			{
				match format.ty() {
					FormatTy::Float | FormatTy::Compressed | FormatTy::Uint | FormatTy::Sint => {
						supported_samples &=
							device.physical_device().limits().framebuffer_color_sample_counts();
					}
					FormatTy::Depth => {
						supported_samples &=
							device.physical_device().limits().framebuffer_depth_sample_counts();
					}
					FormatTy::Stencil => {
						supported_samples &=
							device.physical_device().limits().framebuffer_stencil_sample_counts();
					}
					FormatTy::DepthStencil => {
						supported_samples &=
							device.physical_device().limits().framebuffer_depth_sample_counts();
						supported_samples &=
							device.physical_device().limits().framebuffer_stencil_sample_counts();
					}
				}
			}

			if (num_samples & supported_samples) == 0 {
				let err = ImageCreationError::UnsupportedSamplesCount(num_samples);
				capabilities_error = Some(err);
			}
		}

		// If the `shaderStorageImageMultisample` feature is not enabled and we have
		// `usage_storage` set to true, then the number of samples must be 1.
		if usage.storage && num_samples > 1 {
			if !device.enabled_features().shader_storage_image_multisample {
				return Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled)
			}
		}

		// Decoding the dimensions.
		let (ty, extent, array_layers, flags) = match dimensions {
			ImageDimensions::Dim1D { .. } | ImageDimensions::Dim1DArray { .. } => {
				if dimensions.width() == 0 || dimensions.array_layers() == 0 {
					return Err(ImageCreationError::UnsupportedDimensions(dimensions))
				}
				let extent = vk::Extent3D { width: dimensions.width(), height: 1, depth: 1 };
				(vk::IMAGE_TYPE_1D, extent, dimensions.array_layers(), 0)
			}
			ImageDimensions::Dim2D { .. } | ImageDimensions::Dim2DArray { .. } => {
				if dimensions.width() == 0
					|| dimensions.height() == 0
					|| dimensions.array_layers() == 0
				{
					return Err(ImageCreationError::UnsupportedDimensions(dimensions))
				}
				let extent = vk::Extent3D {
					width: dimensions.width(),
					height: dimensions.height(),
					depth: 1
				};
				(vk::IMAGE_TYPE_2D, extent, dimensions.array_layers(), 0)
			}
			ImageDimensions::Cubemap { .. } | ImageDimensions::CubemapArray { .. } => {
				if dimensions.width() == 0 || dimensions.array_layers() == 0 {
					return Err(ImageCreationError::UnsupportedDimensions(dimensions))
				}
				let extent = vk::Extent3D {
					width: dimensions.width(),
					height: dimensions.width(),
					depth: 1
				};
				(
					vk::IMAGE_TYPE_2D,
					extent,
					dimensions.array_layers(),
					vk::IMAGE_CREATE_CUBE_COMPATIBLE_BIT
				)
			}
			ImageDimensions::Dim3D { width, height, depth } => {
				if width == 0 || height == 0 || depth == 0 {
					return Err(ImageCreationError::UnsupportedDimensions(dimensions))
				}

				let extent = vk::Extent3D { width, height, depth };
				(vk::IMAGE_TYPE_3D, extent, 1, 0)
			}
		};

		// Checking the dimensions against the limits.
		if array_layers > device.physical_device().limits().max_image_array_layers() {
			let err = ImageCreationError::UnsupportedDimensions(dimensions);
			capabilities_error = Some(err);
		}
		match ty {
			vk::IMAGE_TYPE_1D => {
				if extent.width > device.physical_device().limits().max_image_dimension_1d() {
					let err = ImageCreationError::UnsupportedDimensions(dimensions);
					capabilities_error = Some(err);
				}
			}
			vk::IMAGE_TYPE_2D => {
				let limit = device.physical_device().limits().max_image_dimension_2d();
				if extent.width > limit || extent.height > limit {
					let err = ImageCreationError::UnsupportedDimensions(dimensions);
					capabilities_error = Some(err);
				}

				if (flags & vk::IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0 {
					let limit = device.physical_device().limits().max_image_dimension_cube();
					debug_assert_eq!(extent.width, extent.height); // checked above
					if extent.width > limit {
						let err = ImageCreationError::UnsupportedDimensions(dimensions);
						capabilities_error = Some(err);
					}
				}
			}
			vk::IMAGE_TYPE_3D => {
				let limit = device.physical_device().limits().max_image_dimension_3d();
				if extent.width > limit || extent.height > limit || extent.depth > limit {
					let err = ImageCreationError::UnsupportedDimensions(dimensions);
					capabilities_error = Some(err);
				}
			}
			_ => unreachable!()
		};

		let usage = usage.to_usage_bits();

		// Now that all checks have been performed, if any of the check failed we query the Vulkan
		// implementation for additional image capabilities.
		if let Some(capabilities_error) = capabilities_error {
			let tiling =
				if linear_tiling { vk::IMAGE_TILING_LINEAR } else { vk::IMAGE_TILING_OPTIMAL };

			let mut output = mem::uninitialized();
			let physical_device = device.physical_device().internal_object();
			let r = vk_i.GetPhysicalDeviceImageFormatProperties(
				physical_device,
				format as u32,
				ty,
				tiling,
				usage,
				0, // TODO
				&mut output
			);

			match check_errors(r) {
				Ok(_) => (),
				Err(Error::FormatNotSupported) => {
					return Err(ImageCreationError::FormatNotSupported)
				}
				Err(err) => return Err(err.into())
			}

			if extent.width > output.maxExtent.width
				|| extent.height > output.maxExtent.height
				|| extent.depth > output.maxExtent.depth
				|| mipmap_levels > output.maxMipLevels
				|| array_layers > output.maxArrayLayers
				|| (num_samples & output.sampleCounts) == 0
			{
				return Err(capabilities_error)
			}
		}

		// Everything now ok. Creating the image.
		let image = {
			let infos = vk::ImageCreateInfo {
				sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				pNext: ptr::null(),
				flags,
				imageType: ty,
				format: format as u32,
				extent,
				mipLevels: mipmap_levels,
				arrayLayers: array_layers,
				samples: num_samples,
				tiling: if linear_tiling {
					vk::IMAGE_TILING_LINEAR
				} else {
					vk::IMAGE_TILING_OPTIMAL
				},
				usage,
				sharingMode: sh_mode,
				queueFamilyIndexCount: sh_indices.len() as u32,
				pQueueFamilyIndices: sh_indices.as_ptr(),
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
			samples: num_samples,
			mipmap_levels,
			format_features,
			needs_destruction: true
		};

		Ok((image, mem_reqs))
	}

	/// Creates an image from a raw handle. The image won't be destroyed.
	///
	/// This function is for example used at the swapchain's initialization.
	pub unsafe fn from_raw(
		device: Arc<Device>, handle: u64, usage: u32, format: Format, dimensions: ImageDimensions,
		samples: u32, mipmap_levels: u32
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

		assert!(mip_level < self.mipmap_levels);

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

	pub fn usage_transfer_source(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0
	}

	pub fn usage_transfer_destination(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0
	}

	pub fn usage_sampled(&self) -> bool { (self.usage & vk::IMAGE_USAGE_SAMPLED_BIT) != 0 }

	pub fn usage_storage(&self) -> bool { (self.usage & vk::IMAGE_USAGE_STORAGE_BIT) != 0 }

	pub fn usage_color_attachment(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0
	}

	pub fn usage_depth_stencil_attachment(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
	}

	pub fn usage_transient_attachment(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0
	}

	pub fn usage_input_attachment(&self) -> bool {
		(self.usage & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0
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
pub enum ImageCreationError {
	/// Allocating memory failed.
	AllocError(DeviceMemoryAllocError),
	/// A wrong number of mipmaps was provided.
	InvalidMipmapsCount { obtained: u32, valid_range: Range<u32> },
	/// The requested number of samples is not supported, or is 0.
	UnsupportedSamplesCount(u32),
	/// The dimensions are too large, or one of the dimensions is 0.
	UnsupportedDimensions(ImageDimensions),
	/// The requested format is not supported by the Vulkan implementation.
	FormatNotSupported,
	/// The format is supported, but at least one of the requested usages is not supported.
	UnsupportedUsage,
	/// The `shader_storage_image_multisample` feature must be enabled to create such an image.
	ShaderStorageImageMultisampleFeatureNotEnabled
}
impl fmt::Display for ImageCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ImageCreationError::AllocError(e) => write!(f, "Memory allocation failed: {}", e),
			ImageCreationError::InvalidMipmapsCount { obtained, valid_range } => write!(
				f,
				"A wrong number of mipmaps provided: {} valid range: {:?}",
				obtained, valid_range
			),
			ImageCreationError::UnsupportedSamplesCount(samples) => {
				write!(f, "The requested number of sampler is not supported: {}", samples)
			}
			ImageCreationError::UnsupportedDimensions(dims) => {
				write!(f, "The requested dimensions are not supported: {:?}", dims)
			}
			ImageCreationError::FormatNotSupported => {
				write!(f, "The requested format is not supported")
			}
			ImageCreationError::UnsupportedUsage => {
				write!(f, "The requested usage is not supported for requested format")
			}
			ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
				write!(f, "The `shader_storage_image_multisample` feature must be enabled")
			}
		}
	}
}
impl error::Error for ImageCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			ImageCreationError::AllocError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for ImageCreationError {
	fn from(err: OomError) -> ImageCreationError {
		ImageCreationError::AllocError(DeviceMemoryAllocError::OomError(err))
	}
}
impl From<DeviceMemoryAllocError> for ImageCreationError {
	fn from(err: DeviceMemoryAllocError) -> ImageCreationError {
		ImageCreationError::AllocError(err)
	}
}
impl From<Error> for ImageCreationError {
	fn from(err: Error) -> ImageCreationError {
		match err {
			err @ Error::OutOfHostMemory => ImageCreationError::AllocError(err.into()),
			err @ Error::OutOfDeviceMemory => ImageCreationError::AllocError(err.into()),
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}

/// Describes the memory layout of an image with linear tiling.
///
/// Obtained by calling `*_linear_layout` on the image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LinearLayout {
	/// Number of bytes from the start of the memory and the start of the queried subresource.
	pub offset: usize,
	/// Total number of bytes for the queried subresource. Can be used for a safety check.
	pub size: usize,
	/// Number of bytes between two texels or two blocks in adjacent rows.
	pub row_pitch: usize,
	/// Number of bytes between two texels or two blocks in adjacent array layers. This value is
	/// undefined for images with only one array layer.
	pub array_pitch: usize,
	/// Number of bytes between two texels or two blocks in adjacent depth layers. This value is
	/// undefined for images that are not three-dimensional.
	pub depth_pitch: usize
}
