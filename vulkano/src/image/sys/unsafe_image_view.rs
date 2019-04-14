use std::{fmt, mem, ptr, sync::Arc};

use vk_sys as vk;

use crate::{
	check_errors,
	device::Device,
	format::{Format, FormatTy},
	image::{ImageDimensions, ImageSubresourceRange, ImageViewType, Swizzle},
	OomError,
	VulkanObject
};

use super::UnsafeImage;

pub struct UnsafeImageView {
	device: Arc<Device>,

	view: vk::ImageView,
	pub(in crate::image) format: Format,

	pub(in crate::image) dimensions: ImageDimensions,
	pub(in crate::image) subresource_range: ImageSubresourceRange,

	usage: vk::ImageUsageFlagBits,
	pub(in crate::image) swizzle: Swizzle
}

impl UnsafeImageView {
	/// Creates a new view from an image.
	///
	/// Note that you must create the view with identity swizzling if you want to use this view
	/// as a framebuffer attachment.
	///
	/// # Panic
	///
	/// - Panics if `mipmap_levels` or `array_layers` is out of range of the image.
	/// - Panics if the view types doesn't match the dimensions of the image (for example a 2D
	///   view from a 3D image).
	/// - Panics if trying to create a cubemap with a number of array layers different from 6.
	/// - Panics if trying to create a cubemap array with a number of array layers not a multiple of 6.
	pub unsafe fn new(
		image: &UnsafeImage, view_type: ImageViewType, format: Option<Format>, swizzle: Swizzle,
		subresource_range: ImageSubresourceRange
	) -> Result<UnsafeImageView, OomError> {
		let vk = image.device.pointers();

		assert!(subresource_range.array_layers_end().get() <= image.dimensions.array_layers().get());
		assert!(subresource_range.mipmap_levels_end().get() <= image.mipmap_levels.get());

		// TODO: Views can have different formats than their underlying images, but
		// only if certain requirements are met. We need to check those before we
		// allow creating views with different formats.
		let view_format = image.format;

		let aspect_mask = match view_format.ty() {
			FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
				vk::IMAGE_ASPECT_COLOR_BIT
			}
			FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
			FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
			FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
		};

		let view_type_flag = {
			let image_view_type = ImageViewType::from(image.dimensions);

			if !view_type.compatible_with(image_view_type) {
				panic!(
					"Cannot create an image view with type {:?} into an image of type {:?}",
					view_type, image_view_type
				);
			}
			if subresource_range.array_layers.get() > 1
				&& (!view_type.is_array() || !image_view_type.is_array())
			{
				panic!(
					"Cannot create an array image view with type {:?} and {} layers into an image of type {:?}",
					view_type,  subresource_range.array_layers, image_view_type
				);
			}

			match view_type {
				ImageViewType::Dim1D => vk::IMAGE_VIEW_TYPE_1D,
				ImageViewType::Dim1DArray => vk::IMAGE_VIEW_TYPE_1D_ARRAY,

				ImageViewType::Dim2D => vk::IMAGE_VIEW_TYPE_2D,
				ImageViewType::Dim2DArray => vk::IMAGE_VIEW_TYPE_2D_ARRAY,

				ImageViewType::Cubemap => vk::IMAGE_VIEW_TYPE_CUBE,
				ImageViewType::CubemapArray => vk::IMAGE_VIEW_TYPE_CUBE_ARRAY,

				ImageViewType::Dim3D => vk::IMAGE_VIEW_TYPE_CUBE_ARRAY
			}
		};

		let view = {
			let infos = vk::ImageViewCreateInfo {
				sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				pNext: ptr::null(),
				flags: 0, // reserved
				image: image.internal_object(),
				viewType: view_type_flag,
				format: view_format as u32,
				components: swizzle.into(),
				subresourceRange: vk::ImageSubresourceRange {
					aspectMask: aspect_mask,
					baseArrayLayer: subresource_range.array_layers_offset,
					layerCount: subresource_range.array_layers.get(),
					baseMipLevel: subresource_range.mipmap_levels_offset,
					levelCount: subresource_range.mipmap_levels.get()
				}
			};

			let mut output = mem::uninitialized();
			check_errors(vk.CreateImageView(
				image.device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			output
		};

		let dimensions = match view_type {
			ImageViewType::Dim1D => ImageDimensions::Dim1D { width: image.dimensions.width() },
			ImageViewType::Dim1DArray => ImageDimensions::Dim1DArray {
				width: image.dimensions.width(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Dim2D => ImageDimensions::Dim2D {
				width: image.dimensions.width(),
				height: image.dimensions.height()
			},
			ImageViewType::Dim2DArray => ImageDimensions::Dim2DArray {
				width: image.dimensions.width(),
				height: image.dimensions.height(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Cubemap => ImageDimensions::Cubemap { size: image.dimensions.width() },
			ImageViewType::CubemapArray => ImageDimensions::CubemapArray {
				size: image.dimensions.width(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Dim3D => ImageDimensions::Dim3D {
				width: image.dimensions.width(),
				height: image.dimensions.height(),
				depth: image.dimensions.depth()
			}
		};

		Ok(UnsafeImageView {
			device: image.device.clone(),
			view,
			format: view_format,

			dimensions,
			subresource_range,

			usage: image.usage,
			swizzle
		})
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

unsafe impl VulkanObject for UnsafeImageView {
	type Object = vk::ImageView;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT;

	fn internal_object(&self) -> vk::ImageView { self.view }
}

impl fmt::Debug for UnsafeImageView {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(fmt, "<Vulkan image view {:?}>", self.view)
	}
}

impl Drop for UnsafeImageView {
	fn drop(&mut self) {
		unsafe {
			let vk = self.device.pointers();
			vk.DestroyImageView(self.device.internal_object(), self.view, ptr::null());
		}
	}
}
