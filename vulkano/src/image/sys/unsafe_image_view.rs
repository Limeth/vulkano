use std::{fmt, mem, ops::Range, ptr, sync::Arc};

use vk_sys as vk;

use crate::{
	check_errors,
	device::Device,
	format::{Format, FormatTy},
	image::ImageViewType,
	OomError,
	VulkanObject
};

use super::UnsafeImage;

pub struct UnsafeImageView {
	view: vk::ImageView,
	device: Arc<Device>,
	usage: vk::ImageUsageFlagBits,
	identity_swizzle: bool,
	format: Format
}

impl UnsafeImageView {
	/// See the docs of new().
	pub unsafe fn raw(
		image: &UnsafeImage, view_type: ImageViewType, mipmap_levels: Range<u32>,
		array_layers: Range<u32>
	) -> Result<UnsafeImageView, OomError> {
		let vk = image.device.pointers();

		assert!(mipmap_levels.end > mipmap_levels.start);
		assert!(mipmap_levels.end <= image.mipmaps);
		assert!(array_layers.end > array_layers.start);
		assert!(array_layers.end <= image.dimensions.array_layers());

		let aspect_mask = match image.format.ty() {
			FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
				vk::IMAGE_ASPECT_COLOR_BIT
			}
			FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
			FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
			FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
		};

		let view_type = {
			let image_view_type = ImageViewType::from(image.dimensions);
			let layer_count = array_layers.end - array_layers.start;

			if !view_type.compatible_with(image_view_type) {
				panic!(
					"Cannot create an image view with type {:?} into an image of type {:?}",
					view_type, image_view_type
				);
			}
			if layer_count > 1 && (!view_type.is_array() || !image_view_type.is_array()) {
				panic!("Cannot create an array image view with type {:?} and {} layers into an image of type {:?}", view_type, layer_count, image_view_type);
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
				viewType: view_type,
				format: image.format as u32,
				components: vk::ComponentMapping { r: 0, g: 0, b: 0, a: 0 }, // FIXME:
				subresourceRange: vk::ImageSubresourceRange {
					aspectMask: aspect_mask,
					baseMipLevel: mipmap_levels.start,
					levelCount: mipmap_levels.end - mipmap_levels.start,
					baseArrayLayer: array_layers.start,
					layerCount: array_layers.end - array_layers.start
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

		Ok(UnsafeImageView {
			view,
			device: image.device.clone(),
			usage: image.usage,
			identity_swizzle: true, // FIXME:
			format: image.format
		})
	}

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
	/// - Panics if trying to create a cubemap array with a number of array layers not a multiple
	///   of 6.
	/// - Panics if the device or host ran out of memory.
	pub unsafe fn new(
		image: &UnsafeImage, ty: ImageViewType, mipmap_levels: Range<u32>, array_layers: Range<u32>
	) -> UnsafeImageView {
		UnsafeImageView::raw(image, ty, mipmap_levels, array_layers).unwrap()
	}

	pub fn format(&self) -> Format { self.format }

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
