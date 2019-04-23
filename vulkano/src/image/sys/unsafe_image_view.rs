use std::{error, fmt, mem, ops::Range, ptr, sync::Arc};

use vk_sys as vk;

use crate::{
	check_errors,
	device::Device,
	format::{Format, FormatTy},
	image::{ImageDimensions, ImageSubresourceRange, ImageUsage, ImageViewType, Swizzle},
	OomError,
	VulkanObject
};

use super::UnsafeImage;

pub struct UnsafeImageView {
	device: Arc<Device>,

	view: vk::ImageView,

	usage: ImageUsage,
	format: Format,

	dimensions: ImageDimensions,
	subresource_range: ImageSubresourceRange,
	swizzle: Swizzle
}
impl UnsafeImageView {
	/// Creates a new view from an image.
	///
	/// Note that you must create the view with identity swizzling if you want to use this view
	/// as a framebuffer attachment.
	pub unsafe fn new(
		image: &UnsafeImage, view_type: Option<ImageViewType>, format: Option<Format>,
		swizzle: Swizzle, subresource_range: ImageSubresourceRange
	) -> Result<UnsafeImageView, UnsafeImageViewCreationError> {
		let vk = image.device().pointers();

		if subresource_range.array_layers_end().get() > image.dimensions().array_layers().get() {
			return Err(UnsafeImageViewCreationError::ArrayLayersOutOfRange {
				requested: subresource_range.array_layers_range(),
				array_layers: image.dimensions().array_layers().get()
			})
		}
		if subresource_range.mipmap_levels_end().get() > image.mipmap_levels().get() {
			return Err(UnsafeImageViewCreationError::MipmapLevelsOutOfRange {
				requested: subresource_range.mipmap_levels_range(),
				mipmap_levels: image.mipmap_levels().get()
			})
		}

		// TODO: Views can have different formats than their underlying images, but
		// only if certain requirements are met. We need to check those before we
		// allow creating views with different formats.
		let view_format = image.format();

		let aspect_mask = match view_format.ty() {
			FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
				vk::IMAGE_ASPECT_COLOR_BIT
			}
			FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
			FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
			FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
		};

		let (view_type, view_type_flag) = {
			let image_view_type = ImageViewType::from(image.dimensions());

			let view_type = if let Some(view_type) = view_type {
				if !view_type.compatible_with(image_view_type) {
					return Err(UnsafeImageViewCreationError::ImageViewNotCompatible {
						requested: view_type,
						actual: image_view_type
					})
				}
				if subresource_range.array_layers.get() > 1 {
					if !view_type.is_array() {
						return Err(UnsafeImageViewCreationError::ViewNotArrayType {
							view_type,
							array_layers: subresource_range.array_layers.get()
						})
					}
					if !image_view_type.is_array() {
						return Err(UnsafeImageViewCreationError::ImageNotArrayType {
							view_type: image_view_type,
							array_layers: subresource_range.array_layers.get()
						})
					}
				}
				if (view_type == ImageViewType::Cubemap
					&& subresource_range.array_layers.get() != 6)
					|| (view_type == ImageViewType::CubemapArray
						&& subresource_range.array_layers.get() % 6 != 0)
				{
					return Err(UnsafeImageViewCreationError::ArrayLayersCubemapMismatch {
						array_layers: subresource_range.array_layers.get()
					})
				}

				view_type
			} else {
				image_view_type
			};

			let view_type_flag = match view_type {
				ImageViewType::Dim1D => vk::IMAGE_VIEW_TYPE_1D,
				ImageViewType::Dim1DArray => vk::IMAGE_VIEW_TYPE_1D_ARRAY,

				ImageViewType::Dim2D => vk::IMAGE_VIEW_TYPE_2D,
				ImageViewType::Dim2DArray => vk::IMAGE_VIEW_TYPE_2D_ARRAY,

				ImageViewType::Cubemap => vk::IMAGE_VIEW_TYPE_CUBE,
				ImageViewType::CubemapArray => vk::IMAGE_VIEW_TYPE_CUBE_ARRAY,

				ImageViewType::Dim3D => vk::IMAGE_VIEW_TYPE_CUBE_ARRAY
			};

			(view_type, view_type_flag)
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
			match check_errors(vk.CreateImageView(
				image.device().internal_object(),
				&infos,
				ptr::null(),
				&mut output
			)) {
				Err(e) => Err(OomError::from(e))?,
				Ok(_) => ()
			};
			output
		};

		let dimensions = match view_type {
			ImageViewType::Dim1D => ImageDimensions::Dim1D { width: image.dimensions().width() },
			ImageViewType::Dim1DArray => ImageDimensions::Dim1DArray {
				width: image.dimensions().width(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Dim2D => ImageDimensions::Dim2D {
				width: image.dimensions().width(),
				height: image.dimensions().height()
			},
			ImageViewType::Dim2DArray => ImageDimensions::Dim2DArray {
				width: image.dimensions().width(),
				height: image.dimensions().height(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Cubemap => ImageDimensions::Cubemap { size: image.dimensions().width() },
			ImageViewType::CubemapArray => ImageDimensions::CubemapArray {
				size: image.dimensions().width(),
				array_layers: subresource_range.array_layers
			},

			ImageViewType::Dim3D => ImageDimensions::Dim3D {
				width: image.dimensions().width(),
				height: image.dimensions().height(),
				depth: image.dimensions().depth()
			}
		};

		Ok(UnsafeImageView {
			device: image.device().clone(),
			view,

			usage: image.usage(),
			format: view_format,

			dimensions,
			subresource_range,

			swizzle
		})
	}

	/// Usage getter.
	pub fn usage(&self) -> ImageUsage { self.usage }

	/// Format getter.
	pub fn format(&self) -> Format { self.format }

	/// Dimensions getter.
	pub fn dimensions(&self) -> ImageDimensions { self.dimensions }

	/// Subresource range getter.
	pub fn subresource_range(&self) -> ImageSubresourceRange { self.subresource_range }

	/// Swizzle getter.
	pub fn swizzle(&self) -> Swizzle { self.swizzle }
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


#[derive(Debug)]
pub enum UnsafeImageViewCreationError {
	OomError(OomError),

	/// The requested image view type is not compatible with the image view type.
	ImageViewNotCompatible {
		requested: ImageViewType,
		actual: ImageViewType
	},

	/// The requested range of mipmap levels is out of range of actual range of mipmap levels.
	MipmapLevelsOutOfRange {
		requested: Range<u32>,
		mipmap_levels: u32
	},

	/// Requested view type is not array type, but requested multiple array layers.
	ViewNotArrayType {
		view_type: ImageViewType,
		array_layers: u32
	},

	/// Requested image is not array type, but requested multiple array layers.
	ImageNotArrayType {
		view_type: ImageViewType,
		array_layers: u32
	},

	/// The requested range of array layers is out of range of actual range of array layers.
	ArrayLayersOutOfRange {
		requested: Range<u32>,
		array_layers: u32
	},

	/// The requested number of array layers is not compatible with cubemap view type (must be a multiple of 6).
	ArrayLayersCubemapMismatch {
		array_layers: u32
	}
}
impl fmt::Display for UnsafeImageViewCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			UnsafeImageViewCreationError::OomError(e) => e.fmt(f),

			UnsafeImageViewCreationError::ImageViewNotCompatible { requested, actual }
			=> write!(
				f,
				"The requested image view type ({:?}) is not compatible with the image view type ({:?})",
				requested, actual
			),

			UnsafeImageViewCreationError::MipmapLevelsOutOfRange { requested, mipmap_levels }
			=> write!(
				f,
				"The requested range of mipmap levels ({:?}) is out of range of actual range of mipmap levels (0 .. {})",
				requested, mipmap_levels
			),

			UnsafeImageViewCreationError::ViewNotArrayType { view_type, array_layers }
			=> write!(
				f,
				"Requested view type ({:?}) is not array type, but requested multiple array layers ({})",
				view_type, array_layers
			),

			UnsafeImageViewCreationError::ImageNotArrayType { view_type, array_layers }
			=> write!(
				f,
				"Requested image type ({:?}) is not array type, but requested multiple array layers ({})",
				view_type, array_layers
			),

			UnsafeImageViewCreationError::ArrayLayersOutOfRange { requested, array_layers }
			=> write!(
				f,
				"The requested range of array layers ({:?}) is out of range of actual range of array layers (0 .. {})",
				requested, array_layers
			),

			UnsafeImageViewCreationError::ArrayLayersCubemapMismatch { array_layers }
			=> write!(
				f,
				"The requested number of array ({}) layers is not compatible with cubemap view type (must be a multiple of 6)",
				array_layers
			)
		}
	}
}
impl error::Error for UnsafeImageViewCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			UnsafeImageViewCreationError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for UnsafeImageViewCreationError {
	fn from(err: OomError) -> Self { UnsafeImageViewCreationError::OomError(err) }
}
