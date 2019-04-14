use std::{error, fmt, ops::Range};

use crate::{
	buffer::BufferAccess,
	format::FormatDesc,
	image::{
		sys::UnsafeImageView,
		ImageAccess,
		ImageDimensions,
		ImageLayout,
		ImageSubresourceRange,
		ImageViewAccess,
		ImageViewType,
		Swizzle
	},
	OomError
};

/// Image view that holds a reference to the image to keep it alive.
pub struct ImageView<I: ImageAccess> {
	/// We need a reference to the image to keep it alive.
	image: I,

	/// The inner object.
	view: UnsafeImageView,

	/// The layout this view expected the subresource to be in.
	layout: ImageLayout
}
impl<I: ImageAccess> ImageView<I> {
	/// Creates a new view for the image.
	///
	/// TODO: `format` parameter currently doesn't do anything, the image format is used instead.
	pub fn new<F: FormatDesc>(
		image: I, view_type: ImageViewType, format: Option<F>, swizzle: Swizzle,
		subresource_range: ImageSubresourceRange
	) -> Result<ImageView<I>, ImageViewCreationError> {
		let image_dimensions = image.dimensions();
		let image_view_type = ImageViewType::from(image_dimensions);
		if !view_type.compatible_with(image_view_type) {
			return Err(ImageViewCreationError::ImageViewNotCompatible {
				requested: view_type,
				actual: image_view_type
			})
		}

		let image_array_layers = image_dimensions.array_layers();
		if subresource_range.array_layers_end().get() > image_array_layers.get() {
			return Err(ImageViewCreationError::ArrayLayersOutOfRange {
				requested: subresource_range.array_layers_range(),
				image_layers: image_array_layers.get()
			})
		}

		let image_mipmap_levels = image.mipmap_levels();
		if subresource_range.mipmap_levels_end().get() > image_mipmap_levels.get() {
			return Err(ImageViewCreationError::MipmapLevelsOutOfRange {
				requested: subresource_range.mipmap_levels_range(),
				image_levels: image_mipmap_levels.get()
			})
		}

		if subresource_range.array_layers.get() > 1 {
			if !view_type.is_array() {
				return Err(ImageViewCreationError::ViewNotArrayType {
					view_type,
					array_layers: subresource_range.array_layers.get()
				})
			}
			if !image_view_type.is_array() {
				return Err(ImageViewCreationError::ImageNotArrayType {
					view_type: image_view_type,
					array_layers: subresource_range.array_layers.get()
				})
			}
		}

		let view = unsafe {
			UnsafeImageView::new(
				image.inner(),
				view_type,
				format.map(|f| f.format()),
				swizzle,
				subresource_range
			)?
		};

		// TODO: Query Vulkan for view layout
		unimplemented!()

		// Ok(ImageView { image, view })
	}
}
unsafe impl<I: ImageAccess> ImageViewAccess for ImageView<I> {
	fn parent(&self) -> &dyn ImageAccess { &self.image }

	fn inner(&self) -> &UnsafeImageView { &self.view }

	fn dimensions(&self) -> ImageDimensions { self.view.dimensions }

	fn subresource_range(&self) -> ImageSubresourceRange { self.view.subresource_range }

	// TODO: Do we need these?
	fn descriptor_set_storage_image_layout(&self) -> ImageLayout { unimplemented!() }

	// TODO: Do we need these?
	fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout { unimplemented!() }

	// TODO: Do we need these?
	fn descriptor_set_sampled_image_layout(&self) -> ImageLayout { unimplemented!() }

	// TODO: Do we need these?
	fn descriptor_set_input_attachment_layout(&self) -> ImageLayout { unimplemented!() }

	fn identity_swizzle(&self) -> bool { self.view.swizzle.identity() }

	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool { false }

	fn current_layout(&self) -> Result<ImageLayout, ()> {
		self.parent().current_layout(self.subresource_range())
	}

	fn required_layout(&self) -> ImageLayout { ImageLayout::PresentSrc }
}

#[derive(Debug)]
pub enum ImageViewCreationError {
	OomError(OomError),

	/// The requested image view type is not compatible with the image view type.
	ImageViewNotCompatible {
		requested: ImageViewType,
		actual: ImageViewType
	},

	/// The requested range of mipmap levels is out of range of actual range of mipmap levels.
	MipmapLevelsOutOfRange {
		requested: Range<u32>,
		image_levels: u32
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
		image_layers: u32
	}
}
impl fmt::Display for ImageViewCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ImageViewCreationError::OomError(e) => e.fmt(f),

			ImageViewCreationError::ImageViewNotCompatible { requested, actual }
			=> write!(
				f,
				"The requested image view type ({:?}) is not compatible with the image view type ({:?})",
				requested, actual
			),

			ImageViewCreationError::MipmapLevelsOutOfRange { requested, image_levels }
			=> write!(
				f,
				"The requested range of mipmap levels {:?} is out of range of actual range of mipmap levels 0 .. {}",
				requested, image_levels
			),

			ImageViewCreationError::ViewNotArrayType { view_type, array_layers }
			=> write!(
				f,
				"Requested view type {:?} is not array type, but requested multiple array layers ({})",
				view_type, array_layers
			),

			ImageViewCreationError::ImageNotArrayType { view_type, array_layers }
			=> write!(
				f,
				"Requested image {:?} is not array type, but requested multiple array layers ({})",
				view_type, array_layers
			),

			ImageViewCreationError::ArrayLayersOutOfRange { requested, image_layers }
			=> write!(
				f,
				"The requested range of array layers {:?} is out of range of actual range of array layers 0 .. {}",
				requested, image_layers
			)
		}
	}
}
impl error::Error for ImageViewCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			ImageViewCreationError::OomError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for ImageViewCreationError {
	fn from(err: OomError) -> Self { ImageViewCreationError::OomError(err) }
}
