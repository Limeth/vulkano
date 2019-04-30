use std::{error, fmt};

use crate::{
	format::{FormatDesc, Format},
	image::{
		layout::InvalidLayoutUsageError,
		sys::{UnsafeImageView, UnsafeImageViewCreationError},
		ImageAccess,
		ImageSubresourceRange,
		ImageViewAccess,
		ImageViewType,
		RequiredLayouts,
		Swizzle
	}
};

mod attachment;
pub use attachment::AttachmentImageViewCreationError;

/// Image view that holds a reference to the image to keep it alive.
#[derive(Debug)]
pub struct ImageView<I: ImageAccess> {
	/// We need a reference to the image to keep it alive.
	image: I,

	/// The inner object.
	view: UnsafeImageView,

	/// The layouts this view requests to be in.
	required_layouts: RequiredLayouts
}
impl<I: ImageAccess> ImageView<I> {
	/// Creates a new view for the image.
	///
	/// TODO: `format` parameter currently doesn't do anything, the image format is used instead.
	pub fn new<F: FormatDesc>(
		image: I, view_type: Option<ImageViewType>, format: Option<F>, swizzle: Swizzle,
		subresource_range: Option<ImageSubresourceRange>, mut required_layouts: RequiredLayouts
	) -> Result<ImageView<I>, ImageViewCreationError> {
		required_layouts.valid_for_usage(image.usage())?;
		required_layouts.infer_mut(image.usage());

		let subresource_range = match subresource_range {
			Some(r) => r,
			None => ImageSubresourceRange::whole_image(&image)
		};

		let view = unsafe {
			UnsafeImageView::new(
				image.inner(),
				view_type,
				format.map(|f| f.format()),
				swizzle,
				subresource_range
			)?
		};

		Ok(ImageView { image, view, required_layouts })
	}

	/// Creates a new view covering the whole image.
	///
	/// This is equivalent to calling `new` with all parameters defaulted.
	pub fn whole_image(image: I) -> Result<ImageView<I>, ImageViewCreationError> {
		ImageView::new(image, None, None::<Format>, Default::default(), None, Default::default())
	}

	/// Borrows the inner image.
	pub fn borrow_image(&self) -> &I { &self.image }
}
unsafe impl<I: ImageAccess> ImageViewAccess for ImageView<I> {
	fn parent(&self) -> &dyn ImageAccess { &self.image }

	fn inner(&self) -> &UnsafeImageView { &self.view }

	fn required_layouts(&self) -> &RequiredLayouts { &self.required_layouts }
}

#[derive(Debug)]
pub enum ImageViewCreationError {
	Base(UnsafeImageViewCreationError),
	InvalidLayoutUsage(InvalidLayoutUsageError)
}
impl fmt::Display for ImageViewCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			ImageViewCreationError::Base(e) => e.fmt(f),
			ImageViewCreationError::InvalidLayoutUsage(e) => e.fmt(f)
		}
	}
}
impl error::Error for ImageViewCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			ImageViewCreationError::Base(ref e) => Some(e),
			ImageViewCreationError::InvalidLayoutUsage(ref e) => Some(e)
		}
	}
}
impl From<UnsafeImageViewCreationError> for ImageViewCreationError {
	fn from(err: UnsafeImageViewCreationError) -> Self { ImageViewCreationError::Base(err) }
}
impl From<InvalidLayoutUsageError> for ImageViewCreationError {
	fn from(err: InvalidLayoutUsageError) -> Self {
		ImageViewCreationError::InvalidLayoutUsage(err)
	}
}