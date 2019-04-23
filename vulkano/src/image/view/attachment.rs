use std::{error, fmt};

use crate::{
	format::{FormatDesc, FormatTy},
	image::{
		layout::{
			ImageLayoutCombinedImage,
			ImageLayoutEnd,
			ImageLayoutInputAttachment,
			ImageLayoutSampledImage,
			ImageLayoutStorageImage,
			RequiredLayouts
		},
		ImageAccess,
		ImageSubresourceRange,
		ImageViewCreationError,
		Swizzle
	}
};

use super::ImageView;

impl<I: ImageAccess> ImageView<I> {
	/// Create a new attachment view for the image.
	///
	/// This is a shortcut for the `new` function, but it also checks
	/// parameters to ensure that if you get a view, it you will be able
	/// to use it as an attachment.
	///
	/// TODO: `format` parameter currently doesn't do anything, the image format is used instead.
	pub fn new_attachment<F: FormatDesc>(
		image: I, format: Option<F>, subresource_range: ImageSubresourceRange
	) -> Result<ImageView<I>, AttachmentImageViewCreationError> {
		let is_depth_or_stencil = match image.format().ty() {
			FormatTy::Depth => true,
			FormatTy::DepthStencil => true,
			FormatTy::Stencil => true,
			FormatTy::Compressed => {
				return Err(AttachmentImageViewCreationError::CompressedFormatError)
			}
			_ => false
		};

		macro_rules! check_required_layout {
			($thing: expr) => {
				if $thing.valid_for_usage(image.usage()).is_ok() {
					Some($thing)
				} else {
					None
					}
			};
		}

		let required_layouts = {
			if is_depth_or_stencil {
				RequiredLayouts {
					global: Some(ImageLayoutEnd::DepthStencilAttachmentOptimal),
					storage: check_required_layout!(ImageLayoutStorageImage::General),
					sampled: check_required_layout!(
						ImageLayoutSampledImage::DepthStencilReadOnlyOptimal
					),
					combined: check_required_layout!(
						ImageLayoutCombinedImage::DepthStencilReadOnlyOptimal
					),
					input_attachment: check_required_layout!(
						ImageLayoutInputAttachment::DepthStencilReadOnlyOptimal
					)
				}
			} else {
				RequiredLayouts {
					global: Some(ImageLayoutEnd::ColorAttachmentOptimal),
					storage: check_required_layout!(ImageLayoutStorageImage::General),
					sampled: check_required_layout!(ImageLayoutSampledImage::ShaderReadOnlyOptimal),
					combined: check_required_layout!(
						ImageLayoutCombinedImage::ShaderReadOnlyOptimal
					),
					input_attachment: check_required_layout!(
						ImageLayoutInputAttachment::ShaderReadOnlyOptimal
					)
				}
			}
		};

		Ok(ImageView::new(
			image,
			None,
			format,
			Swizzle::identity(),
			subresource_range,
			required_layouts
		)?)
	}
}

#[derive(Debug)]
pub enum AttachmentImageViewCreationError {
	Base(ImageViewCreationError),

	/// Attachment image view cannot be in compressed format.
	CompressedFormatError
}
impl fmt::Display for AttachmentImageViewCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			AttachmentImageViewCreationError::Base(e) => e.fmt(f),

			AttachmentImageViewCreationError::CompressedFormatError => {
				write!(f, "Attachment image view cannot be in compressed format")
			}
		}
	}
}
impl error::Error for AttachmentImageViewCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			AttachmentImageViewCreationError::Base(e) => e.source(),
			_ => None
		}
	}
}
impl From<ImageViewCreationError> for AttachmentImageViewCreationError {
	fn from(err: ImageViewCreationError) -> Self { AttachmentImageViewCreationError::Base(err) }
}
