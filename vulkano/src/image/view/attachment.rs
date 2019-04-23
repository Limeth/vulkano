use std::{error, fmt};

use crate::{
	format::{FormatDesc, FormatTy},
	image::{
		layout::{
			ImageLayoutCombinedImage,
			ImageLayoutEnd,
			ImageLayoutInputAttachment,
			ImageLayoutSampledImage,
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

		let (global_layout, sampled_layout, combined_layout, input_layout) = if is_depth_or_stencil
		{
			(
				ImageLayoutEnd::DepthStencilAttachmentOptimal,
				ImageLayoutSampledImage::DepthStencilReadOnlyOptimal,
				ImageLayoutCombinedImage::DepthStencilReadOnlyOptimal,
				ImageLayoutInputAttachment::DepthStencilReadOnlyOptimal
			)
		} else {
			(
				ImageLayoutEnd::ColorAttachmentOptimal,
				ImageLayoutSampledImage::ShaderReadOnlyOptimal,
				ImageLayoutCombinedImage::ShaderReadOnlyOptimal,
				ImageLayoutInputAttachment::ShaderReadOnlyOptimal
			)
		};

		Ok(ImageView::new(
			image,
			None,
			format,
			Swizzle::identity(),
			subresource_range,
			RequiredLayouts {
				global: Some(global_layout),
				storage: None,
				sampled: Some(sampled_layout),
				combined: Some(combined_layout),
				input_attachment: Some(input_layout)
			}
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
