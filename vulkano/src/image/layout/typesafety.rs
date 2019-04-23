//! Provides typelevel safety for `ImageLayout` variants and subsets.
//!

use crate::image::ImageUsage;

use super::{ImageLayout, InvalidLayoutUsageError};

macro_rules! impl_typesafe_layout {
	(
		$name: ident {
			$(
				$key: ident
			),+
		} $(: $required_usage: ident )?
	) => {
		/// Typelevel protection against passing invalid values into
		/// certain Vulkan API functions.
		///
		/// See [`ImageLayout`](../enum.ImageLayout.html)
		#[derive(Debug, Copy, Clone, PartialEq, Eq)]
		#[repr(u32)]
		pub enum $name {
			$(
				$key = ImageLayout::$key as u32
			),+
		}
		impl $name {
			/// Returns true if given layout is valid for given usage.
			pub fn valid_for_usage(&self, usage: ImageUsage) -> Result<(), InvalidLayoutUsageError> {
				$(
					if !usage.$required_usage {
						return Err(InvalidLayoutUsageError { layout: ImageLayout::from(*self), usage })
					}
				)?

				ImageLayout::from(*self).valid_for_usage(usage)
			}
		}
	}
}

/// UNSAFE, use with care.
///
/// You must ensure that the `$from` is a `#[repr(u32)]` subset
/// of `$target`. Otherwise it's undefined behaviour.
macro_rules! impl_from_typesafe {
	($from: ident, $target: ident) => {
		impl From<$from> for $target {
			fn from(other: $from) -> Self { unsafe { std::mem::transmute(other) } }
		}
	};
}

// General manipulation.
impl_typesafe_layout!(ImageLayoutEnd {
	General,
	ColorAttachmentOptimal,
	DepthStencilAttachmentOptimal,
	DepthStencilReadOnlyOptimal,
	ShaderReadOnlyOptimal,
	TransferSrcOptimal,
	TransferDstOptimal,
	PresentSrc,
	DepthReadOnlyStencilAttachmentOptimal,
	DepthAttachmentStencilReadOnlyOptimal
});
impl_from_typesafe!(ImageLayoutEnd, ImageLayout);
impl ImageLayoutEnd {
	pub fn try_from_image_layout(layout: ImageLayout) -> Option<Self> {
		match layout {
			ImageLayout::General => Some(ImageLayoutEnd::General),
			ImageLayout::ColorAttachmentOptimal => Some(ImageLayoutEnd::ColorAttachmentOptimal),
			ImageLayout::DepthStencilAttachmentOptimal => {
				Some(ImageLayoutEnd::DepthStencilAttachmentOptimal)
			}
			ImageLayout::DepthStencilReadOnlyOptimal => {
				Some(ImageLayoutEnd::DepthStencilReadOnlyOptimal)
			}
			ImageLayout::ShaderReadOnlyOptimal => Some(ImageLayoutEnd::ShaderReadOnlyOptimal),
			ImageLayout::TransferSrcOptimal => Some(ImageLayoutEnd::TransferSrcOptimal),
			ImageLayout::TransferDstOptimal => Some(ImageLayoutEnd::TransferDstOptimal),
			ImageLayout::PresentSrc => Some(ImageLayoutEnd::PresentSrc),
			ImageLayout::DepthReadOnlyStencilAttachmentOptimal => {
				Some(ImageLayoutEnd::DepthReadOnlyStencilAttachmentOptimal)
			}
			ImageLayout::DepthAttachmentStencilReadOnlyOptimal => {
				Some(ImageLayoutEnd::DepthAttachmentStencilReadOnlyOptimal)
			}

			_ => None
		}
	}
}

impl_typesafe_layout!(
	ImageLayoutStorageImage {
	General
	// SharedPresent
}: storage
);
impl_from_typesafe!(ImageLayoutStorageImage, ImageLayout);
impl_from_typesafe!(ImageLayoutStorageImage, ImageLayoutEnd);

impl_typesafe_layout!(
	ImageLayoutSampledImage {
		General,
		DepthStencilReadOnlyOptimal,
		ShaderReadOnlyOptimal,
		DepthReadOnlyStencilAttachmentOptimal,
		DepthAttachmentStencilReadOnlyOptimal // SharedPresent
	}: sampled
);
impl_from_typesafe!(ImageLayoutSampledImage, ImageLayout);
impl_from_typesafe!(ImageLayoutSampledImage, ImageLayoutEnd);
impl_typesafe_layout!(
	ImageLayoutCombinedImage {
		General,
		DepthStencilReadOnlyOptimal,
		ShaderReadOnlyOptimal,
		DepthReadOnlyStencilAttachmentOptimal,
		DepthAttachmentStencilReadOnlyOptimal // SharedPresent
	}: sampled
);
impl_from_typesafe!(ImageLayoutCombinedImage, ImageLayout);
impl_from_typesafe!(ImageLayoutCombinedImage, ImageLayoutEnd);
impl_typesafe_layout!(
	ImageLayoutInputAttachment {
		General,
		DepthStencilReadOnlyOptimal,
		ShaderReadOnlyOptimal,
		DepthReadOnlyStencilAttachmentOptimal,
		DepthAttachmentStencilReadOnlyOptimal // SharedPresent
	}: input_attachment
);
impl_from_typesafe!(ImageLayoutInputAttachment, ImageLayout);
impl_from_typesafe!(ImageLayoutInputAttachment, ImageLayoutEnd);

// Commands
impl_typesafe_layout!(
	ImageLayoutImageSrc {
		General,
		TransferSrcOptimal // SharedPresent
	}: transfer_source
);
impl_from_typesafe!(ImageLayoutImageSrc, ImageLayout);
impl_from_typesafe!(ImageLayoutImageSrc, ImageLayoutEnd);

impl_typesafe_layout!(
	ImageLayoutImageDst {
		General,
		TransferDstOptimal // SharedPresent
	}: transfer_destination
);
impl_from_typesafe!(ImageLayoutImageDst, ImageLayout);
impl_from_typesafe!(ImageLayoutImageDst, ImageLayoutEnd);
