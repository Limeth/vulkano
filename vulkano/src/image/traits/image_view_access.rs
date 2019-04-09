use crate::{
	format::Format,
	image::{sys::UnsafeImageView, ImageDimensions, ImageLayout},
	sampler::Sampler,
	SafeDeref
};

use super::ImageAccess;

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAccess {
	fn parent(&self) -> &ImageAccess;

	/// Returns the dimensions of the image view.
	fn dimensions(&self) -> ImageDimensions;

	/// Returns the inner unsafe image view object used by this image view.
	fn inner(&self) -> &UnsafeImageView;

	/// Returns the format of this view. This can be different from the parent's format.
	fn format(&self) -> Format {
		// TODO: remove this default impl
		self.inner().format()
	}

	fn samples(&self) -> u32 { self.parent().samples() }

	/// Returns the image layout to use in a descriptor with the given subresource.
	fn descriptor_set_storage_image_layout(&self) -> ImageLayout;
	/// Returns the image layout to use in a descriptor with the given subresource.
	fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout;
	/// Returns the image layout to use in a descriptor with the given subresource.
	fn descriptor_set_sampled_image_layout(&self) -> ImageLayout;
	/// Returns the image layout to use in a descriptor with the given subresource.
	fn descriptor_set_input_attachment_layout(&self) -> ImageLayout;

	/// Returns true if the view doesn't use components swizzling.
	///
	/// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
	/// the other thing.
	fn identity_swizzle(&self) -> bool;

	/// Returns true if the given sampler can be used with this image view.
	///
	/// This method should check whether the sampler's configuration can be used with the format
	/// of the view.
	// TODO: return a Result and propagate it when binding to a descriptor set
	fn can_be_sampled(&self, _sampler: &Sampler) -> bool {
		true // FIXME
	}

	// fn usable_as_render_pass_attachment(&self, ???) -> Result<(), ???>;
}

unsafe impl<T> ImageViewAccess for T
where
	T: SafeDeref,
	T::Target: ImageViewAccess
{
	fn parent(&self) -> &ImageAccess { (**self).parent() }

	fn inner(&self) -> &UnsafeImageView { (**self).inner() }

	fn dimensions(&self) -> ImageDimensions { (**self).dimensions() }

	fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
		(**self).descriptor_set_storage_image_layout()
	}

	fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
		(**self).descriptor_set_combined_image_sampler_layout()
	}

	fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
		(**self).descriptor_set_sampled_image_layout()
	}

	fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
		(**self).descriptor_set_input_attachment_layout()
	}

	fn identity_swizzle(&self) -> bool { (**self).identity_swizzle() }

	fn can_be_sampled(&self, sampler: &Sampler) -> bool { (**self).can_be_sampled(sampler) }
}

pub unsafe trait AttachmentImageView: ImageViewAccess {
	fn accept(&self, initial_layout: ImageLayout, final_layout: ImageLayout) -> bool;
}
