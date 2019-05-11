use crate::{
	buffer::BufferAccess,
	format::Format,
	image::{
		layout::{ImageLayout, ImageLayoutEnd},
		sys::UnsafeImageView,
		ImageDimensions,
		ImageSubresourceLayoutError,
		ImageSubresourceRange,
		ImageUsage,
		RequiredLayouts
	},
	sampler::Sampler,
	sync::AccessError,
	SafeDeref
};

use super::ImageAccess;

/// Trait for types that represent the GPU can access an image view.
///
/// Image views are the way to work with images in Vulkan and Vulkano, you need
/// one to perform any operation on images.
///
/// The power of image views comes from the fact that they can be created for
/// subresource ranges. That is, they can be created for certain ranges of array layers
/// and mipmap levels. For example, having a cubemap, you could create a view for each face
/// and then bind those views as framebuffer attachments, rendering to them or reading from
/// them as if they were separate textures (because they basically are).
///
/// There is one promiment problem that can arise, however. How do we synchronize accesses
/// of these views. Each subresource range can have a different layout. That means that if
/// you create overlapping views you could run into a case where your view is no longer in one
/// layout but in multiple different ones. If this happens, the `current_layout` function will
/// return an error. If you pass such view into a command buffer, you will get an error.
///
/// For more info on how Vulkano keeps track of layouts, see documentation of the crate::image::sync module.
pub unsafe trait ImageViewAccess: std::fmt::Debug {
	/// Returns a dynamic reference to the parent image.
	fn parent(&self) -> &dyn ImageAccess;
	/// Returns the inner unsafe image view object used by this image view.
	fn inner(&self) -> &UnsafeImageView;

	/// Returns the usage of this view.
	// TODO: Can this be different from parent?
	fn usage(&self) -> ImageUsage { self.inner().usage() }

	/// Returns the format of this view. This can be different from the parent's format.
	fn format(&self) -> Format { self.inner().format() }

	/// Returns the dimensions of the image view.
	fn dimensions(&self) -> ImageDimensions { self.inner().dimensions() }
	/// Returns the subresource range for this view.
	fn subresource_range(&self) -> ImageSubresourceRange { self.inner().subresource_range() }

	/// Returns true if the view doesn't use components swizzling.
	///
	/// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
	/// the other thing.
	fn identity_swizzle(&self) -> bool { self.inner().swizzle().is_identity() }

	/// Returns true if the given sampler can be used with this image view.
	///
	/// This method should check whether the sampler's configuration can be used with the format
	/// of the view.
	// TODO: return a Result and propagate it when binding to a descriptor set
	fn can_be_sampled(&self, _sampler: &Sampler) -> bool {
		true // FIXME
	}

	/// Returns true if an access to `self` potentially overlaps the same memory as an
	/// access to `other`.
	///
	/// If this function returns `false`, this means that we are allowed to access the content
	/// of `self` at the same time as the content of `other` without causing a data race.
	///
	/// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
	/// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
		false // TODO
	}

	/// Returns true if an access to `self` potentially overlaps the same memory as an
	/// access to `other`.
	///
	/// If this function returns `false`, this means that we are allowed to access the content
	/// of `self` at the same time as the content of `other` without causing a data race.
	///
	/// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
	/// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
	fn conflicts_image(&self, other: &dyn ImageViewAccess) -> bool {
		self.parent().conflicts_image(
			self.subresource_range(),
			other.parent(),
			other.subresource_range()
		)
	}

	fn conflict_key(&self) -> u64 { self.parent().conflict_key() }

	/// Equivalent to `self.parent().initiate_gpu_lock(self.subresource_range, exclusive_access, expected_layout)`.
	fn initiate_gpu_lock(
		&self, exclusive_access: bool, expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		self.parent().initiate_gpu_lock(self.subresource_range(), exclusive_access, expected_layout)
	}

	/// Equivalent to `self.parent().increase_gpu_lock(self.subresource_range())`.
	unsafe fn increase_gpu_lock(&self) { self.parent().increase_gpu_lock(self.subresource_range()) }

	/// Equivalent to `self.parent().decrease_gpu_lock(self.subresource_range(), transitioned_layout)`.
	unsafe fn decrease_gpu_lock(&self, transitioned_layout: Option<ImageLayoutEnd>) {
		self.parent().decrease_gpu_lock(self.subresource_range(), transitioned_layout)
	}

	/// Reports the current layout of the view.
	///
	/// If this value is incorrect, bad things can happen.
	fn current_layout(&self) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		self.parent().current_layout(self.subresource_range())
	}

	/// Getter for required layouts.
	fn required_layouts(&self) -> &RequiredLayouts;
}

unsafe impl<T> ImageViewAccess for T
where
	T: std::fmt::Debug + SafeDeref,
	T::Target: ImageViewAccess
{
	fn parent(&self) -> &dyn ImageAccess { (**self).parent() }

	fn inner(&self) -> &UnsafeImageView { (**self).inner() }

	fn dimensions(&self) -> ImageDimensions { (**self).dimensions() }

	fn subresource_range(&self) -> ImageSubresourceRange { (**self).subresource_range() }

	fn identity_swizzle(&self) -> bool { (**self).identity_swizzle() }

	fn can_be_sampled(&self, sampler: &Sampler) -> bool { (**self).can_be_sampled(sampler) }

	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool {
		(**self).conflicts_buffer(other)
	}

	fn conflicts_image(&self, other: &dyn ImageViewAccess) -> bool {
		(**self).conflicts_image(other)
	}

	fn conflict_key(&self) -> u64 { (**self).conflict_key() }

	fn current_layout(&self) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		(**self).current_layout()
	}

	fn required_layouts(&self) -> &RequiredLayouts { (**self).required_layouts() }
}
