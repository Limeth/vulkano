use crate::{
	buffer::BufferAccess,
	format::{
		Format,
		PossibleDepthFormatDesc,
		PossibleDepthStencilFormatDesc,
		PossibleFloatFormatDesc,
		PossibleSintFormatDesc,
		PossibleStencilFormatDesc,
		PossibleUintFormatDesc
	},
	image::{
		sys::UnsafeImageView,
		ImageDimensions,
		ImageLayout,
		ImageSubresourceLayoutError,
		ImageSubresourceRange,
		ImageUsage
	},
	sampler::Sampler,
	sync::AccessError,
	SafeDeref
};

use super::ImageAccess;

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAccess {
	/// Returns a dynamic reference to the parent image.
	fn parent(&self) -> &dyn ImageAccess;
	/// Returns the inner unsafe image view object used by this image view.
	fn inner(&self) -> &UnsafeImageView;

	/// Returns the usage of this view.
	// TODO: Can this be different from parent?
	fn usage(&self) -> ImageUsage { self.inner().usage() }

	/// Returns the format of this view. This can be different from the parent's format.
	fn format(&self) -> Format { self.inner().format() }
	/// Returns true if this view format is color.
	fn has_color(&self) -> bool {
		let format = self.format();
		format.is_float() || format.is_uint() || format.is_sint()
	}
	/// Returns true if this view format is depth or depth_stencil.
	fn has_depth(&self) -> bool {
		let format = self.format();
		format.is_depth() || format.is_depth_stencil()
	}
	/// Returns true if this view format is stencil or depth_stencil.
	fn has_stencil(&self) -> bool {
		let format = self.format();
		format.is_stencil() || format.is_depth_stencil()
	}

	/// Returns the dimensions of the image view.
	fn dimensions(&self) -> ImageDimensions;
	/// Returns the subresource range for this view.
	fn subresource_range(&self) -> ImageSubresourceRange { self.inner().subresource_range() }

	/// Returns true if the view doesn't use components swizzling.
	///
	/// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
	/// the other thing.
	fn identity_swizzle(&self) -> bool { self.inner().swizzle().identity() }

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
	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool;

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
	unsafe fn decrease_gpu_lock(&self, transitioned_layout: Option<ImageLayout>) {
		self.parent().decrease_gpu_lock(self.subresource_range(), transitioned_layout)
	}

	/// Reports the current layout of the view.
	///
	/// If this value is incorrect, bad things can happen.
	fn current_layout(&self) -> Result<ImageLayout, ImageSubresourceLayoutError>;

	/// Reports to vulkano which layout the view wants to be at the end of an auto command buffer.
	///
	/// Returning `ImageLayout::Undefined` means the view doesn't have a requirement.
	fn required_layout(&self) -> ImageLayout;

	/// Reports to vulkano which layout the view wants to be when used as a storage image
	/// in descriptor set.
	fn required_layout_descriptor_storage(&self) -> ImageLayout;
	/// Reports to vulkano which layout the view wants to be when used as a sampled image
	/// in descriptor set.
	fn required_layout_descriptor_sampled(&self) -> ImageLayout;
	/// Reports to vulkano which layout the view wants to be when used as a combined
	/// image and sampler in descriptor set.
	fn required_layout_descriptor_combined(&self) -> ImageLayout;
	/// Reports to vulkano which layout the view wants to be when used as an input
	/// attachment in descriptor set.
	fn required_layout_descriptor_input_attachment(&self) -> ImageLayout;
}

unsafe impl<T> ImageViewAccess for T
where
	T: SafeDeref,
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

	fn required_layout(&self) -> ImageLayout { (**self).required_layout() }

	fn required_layout_descriptor_storage(&self) -> ImageLayout {
		(**self).required_layout_descriptor_storage()
	}

	fn required_layout_descriptor_sampled(&self) -> ImageLayout {
		(**self).required_layout_descriptor_sampled()
	}

	fn required_layout_descriptor_combined(&self) -> ImageLayout {
		(**self).required_layout_descriptor_combined()
	}

	fn required_layout_descriptor_input_attachment(&self) -> ImageLayout {
		(**self).required_layout_descriptor_input_attachment()
	}
}
