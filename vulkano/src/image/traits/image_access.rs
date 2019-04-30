use std::{num::NonZeroU32, sync::Arc};

use crate::{
	buffer::BufferAccess,
	device::Device,
	format::Format,
	image::{
		layout::{ImageLayout, ImageLayoutEnd},
		sys::UnsafeImage,
		ImageDimensions,
		ImageSubresourceLayoutError,
		ImageSubresourceRange,
		ImageUsage
	},
	sync::AccessError,
	SafeDeref
};

/// Trait for types that represent the way a GPU can access an image.
pub unsafe trait ImageAccess: std::fmt::Debug {
	/// Returns the inner unsafe image object used by this image.
	fn inner(&self) -> &UnsafeImage;

	fn device(&self) -> &Arc<Device> { &self.inner().device() }

	/// Returns the usage this image was created with.
	fn usage(&self) -> ImageUsage { self.inner().usage() }

	/// Returns the format of this image.
	fn format(&self) -> Format { self.inner().format() }


	/// Returns the dimensions of the image.
	fn dimensions(&self) -> ImageDimensions { self.inner().dimensions() }
	/// Returns the number of mipmap levels of this image.
	fn mipmap_levels(&self) -> NonZeroU32 { self.inner().mipmap_levels() }
	/// Returns the number of samples of this image.
	fn samples(&self) -> NonZeroU32 { self.inner().samples() }


	/// Returns true if the image can be used as a source for blits.
	fn supports_blit_source(&self) -> bool { self.inner().supports_blit_source() }
	/// Returns true if the image can be used as a destination for blits.
	fn supports_blit_destination(&self) -> bool { self.inner().supports_blit_destination() }


	/// Returns a key that uniquely identifies the memory content of the image.
	/// Two ranges that potentially overlap in memory must return the same key.
	///
	/// The key is shared amongst all buffers and images, which means that you can make several
	/// different image objects share the same memory, or make some image objects share memory
	/// with buffers, as long as they return the same key.
	///
	/// Since it is possible to accidentally return the same key for memory ranges that don't
	/// overlap, the `conflicts_image` or `conflicts_buffer` function should always be called to
	/// verify whether they actually overlap.
	fn conflict_key(&self) -> u64 { self.inner().key() }

	/// Returns true if an access to `self` potentially overlaps the same memory as an
	/// access to `other`.
	///
	/// If this function returns `false`, this means that we are allowed to access the content
	/// of `self` at the same time as the content of `other` without causing a data race.
	///
	/// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
	/// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool;

	/// Returns true if an access to a subresource range of `self` potentially overlaps
	/// the same memory as an access to a subresource range of `other`.
	///
	/// If this function returns `false`, this means that we are allowed to access the content
	/// of `self` at the same time as the content of `other` without causing a data race.
	///
	/// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
	/// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
	fn conflicts_image(
		&self, subresource_range: ImageSubresourceRange, other: &dyn ImageAccess,
		other_subresource_range: ImageSubresourceRange
	) -> bool;


	/// A proxy method for the internal `ImageResourceLocker::current_layout` implementation.
	///
	/// The implementation isn't required to use `ImageResourceLocker` implementation to handle
	/// locking and unlocking, however this is recommended and used throughout vulkano.
	fn current_layout(
		&self, subresource_range: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError>;

	/// A proxy method for the internal `ImageResourceLocker::initiate_gpu_lock` implementation.
	///
	/// The implementation isn't required to use `ImageResourceLocker` implementation to handle
	/// locking and unlocking, however this is recommended and used throughout vulkano.
	fn initiate_gpu_lock(
		&self, subresource_range: ImageSubresourceRange, exclusive_access: bool,
		expected_layout: ImageLayout
	) -> Result<(), AccessError>;

	/// A proxy method for the internal `ImageResourceLocker::increase_gpu_lock` implementation.
	///
	/// The implementation isn't required to use `ImageResourceLocker` implementation to handle
	/// locking and unlocking, however this is recommended and used throughout vulkano.
	unsafe fn increase_gpu_lock(&self, subresource_range: ImageSubresourceRange);

	/// A proxy method for the internal `ImageResourceLocker::decrease_gpu_lock` implementation.
	///
	/// The implementation isn't required to use `ImageResourceLocker` implementation to handle
	/// locking and unlocking, however this is recommended and used throughout vulkano.
	unsafe fn decrease_gpu_lock(
		&self, subresource_range: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
	);
}
unsafe impl<T> ImageAccess for T
where
	T: std::fmt::Debug + SafeDeref,
	T::Target: ImageAccess
{
	fn inner(&self) -> &UnsafeImage { (**self).inner() }

	fn conflicts_buffer(&self, other: &BufferAccess) -> bool { (**self).conflicts_buffer(other) }

	fn conflicts_image(
		&self, subresource_range: ImageSubresourceRange, other: &dyn ImageAccess,
		other_subresource_range: ImageSubresourceRange
	) -> bool {
		(**self).conflicts_image(subresource_range, other, other_subresource_range)
	}

	fn conflict_key(&self) -> u64 { (**self).conflict_key() }

	fn current_layout(
		&self, subresource_range: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		(**self).current_layout(subresource_range)
	}

	fn initiate_gpu_lock(
		&self, subresource_range: ImageSubresourceRange, exclusive_access: bool,
		expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		(**self).initiate_gpu_lock(subresource_range, exclusive_access, expected_layout)
	}

	unsafe fn increase_gpu_lock(&self, subresource_range: ImageSubresourceRange) {
		(**self).increase_gpu_lock(subresource_range)
	}

	unsafe fn decrease_gpu_lock(
		&self, subresource_range: ImageSubresourceRange,
		transitioned_layout: Option<ImageLayoutEnd>
	) {
		(**self).decrease_gpu_lock(subresource_range, transitioned_layout)
	}
}
