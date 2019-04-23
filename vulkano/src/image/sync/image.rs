use std::num::NonZeroU32;

use std::{error, fmt, sync::Arc};

use crate::{
	buffer::BufferAccess,
	device::Device,
	format::FormatDesc,
	image::{
		sync::locker::{ImageResourceLocker, SimpleImageResourceLocker},
		sys::{UnsafeImage, UnsafeImageCreationError},
		ImageAccess,
		ImageDimensions,
		ImageLayout,
		ImageLayoutEnd,
		ImageSubresourceLayoutError,
		ImageSubresourceRange,
		ImageUsage,
		MipmapsCount
	},
	instance::QueueFamily,
	memory::{
		pool::{
			AllocFromRequirementsFilter,
			AllocLayout,
			MappingRequirement,
			MemoryPool,
			MemoryPoolAlloc,
			PotentialDedicatedAllocation,
			StdMemoryPoolAlloc
		},
		DedicatedAlloc
	},
	sync::{AccessError, Sharing}
};

pub type ImageCreationError = UnsafeImageCreationError;

/// A wrapper around `UnsafeImage` which implements access synchronization.
///
/// Note that by default the type parameter L is `SimpleImageResourceLocker` which
/// is simple and fast but not always desired.
// TODO: Rework allocators
#[derive(Debug)]
pub struct SyncImage<L = SimpleImageResourceLocker>
where
	L: ImageResourceLocker
{
	/// The inner image.
	inner: UnsafeImage,

	/// The memory bound to the image.
	memory: PotentialDedicatedAllocation<StdMemoryPoolAlloc>,

	/// The locker implementation.
	locker: L /* TODO: In the future, we might want to provide
	           * typelevel safety for iamge formats
	           * _phantom_format: std::marker::PhantomData<F> */
}
impl<L> SyncImage<L>
where
	L: ImageResourceLocker
{
	/// Creates a new synchronized image.
	///
	/// This function is a little simplified version of `new_sharing` with the
	/// `queue_families` and `preinitialized_layout` parameters left out.
	///
	/// The reason for leaving out `queue_families` is that most of the time only
	/// exclusive sharing is desired.
	///
	/// The reason for leaving out the `preinitialized_layout` parameter is described in
	/// the `new_sharing` constructor.
	pub fn new<F: FormatDesc>(
		device: Arc<Device>, usage: ImageUsage, format: F, dimensions: ImageDimensions,
		samples: NonZeroU32, mipmaps: MipmapsCount
	) -> Result<Self, ImageCreationError> {
		SyncImage::new_sharing(
			device,
			None::<QueueFamily>,
			usage,
			format,
			dimensions,
			samples,
			mipmaps,
			false
		)
	}

	/// Creates a new synchronized image.
	///
	/// This function creates an inner `UnsafeImage` with given parameters
	/// and uses the standard memory pool.
	///
	/// The `preinitialized_layout` should be set to true only in cases
	/// where you know what you are doing, otherwise it might have negative
	/// performance impact.
	pub fn new_sharing<'a, I, F>(
		device: Arc<Device>, queue_families: I, usage: ImageUsage, format: F,
		dimensions: ImageDimensions, samples: NonZeroU32, mipmaps: MipmapsCount,
		preinitialized_layout: bool
	) -> Result<Self, ImageCreationError>
	where
		I: IntoIterator<Item = QueueFamily<'a>>,
		I::IntoIter: ExactSizeIterator,
		F: FormatDesc
	{
		let (image, memory_requirements) = unsafe {
			let queue_families = queue_families.into_iter();
			let sharing = if queue_families.len() >= 2 {
				Sharing::Concurrent(queue_families.map(|f| f.id()))
			} else {
				Sharing::Exclusive
			};

			UnsafeImage::new(
				device.clone(),
				sharing,
				usage,
				format.format(),
				dimensions,
				samples,
				mipmaps,
				preinitialized_layout,
				false
			)?
		};

		let memory = MemoryPool::alloc_from_requirements(
			&Device::standard_pool(&device), // TODO: custom allocator/memory pool
			&memory_requirements,
			AllocLayout::Optimal,
			MappingRequirement::DoNotMap,
			DedicatedAlloc::Image(&image),
			|t| {
				if t.is_device_local() {
					AllocFromRequirementsFilter::Preferred
				} else {
					AllocFromRequirementsFilter::Allowed
				}
			}
		)?;
		debug_assert!((memory.offset() % memory_requirements.alignment) == 0);
		unsafe {
			image.bind_memory(memory.memory(), memory.offset())?;
		}

		let locker =
			L::new(preinitialized_layout, dimensions.array_layers(), image.mipmap_levels());

		Ok(SyncImage { inner: image, memory, locker })
	}

	/// Adopts another `SyncImage` but replaces the locker implementation.
	///
	/// An Err is returned if the old image is locked or if the complexity of
	/// the new locker is too low to represent the current state in the old locker.
	///
	/// Note that since images are usually manipulated using `Arc`s (for example
	/// in `ImageView`), you must first call `Arc::try_unwrap`. To ensure that
	/// you are holding the only copy of the `Arc`, you need to drop all outstanding
	/// `ImageView`s and `CommandBuffer`s and call `cleanup_finished()` on a fence GPU future
	/// that used any `CommandBuffer`s holding a reference to the image.
	///
	/// While this process can be tedious, the intended way to initialize mipmaps for
	/// an image is to create an image with the `MatrixImageResourceLocker` locker, upload
	/// the base level, blit all other levels and then change to locker to a less complex
	/// one to avoid the overhead. This works especially well with the semantical "immutable"
	/// images.
	///
	/// # Example:
	/// ```
	/// // Some uses left out
	/// use vulkano::image::{SyncImage, ImageView};
	/// use vulkano::image::sync::locker::{SimpleImageResourceLocker, MatrixImageResourceLocker};
	///
	/// // Obtain the original image somehow.
	/// let image: Arc<SyncImage<MatrixImageResourceLocker>> = Arc::new(SyncImage::new(...));
	/// // Create a view or more, depending on your needs.
	/// let view1 = Arc<ImageView> = Arc::new(ImageView::new(...));
	/// let view2 = Arc<ImageView> = Arc::new(ImageView::new(...));
	///
	/// // Blit mipmaps or prepare the image somehow.
	/// let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(...).unwrap()
	/// .blit_image(view1, ..., view2, ...).unwrap()
	/// .build().unwrap();
	///
	/// let future = previous_frame.then_execute(..., command_buffer).then_signal_fence_and_flush();
	///
	/// // Need to ensure that the future has already happened somehow, so we use the wait function.
	/// // If we didn't want to block, we could poll the future periodically using the duration parameter.
	/// future.wait(None).unwrap();
	///
	/// // Cleanup all the references.
	/// future.cleanup_finished();
	/// // Drop the views (or just let them go out of scope).
	/// std::mem::drop(view1);
	/// std::mem::drop(view2);
	///
	/// // At this point there should be no outstanding references to the orginal image except for
	/// // the one we have in image variable.
	/// let bare_image: SyncImage<MatrixImageResourceLocker> = image.try_unwrap().expect("We were wrong");
	///
	/// let image: Arc<SyncImage<SimpleImageResourceLocker>> = Arc::new(
	/// 	SyncImage::new_change_locker(bare_image)
	/// 	).expect("We left the image in a state that is too complex for this locker to handle");
	/// // The ImageSubresourceLayoutError error above can be safely avoided if we ensure that we
	/// // leave the whole image in the same layout at the end of the command buffer.
	/// // This is, however, out of the scope of this example.
	/// ```
	pub fn new_change_locker(
		other: SyncImage<impl ImageResourceLocker>
	) -> Result<Self, LockerChangeError> {
		let lock = other.initiate_gpu_lock(
			ImageSubresourceRange::whole_image(&other),
			true,
			ImageLayout::Undefined
		)?;

		let array_layers = other.dimensions().array_layers();
		let mipmap_levels = other.mipmap_levels();
		let locker = L::try_from_locker(other.locker, array_layers, mipmap_levels)?;

		Ok(SyncImage { inner: other.inner, memory: other.memory, locker })
	}
}
unsafe impl<L: ImageResourceLocker> ImageAccess for SyncImage<L> {
	fn inner(&self) -> &UnsafeImage { &self.inner }

	fn conflicts_buffer(&self, other: &dyn BufferAccess) -> bool { false }

	fn conflicts_image(
		&self, subresource_range: ImageSubresourceRange, other: &dyn ImageAccess,
		other_subresource_range: ImageSubresourceRange
	) -> bool {
		self.conflict_key() == other.conflict_key()
			&& subresource_range.overlaps_with(&other_subresource_range)
	}

	fn current_layout(
		&self, subresource_range: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		self.locker.current_layout(subresource_range)
	}

	fn initiate_gpu_lock(
		&self, subresource_range: ImageSubresourceRange, exclusive_access: bool,
		expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		self.locker.initiate_gpu_lock(subresource_range, exclusive_access, expected_layout)
	}

	unsafe fn increase_gpu_lock(&self, subresource_range: ImageSubresourceRange) {
		self.locker.increase_gpu_lock(subresource_range)
	}

	unsafe fn decrease_gpu_lock(
		&self, subresource_range: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
	) {
		self.locker.decrease_gpu_lock(subresource_range, new_layout)
	}
}

#[derive(Debug)]
pub enum LockerChangeError {
	ImageSubresourceLayoutError(ImageSubresourceLayoutError),
	AccessError(AccessError)
}
impl fmt::Display for LockerChangeError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			LockerChangeError::ImageSubresourceLayoutError(e) => e.fmt(f),
			LockerChangeError::AccessError(e) => e.fmt(f)
		}
	}
}
impl error::Error for LockerChangeError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			LockerChangeError::ImageSubresourceLayoutError(e) => e.source(),
			LockerChangeError::AccessError(e) => e.source()
		}
	}
}
impl From<ImageSubresourceLayoutError> for LockerChangeError {
	fn from(err: ImageSubresourceLayoutError) -> Self {
		LockerChangeError::ImageSubresourceLayoutError(err)
	}
}
impl From<AccessError> for LockerChangeError {
	fn from(err: AccessError) -> Self { LockerChangeError::AccessError(err) }
}
