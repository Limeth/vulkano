use std::{
	num::NonZeroU32,
	sync::atomic::{AtomicIsize, Ordering}
};

use super::ImageResourceLocker;
use crate::{
	image::{ImageLayout, ImageSubresourceRange},
	sync::AccessError
};

// TODO: AtomicU32 stable pls
#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
type AtomicU32 = std::sync::atomic::AtomicUsize;


// This code will not compile on platforms with different pointer_width than 32 or 64.
// In case such a platform needs to be supported, the following alias can be used,
// but some of the code will need to be edited too.
// #[cfg(not(any(target_pointer_width="32", target_pointer_width="64")))]
// type AtomicU32 = crossbeam::atomic::AtomicCell<u32>;

/// A simple image resource locker.
///
/// This resource locker always locks the whole image.
/// It's advantage over more granular lockers is that it's very simple
/// and only uses atomics, so it's actually lock free.
/// The whole image must always be in the same layout.
#[derive(Debug)]
pub struct SimpleImageResourceLocker {
	implicit_layout: ImageLayout,
	array_layers: NonZeroU32,
	mipmap_levels: NonZeroU32,

	/// Used as a lock for this resource.
	///
	/// Positive numbers mean read-only locks.
	/// Negative numbers mean an exclusive lock.
	lock: AtomicIsize,

	/// The layout that the resource currently has.
	///
	/// The `ImageLayout` enum is #[repr(u32)] so it's okay to represent it as
	/// atomic integer. However this will all break down on platforms where
	/// usize is less that 32 bits.
	current_layout: AtomicU32
}
unsafe impl ImageResourceLocker for SimpleImageResourceLocker {
	fn new(preinitialized: bool, array_layers: NonZeroU32, mipmap_levels: NonZeroU32) -> Self {
		let implicit_layout =
			if preinitialized { ImageLayout::Preinitialized } else { ImageLayout::Undefined };

		let current_layout = AtomicU32::new(implicit_layout as u32 as usize);

		SimpleImageResourceLocker {
			implicit_layout,
			array_layers,
			mipmap_levels,

			lock: AtomicIsize::new(0),
			current_layout
		}
	}

	fn initiate_gpu_lock(
		&self, _: ImageSubresourceRange, exclusive: bool, expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		let current_layout = {
			// TODO: AtomicU32 stable pls
			let current_num = self.current_layout.load(Ordering::Acquire) as u32;

			// We never store anything else but `ImageLayout as u32` in the atomic.
			unsafe { std::mem::transmute(current_num) }
		};
		if expected_layout != ImageLayout::Undefined && current_layout != expected_layout {
			return Err(AccessError::ImageLayoutMismatch {
				requested: expected_layout,
				actual: current_layout
			})
		}

		// TODO: Could this ordering be a less strict one (like Acquire)?
		let mut lock_value = self.lock.load(Ordering::SeqCst);
		loop {
			if lock_value < 0 {
				return Err(AccessError::AlreadyInUse)
			}

			if exclusive {
				if lock_value > 0 {
					return Err(AccessError::ExclusiveDenied)
				}
				lock_value = self.lock.compare_and_swap(0, -1, Ordering::SeqCst);
				if lock_value == -1 {
					return Ok(())
				}
			} else {
				let wanted_lock_value = lock_value + 1;
				lock_value =
					self.lock.compare_and_swap(lock_value, wanted_lock_value, Ordering::SeqCst);
				if lock_value == wanted_lock_value {
					return Ok(())
				}
			}
		}
	}

	unsafe fn increase_gpu_lock(&self, _: ImageSubresourceRange) {
		// TODO: Could this ordering be a less strict one (like Acquire)?
		let mut lock_value = self.lock.load(Ordering::SeqCst);
		loop {
			if lock_value == 0 {
				panic!("The lock must already be locked before it can be increased")
			}

			let wanted_lock_value = if lock_value < 0 { lock_value - 1 } else { lock_value + 1 };
			lock_value =
				self.lock.compare_and_swap(lock_value, wanted_lock_value, Ordering::SeqCst);
			if lock_value == wanted_lock_value {
				return
			}
		}
	}

	unsafe fn decrease_gpu_lock(
		&self, range: ImageSubresourceRange, new_layout: Option<ImageLayout>
	) {
		// TODO: Could this ordering be a less strict one (like Acquire)?
		let mut lock_value = self.lock.load(Ordering::SeqCst);
		loop {
			if lock_value == 0 {
				panic!("The lock must be locked before it can be unlocked")
			}
			if let Some(new_layout) = new_layout {
				if lock_value >= 0 {
					panic!("The lock must be exclusive for the layout to change")
				}

				if range.array_layers_offset != 0
					|| range.mipmap_levels_offset != 0
					|| range.array_layers != self.array_layers
					|| range.mipmap_levels != self.mipmap_levels
				{
					panic!("The simple locker doesn't allow changing layouts for subrange of the image")
				}

				// TODO: AtomicU32 stable pls
				self.current_layout.store(new_layout as u32 as usize, Ordering::Release);
			}

			let wanted_lock_value = if lock_value < 0 { lock_value + 1 } else { lock_value - 1 };
			lock_value =
				self.lock.compare_and_swap(lock_value, wanted_lock_value, Ordering::SeqCst);
			if lock_value == wanted_lock_value {
				break
			}
		}
	}
}

#[cfg(test)]
mod tests {
	// TODO
}
