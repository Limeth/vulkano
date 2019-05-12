use std::{num::NonZeroU32, sync::Mutex};

use super::ImageResourceLocker;
use crate::{
	image::{
		layout::ImageLayoutMatrix,
		ImageLayout,
		ImageLayoutEnd,
		ImageSubresourceLayoutError,
		ImageSubresourceRange
	},
	sync::AccessError
};

#[derive(Debug)]
pub struct MatrixImageResourceLocker {
	inner: Mutex<InnerLocker>
}
unsafe impl ImageResourceLocker for MatrixImageResourceLocker {
	fn new(preinitialized: bool, array_layers: NonZeroU32, mipmap_levels: NonZeroU32) -> Self {
		MatrixImageResourceLocker {
			inner: Mutex::new(InnerLocker::new(preinitialized, array_layers, mipmap_levels))
		}
	}

	fn try_from_locker(
		other: impl ImageResourceLocker, array_layers: NonZeroU32, mipmap_levels: NonZeroU32
	) -> Result<Self, ImageSubresourceLayoutError>
	where
		Self: Sized
	{
		Ok(MatrixImageResourceLocker {
			inner: Mutex::new(InnerLocker::try_from_locker(other, array_layers, mipmap_levels)?)
		})
	}

	fn current_layout(
		&self, subresource_range: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		let locker = self.inner.lock().expect("Mutex poisoned");
		locker.current_layout(subresource_range)
	}

	fn initiate_gpu_lock(
		&self, range: ImageSubresourceRange, exclusive: bool, expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		let mut locker = self.inner.lock().expect("Mutex poisoned");
		locker.initiate_lock(range, exclusive, expected_layout)
	}

	unsafe fn increase_gpu_lock(&self, range: ImageSubresourceRange) {
		let mut locker = self.inner.lock().expect("Mutex poisoned");
		locker.increase_lock(range)
	}

	unsafe fn decrease_gpu_lock(
		&self, range: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
	) {
		let mut locker = self.inner.lock().expect("Mutex poisoned");
		locker.decrease_lock(range, new_layout)
	}
}

#[derive(Debug, Clone, Copy)]
struct InnerEntry {
	pub count: u8, // Hopefully noone needs to lock this more than 255 times.
	pub exclusive: bool
}
impl InnerEntry {
	pub const fn new() -> Self { InnerEntry { count: 0, exclusive: false } }

	pub fn initiate(&mut self, exclusive: bool) {
		debug_assert!(!self.exclusive);
		debug_assert!(!(exclusive && self.count > 0));

		self.count += 1;
		self.exclusive = exclusive;
	}

	pub fn increase(&mut self) {
		debug_assert_ne!(
			self.count, 0,
			"The lock must already be locked before it can be increased"
		);

		self.count += 1;
	}

	pub fn decrease(&mut self) {
		debug_assert_ne!(self.count, 0, "The lock must be locked before it can be unlocked");

		if self.count == 1 {
			self.exclusive = false;
			self.count = 0;
		} else {
			self.count -= 1;
		}
	}
}

/// The actual implementation of the locker.
#[derive(Debug)]
struct InnerLocker {
	matrix: ImageLayoutMatrix<InnerEntry>
}
impl InnerLocker {
	pub fn new(preinitialized: bool, array_layers: NonZeroU32, mipmap_levels: NonZeroU32) -> Self {
		let implicit_layout =
			if preinitialized { ImageLayout::Preinitialized } else { ImageLayout::Undefined };

		let size = array_layers.get() as usize * mipmap_levels.get() as usize;

		let matrix = ImageLayoutMatrix::new_data(
			array_layers.get(),
			implicit_layout,
			vec![InnerEntry::new(); size].into_iter()
		)
		.unwrap();

		InnerLocker { matrix }
	}

	pub fn try_from_locker(
		other: impl ImageResourceLocker, array_layers: NonZeroU32, mipmap_levels: NonZeroU32
	) -> Result<Self, ImageSubresourceLayoutError> {
		let size = array_layers.get() as usize * mipmap_levels.get() as usize;

		let mut layouts_data = Vec::with_capacity(size);
		for a in 0 .. array_layers.get() {
			for m in 0 .. mipmap_levels.get() {
				let layout = other.current_layout(ImageSubresourceRange {
					array_layers: crate::NONZERO_ONE,
					array_layers_offset: a,

					mipmap_levels: crate::NONZERO_ONE,
					mipmap_levels_offset: m
				})?;

				layouts_data.push((layout, InnerEntry::new()));
			}
		}

		let matrix =
			ImageLayoutMatrix::new_layouts_data(array_layers.get(), layouts_data.into_iter())
				.unwrap();

		Ok(InnerLocker { matrix })
	}

	fn current_layout(
		&self, subresource_range: ImageSubresourceRange
	) -> Result<ImageLayout, ImageSubresourceLayoutError> {
		let mut iter = self.matrix.iter_subresource_range(subresource_range);
		let layout = match iter.next() {
			None => unreachable!(),
			Some(entry) => entry.layout
		};

		for entry in iter {
			if entry.layout != layout {
				return Err(ImageSubresourceLayoutError::MultipleLayouts)
			}
		}

		return Ok(layout)
	}

	pub fn initiate_lock(
		&mut self, range: ImageSubresourceRange, exclusive: bool, expected_layout: ImageLayout
	) -> Result<(), AccessError> {
		let mut last = 0;
		let mut result = Ok(());
		for entry in self.matrix.iter_subresource_range_mut(range) {
			if entry.data.exclusive {
				result = Err(AccessError::AlreadyInUseExclusive);
				break
			}
			if exclusive && entry.data.count > 0 {
				result = Err(AccessError::ExclusiveDenied);
				break
			}
			if entry.layout != expected_layout {
				result = Err(AccessError::ImageLayoutMismatch {
					expected: expected_layout,
					actual: entry.layout
				});
				break
			}

			entry.data.initiate(exclusive);
			last += 1;
		}

		// Rollback
		if result.is_err() {
			for entry in self.matrix.iter_subresource_range_mut(range) {
				if last == 0 {
					break
				}

				entry.data.decrease();

				last -= 1;
			}
		}

		result
	}

	pub fn increase_lock(&mut self, range: ImageSubresourceRange) {
		self.matrix.iter_subresource_range_mut(range).for_each(|entry| entry.data.increase());
	}

	pub fn decrease_lock(
		&mut self, range: ImageSubresourceRange, new_layout: Option<ImageLayoutEnd>
	) {
		if let Some(new_layout) = new_layout {
			for entry in self.matrix.iter_subresource_range_mut(range) {
				if !entry.data.exclusive {
					panic!("The lock must be exclusive for the layout to change")
				}
				entry.data.decrease();
				entry.layout = new_layout.into();
			}
		} else {
			self.matrix.iter_subresource_range_mut(range).for_each(|entry| {
				entry.data.decrease();
			});
		}
	}
}

#[cfg(test)]
mod tests {
	// TODO
}
