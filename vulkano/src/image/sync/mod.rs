//! Traits and structs handling CPU to GPU image synchronization.
//!
//! See the `ImageResourceLocker` trait doc for more info.
//!
//! This module provides three implementations of `ImageResourceLocker` trait of
//! increasing complexity.
//!
//! The `SimpleImageResourceLocker` is a very fast lock-free locker
//! that doesn't allow multiple locks for one image. This means that you won't
//! be able to copy between image layers, for instance. It also requires the whole
//! image to be in the same layout.
//!
//! TODO: ArrayImageResourceLocker - implement and document a locker that
//! only allows access to one array layer at a time. This will be a compromise
//! between the simple locker and the matrix locker. It will also be able to only
//! use atomics and hence be lock-free.
//!
//! The `MatrixImageResourceLocker` is the exact opposite of the simple locker.
//! It stores layout state for each mipmap of each array layer in a matrix of size
//! `array_layers * mipmap_levels` (see `crate::image::layout::ImageLayoutMatrix`).
//! It also uses a mutex internally so it should be used with care. It can, however,
//! handle even overlapping image views correctly.
//!

use std::num::NonZeroU32;

use crate::{
	image::{ImageLayout, ImageSubresourceRange},
	sync::AccessError
};

mod image;
mod matrix_locker;
mod simple_locker;

pub use image::SyncImage;
pub use matrix_locker::MatrixImageResourceLocker;
pub use simple_locker::SimpleImageResourceLocker;

/// Trait for image resource lockers.
///
/// A resource locker is an object that handles CPU to GPU locking and unlocking
/// of image resources or subresources. Each time a resource (an image or an image view)
/// is to be sent to the GPU, it is marked as locked in a locker. This ensures that the
/// resource doesn't have race conditions while the GPU is using it.
///
/// The locker is provided with a subresource range and expected layout. If the locker
/// detects a conflict while attempting to lock a resource, it will return an error.
///
/// This trait is unsafe because incorrect implementation can result in race conditions
/// on both the CPU side and the GPU side.
pub unsafe trait ImageResourceLocker {
	/// Initialized a new resource locker.
	///
	/// The locker assumes that the whole resource is in the implicit layout
	/// (`ImageLayout::Uninitialized` or `ImageLayout::Preinitialized` if `preinitialized == true`).
	fn new(preinitialized: bool, array_layers: NonZeroU32, mipmap_levels: NonZeroU32) -> Self;

	/// Locks the subresource for usage on the GPU. Returns an error if the lock can't be acquired.
	///
	/// After this function returns `Ok`, you are authorized to use the image subresource on the GPU. If the
	/// GPU operation requires an exclusive access to the image subresource (which includes image layout
	/// transitions) then `exclusive_access` should be true.
	///
	/// The `expected_layout` is the layout we expect the image subresource to be in when we lock it. If the
	/// actual layout doesn't match this expected layout, then an error should be returned. If
	/// `Undefined` is passed, that means that the caller doesn't care about the actual layout,
	/// and that a layout mismatch shouldn't return an error.
	///
	/// This function exists to prevent the user from causing a data race by reading and writing
	/// to the same resource at the same time.
	///
	/// If you call this function, you should call `decrease_gpu_lock()` once the subresource
	/// is no longer in use by the GPU. The implementation is not expected to automatically
	/// perform any unlocking and can rely on the fact that `decrease_gpu_lock()` is going to be called.
	fn initiate_gpu_lock(
		&self, range: ImageSubresourceRange, exclusive: bool, expected_layout: ImageLayout
	) -> Result<(), AccessError>;

	/// Increases the lock counter by one.
	///
	/// This function will panic if it's called on an unlocked subresource.
	///
	/// This is unsafe because it must only be called after `initiate_gpu_lock()` succeeded.
	///
	/// If you call this function, you should call `decrease_gpu_lock()` once the subresource is no longer in use
	/// by the GPU. The implementation is not expected to automatically perform any unlocking and
	/// can rely on the fact that `decrease_gpu_lock()` is going to be called.
	unsafe fn increase_gpu_lock(&self, range: ImageSubresourceRange);

	/// Decreases the lock counter previously increased by `initiate_gpu_lock` or `increase_gpu_lock`.
	///
	/// This function will panic if it's called on an unlocked subresource.
	///
	/// If the the subresource has been transition to another layout while in use by the GPU, then the new
	/// layout should be passes as a parameter.
	/// This function will panic if the lock isn't exclusive but a new layout was passes.
	///
	/// This function is unsafe because the transitioned layout must also be supported
	/// by the image subresource and must not be `Undefined`.
	unsafe fn decrease_gpu_lock(
		&self, range: ImageSubresourceRange, new_layout: Option<ImageLayout>
	);
}
