// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::format::ClearValue;

mod image_access;
mod image_view_access;

pub use image_access::ImageAccess;
pub use image_view_access::ImageViewAccess;

/// Extension trait for images. Checks whether the value `T` can be used as a clear value for the
/// given image.
// TODO: isn't that for image views instead?
pub unsafe trait ImageClearValue<T>: ImageAccess {
	fn decode(&self, t: T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: ImageAccess {
	/// Checks whether pixels of type `P` match the format of the image.
	fn matches_format(&self) -> bool;
}
