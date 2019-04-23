use super::ImageLayout;
use crate::image::ImageSubresourceRange;

/// Struct that stores information about image layouts for individual mipmap levels and array layers.
///
/// An image with A array layers and M mipmap levels can be represented as AxM matrix.
/// This struct uses such matrix to remember which subresource has what layout.
///
/// The type D allows storing arbitrary user data in each field of the matrix.
/// This is used in ImageResourceLocker implementations.
#[derive(Debug)]
pub struct ImageLayoutMatrix<D = ()> {
	matrix: Vec<ImageLayoutMatrixEntry<D>>,

	width: u32
}
impl ImageLayoutMatrix<()> {
	/// Creates a new layout matrix with `D = ()`.
	pub fn new(width: u32, height: u32, layout: ImageLayout) -> Self {
		let matrix =
			vec![ImageLayoutMatrixEntry { layout, data: () }; width as usize * height as usize];

		ImageLayoutMatrix { matrix, width }
	}
}
impl<D> ImageLayoutMatrix<D> {
	/// Creates a new layout matrix with custom data. Returns Err if `data.len() % width != 0`.
	pub fn new_data<I>(width: u32, layout: ImageLayout, data: I) -> Result<Self, ()>
	where
		I: ExactSizeIterator<Item = D>
	{
		if data.len() % width as usize != 0 {
			return Err(())
		}

		let matrix = data.map(|d| ImageLayoutMatrixEntry { layout, data: d }).collect();

		Ok(ImageLayoutMatrix { matrix, width })
	}

	/// Creates a new layout matrix with custom layouts and custom data.
	/// Returns Err if `layouts_data.len() % width != 0`.
	pub fn new_layouts_data<I>(width: u32, layouts_data: I) -> Result<Self, ()>
	where
		I: ExactSizeIterator<Item = (ImageLayout, D)>
	{
		if layouts_data.len() % width as usize != 0 {
			return Err(())
		}

		let matrix =
			layouts_data.map(|(l, d)| ImageLayoutMatrixEntry { layout: l, data: d }).collect();

		Ok(ImageLayoutMatrix { matrix, width })
	}

	pub fn height(&self) -> u32 { self.matrix.len() as u32 / self.width }

	/// Panics if range is out of bounds.
	pub fn iter_subresource_range(&self, range: ImageSubresourceRange) -> ImageLayoutMatrixIter<D> {
		if range.array_layers_offset >= self.width || range.mipmap_levels_offset >= self.height() {
			panic!(
				"Subresource range {:?} out of bounds of layout matrix {}x{}",
				range,
				self.width,
				self.height()
			)
		}

		ImageLayoutMatrixIter::new(&self.matrix, self.width as usize, range)
	}

	/// Panics if range is out of bounds.
	pub fn iter_subresource_range_mut(
		&mut self, range: ImageSubresourceRange
	) -> ImageLayoutMatrixIterMut<D> {
		if range.array_layers_offset >= self.width || range.mipmap_levels_offset >= self.height() {
			panic!(
				"Subresource range {:?} out of bounds of layout matrix {}x{}",
				range,
				self.width,
				self.height()
			)
		}

		ImageLayoutMatrixIterMut::new(&mut self.matrix, self.width as usize, range)
	}
}

// Implements iterator and mut iterator for layout matrix.
//
// Macros can't expand to multiple items, so we have to call the macro
// three times for each iterator.
// Also can't expand to incomplete items, so the @inner_return parts
// are a little hacky.
macro_rules! iterator_impl_macro {
	(@strct $vi: vis $name: ident $($ref_type: ident)?) => {
		#[derive(Debug)]
		$vi struct $name<'a, D> {
			matrix: &'a $( $ref_type )? Vec<ImageLayoutMatrixEntry<D>>,
			base_offset: usize,
			matrix_width: usize,

			width: usize,
			height: usize,
			current: usize
		}
	};
	(@strct_impl $vi: vis $name: ident $($ref_type: ident)?) => {
		impl<'a, D> $name<'a, D> {
			$vi fn new(
				matrix: &'a $( $ref_type )? Vec<ImageLayoutMatrixEntry<D>>,
				matrix_width: usize,
				range: ImageSubresourceRange
			) -> Self {
				$name {
					matrix,
					base_offset: range.array_layers_offset as usize
						+ range.mipmap_levels_offset as usize * matrix_width,
					matrix_width,

					width: range.array_layers.get() as usize,
					height: range.mipmap_levels.get() as usize,

					current: 0
				}
			}
		}
	};
	(@iterator $name: ident $($ref_type: ident)?) => {
		impl<'a, D> Iterator for $name<'a, D> {
			type Item = &'a $( $ref_type )? ImageLayoutMatrixEntry<D>;

			fn next(&mut self) -> Option<Self::Item> {
				let x_current = self.current % self.width;
				let y_current = self.current / self.width;
				if y_current >= self.height {
					None
				} else {
					let current_index = self.base_offset + y_current * self.matrix_width + x_current;
					self.current += 1;

					iterator_impl_macro!(@inner_return self.matrix[current_index], $( $ref_type )?)
				}
			}

			fn size_hint(&self) -> (usize, Option<usize>) {
				let remaining = self.width * self.height - self.current;
				(remaining, Some(remaining))
			}
		}
	};
	(@exactsizeiterator $name: ident $($ref_type: ident)?) => {
		impl<'a, D> ExactSizeIterator for $name<'a, D> {
			fn len(&self) -> usize {
				self.width * self.height - self.current
			}
		}
	};

	(@inner_return $self_matrix_current_index: expr,) => {
		Some(&$self_matrix_current_index)
	};
	(@inner_return $self_matrix_current_index: expr, $ref_type: ident) => {
		unsafe {
			let pointer = (&mut $self_matrix_current_index) as *mut _;
			let unbounded_reference = &mut *pointer;

			Some(unbounded_reference)
		}
	};
}
iterator_impl_macro!(@strct pub ImageLayoutMatrixIter);
iterator_impl_macro!(@strct_impl ImageLayoutMatrixIter);
iterator_impl_macro!(@iterator ImageLayoutMatrixIter);
iterator_impl_macro!(@exactsizeiterator ImageLayoutMatrixIter);

iterator_impl_macro!(@strct pub ImageLayoutMatrixIterMut mut);
iterator_impl_macro!(@strct_impl ImageLayoutMatrixIterMut mut);
iterator_impl_macro!(@iterator ImageLayoutMatrixIterMut mut);
iterator_impl_macro!(@exactsizeiterator ImageLayoutMatrixIterMut mut);


/// One field of the ImageLayoutMatrix.
#[derive(Debug)]
pub struct ImageLayoutMatrixEntry<D> {
	/// The layout of this field.
	pub layout: ImageLayout,
	/// The user data.
	pub data: D
}
impl<D: Clone> Clone for ImageLayoutMatrixEntry<D> {
	fn clone(&self) -> Self {
		ImageLayoutMatrixEntry { layout: self.layout, data: self.data.clone() }
	}
}
impl<D: Copy + Clone> Copy for ImageLayoutMatrixEntry<D> {}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_image_layout_matrix() {
		let mut matrix =
			ImageLayoutMatrix::new_data(4, ImageLayout::Undefined, (0 .. 16).into_iter()).unwrap();

		let mut iter = matrix.iter_subresource_range_mut(ImageSubresourceRange {
			array_layers: crate::NONZERO_ONE,
			array_layers_offset: 1,

			mipmap_levels: unsafe { std::num::NonZeroU32::new_unchecked(2) },
			mipmap_levels_offset: 1
		});

		assert_eq!(iter.len(), 2);
		assert_eq!(iter.next().unwrap().data, 5);
		assert_eq!(iter.next().unwrap().data, 9);
		assert!(iter.next().is_none());
	}
}
