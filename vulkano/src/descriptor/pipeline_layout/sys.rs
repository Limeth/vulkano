// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::{error, fmt, mem, ptr, sync::Arc};

use crate::{check_errors, Error, OomError, VulkanObject};
use vk_sys as vk;

use crate::{
	descriptor::{
		descriptor::{DescriptorDesc, ShaderStages},
		descriptor_set::UnsafeDescriptorSetLayout,
		pipeline_layout::{
			PipelineLayoutDesc,
			PipelineLayoutDescPcRange,
			PipelineLayoutLimitsError
		}
	},
	device::{Device, DeviceOwned}
};

/// Wrapper around the `PipelineLayout` Vulkan object. Describes to the Vulkan implementation the
/// descriptor sets and push constants available to your shaders
pub struct PipelineLayout {
	device: Arc<Device>,
	layout: vk::PipelineLayout,
	layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
	desc: PipelineLayoutDescAggregation,
}

impl PipelineLayout
{
	/// Creates a new `PipelineLayout`.
	pub fn new<L: PipelineLayoutDesc>(
		device: Arc<Device>, desc: L
	) -> Result<Arc<PipelineLayout>, PipelineLayoutCreationError> {
		let desc = desc.aggregate();
		let vk = device.pointers();

		desc.check_against_limits(&device)?;

		// Building the list of `UnsafeDescriptorSetLayout` objects.
		let layouts = {
			let mut layouts: SmallVec<[_; 16]> = SmallVec::new();
			for num in 0 .. desc.num_sets() {
				let sets_iter = 0 .. desc.num_bindings_in_set(num).unwrap_or(0);
				let desc_iter = sets_iter.map(|d| desc.descriptor(num, d));
				let layout = Arc::new(UnsafeDescriptorSetLayout::new(device.clone(), desc_iter)?);

				layouts.push(layout);
			}
			layouts
		};

		// Grab the list of `vkDescriptorSetLayout` objects from `layouts`.
		let layouts_ids =
			layouts.iter().map(|l| l.internal_object()).collect::<SmallVec<[_; 16]>>();

		// Builds a list of `vkPushConstantRange` that describe the push constants.
		let push_constants = {
			let mut out: SmallVec<[_; 8]> = SmallVec::new();

			for pc_id in 0 .. desc.num_push_constants_ranges() {
				let PipelineLayoutDescPcRange { offset, size, stages } = {
					match desc.push_constants_range(pc_id) {
						Some(o) => o,
						None => continue
					}
				};

				if stages == ShaderStages::none() || size == 0 || (size % 4) != 0 {
					return Err(PipelineLayoutCreationError::InvalidPushConstant)
				}

				out.push(vk::PushConstantRange {
					stageFlags: stages.into_vulkan_bits(),
					offset: offset as u32,
					size: size as u32
				});
			}

			out
		};

		// Each bit of `stageFlags` must only be present in a single push constants range.
		// We check that with a debug_assert because it's supposed to be enforced by the
		// `PipelineLayoutDesc`.
		debug_assert!({
			let mut stages = 0;
			let mut outcome = true;
			for pc in push_constants.iter() {
				if (stages & pc.stageFlags) != 0 {
					outcome = false;
					break
				}
				stages &= pc.stageFlags;
			}
			outcome
		});

		// FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device doesn't
		//        have tess shaders enabled

		// Build the final object.
		let layout = unsafe {
			let infos = vk::PipelineLayoutCreateInfo {
				sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				pNext: ptr::null(),
				flags: 0, // reserved
				setLayoutCount: layouts_ids.len() as u32,
				pSetLayouts: layouts_ids.as_ptr(),
				pushConstantRangeCount: push_constants.len() as u32,
				pPushConstantRanges: push_constants.as_ptr()
			};

			let mut output = mem::uninitialized();
			check_errors(vk.CreatePipelineLayout(
				device.internal_object(),
				&infos,
				ptr::null(),
				&mut output
			))?;
			output
		};

		Ok(Arc::new(PipelineLayout { device: device.clone(), layout, layouts, desc }))
	}

	/// Returns the description of the pipeline layout.
	pub fn desc(&self) -> &PipelineLayoutDescAggregation {
		&self.desc
	}

	pub fn sys(&self) -> PipelineLayoutSys { PipelineLayoutSys(&self.layout) }

	pub fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
		self.layouts.get(index)
	}
}

unsafe impl PipelineLayoutDesc for PipelineLayout {
	fn num_sets(&self) -> usize {
		self.desc.num_sets()
	}

	fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
		self.desc.num_bindings_in_set(set)
	}

	fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
		self.desc.descriptor(set, binding)
	}

	fn num_push_constants_ranges(&self) -> usize {
		self.desc.num_push_constants_ranges()
	}

	fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
		self.desc.push_constants_range(num)
	}

	fn aggregate(self) -> PipelineLayoutDescAggregation {
		self.desc.clone()
	}
}

unsafe impl DeviceOwned for PipelineLayout {
	fn device(&self) -> &Arc<Device> { &self.device }
}

impl fmt::Debug for PipelineLayout
{
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt.debug_struct("PipelineLayout")
			.field("raw", &self.layout)
			.field("device", &self.device)
			.field("desc", &self.desc)
			.finish()
	}
}

impl Drop for PipelineLayout {
	fn drop(&mut self) {
		unsafe {
			let vk = self.device.pointers();
			vk.DestroyPipelineLayout(self.device.internal_object(), self.layout, ptr::null());
		}
	}
}

/// Opaque object that is borrowed from a `PipelineLayout`.
///
/// This object exists so that we can pass it around without having to be generic over the template
/// parameter of the `PipelineLayout`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct PipelineLayoutSys<'a>(&'a vk::PipelineLayout);

unsafe impl<'a> VulkanObject for PipelineLayoutSys<'a> {
	type Object = vk::PipelineLayout;

	const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT;

	fn internal_object(&self) -> vk::PipelineLayout { *self.0 }
}

/// Error that can happen when creating a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutCreationError {
	/// Not enough memory.
	OomError(OomError),
	/// The pipeline layout description doesn't fulfill the limit requirements.
	LimitsError(PipelineLayoutLimitsError),
	/// The list of stages must not be empty,
	/// the size must not be 0, and the size must be a multiple or 4.
	InvalidPushConstant
}
impl fmt::Display for PipelineLayoutCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			PipelineLayoutCreationError::OomError(e) => e.fmt(f),
			PipelineLayoutCreationError::LimitsError(e) => e.fmt(f),
			PipelineLayoutCreationError::InvalidPushConstant
			=> write!(f, "The list of stages must not be empty, the size must not be 0, and the size must be a multiple or 4")
		}
	}
}
impl error::Error for PipelineLayoutCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			PipelineLayoutCreationError::OomError(e) => e.source(),
			PipelineLayoutCreationError::LimitsError(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for PipelineLayoutCreationError {
	fn from(err: OomError) -> PipelineLayoutCreationError {
		PipelineLayoutCreationError::OomError(err)
	}
}
impl From<PipelineLayoutLimitsError> for PipelineLayoutCreationError {
	fn from(err: PipelineLayoutLimitsError) -> PipelineLayoutCreationError {
		PipelineLayoutCreationError::LimitsError(err)
	}
}
impl From<Error> for PipelineLayoutCreationError {
	fn from(err: Error) -> PipelineLayoutCreationError {
		match err {
			Error::OutOfHostMemory => PipelineLayoutCreationError::OomError(OomError::from(err)),
			Error::OutOfDeviceMemory => PipelineLayoutCreationError::OomError(OomError::from(err)),
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PipelineLayoutDescAggregation {
	descriptor_sets: Vec<Vec<DescriptorDesc>>,
	push_constants_ranges: Vec<PipelineLayoutDescPcRange>,
}

impl PipelineLayoutDescAggregation {
	pub fn from(other: impl PipelineLayoutDesc) -> Self {
		let num_sets = other.num_sets();
		let mut descriptor_sets = Vec::with_capacity(num_sets);

		for set in 0..other.num_sets() {
			let num_bindings = other.num_bindings_in_set(set).unwrap();
			let mut bindings = Vec::with_capacity(num_bindings);

			for binding in 0..num_bindings {
				bindings.push(other.descriptor(set, binding).unwrap());
			}

			descriptor_sets.push(bindings);
		}

		let num_pcrs = other.num_push_constants_ranges();
		let mut pcrs = Vec::with_capacity(num_pcrs);

		for pcr in 0..num_pcrs {
			pcrs.push(other.push_constants_range(pcr).unwrap());
		}

		PipelineLayoutDescAggregation {
			descriptor_sets,
			push_constants_ranges: pcrs,
		}
	}
}

unsafe impl PipelineLayoutDesc for PipelineLayoutDescAggregation {
	fn num_sets(&self) -> usize {
		self.descriptor_sets.len()
	}

	fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
		self.descriptor_sets.get(set).map(Vec::len)
	}

	fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
		self.descriptor_sets.get(set).and_then(|set| set.get(binding)).map(Clone::clone)
	}

	fn num_push_constants_ranges(&self) -> usize {
		self.push_constants_ranges.len()
	}

	fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
		self.push_constants_ranges.get(num).map(Clone::clone)
	}

	fn aggregate(self) -> Self {
		self
	}
}

// TODO: restore
// #[cfg(test)]
// mod tests {
// use std::iter;
// use std::sync::Arc;
// use crate::descriptor::descriptor::ShaderStages;
// use crate::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
// use crate::descriptor::pipeline_layout::sys::PipelineLayout;
// use crate::descriptor::pipeline_layout::sys::PipelineLayoutCreationError;
//
// #[test]
// fn empty() {
// let (device, _) = gfx_dev_and_queue!();
// let _layout = PipelineLayout::new(&device, iter::empty(), iter::empty()).unwrap();
// }
//
// #[test]
// fn wrong_device_panic() {
// let (device1, _) = gfx_dev_and_queue!();
// let (device2, _) = gfx_dev_and_queue!();
//
// let set = match UnsafeDescriptorSetLayout::raw(device1, iter::empty()) {
// Ok(s) => Arc::new(s),
// Err(_) => return
// };
//
// assert_should_panic!({
// let _ = PipelineLayout::new(&device2, Some(&set), iter::empty());
// });
// }
//
// #[test]
// fn invalid_push_constant_stages() {
// let (device, _) = gfx_dev_and_queue!();
//
// let push_constant = (0, 8, ShaderStages::none());
//
// match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
// Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
// _ => panic!()
// }
// }
//
// #[test]
// fn invalid_push_constant_size1() {
// let (device, _) = gfx_dev_and_queue!();
//
// let push_constant = (0, 0, ShaderStages::all_graphics());
//
// match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
// Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
// _ => panic!()
// }
// }
//
// #[test]
// fn invalid_push_constant_size2() {
// let (device, _) = gfx_dev_and_queue!();
//
// let push_constant = (0, 11, ShaderStages::all_graphics());
//
// match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
// Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
// _ => panic!()
// }
// }
// }
