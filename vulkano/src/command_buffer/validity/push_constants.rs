// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt};

use crate::descriptor::pipeline_layout::{
	PipelineLayoutDesc,
	PipelineLayoutPushConstantsCompatible
};

/// Checks whether push constants are compatible with the pipeline.
pub fn check_push_constants_validity<Pl, Pc>(
	pipeline: &Pl, push_constants: &Pc
) -> Result<(), CheckPushConstantsValidityError>
where
	Pl: ?Sized + PipelineLayoutDesc,
	Pc: ?Sized
{
	if !pipeline.is_compatible(push_constants) {
		return Err(CheckPushConstantsValidityError::IncompatiblePushConstants)
	}

	Ok(())
}

/// Error that can happen when checking push constants validity.
#[derive(Debug, Copy, Clone)]
pub enum CheckPushConstantsValidityError {
	/// The push constants are incompatible with the pipeline layout.
	IncompatiblePushConstants
}
impl fmt::Display for CheckPushConstantsValidityError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			CheckPushConstantsValidityError::IncompatiblePushConstants => {
				write!(f, "The push constants are incompatible with the pipeline layout")
			}
		}
	}
}
impl error::Error for CheckPushConstantsValidityError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> { None }
}
