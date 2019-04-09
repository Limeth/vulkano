// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{error, fmt, u32};

use crate::{
	descriptor::pipeline_layout::PipelineLayoutNotSupersetError,
	pipeline::{
		input_assembly::PrimitiveTopology,
		shader::ShaderInterfaceMismatchError,
		vertex::IncompatibleVertexDefinitionError
	},
	Error,
	OomError
};

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsPipelineCreationError {
	/// Not enough memory.
	OomError(OomError),

	/// The pipeline layout is not compatible with what the shaders expect.
	IncompatiblePipelineLayout(PipelineLayoutNotSupersetError),

	/// The interface between the vertex shader and the geometry shader mismatches.
	VertexGeometryStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the vertex shader and the tessellation control shader mismatches.
	VertexTessControlStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the vertex shader and the fragment shader mismatches.
	VertexFragmentStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the tessellation control shader and the tessellation evaluation
	/// shader mismatches.
	TessControlTessEvalStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the tessellation evaluation shader and the geometry shader
	/// mismatches.
	TessEvalGeometryStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the tessellation evaluation shader and the fragment shader
	/// mismatches.
	TessEvalFragmentStagesMismatch(ShaderInterfaceMismatchError),

	/// The interface between the geometry shader and the fragment shader mismatches.
	GeometryFragmentStagesMismatch(ShaderInterfaceMismatchError),

	/// The output of the fragment shader is not compatible with what the render pass subpass expects.
	FragmentShaderRenderPassIncompatible,

	/// The vertex definition is not compatible with the input of the vertex shader.
	IncompatibleVertexDefinition(IncompatibleVertexDefinitionError),

	/// The maximum stride value for vertex input has been exceeded.
	MaxVertexInputBindingStrideExceeded {
		/// Index of the faulty binding.
		binding: usize,
		/// Maximum allowed value.
		max: usize,
		/// Value that was passed.
		obtained: usize
	},

	/// The maximum number of vertex sources has been exceeded.
	MaxVertexInputBindingsExceeded {
		/// Maximum allowed value.
		max: usize,
		/// Value that was passed.
		obtained: usize
	},

	/// The maximum offset for a vertex attribute has been exceeded.
	///
	/// This means that your vertex struct is too large.
	MaxVertexInputAttributeOffsetExceeded {
		/// Maximum allowed value.
		max: usize,
		/// Value that was passed.
		obtained: usize
	},

	/// The maximum number of vertex attributes has been exceeded.
	MaxVertexInputAttributesExceeded {
		/// Maximum allowed value.
		max: usize,
		/// Value that was passed.
		obtained: usize
	},

	/// The user requested to use primitive restart, but the primitive topology doesn't support it.
	PrimitiveDoesntSupportPrimitiveRestart {
		/// The topology that doesn't support primitive restart.
		primitive: PrimitiveTopology
	},

	/// The `multi_viewport` feature must be enabled in order to use multiple viewports at once.
	MultiViewportFeatureNotEnabled,

	/// The maximum number of viewports has been exceeded.
	MaxViewportsExceeded {
		/// Maximum allowed value.
		max: u32,
		/// Value that was passed.
		obtained: u32
	},

	/// The maximum dimensions of viewports has been exceeded.
	MaxViewporImageDimensionsExceeded,

	/// The minimum or maximum bounds of viewports have been exceeded.
	ViewportBoundsExceeded,

	/// The `wide_lines` feature must be enabled in order to use a line width greater than 1.0.
	WideLinesFeatureNotEnabled,

	/// The `depth_clamp` feature must be enabled in order to use depth clamping.
	DepthClampFeatureNotEnabled,

	/// The `depth_bias_clamp` feature must be enabled in order to use a depth bias clamp different from 0.0.
	DepthBiasClampFeatureNotEnabled,

	/// The `fill_mode_non_solid` feature must be enabled in order to use a polygon mode different from `Fill`.
	FillModeNonSolidFeatureNotEnabled,

	/// The `depth_bounds` feature must be enabled in order to use depth bounds testing.
	DepthBoundsFeatureNotEnabled,

	/// The requested stencil test is invalid.
	WrongStencilState,

	/// The primitives topology does not match what the geometry shader expects.
	TopologyNotMatchingGeometryShader,

	/// The `geometry_shader` feature must be enabled in order to use geometry shaders.
	GeometryShaderFeatureNotEnabled,

	/// The `tessellation_shader` feature must be enabled in order to use tessellation shaders.
	TessellationShaderFeatureNotEnabled,

	/// The number of attachments specified in the blending does not match the number of attachments in the subpass.
	MismatchBlendingAttachmentsCount,

	/// The `independent_blend` feature must be enabled in order to use different blending operations per attachment.
	IndependentBlendFeatureNotEnabled,

	/// The `logic_op` feature must be enabled in order to use logic operations.
	LogicOpFeatureNotEnabled,

	/// The depth test requires a depth attachment but render pass has no depth attachment, or depth writing is enabled and the depth attachment is read-only.
	NoDepthAttachment,

	/// The stencil test requires a stencil attachment but render pass has no stencil attachment, or stencil writing is enabled and the stencil attachment is read-only.
	NoStencilAttachment,

	/// Tried to use a patch list without a tessellation shader, or a non-patch-list with a tessellation shader.
	InvalidPrimitiveTopology,

	/// The `maxTessellationPatchSize` limit was exceeded.
	MaxTessellationPatchSizeExceeded,

	/// The wrong type of shader has been passed.
	///
	/// For example you passed a vertex shader as the fragment shader.
	WrongShaderType,

	/// The `sample_rate_shading` feature must be enabled in order to use sample shading.
	SampleRateShadingFeatureNotEnabled,

	/// The `alpha_to_one` feature must be enabled in order to use alpha-to-one.
	AlphaToOneFeatureNotEnabled
}
impl fmt::Display for GraphicsPipelineCreationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			GraphicsPipelineCreationError::OomError(e) => e.fmt(f),
			GraphicsPipelineCreationError::IncompatiblePipelineLayout(e) => e.fmt(f),
			GraphicsPipelineCreationError::VertexGeometryStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::VertexTessControlStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::VertexFragmentStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(e) => e.fmt(f),
			GraphicsPipelineCreationError::IncompatibleVertexDefinition(e) => e.fmt(f),

			GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible
			=> write!(f, "The output of the fragment shader is not compatible with what the render pass subpass expects"),
			GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { binding, max, obtained }
			=> write!(f, "The maximum stride value ({}) for vertex input has been exceeded ({}): binding = {}", max, obtained, binding),
			GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { max, obtained }
			=> write!(f, "The maximum number of vertex sources ({}) has been exceeded ({})", max, obtained),
			GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { max, obtained }
			=> write!(f, "The maximum offset for a vertex attribute ({}) has been exceeded ({})", max, obtained),
			GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { max, obtained }
			=> write!(f, "The maximum number of vertex attributes ({}) has been exceeded ({})", max, obtained),
			GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { primitive }
			=> write!(f, "The user requested to use primitive restart, but the primitive topology ({:?}) doesn't support it", primitive),
			GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled
			=> write!(f, "The `multi_viewport` feature must be enabled in order to use multiple viewports at once"),
			GraphicsPipelineCreationError::MaxViewportsExceeded { max, obtained }
			=> write!(f, "The maximum number of viewports ({}) has been exceeded ({})", max, obtained),
			GraphicsPipelineCreationError::MaxViewporImageDimensionsExceeded
			=> write!(f, "The maximum dimensions of viewports has been exceeded"),
			GraphicsPipelineCreationError::ViewportBoundsExceeded
			=> write!(f, "The minimum or maximum bounds of viewports have been exceeded"),
			GraphicsPipelineCreationError::WideLinesFeatureNotEnabled
			=> write!(f, "The `wide_lines` feature must be enabled in order to use a line width greater than 1.0"),
			GraphicsPipelineCreationError::DepthClampFeatureNotEnabled
			=> write!(f, "The `depth_clamp` feature must be enabled in order to use depth clamping"),
			GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled
			=> write!(f, "The `depth_bias_clamp` feature must be enabled in order to use a depth bias clamp different from 0.0"),
			GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled
			=> write!(f, "The `fill_mode_non_solid` feature must be enabled in order to use a polygon mode different from `Fill`"),
			GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled
			=> write!(f, "The `depth_bounds` feature must be enabled in order to use depth bounds testing"),
			GraphicsPipelineCreationError::WrongStencilState
			=> write!(f, "The requested stencil test is invalid"),
			GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader
			=> write!(f, "The primitives topology does not match what the geometry shader expects"),
			GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled
			=> write!(f, "The `geometry_shader` feature must be enabled in order to use geometry shaders"),
			GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled
			=> write!(f, "The `tessellation_shader` feature must be enabled in order to use tessellation shaders"),
			GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount
			=> write!(f, "The number of attachments specified in the blending does not match the number of attachments in the subpass"),
			GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled
			=> write!(f, "The `independent_blend` feature must be enabled in order to use different blending operations per attachment"),
			GraphicsPipelineCreationError::LogicOpFeatureNotEnabled
			=> write!(f, "The `logic_op` feature must be enabled in order to use logic operations"),
			GraphicsPipelineCreationError::NoDepthAttachment
			=> write!(f, "The depth test requires a depth attachment but render pass has no depth attachment, or depth writing is enabled and the depth attachment is read-only"),
			GraphicsPipelineCreationError::NoStencilAttachment
			=> write!(f, "The stencil test requires a stencil attachment but render pass has no stencil attachment, or stencil writing is enabled and the stencil attachment is read-only"),
			GraphicsPipelineCreationError::InvalidPrimitiveTopology
			=> write!(f, "Tried to use a patch list without a tessellation shader, or a non-patch-list with a tessellation shader"),
			GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded
			=> write!(f, "The `maxTessellationPatchSize` limit was exceeded"),
			GraphicsPipelineCreationError::WrongShaderType
			=> write!(f, "The wrong type of shader has been passed"),
			GraphicsPipelineCreationError::SampleRateShadingFeatureNotEnabled
			=> write!(f, "The `sample_rate_shading` feature must be enabled in order to use sample shading"),
			GraphicsPipelineCreationError::AlphaToOneFeatureNotEnabled
			=> write!(f, "The `alpha_to_one` feature must be enabled in order to use alpha-to-one"),
		}
	}
}
impl error::Error for GraphicsPipelineCreationError {
	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match self {
			GraphicsPipelineCreationError::OomError(e) => e.source(),
			GraphicsPipelineCreationError::IncompatiblePipelineLayout(e) => e.source(),
			GraphicsPipelineCreationError::VertexGeometryStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::VertexTessControlStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::VertexFragmentStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(e) => e.source(),
			GraphicsPipelineCreationError::IncompatibleVertexDefinition(e) => e.source(),
			_ => None
		}
	}
}
impl From<OomError> for GraphicsPipelineCreationError {
	fn from(err: OomError) -> GraphicsPipelineCreationError {
		GraphicsPipelineCreationError::OomError(err)
	}
}
impl From<PipelineLayoutNotSupersetError> for GraphicsPipelineCreationError {
	fn from(err: PipelineLayoutNotSupersetError) -> GraphicsPipelineCreationError {
		GraphicsPipelineCreationError::IncompatiblePipelineLayout(err)
	}
}
impl From<IncompatibleVertexDefinitionError> for GraphicsPipelineCreationError {
	fn from(err: IncompatibleVertexDefinitionError) -> GraphicsPipelineCreationError {
		GraphicsPipelineCreationError::IncompatibleVertexDefinition(err)
	}
}
impl From<Error> for GraphicsPipelineCreationError {
	fn from(err: Error) -> GraphicsPipelineCreationError {
		match err {
			Error::OutOfHostMemory => {
				GraphicsPipelineCreationError::OomError(OomError::from(err))
			}
			Error::OutOfDeviceMemory => {
				GraphicsPipelineCreationError::OomError(OomError::from(err))
			}
			_ => panic!("unexpected error: {:?}", err)
		}
	}
}
