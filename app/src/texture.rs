use core::mem::*;
use std::{
    num::{NonZeroU32, NonZeroU8},
    path,
};

use image::GenericImageView;

#[derive(Debug)]
pub struct Texture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
}

impl Texture {
    pub const SIZE: wgpu::BufferAddress = size_of::<[f32; 2]>() as wgpu::BufferAddress;
    pub const DESCRIPTOR: wgpu::VertexBufferDescriptor<'static> = wgpu::VertexBufferDescriptor {
        stride: Self::SIZE,
        step_mode: wgpu::InputStepMode::Vertex,
        attributes: &[wgpu::VertexAttributeDescriptor {
            offset: 0,
            shader_location: 1,
            format: wgpu::VertexFormat::Float2,
        }],
    };

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// NOTE(alex): Special kind of texture that is used for depth testing (**Depth Buffer**).
    ///
    /// This texture will be created as a render output (`OUTPUT_ATTACHMENT`, similar to how a
    /// `SwapChain` is a render target).
    ///
    /// Just relying on Z-axis ordering doesn't work very well in 3D, we're getting weird
    /// rendering behaviour where some sides disappear, even though they're still half in view.
    /// Depth testing exists to solve this issue.
    ///
    /// The main idea here is to have a black and white texture that stores the _z_-coordinate of
    /// the rendered pixels, and check (via the pipeline `depth_stencil_state.depth_compare`)
    /// whether to keep or replace the data.
    /// No need to sort objects to try and maintain a _z_-ordered set of draw calls.
    pub fn create_depth_texture(
        device: &wgpu::Device,
        swap_chain_descriptor: &wgpu::SwapChainDescriptor,
    ) -> Texture {
        // NOTE(alex): Depth texture must be the same size of the screen, again, it kinds comes
        // back to how similar this is to the swap chain image idea.
        let size = wgpu::Extent3d {
            width: swap_chain_descriptor.width,
            height: swap_chain_descriptor.height,
            depth: 1,
        };

        let descriptor = &wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            // NOTE(alex): `TextureUsage::OUTPUT_ATTACHMENT` is the same as we have in the
            // `SwapChainDescriptor`, as this usage means we're rendering to this texture.
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        };
        let texture = device.create_texture(&descriptor);

        let view_descriptor = wgpu::TextureViewDescriptor {
            label: Some("Depth Texture view"),
            format: Some(Self::DEPTH_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: NonZeroU32::new(1),
            base_array_layer: 0,
            array_layer_count: NonZeroU32::new(1),
        };
        let view = texture.create_view(&view_descriptor);

        // TODO(alex): Some of these `_filter`s have values specified in the glTF file.
        // I need to find a way to create these descriptors in a more dynamic fashion, allowing
        // it to properly handle different attributes for the many imported models.
        let sampler_descriptor = wgpu::SamplerDescriptor {
            label: Some("Depth Texture sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            // NOTE(alex): This is not the actual depth testing compare function!
            compare: Some(wgpu::CompareFunction::LessEqual),
            // TODO(alex): Is there an equivalent to vulkan's
            // `VkPhysicalDeviceProperties.limits.maxSamplerAnisotropy`?
            anisotropy_clamp: NonZeroU8::new(16),
        };
        let sampler = device.create_sampler(&sampler_descriptor);

        Self {
            texture,
            view,
            sampler,
        }
    }

    pub fn load<P: AsRef<path::Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: P,
    ) -> Result<Self, String> {
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str();

        let image = image::open(path).map_err(|err| err.to_string())?;
        Self::from_image(device, queue, &image)
    }

    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> Result<Texture, String> {
        let image = image::load_from_memory(bytes).map_err(|err| err.to_string())?;
        Self::from_image(device, queue, &image)
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
    ) -> Result<Texture, String> {
        let rgba = image.to_rgba8();
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };
        // NOTE(alex): This is somewhat equivalent to the vulkan `VkImage`, the main difference is
        // that memory handling is easier in wgpu.
        // NOTE(alex): 1D images can be used to store an array of data or gradient, 2D are mainly
        // used for textures (here), while 3D images can be used to store voxel volumes.
        // NOTE(alex): You could use the shader to access the buffer of pixels directly, but
        // it isn't optimal.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // TODO(alex): zeux said that this is the modern format that should be used, why?
            // format: wgpu::TextureFormat::Depth32Float,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // NOTE(alex): `TextureUsage::SAMPLED` means optimal for shader, while
            // `TextureUsage::COPY_DST` is optimal as the destination in a transfer op (copy to).
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });
        // NOTE(alex): We only have to write this image to a buffer once (if we're not changing the
        // image), so copying the data during the renderer initialization is fine.
        // There is a legacy way of doing it, that involves copying it as a buffer, and doing the
        // `encoder`, `staging_belt` dance (this is how vulkan works with its staging buffers).
        // Notice that we can treat the texture as a buffer of bytes here, and later think about
        // how we want to apply modifications to it. The `TextureView` is how the data will be
        // actually used, so before we just need to specify what kind of bytes we have, how are
        // they formatted, size, and usage.
        // Later the `TextureView` will be created based on the `Texture`, but it doesn't really
        // care about buffer stuff.
        // And the `Sampler`s are even more independent, as they have no connection to neither
        // `Texture` or `TextureView`.
        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * size.width,
                rows_per_image: size.height,
            },
            size,
        );
        // NOTE(alex): Images are accessed by views rather than directly.
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture view"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: NonZeroU32::new(1),
            base_array_layer: 0,
            array_layer_count: NonZeroU32::new(1),
        });
        // NOTE(alex): How the texture (the texels) will be mapped into geometry, what kind of
        // filters should it apply.
        // The sampler is independent of the image (texture), it's a descriptor that can be used
        // for any image we want and it will apply these properties (filters) to it. It's an
        // interface to extract colors from a texture.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            // NOTE(alex): Texels will be compared to a value, and then the result will be used in
            // filtering operations (useful for shadow maps).
            compare: None,
            // TODO(alex): Is there an equivalent to vulkan's
            // `VkPhysicalDeviceProperties.limits.maxSamplerAnisotropy`?
            anisotropy_clamp: NonZeroU8::new(16),
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}
