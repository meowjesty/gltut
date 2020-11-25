
use anyhow::*;
use glob::glob;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fs::{read_to_string, write};
use std::path::PathBuf;

struct ShaderData {
    src: String,
    src_path: PathBuf,
    spirv_path: PathBuf,
    kind: shaderc::ShaderKind,
}

impl ShaderData {
    pub fn load(src_path: PathBuf) -> Result<Self> {
        let extension = src_path
            .extension()
            .context("File has no extension")?
            .to_str()
            .context("Extension cannot be converted to &str")?;
        let kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            _ => bail!("Unsupported shader: {}", src_path.display()),
        };

        let src = read_to_string(src_path.clone())?;
        let spirv_path = src_path.with_extension(format!("{}.spv", extension));

        Ok(Self {
            src,
            src_path,
            spirv_path,
            kind,
        })
    }
}

fn main() -> Result<()> {
    // Collect all shaders recursively within /src/
    let mut shader_paths = Vec::with_capacity(16);
    shader_paths.extend(glob("./src/**/*.vert")?);
    shader_paths.extend(glob("./src/**/*.frag")?);
    shader_paths.extend(glob("./src/**/*.comp")?);

    // This could be parallelized
    let shaders = shader_paths
        .into_par_iter()
        .map(|glob_result| ShaderData::load(glob_result?))
        .collect::<Vec<Result<_>>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let mut compiler = shaderc::Compiler::new().context("Unable to create shader compiler")?;

    // This can't be parallelized. The [shaderc::Compiler] is not
    // thread safe. Also, it creates a lot of resources. You could
    // spawn multiple processes to handle this, but it would probably
    // be better just to only compile shaders that have been changed
    // recently.
    for shader in shaders {
        // This tells cargo to rerun this script if something in /src/ changes.
        println!("cargo:rerun-if-changed={:?}", shader.src_path);

        let compiled = compiler.compile_into_spirv(
            &shader.src,
            shader.kind,
            &shader.src_path.to_str().unwrap(),
            "main",
            None,
        )?;
        write(shader.spirv_path, compiled.as_binary_u8())?;
    }

    Ok(())
}
