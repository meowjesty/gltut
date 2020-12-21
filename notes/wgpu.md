# Notes on the WGPU (the JS spec)

## 3.4 Programming Model

### 3.4.1 Timelines

When trying to write data into a GPU buffer via `GPUBuffer.mapAsync()`, the user is responsible
for checking when the buffer is accessible for writing. This function returns a promise that the
user must handle to detect if the buffer is currently being used.

More details at example 3 of
[programming-model-timelines](https://gpuweb.github.io/gpuweb/#programming-model-timelines).

## Staging Belt

The code in `vange-rs` uses staging buffers instead of the staging belt. It has a global staging
buffer and uses the `encoder.copy_buffer_to_buffer` approach as seem
[here](https://github.com/kvark/vange-rs/blob/master/src/render/mod.rs#L573-L584).
