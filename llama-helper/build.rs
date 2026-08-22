// llama-helper/build.rs
//
// Only the `cuda` feature needs `libnccl`. Upstream llama-cpp-sys-2 0.1.146
// (the version pinned in Cargo.lock) doesn't set GGML_CUDA_NCCL=OFF the way
// later releases do (see https://github.com/utilityai/llama-cpp-rs/issues/1055),
// so on a host where CMake's find_package(NCCL) succeeds, ggml-cuda ends up
// depending on NCCL symbols that the crate's own build.rs never links against
// (a static library's PRIVATE link deps don't propagate to the final Rust
// binary). This is the same consumer-side workaround the upstream issue
// documents: explicitly link nccl ourselves, but only when the cuda feature
// is actually enabled — unconditionally emitting `-lnccl` broke the cpu and
// vulkan builds, which have no NCCL installed at all.
fn main() {
    if cfg!(feature = "cuda") {
        // NVIDIA's official CUDA apt repo (see .github/workflows/build-linux.yml,
        // "Install CUDA Toolkit" step) installs libnccl under the versioned CUDA
        // toolkit lib directory, not /usr/lib. Search common locations so both
        // CI (NVIDIA apt repo) and a local dev machine (e.g. an nvidia-cuda-toolkit
        // apt/pacman install, which does use /usr/lib) can find it.
        for path in [
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
        ] {
            println!("cargo:rustc-link-search=native={path}");
        }
        println!("cargo:rustc-link-lib=nccl");
    }
}
