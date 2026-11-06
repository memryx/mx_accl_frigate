# MemryX Runtime Libraries for Frigate

This repository contains the **native runtime libraries** for integrating MemryX SDK 3.0 with [Frigate](https://github.com/blakeblackshear/frigate).

The Python runtime is provided separately by the official `memryx-runtime` and `memryx-core` packages. The compiler and full SDK are not included in this repository.

## Included Libraries

The `memryx/x86/` directory contains the x86-64 libraries from SDK packages `3.0.0-10`:

- `libmemx.so.3.0.0`, with the `libmemx.so` symlink
- `libmx_accl.so.2`, with the `libmx_accl.so` symlink

The `memryx/arm/` directory will contain the matching ARM64 libraries with the same filenames and symlinks. ARM64 files are pending and will be added later.

These libraries are shared across Python versions, but each architecture requires its own binaries. Python-specific extensions come from the official runtime wheels.

## Status

The `sdk-3.0` branch is for SDK 3.0 development and testing. ARM64 libraries will be added later.

Frigate's MemryX runtime manifest must be updated to install these libraries alongside the official Python runtime packages. The existing SDK 2.1 manifest does not support this layout.
