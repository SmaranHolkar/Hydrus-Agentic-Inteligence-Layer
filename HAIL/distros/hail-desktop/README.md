# HAIL Desktop — Tauri Distribution

This distribution packages **HAIL Core** (`hail_core`) with **Tauri v2**, providing a native, lightweight, GPU-accelerated desktop application for macOS, Windows, and Linux.

## Architecture

- **Frontend Host**: Native WebKit/WebView2 window displaying the editorial brutalist interface ([distros/hail-web/](file:///d:/HydrusOPT/distros/hail-web/)).
- **Rust Host (`src-tauri/src/main.rs`)**: Manages desktop windows, native menu bars, system tray, and IPC bridges.
- **Python Sidecar (`sidecar.py`)**: Executes `hail_core` memory lattice persistence, skills engine, and local model runner interfaces over JSON-RPC.

## Prerequisites

1. **Rust Toolchain**: Install Rust via [rustup.rs](https://rustup.rs/) (`rustc >= 1.75`).
2. **Node.js / Tauri CLI**: Install `@tauri-apps/cli`:
   ```bash
   npm install -g @tauri-apps/cli@latest
   ```

## Development & Building

### 1. Test Python Sidecar IPC Bridge
```bash
$env:PYTHONPATH="src"
python distros/hail-desktop/sidecar.py "{\"action\": \"status\"}"
```

### 2. Launch Desktop App in Development Mode
```bash
cd distros/hail-desktop
cargo tauri dev
```

### 3. Build Standalone Installer Package (.exe / .dmg / .deb)
```bash
cd distros/hail-desktop
cargo tauri build
```
The compiled single-file installer will be output in `src-tauri/target/release/bundle/`.
