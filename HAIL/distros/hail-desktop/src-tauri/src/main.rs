// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct SidecarResponse {
    status: Option<String>,
    distro: Option<String>,
    kernel: Option<String>,
    skills_count: Option<usize>,
    error: Option<String>,
}

#[tauri::command]
fn get_status() -> String {
    format!(
        "{{\"status\": \"online\", \"distro\": \"HAIL Desktop (Tauri Host)\", \"kernel\": \"HAIL Core v0.1\"}}"
    )
}

#[tauri::command]
fn recall_memory(query: String) -> String {
    format!(
        "{{\"status\": \"success\", \"query\": \"{}\", \"results\": [\"Memory recalled from HAIL Core\"]}}",
        query
    )
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_status, recall_memory])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
