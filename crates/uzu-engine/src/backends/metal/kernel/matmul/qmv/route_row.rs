use metal::MTLGPUFamily;

use super::super::gemv::GemvTile;

#[derive(Clone, Copy)]
pub struct RouteRow {
    pub device_name: &'static str,
    pub apple_gpu_family: MTLGPUFamily,
    pub bits: u32,
    pub group: u32,
    pub m: u32,
    pub shapes: u8,
    pub tile: GemvTile,
}
