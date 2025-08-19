use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Deserialize, Serialize)]
pub struct Settings {
    pub window_title: String,
    pub resolution: f64,
    pub rescan_delay: u64,
    pub reference_width: i32,
    pub reference_height: i32,
    pub convert_to_grayscale: bool,
    pub templates: Vec<TemplateSettings>,
    pub random_offset: RandomOffsetSettings,
    pub human_like_movement: HumanLikeMovementSettings,
    pub automation: Automation,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Automation {
    pub merge: Merge,
    pub shtorm: Shtorm,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Merge {
    pub enabled: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Shtorm {
    pub enabled: bool,
    pub retries: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RandomOffsetSettings {
    pub enabled: bool,
    pub max_x_offset: i32,
    pub max_y_offset: i32,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HumanLikeMovementSettings {
    pub enabled: bool,
    pub max_deviation: f64, // Максимальное отклонение от прямой линии (в пикселях)
    pub speed_variation: f64, // Вариация скорости (0.0 - 1.0)
    pub curve_smoothness: usize, // Количество промежуточных точек для кривой
    pub min_pause_ms: u64,  // Минимальная пауза между движениями
    pub max_pause_ms: u64,  // Максимальная пауза между движениями
    pub base_speed: f64,
    pub min_down_ms: u64,
    pub max_down_ms: u64,
    pub min_up_ms: u64,
    pub max_up_ms: u64,
    pub min_move_delay_ms: u64,
    pub max_move_delay_ms: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TemplateSettings {
    pub name: String,
    pub path: String,
    pub threshold: f64,
    pub min_distance: f32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub resolution: Option<f64>,
    #[serde(default)]
    pub always_active: bool,
}
