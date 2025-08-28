use crossterm::{execute, terminal::SetTitle};
use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::prelude::MatTraitConst;
use scrap2_bot::capture::AppError;
use scrap2_bot::capture::AppResult;
use scrap2_bot::capture::capture_window_by_title;
use scrap2_bot::capture::get_window_size;
use scrap2_bot::capture::is_cursor_in_window;
use scrap2_bot::drawing::display_results_as_table;
use scrap2_bot::moving::human_like_move;
use scrap2_bot::objectdetector::DetectionResult;
use scrap2_bot::objectdetector::ObjectDetector;
use scrap2_bot::processors::process_barrels;
use scrap2_bot::processors::process_magnets_cloud;
use scrap2_bot::settings::AntiCaptcha;
use scrap2_bot::settings::Automation;
use scrap2_bot::settings::HumanLikeMovementSettings;
use scrap2_bot::settings::Merge;
use scrap2_bot::settings::RandomOffsetSettings;
use scrap2_bot::settings::Settings;
use scrap2_bot::settings::Shtorm;
use scrap2_bot::utils;
use scrap2_bot::utils::check_and_suggest_window_size;
use scrap2_bot::utils::clear_screen;
use std::env;
use std::fs;
use std::thread;
use std::time::Duration;

fn load_or_create_settings(window_title: &str) -> AppResult<Settings> {
    let settings_path = "settings.json";

    if let Ok(settings_content) = fs::read_to_string(settings_path) {
        serde_json::from_str(&settings_content).map_err(Into::into)
    } else {
        let (width, height) = get_window_size(window_title)?;

        let settings = Settings {
            window_title: window_title.to_string(),
            resolution: 0.38,
            rescan_delay: 250,
            reference_width: width,
            reference_height: height,
            convert_to_grayscale: true,
            random_offset: RandomOffsetSettings {
                enabled: true,
                max_x_offset: 5,
                max_y_offset: 5,
            },
            human_like_movement: HumanLikeMovementSettings {
                // Добавлено
                enabled: true,
                max_deviation: 10.0,
                speed_variation: 0.3,
                curve_smoothness: 5,
                min_pause_ms: 10,
                max_pause_ms: 50,
                base_speed: 0.1,
                min_down_ms: 2,
                max_down_ms: 5,
                min_up_ms: 2,
                max_up_ms: 6,
                min_move_delay_ms: 5,
                max_move_delay_ms: 12,
            },
            automation: Automation {
                merge: Merge { enabled: true },
                shtorm: Shtorm {
                    enabled: true,
                    retries: 1,
                },
                anticaptcha: AntiCaptcha {
                    enabled: true,
                    mode: "mask"
                }
            },
            templates: Vec::new(),
        };

        let serialized = serde_json::to_string_pretty(&settings)?;
        fs::write(settings_path, serialized)?;

        Ok(settings)
    }
}

fn main() -> AppResult<()> {
    let args: Vec<String> = env::args().collect();
    let infinite_mode = args.iter().any(|arg| arg == "--infinite" || arg == "-i");
    let debug_mode = args.iter().any(|arg| arg == "--debug" || arg == "-d");
    execute!(std::io::stdout(), SetTitle("Scrap II Bot"))?;

    let window_title = "M2006C3MNG";
    let settings = load_or_create_settings(window_title)?;

    check_and_suggest_window_size(
        &settings.window_title,
        settings.reference_width,
        settings.reference_height,
    )?;

    let mut detector = ObjectDetector::new(settings.resolution);

    for template_settings in settings.templates.iter() {
        detector.add_template(
            &template_settings.name,
            &template_settings.path,
            template_settings.threshold,
            template_settings.min_distance,
            template_settings.red,
            template_settings.green,
            template_settings.blue,
            template_settings.resolution,
            template_settings.always_active,
        )?;
    }

    // Инициализируем начальный диапазон
    detector.active_range = (0, 50); // Начинаем с Empty + первые 5 бочек
    let mut last_frame_time = std::time::Instant::now();
    loop {
        let screenshot_path = "screenshot.png";
        let (window_x, window_y) =
            capture_window_by_title(&settings.window_title, screenshot_path)?;
        let mut image = imgcodecs::imread(screenshot_path, IMREAD_COLOR)
            .map_err(|e| AppError::ImageProcessing(format!("Failed to load screenshot: {}", e)))?;

        if image.empty() {
            return Err(AppError::ImageProcessing(
                "Loaded image is empty".to_string(),
            ));
        }

        let (detections, detection_time) =
            detector.detect_objects_optimized(&image, settings.convert_to_grayscale)?;

        let (window_width, window_height) = get_window_size(&settings.window_title)?;

        let is_on_window = is_cursor_in_window(window_x, window_y, window_width, window_height)?;

        let current_time = std::time::Instant::now();
        let frame_time = current_time.duration_since(last_frame_time).as_secs_f64();
        let fps = 1.0 / frame_time;
        last_frame_time = current_time;

        if debug_mode {
            detector.draw_detections(&mut image, &detections)?;
            imgcodecs::imwrite("result.png", &image, &Vector::new())?;
        }

        let (original_x, original_y) = utils::get_currect_mouse_potision()?;

        // Обработка облака мангинитов
        let cloud: Vec<DetectionResult> = detections
            .clone()
            .into_iter()
            .filter(|d| d.object_name.starts_with("Cloud"))
            .collect();

        if cloud.len() > 0 && settings.automation.shtorm.enabled {
            for _ in 1..settings.automation.shtorm.retries {
                process_magnets_cloud(window_x, window_y, window_width, window_height)?;

                // После обработки облака продолжаем основной цикл
                thread::sleep(Duration::from_millis(3));
            }

            if !is_on_window && settings.human_like_movement.enabled {
                human_like_move(original_x, original_y, &settings.human_like_movement)?;
            } else if !&settings.human_like_movement.enabled {
                human_like_move(original_x, original_y, &settings.human_like_movement)?;
            }
            continue;
        }

        // Обработка бочек
        let mut barrels: Vec<DetectionResult> = detections
            .clone()
            .into_iter()
            .filter(|d| d.object_name.starts_with("Barrel"))
            .collect();

        if barrels.len() > 0 && settings.automation.merge.enabled {
            // Очищаем терминал и выводим информацию
            clear_screen()?;
            display_results_as_table(
                &detections,
                4,
                5,
                &detector.templates,
                detection_time.try_into().unwrap_or(0),
                fps,
            );

            // Сортируем бочки по номеру (от меньшего к большему)
            barrels.sort_by(|a, b| {
                let num_a = a
                    .object_name
                    .split_whitespace()
                    .last()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                let num_b = b
                    .object_name
                    .split_whitespace()
                    .last()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                num_a.cmp(&num_b)
            });

            let _ = process_barrels(window_x, window_y, barrels, &mut detector, &settings)?;

            if !is_on_window && settings.human_like_movement.enabled {
                human_like_move(original_x, original_y, &settings.human_like_movement)?;
            } else if !&settings.human_like_movement.enabled {
                human_like_move(original_x, original_y, &settings.human_like_movement)?;
            }
            if !infinite_mode {
                break;
            }

            thread::sleep(Duration::from_millis(settings.rescan_delay));
        }
        thread::sleep(Duration::from_millis(5));
    }

    Ok(())
}
