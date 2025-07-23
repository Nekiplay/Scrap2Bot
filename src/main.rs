use Scrap2Bot::capture::AppError;
use Scrap2Bot::capture::AppResult;
use Scrap2Bot::capture::capture_window_by_title;
use Scrap2Bot::capture::get_window_size;
use Scrap2Bot::objectdetector::DetectionResult;
use Scrap2Bot::objectdetector::ObjectDetector;
use Scrap2Bot::objectdetector::ObjectTemplate;
use Scrap2Bot::settings::HumanLikeMovementSettings;
use Scrap2Bot::settings::RandomOffsetSettings;
use Scrap2Bot::settings::Settings;
use crossterm::{execute, terminal::SetTitle};
use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::prelude::MatTraitConst;
use rand::Rng;
use std::f64::consts::PI;
use std::fs;
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn change_window_title(partial_title: &str, new_title: &str) -> AppResult<()> {
    // 1. Пробуем найти через wmctrl (более надежный)
    let wmctrl_output = Command::new("wmctrl").args(&["-l"]).output()?;

    if !wmctrl_output.status.success() {
        return Err(AppError::WindowNotFound(
            "Failed to list windows with wmctrl".to_string(),
        ));
    }

    let output_str = String::from_utf8(wmctrl_output.stdout)?;
    let mut window_id = None;

    // Ищем окно, содержащее искомую подстроку
    for line in output_str.lines() {
        if line.contains(partial_title) {
            if let Some(id) = line.split_whitespace().next() {
                window_id = Some(id.to_string());
                break;
            }
        }
    }

    // 2. Если не нашли через wmctrl, пробуем xdotool
    let window_id = match window_id {
        Some(id) => id,
        None => {
            let xdotool_output = Command::new("xdotool")
                .args(&["search", "--name", partial_title, "--limit", "1"])
                .output()?;

            if !xdotool_output.status.success() {
                return Err(AppError::WindowNotFound(format!(
                    "Window containing '{}' not found with either wmctrl or xdotool",
                    partial_title
                )));
            }

            String::from_utf8(xdotool_output.stdout)?.trim().to_string()
        }
    };

    if window_id.is_empty() {
        return Err(AppError::WindowNotFound(format!(
            "Window containing '{}' not found",
            partial_title
        )));
    }

    // 3. Пробуем изменить название через wmctrl
    let wmctrl_status = Command::new("wmctrl")
        .args(&["-ir", &window_id, "-T", new_title])
        .status();

    // 4. Если wmctrl не сработал, пробуем xdotool
    if wmctrl_status.is_err() || !wmctrl_status.unwrap().success() {
        Command::new("xdotool")
            .args(&["set_window", "--name", new_title, &window_id])
            .status()?;
    }

    Ok(())
}

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
            templates: Vec::new(),
        };

        let serialized = serde_json::to_string_pretty(&settings)?;
        fs::write(settings_path, serialized)?;

        Ok(settings)
    }
}

// Функция для генерации кривой Безье с человеческими характеристиками
fn generate_human_like_path(
    start: (i32, i32),
    end: (i32, i32),
    settings: &HumanLikeMovementSettings,
) -> Vec<(i32, i32)> {
    let mut rng = rand::thread_rng();
    let mut path = Vec::new();

    if !settings.enabled {
        path.push(start);
        path.push(end);
        return path;
    }

    // Добавляем начальную точку
    path.push(start);

    // Создаем контрольные точки для кривой Безье
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;

    // Количество промежуточных точек
    let num_points = settings.curve_smoothness.max(2);

    // Генерируем небольшие отклонения
    for i in 1..num_points {
        let t = i as f64 / num_points as f64;

        // Базовое линейное перемещение
        let x = start.0 as f64 + dx as f64 * t;
        let y = start.1 as f64 + dy as f64 * t;

        // Добавляем случайное отклонение
        let deviation_scale = (t * PI).sin().abs() * settings.max_deviation;
        let dev_x = rng.gen_range(-deviation_scale..deviation_scale);
        let dev_y = rng.gen_range(-deviation_scale..deviation_scale);

        path.push(((x + dev_x).round() as i32, (y + dev_y).round() as i32));
    }

    // Добавляем конечную точку
    path.push(end);

    path
}

// Модифицированная функция перемещения
fn human_like_move(x: i32, y: i32, settings: &HumanLikeMovementSettings) -> AppResult<()> {
    let mut rng = rand::thread_rng();

    if !settings.enabled {
        Command::new("xdotool")
            .args(&["mousemove", &x.to_string(), &y.to_string()])
            .status()?;
        return Ok(());
    }

    // Получаем текущую позицию курсора
    let output = Command::new("xdotool")
        .args(&["getmouselocation", "--shell"])
        .output()?;

    let output_str = String::from_utf8(output.stdout)?;
    let mut current_x = 0;
    let mut current_y = 0;

    for line in output_str.lines() {
        if line.starts_with("X=") {
            current_x = line[2..].parse().unwrap_or(0);
        } else if line.starts_with("Y=") {
            current_y = line[2..].parse().unwrap_or(0);
        }
    }

    // Генерируем путь
    let path = generate_human_like_path((current_x, current_y), (x, y), settings);

    // Двигаемся по пути с переменной скоростью
    for i in 0..path.len() - 1 {
        let (from_x, from_y) = path[i];
        let (to_x, to_y) = path[i + 1];

        // Вычисляем расстояние между точками
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let distance = ((dx * dx + dy * dy) as f64).sqrt();

        // Базовое время движения (миллисекунды на пиксель)
        let base_speed = settings.base_speed
            + rng.gen_range(-settings.speed_variation..settings.speed_variation);
        let move_time = (distance * base_speed).max(1.0) as u64;

        // Плавное перемещение между точками
        Command::new("xdotool")
            .args(&["mousemove_relative", "--", &dx.to_string(), &dy.to_string()])
            .status()?;

        // Случайная пауза для имитации человеческой реакции
        if i < path.len() - 2 {
            let pause_time = rng.gen_range(settings.min_pause_ms..settings.max_pause_ms);
            thread::sleep(Duration::from_millis(pause_time));
        }

        thread::sleep(Duration::from_millis(move_time));
    }

    Ok(())
}

fn display_results_as_table(
    detections: &[DetectionResult],
    cols: usize,
    rows: usize,
    templates: &[Arc<ObjectTemplate>],
    detection_time: usize,
) {
    if detections.is_empty() {
        print!("No objects detected\n");
        return;
    }

    // Находим минимальные и максимальные координаты
    let min_x = detections.iter().map(|d| d.location.x).min().unwrap_or(0);
    let max_x = detections.iter().map(|d| d.location.x).max().unwrap_or(0);
    let min_y = detections.iter().map(|d| d.location.y).min().unwrap_or(0);
    let max_y = detections.iter().map(|d| d.location.y).max().unwrap_or(0);

    // Вычисляем ширину и высоту ячейки
    let cell_width = (max_x - min_x) as f32 / (cols - 1) as f32;
    let cell_height = (max_y - min_y) as f32 / (rows - 1) as f32;

    // Создаем таблицу с дополнительной информацией о цвете
    let mut table: Vec<Vec<Option<(u32, (f32, f32, f32))>>> = vec![vec![None; cols]; rows];

    // Заполняем таблицу
    for detection in detections {
        let col = ((detection.location.x - min_x) as f32 / cell_width).round() as usize;
        let row = ((detection.location.y - min_y) as f32 / cell_height).round() as usize;

        let number = detection
            .object_name
            .chars()
            .filter_map(|c| c.to_digit(10))
            .fold(0, |acc, digit| acc * 10 + digit);

        if row < rows && col < cols {
            if let Some(template) = templates.iter().find(|t| t.name == detection.object_name) {
                table[row][col] = Some((number, (template.red, template.green, template.blue)));
            }
        }
    }

    // Фильтруем бочки и собираем в новый Vec
    let barrels: Vec<DetectionResult> = detections
        .iter()
        .filter(|d| d.object_name.starts_with("Barrel"))
        .cloned()
        .collect();

    let (min_lvl, max_lvl, merges_remaining) = calculate_required_merges(&barrels);

    // Table drawing with fixed cell width
    let cell_width = 5;
    let empty_cell = format!(" {:^3} ", ""); // Centered empty cell

    // Print table header
    print!("╔");
    for c in 0..cols {
        print!("{}", "═".repeat(cell_width));
        if c < cols - 1 {
            print!("╦");
        }
    }
    println!("╗");

    // Print table rows
    for row in 0..rows {
        print!("║");
        for col in 0..cols {
            if let Some((num, (r, g, b))) = &table[row][col] {
                if *num != 0 {
                    print!(" \x1b[48;2;{:.0};{:.0};{:.0}m{:^3}\x1b[0m ", r, g, b, num);
                } else {
                    print!("{}", empty_cell);
                }
            } else {
                print!("{}", empty_cell);
            }
            print!("║");
        }
        println!();

        // Print row separator if not last row
        if row < rows - 1 {
            print!("╠");
            for c in 0..cols {
                print!("{}", "═".repeat(cell_width));
                if c < cols - 1 {
                    print!("╬");
                }
            }
            println!("╣");
        }
    }

    // Print table footer
    print!("╚");
    for c in 0..cols {
        print!("{}", "═".repeat(cell_width));
        if c < cols - 1 {
            print!("╩");
        }
    }
    println!("╝ {}ms", detection_time);

    // Statistics section
let min_w = 5;      // Ширина для min_lvl
let max_w = 5;      // Ширина для max_lvl
let target_w = 5;   // Ширина для target
let merges_w = 10;   // Увеличил ширину для merges_remaining (было 7)
let total_width = min_w + max_w + target_w + merges_w + 5; // 3 пробела между колонками

println!("╔{}╗", "═".repeat(total_width));
println!(
    "║ {:^min_w$} {:^max_w$} {:^target_w$} {:>merges_w$} ║",
    format!("⭣{}", min_lvl),
    format!("⭡{}", max_lvl),
    format!("⭢{}", max_lvl + 1),
    format!("⭤{}", merges_remaining)
);
println!("╚{}╝", "═".repeat(total_width));
}

fn check_and_suggest_window_size(
    window_title: &str,
    recommended_width: i32,
    recommended_height: i32,
) -> AppResult<()> {
    let (current_width, current_height) = get_window_size(window_title)?;

    // Define tolerance (5 pixels in each direction)
    const TOLERANCE: i32 = 5;
    let width_diff = (current_width - recommended_width).abs();
    let height_diff = (current_height - recommended_height).abs();

    if width_diff > TOLERANCE || height_diff > TOLERANCE {
        print!(
            "Current window size: {}x{}\n",
            current_width, current_height
        );
        print!(
            "Recommended window size: {}x{} (with ±{}px tolerance)\n",
            recommended_width, recommended_height, TOLERANCE
        );

        // Show exact difference information
        if width_diff > TOLERANCE {
            print!(
                "Width difference: {}px (tolerance: {}px)\n",
                width_diff, TOLERANCE
            );
        }
        if height_diff > TOLERANCE {
            print!(
                "Height difference: {}px (tolerance: {}px)\n",
                height_diff, TOLERANCE
            );
        }

        print!("Would you like to resize the window to the recommended size? (y/n)\n");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            Command::new("wmctrl")
                .args(&[
                    "-r",
                    window_title,
                    "-e",
                    &format!("0,-1,-1,{},{}", recommended_width, recommended_height),
                ])
                .status()?;
            print!(
                "Window size changed to {}x{}. Please restart the program.\n",
                recommended_width, recommended_height
            );
            std::process::exit(0);
        } else {
            print!(
                "Continuing with current window size. Detection results may be less accurate.\n"
            );
        }
    }

    Ok(())
}

use std::env;

fn is_cursor_in_window(
    window_x: i32,
    window_y: i32,
    window_width: i32,
    window_height: i32,
) -> AppResult<bool> {
    let output = Command::new("xdotool")
        .args(&["getmouselocation", "--shell"])
        .output()?;

    let output_str = String::from_utf8(output.stdout)?;
    let mut cursor_x = 0;
    let mut cursor_y = 0;

    for line in output_str.lines() {
        if line.starts_with("X=") {
            cursor_x = line[2..].parse().unwrap_or(0);
        } else if line.starts_with("Y=") {
            cursor_y = line[2..].parse().unwrap_or(0);
        }
    }

    Ok(cursor_x >= window_x
        && cursor_x <= window_x + window_width
        && cursor_y >= window_y
        && cursor_y <= window_y + window_height)
}

fn calculate_required_merges(barrels: &[DetectionResult]) -> (u32, u32, u32) {
    // Собираем статистику по уровням бочек
    let mut level_counts = std::collections::HashMap::new();
    for barrel in barrels {
        let level = barrel
            .object_name
            .split_whitespace()
            .last()
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        *level_counts.entry(level).or_insert(0) += 1;
    }

    let min_level = *level_counts.keys().min().unwrap_or(&0);
    let max_level = *level_counts.keys().max().unwrap_or(&0);

    // Вычисляем сколько соединений нужно с учетом уже имеющихся бочек
    let mut merges_needed = 0;
    let mut needed = 2; // Для получения 1 бочки следующего уровня нужно 2 текущего

    for level in (min_level..=max_level).rev() {
        let available = *level_counts.get(&level).unwrap_or(&0);

        if available >= needed {
            // Хватает бочек этого уровня
            merges_needed += needed / 2;
            needed = 0;
            break;
        } else {
            // Не хватает, считаем сколько нужно получить из более низких уровней
            let missing = needed - available;
            merges_needed += available / 2;
            needed = missing * 2;
        }
    }

    // Если все еще нужно бочки (не хватило даже минимальных)
    if needed > 0 {
        merges_needed += needed / 2;
    }

    (min_level, max_level, merges_needed)
}

fn process_barrels(
    window_x: i32,
    window_y: i32,
    mut barrels: Vec<DetectionResult>,
    detector: &mut ObjectDetector,
    settings: &Settings,
) -> AppResult<(Vec<DetectionResult>, (i32, i32))> {
    let mut rng = rand::thread_rng();

    // Получаем текущую позицию курсора только один раз в начале
    let original_pos = Command::new("xdotool")
        .args(&["getmouselocation", "--shell"])
        .output()?;

    let original_pos = String::from_utf8(original_pos.stdout)?;
    let mut original_x = 0;
    let mut original_y = 0;

    for line in original_pos.lines() {
        if line.starts_with("X=") {
            original_x = line[2..].parse().unwrap_or(0);
        } else if line.starts_with("Y=") {
            original_y = line[2..].parse().unwrap_or(0);
        }
    }

    let mut merged = true;
    while merged {
        merged = false;

        let mut merges: Vec<(usize, usize, u32)> = Vec::new();

        for i in 0..barrels.len() {
            for j in (i + 1)..barrels.len() {
                if barrels[i].object_name == barrels[j].object_name {
                    let current_level: u32 = barrels[i]
                        .object_name
                        .split_whitespace()
                        .last()
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0);
                    let next_level = current_level + 1;

                    if detector
                        .templates
                        .iter()
                        .any(|t| t.name == format!("Barrel {}", next_level))
                    {
                        merges.push((i, j, next_level));
                        break;
                    }
                }
            }
        }

        if let Some((i, j, next_level)) = merges.first() {
            let from = &barrels[*i];
            let to = &barrels[*j];

            let from_template = detector
                .templates
                .iter()
                .find(|t| t.name == from.object_name)
                .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;

            let to_template = detector
                .templates
                .iter()
                .find(|t| t.name == to.object_name)
                .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;

            let from_size = (from_template.template.cols(), from_template.template.rows());
            let to_size = (to_template.template.cols(), to_template.template.rows());

            // Вычисляем целевые позиции с учетом случайного смещения
            let (from_offset_x, from_offset_y) = if settings.random_offset.enabled {
                (
                    rng.gen_range(
                        -settings.random_offset.max_x_offset..=settings.random_offset.max_x_offset,
                    ),
                    rng.gen_range(
                        -settings.random_offset.max_y_offset..=settings.random_offset.max_y_offset,
                    ),
                )
            } else {
                (0, 0)
            };

            let (to_offset_x, to_offset_y) = if settings.random_offset.enabled {
                (
                    rng.gen_range(
                        -settings.random_offset.max_x_offset..=settings.random_offset.max_x_offset,
                    ),
                    rng.gen_range(
                        -settings.random_offset.max_y_offset..=settings.random_offset.max_y_offset,
                    ),
                )
            } else {
                (0, 0)
            };

            let abs_from_x = window_x + from.location.x + from_size.0 / 2 + from_offset_x;
            let abs_from_y = window_y + from.location.y + from_size.1 / 2 + from_offset_y;

            let abs_to_x = window_x + to.location.x + to_size.0 / 2 + to_offset_x;
            let abs_to_y = window_y + to.location.y + to_size.1 / 2 + to_offset_y;

            // Перемещаемся к начальной точке
            human_like_move(abs_from_x, abs_from_y, &settings.human_like_movement)?;

            // Небольшая пауза перед кликом
            if settings.human_like_movement.enabled {
                thread::sleep(Duration::from_millis(rng.gen_range(
                    settings.human_like_movement.min_down_ms
                        ..settings.human_like_movement.max_down_ms,
                )));
            } else {
                thread::sleep(Duration::from_millis(rng.gen_range(2..5)));
            }

            // Нажимаем кнопку мыши
            Command::new("xdotool").args(&["mousedown", "1"]).status()?;

            // Небольшая пауза перед началом перемещения
            if settings.human_like_movement.enabled {
                thread::sleep(Duration::from_millis(rng.gen_range(
                    settings.human_like_movement.min_move_delay_ms
                        ..settings.human_like_movement.max_move_delay_ms,
                )));
            } else {
                thread::sleep(Duration::from_millis(rng.gen_range(5..12)));
            }

            // Перемещаемся к конечной точке
            human_like_move(abs_to_x, abs_to_y, &settings.human_like_movement)?;

            // Небольшая пауза перед отпусканием
            thread::sleep(Duration::from_millis(rng.gen_range(6..12)));
            if settings.human_like_movement.enabled {
                thread::sleep(Duration::from_millis(rng.gen_range(
                    settings.human_like_movement.min_up_ms..settings.human_like_movement.max_up_ms,
                )));
            } else {
                thread::sleep(Duration::from_millis(rng.gen_range(5..12)));
            }

            // Отпускаем кнопку мыши
            Command::new("xdotool").args(&["mouseup", "1"]).status()?;

            // Сохраняем позицию для новой бочки
            let new_location = to.location;
            let conf = to.confidence.clone();

            // Удаляем бочки (сначала бОльший индекс)
            barrels.remove(*j);
            barrels.remove(*i);

            // Добавляем новую бочку
            barrels.push(DetectionResult {
                object_name: format!("Barrel {}", next_level),
                location: new_location,
                confidence: conf,
            });

            merged = true;
        }
    }

    Ok((barrels, (original_x, original_y)))
}

fn main() -> AppResult<()> {
    let args: Vec<String> = env::args().collect();
    let infinite_mode = args.iter().any(|arg| arg == "--infinite" || arg == "-i");
    execute!(std::io::stdout(), SetTitle("Scrap II Bot"))?;

    let window_title = "M2006C3MNG";
    let mut settings = load_or_create_settings(window_title)?;

    let _ = change_window_title(&settings.window_title, "Scrap II");
    settings.window_title = String::from("Scrap II");

    check_and_suggest_window_size(
        &settings.window_title,
        settings.reference_width,
        settings.reference_height,
    )?;

    let mut detector = ObjectDetector::new(settings.resolution);

    for (i, template_settings) in settings.templates.iter().enumerate() {
        detector.add_template(
            &template_settings.name,
            &template_settings.path,
            template_settings.threshold,
            template_settings.min_distance,
            template_settings.red,
            template_settings.green,
            template_settings.blue,
            template_settings.resolution,
        )?;

        if template_settings.name == "Empty" {
            detector.empty_template_index = i;
        } else if template_settings.name == "Cloud" {
            detector.cloud_template_index = i;
        }
    }

    // Инициализируем начальный диапазон
    detector.active_range = (
        detector.empty_template_index,
        detector.empty_template_index + 5,
    ); // Начинаем с Empty + первые 5 бочек

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

        detector.draw_detections(&mut image, &detections)?;
        imgcodecs::imwrite("result.png", &image, &Vector::new())?;

        // Обработка облака мангинитов
        let cloud: Vec<DetectionResult> = detections
            .clone()
            .into_iter()
            .filter(|d| d.object_name.starts_with("Cloud"))
            .collect();

        if cloud.len() > 0 {
            let fast_movement_settings = HumanLikeMovementSettings {
                enabled: true, // Включено для горизонтальных движений
                max_deviation: 0.000001,
                speed_variation: 0.000001,
                curve_smoothness: 15,
                min_pause_ms: 0,
                max_pause_ms: 1,
                base_speed: 0.000001,
                min_down_ms: 0,
                max_down_ms: 1,
                min_up_ms: 0,
                max_up_ms: 1,
                min_move_delay_ms: 0,
                max_move_delay_ms: 1,
            };

            // Получаем размеры окна
            let (window_width, window_height) = get_window_size(&settings.window_title)?;

            // Параметры отрисовки
            let cell_width = 4;
            let line_length = 4 * (cell_width + 2) + 1;
            let mut drop_positions = vec![0usize; 5]; // Позиции 5 капель

            // Функция отрисовки облака с каплями
            let draw_cloud = |drop_positions: &[usize], is_moving_right: bool| {
                print!("\x1B[2J\x1B[1;1H"); // Очистка экрана

                // Верхняя граница (как в таблице)
                print!("╔");
                for _ in 0..(line_length - 2) {
                    print!("═");
                }
                println!("╗");

                // Облако
                println!("║{:^width$}║", "  .-~~~-.", width = line_length - 2);
                println!("║{:^width$}║", " .'       '.", width = line_length - 2);
                println!("║{:^width$}║", "(           )", width = line_length - 2);
                println!("║{:^width$}║", " `~-._____.-'", width = line_length - 2);

                // Капли дождя
                for &pos in drop_positions {
                    // Ограничиваем позицию внутри границ
                    let pos = pos.min(line_length - 4);
                    println!(
                        "║ {}{}{} ║",
                        " ".repeat(pos),
                        "\x1B[34m|\x1B[0m", // Синяя капля
                        " ".repeat(line_length - 4 - pos - 1)
                    );
                }

                // Нижняя граница (как в таблице)
                print!("╚");
                for _ in 0..(line_length - 2) {
                    print!("═");
                }
                println!("╝");

                // Статус с правильным выравниванием
                let status = if is_moving_right {
                    "\x1B[36mCollecting magnets >>\x1B[0m"
                } else {
                    "\x1B[36mCollecting magnets <<\x1B[0m"
                };

                // Удаляем escape-последовательности для расчета ширины
                let clean_status = status.replace("\x1B[36m", "").replace("\x1B[0m", "");
                let padding = line_length.saturating_sub(clean_status.len() + 2) / 2;

                // Статусная строка с границами
                print!("╔");
                for _ in 0..(line_length - 2) {
                    print!("═");
                }
                println!("╗");

                println!(
                    "║{}{}{}║",
                    " ".repeat(padding),
                    status,
                    " ".repeat(line_length - clean_status.len() - 2 - padding)
                );

                print!("╚");
                for _ in 0..(line_length - 2) {
                    print!("═");
                }
                println!("╝");
            };

            // Вычисляем шаг для зигзага (примерно 1/8 высоты окна)
            let step_height = (window_height - 260) / 9;

            // Создаем зигзагообразный маршрут от верха до низа окна
            let mut current_y = window_y + 50 + step_height;
            let left_x = window_x + (4 + settings.random_offset.max_x_offset);
            let right_x = window_x + window_width - (4 + settings.random_offset.max_x_offset);

            // 1. Перемещаемся к начальной точке (левый верхний угол) с human-like движением
            human_like_move(left_x, current_y, &fast_movement_settings)?;
            for i in 0..5 {
                drop_positions[i as usize] = (i * 3) % (line_length - 4);
            }
            draw_cloud(&drop_positions, true);
            thread::sleep(Duration::from_millis(1));

            // 2. Нажимаем кнопку мыши
            Command::new("xdotool").args(&["mousedown", "1"]).status()?;

            while current_y < window_y + window_height - step_height {
                // Движение вправо - с human-like движением
                human_like_move(right_x, current_y, &fast_movement_settings)?;
                for i in 0..5 {
                    drop_positions[i as usize] =
                        (drop_positions[i as usize] + 5) % (line_length - 4);
                }
                draw_cloud(&drop_positions, true);
                thread::sleep(Duration::from_millis(1));

                // Движение вниз - прямое перемещение без human-like
                current_y += step_height;
                Command::new("xdotool")
                    .args(&["mousemove", &right_x.to_string(), &current_y.to_string()])
                    .status()?;
                for i in 0..5 {
                    drop_positions[i as usize] =
                        (drop_positions[i as usize] + 2) % (line_length - 4);
                }
                draw_cloud(&drop_positions, true);
                thread::sleep(Duration::from_millis(1));

                // Движение влево - с human-like движением
                human_like_move(left_x, current_y, &fast_movement_settings)?;
                for i in 0..5 {
                    drop_positions[i as usize] =
                        (drop_positions[i as usize].max(5) - 5) % (line_length - 4);
                }
                draw_cloud(&drop_positions, false);
                thread::sleep(Duration::from_millis(1));

                // Движение вниз (если не вышли за границы) - прямое перемещение без human-like
                if current_y < window_y + window_height - step_height {
                    current_y += step_height;
                    Command::new("xdotool")
                        .args(&["mousemove", &left_x.to_string(), &current_y.to_string()])
                        .status()?;
                    for i in 0..5 {
                        drop_positions[i as usize] =
                            (drop_positions[i as usize] + 3) % (line_length - 4);
                    }
                    draw_cloud(&drop_positions, false);
                    thread::sleep(Duration::from_millis(1));
                }
            }

            // Отпускаем кнопку мыши
            Command::new("xdotool").args(&["mouseup", "1"]).status()?;

            // После обработки облака продолжаем основной цикл
            thread::sleep(Duration::from_millis(3));
            continue;
        }

        // Обработка бочек
        let mut barrels: Vec<DetectionResult> = detections
            .clone()
            .into_iter()
            .filter(|d| d.object_name.starts_with("Barrel"))
            .collect();

        if barrels.len() > 0 {
            display_results_as_table(
                &detections,
                4,
                5,
                &detector.templates,
                detection_time.try_into().unwrap_or(0),
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

            let (barrels, (original_x, original_y)) =
                process_barrels(window_x, window_y, barrels, &mut detector, &settings)?;

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
