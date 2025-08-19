use crate::capture::{AppError, AppResult};
use crate::drawing::draw_cloud;
use crate::moving::human_like_move;
use crate::objectdetector::{DetectionResult, ObjectDetector};
use crate::settings::{HumanLikeMovementSettings, Settings};
use opencv::prelude::MatTraitConst;
use rand::Rng;
use std::process::Command;
use std::thread;
use std::time::Duration;

pub fn calculate_required_merges(barrels: &[DetectionResult]) -> (u32, u32, u32) {
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

pub fn process_barrels(
    window_x: i32,
    window_y: i32,
    mut barrels: Vec<DetectionResult>,
    detector: &mut ObjectDetector,
    settings: &Settings,
) -> AppResult<(Vec<DetectionResult>, (i32, i32))> {
    let mut rng = rand::thread_rng();

    // Получаем текущую позицию курсора
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

        // Находим ВСЕ возможные пары для слияния
        let mut merges: Vec<(usize, usize, u32)> = Vec::new();
        let mut used_indices = std::collections::HashSet::new();

        for i in 0..barrels.len() {
            if used_indices.contains(&i) {
                continue;
            }

            for j in (i + 1)..barrels.len() {
                if used_indices.contains(&j) {
                    continue;
                }

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
                        used_indices.insert(i);
                        used_indices.insert(j);
                        break; // Переходим к следующей i после нахождения пары
                    }
                }
            }
        }

        // Обрабатываем ВСЕ найденные пары за одну итерацию
        if !merges.is_empty() {
            // Сортируем пары по индексам в обратном порядке (чтобы удаление не сбивало индексы)
            merges.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

            let mut new_barrels = Vec::new();
            let mut barrels_to_remove: Vec<usize> =
                merges.iter().flat_map(|(i, j, _)| vec![*i, *j]).collect();
            barrels_to_remove.sort();
            barrels_to_remove.dedup();

            // Сначала обрабатываем все слияния
            for (i, j, next_level) in merges {
                let from = &barrels[i];
                let to = &barrels[j];

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
                            -settings.random_offset.max_x_offset
                                ..=settings.random_offset.max_x_offset,
                        ),
                        rng.gen_range(
                            -settings.random_offset.max_y_offset
                                ..=settings.random_offset.max_y_offset,
                        ),
                    )
                } else {
                    (0, 0)
                };

                let (to_offset_x, to_offset_y) = if settings.random_offset.enabled {
                    (
                        rng.gen_range(
                            -settings.random_offset.max_x_offset
                                ..=settings.random_offset.max_x_offset,
                        ),
                        rng.gen_range(
                            -settings.random_offset.max_y_offset
                                ..=settings.random_offset.max_y_offset,
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
                    thread::sleep(Duration::from_millis(rng.gen_range(15..17)));
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
                    thread::sleep(Duration::from_millis(rng.gen_range(15..17)));
                }

                // Перемещаемся к конечной точке
                human_like_move(abs_to_x, abs_to_y, &settings.human_like_movement)?;

                // Небольшая пауза перед отпусканием
                thread::sleep(Duration::from_millis(rng.gen_range(20..25)));
                if settings.human_like_movement.enabled {
                    thread::sleep(Duration::from_millis(rng.gen_range(
                        settings.human_like_movement.min_up_ms
                            ..settings.human_like_movement.max_up_ms,
                    )));
                } else {
                    thread::sleep(Duration::from_millis(rng.gen_range(20..25)));
                }

                // Отпускаем кнопку мыши
                Command::new("xdotool").args(&["mouseup", "1"]).status()?;

                // Сохраняем новую бочку
                new_barrels.push(DetectionResult {
                    object_name: format!("Barrel {}", next_level),
                    location: to.location.clone(),
                    confidence: to.confidence.clone(),
                });

                thread::sleep(Duration::from_millis(rng.gen_range(5..10)));
            }

            // Теперь добавляем все несмерженные бочки
            for (index, barrel) in barrels.iter().enumerate() {
                if !barrels_to_remove.contains(&index) {
                    new_barrels.push(barrel.clone());
                }
            }

            barrels = new_barrels;
            merged = true;
        }
    }

    Ok((barrels, (original_x, original_y)))
}

pub fn process_magnets_cloud(
    window_x: i32,
    window_y: i32,
    window_width: i32,
    window_height: i32,
    settings: &Settings,
) -> AppResult<()> {
    // Получаем текущую позицию курсора
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

    let fast_movement_settings = HumanLikeMovementSettings {
        enabled: true,
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

    // Параметры отрисовки
    let cell_width = 4;
    let line_length = 4 * (cell_width + 2) + 1;
    let mut drop_positions = vec![0usize; 5]; // Позиции 5 капель

    // Вычисляем шаг для зигзага (примерно 1/8 высоты окна)
    let step_height = window_height / 11;

    // Создаем зигзагообразный маршрут от верха до низа окна
    let mut current_y = window_y + 50 + step_height;
    let left_x = window_x + (4 + settings.random_offset.max_x_offset);
    let right_x = window_x + window_width - (4 + settings.random_offset.max_x_offset);

    // 1. Перемещаемся к начальной точке (левый верхний угол) с human-like движением
    human_like_move(left_x, current_y, &fast_movement_settings)?;
    for i in 0..5 {
        drop_positions[i as usize] = (i * 3) % (line_length - 4);
    }
    draw_cloud(&drop_positions, true, line_length);
    thread::sleep(Duration::from_millis(1));

    // 2. Нажимаем кнопку мыши
    Command::new("xdotool").args(&["mousedown", "1"]).status()?;

    while current_y < window_y + window_height - 80 - step_height {
        // Движение вправо - с human-like движением
        human_like_move(right_x, current_y, &fast_movement_settings)?;
        for i in 0..5 {
            drop_positions[i as usize] = (drop_positions[i as usize] + 5) % (line_length - 4);
        }
        draw_cloud(&drop_positions, true, line_length);
        thread::sleep(Duration::from_millis(1));

        // Движение вниз - прямое перемещение без human-like
        current_y += step_height;
        Command::new("xdotool")
            .args(&["mousemove", &right_x.to_string(), &current_y.to_string()])
            .status()?;
        for i in 0..5 {
            drop_positions[i as usize] = (drop_positions[i as usize] + 2) % (line_length - 4);
        }
        draw_cloud(&drop_positions, true, line_length);
        thread::sleep(Duration::from_millis(1));

        // Движение влево - с human-like движением
        human_like_move(left_x, current_y, &fast_movement_settings)?;
        for i in 0..5 {
            drop_positions[i as usize] =
                (drop_positions[i as usize].max(5) - 5) % (line_length - 4);
        }
        draw_cloud(&drop_positions, false, line_length);
        thread::sleep(Duration::from_millis(1));

        // Движение вниз (если не вышли за границы) - прямое перемещение без human-like
        if current_y < window_y + window_height - step_height {
            current_y += step_height;
            Command::new("xdotool")
                .args(&["mousemove", &left_x.to_string(), &current_y.to_string()])
                .status()?;
            for i in 0..5 {
                drop_positions[i as usize] = (drop_positions[i as usize] + 3) % (line_length - 4);
            }
            draw_cloud(&drop_positions, false, line_length);
            thread::sleep(Duration::from_millis(1));
        }
    }

    // Отпускаем кнопку мыши
    Command::new("xdotool").args(&["mouseup", "1"]).status()?;
    human_like_move(original_x, original_y, &settings.human_like_movement)?;

    Ok(())
}
