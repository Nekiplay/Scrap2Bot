use crate::capture::AppResult;
use crate::settings::HumanLikeMovementSettings;
use rand::Rng;
use std::f64::consts::PI;
use std::process::Command;
use std::thread;
use std::time::Duration;

// Функция для генерации кривой Безье с человеческими характеристиками
pub fn generate_human_like_path(
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
pub fn human_like_move(x: i32, y: i32, settings: &HumanLikeMovementSettings) -> AppResult<()> {
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
