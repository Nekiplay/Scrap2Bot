use crate::settings::HumanLikeMovementSettings;
use rand::Rng;
use std::f64::consts::PI;
use std::process::Command;
use std::thread;
use std::time::Duration;
use winapi::shared::windef::POINT;
use winapi::shared::minwindef::BOOL;
use winapi::um::winuser::MOUSEEVENTF_LEFTUP;
use winapi::um::winuser::mouse_event;
use winapi::um::winuser::MOUSEEVENTF_LEFTDOWN;
use winapi::um::winuser::SM_CYSCREEN;
use winapi::um::winuser::GetSystemMetrics;
use winapi::um::winuser::SM_CXSCREEN;
use winapi::um::winuser::SetCursorPos;
use winapi::um::winuser::GetCursorPos;
use std::error::Error;
use std::fmt; 

#[derive(Debug)]
pub enum MouseError {
    WinApiError(String),
    InvalidCoordinates(String),
}

impl fmt::Display for MouseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MouseError::WinApiError(msg) => write!(f, "WinAPI error: {}", msg),
            MouseError::InvalidCoordinates(msg) => write!(f, "Invalid coordinates: {}", msg),
        }
    }
}

impl Error for MouseError {}

pub type MouseResult<T> = std::result::Result<T, MouseError>;

/// Получает текущую позицию курсора
pub fn get_cursor_position() -> MouseResult<(i32, i32)> {
    let mut point = POINT { x: 0, y: 0 };
    
    let result = unsafe { GetCursorPos(&mut point) };
    
    if result == 0 {
        return Err(MouseError::WinApiError("Failed to get cursor position".to_string()));
    }
    
    Ok((point.x, point.y))
}

/// Устанавливает позицию курсора
pub fn set_cursor_position(x: i32, y: i32) -> MouseResult<()> {
    let result = unsafe { SetCursorPos(x, y) };
    
    if result == 0 {
        return Err(MouseError::WinApiError("Failed to set cursor position".to_string()));
    }
    
    Ok(())
}

/// Получает размеры экрана
pub fn get_screen_size() -> MouseResult<(i32, i32)> {
    let width = unsafe { GetSystemMetrics(SM_CXSCREEN) };
    let height = unsafe { GetSystemMetrics(SM_CYSCREEN) };
    
    if width == 0 || height == 0 {
        return Err(MouseError::WinApiError("Failed to get screen dimensions".to_string()));
    }
    
    Ok((width, height))
}

/// Проверяет, находятся ли координаты в пределах экрана
pub fn validate_coordinates(x: i32, y: i32) -> MouseResult<()> {
    let (screen_width, screen_height) = get_screen_size()?;
    
    if x < 0 || x >= screen_width || y < 0 || y >= screen_height {
        return Err(MouseError::InvalidCoordinates(
            format!("Coordinates ({}, {}) are outside screen bounds ({}x{})", 
                    x, y, screen_width, screen_height)
        ));
    }
    
    Ok(())
}

/// Генерирует человекоподобный путь движения мыши
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
    let distance = ((dx * dx + dy * dy) as f64).sqrt();

    // Количество промежуточных точек базируется на расстоянии
    let num_points = (settings.curve_smoothness.max(2) as f64 * (distance / 100.0).sqrt()).ceil() as usize;

    // Генерируем контрольные точки для кривой Безье
    let control_offset = distance * 0.2; // 20% от общего расстояния
    let control_angle = rng.gen_range(-PI / 4.0..PI / 4.0); // ±45 градусов
    
    let control1_x = start.0 as f64 + control_offset * control_angle.cos();
    let control1_y = start.1 as f64 + control_offset * control_angle.sin();
    
    let control2_x = end.0 as f64 - control_offset * control_angle.cos();
    let control2_y = end.1 as f64 - control_offset * control_angle.sin();

    // Генерируем точки по кривой Безье
    for i in 1..num_points {
        let t = i as f64 / num_points as f64;
        
        // Кубическая кривая Безье
        let x = cubic_bezier(start.0 as f64, control1_x, control2_x, end.0 as f64, t);
        let y = cubic_bezier(start.1 as f64, control1_y, control2_y, end.1 as f64, t);

        // Добавляем случайное отклонение для имитации человеческой неточности
        let deviation_scale = (t * PI).sin().abs() * settings.max_deviation;
        let dev_x = rng.gen_range(-deviation_scale..deviation_scale);
        let dev_y = rng.gen_range(-deviation_scale..deviation_scale);

        let final_x = (x + dev_x).round() as i32;
        let final_y = (y + dev_y).round() as i32;

        // Проверяем, что координаты в пределах экрана
        if validate_coordinates(final_x, final_y).is_ok() {
            path.push((final_x, final_y));
        }
    }

    // Добавляем конечную точку
    path.push(end);
    path
}

/// Кубическая функция Безье
fn cubic_bezier(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;

    uuu * p0 + 3.0 * uu * t * p1 + 3.0 * u * tt * p2 + ttt * p3
}

/// Человекоподобное движение мыши
pub fn human_like_move(x: i32, y: i32, settings: &HumanLikeMovementSettings) -> MouseResult<()> {
    let mut rng = rand::thread_rng();

    // Проверяем валидность целевых координат
    validate_coordinates(x, y)?;

    if !settings.enabled {
        return set_cursor_position(x, y);
    }

    // Получаем текущую позицию курсора
    let current_pos = get_cursor_position()?;

    // Если уже в нужной позиции, ничего не делаем
    if current_pos.0 == x && current_pos.1 == y {
        return Ok(());
    }

    // Генерируем путь
    let path = generate_human_like_path(current_pos, (x, y), settings);

    // Двигаемся по пути с переменной скоростью
    for i in 0..path.len() - 1 {
        let (from_x, from_y) = path[i];
        let (to_x, to_y) = path[i + 1];

        // Вычисляем расстояние между точками
        let dx = to_x - from_x;
        let dy = to_y - from_y;
        let distance = ((dx * dx + dy * dy) as f64).sqrt();

        // Устанавливаем новую позицию
        set_cursor_position(to_x, to_y)?;

        // Вычисляем время паузы на основе скорости и расстояния
        let base_speed = settings.base_speed
            + rng.gen_range(-settings.speed_variation..settings.speed_variation);
        let move_time = (distance * base_speed).max(1.0) as u64;

        // Случайная пауза для имитации человеческой реакции
        if i < path.len() - 2 {
            let pause_time = rng.gen_range(settings.min_pause_ms..=settings.max_pause_ms);
            thread::sleep(Duration::from_millis(pause_time));
        }

        // Основная пауза движения
        thread::sleep(Duration::from_millis(move_time));
    }

    Ok(())
}

/// Плавное движение мыши с линейной интерполяцией (более быстрый вариант)
pub fn smooth_move(x: i32, y: i32, duration_ms: u64) -> MouseResult<()> {
    validate_coordinates(x, y)?;
    
    let current_pos = get_cursor_position()?;
    let (start_x, start_y) = current_pos;
    
    if start_x == x && start_y == y {
        return Ok(());
    }
    
    let steps = (duration_ms / 16).max(1); // ~60 FPS
    let step_duration = duration_ms / steps;
    
    for i in 0..=steps {
        let progress = i as f64 / steps as f64;
        
        // Применяем easing функцию для более естественного движения
        let eased_progress = ease_in_out_cubic(progress);
        
        let current_x = start_x as f64 + (x - start_x) as f64 * eased_progress;
        let current_y = start_y as f64 + (y - start_y) as f64 * eased_progress;
        
        set_cursor_position(current_x.round() as i32, current_y.round() as i32)?;
        
        if i < steps {
            thread::sleep(Duration::from_millis(step_duration));
        }
    }
    
    Ok(())
}

/// Easing функция для плавного ускорения и замедления
fn ease_in_out_cubic(t: f64) -> f64 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let f = 2.0 * t - 2.0;
        1.0 + f * f * f / 2.0
    }
}

/// Движение мыши по окружности (для демонстрации)
pub fn move_in_circle(center_x: i32, center_y: i32, radius: i32, duration_ms: u64) -> MouseResult<()> {
    validate_coordinates(center_x - radius, center_y - radius)?;
    validate_coordinates(center_x + radius, center_y + radius)?;
    
    let steps = (duration_ms / 16).max(1);
    let step_duration = duration_ms / steps;
    let angle_step = 2.0 * PI / steps as f64;
    
    for i in 0..steps {
        let angle = i as f64 * angle_step;
        let x = center_x + (radius as f64 * angle.cos()) as i32;
        let y = center_y + (radius as f64 * angle.sin()) as i32;
        
        set_cursor_position(x, y)?;
        thread::sleep(Duration::from_millis(step_duration));
    }
    
    Ok(())
}

/// Случайное дрожание мыши (для имитации нервозности)
pub fn mouse_jitter(center_x: i32, center_y: i32, max_offset: i32, duration_ms: u64) -> MouseResult<()> {
    let mut rng = rand::thread_rng();
    let steps = duration_ms / 50; // Каждые 50ms
    
    for _ in 0..steps {
        let offset_x = rng.gen_range(-max_offset..=max_offset);
        let offset_y = rng.gen_range(-max_offset..=max_offset);
        
        let new_x = center_x + offset_x;
        let new_y = center_y + offset_y;
        
        if validate_coordinates(new_x, new_y).is_ok() {
            set_cursor_position(new_x, new_y)?;
        }
        
        thread::sleep(Duration::from_millis(50));
    }
    
    // Возвращаемся в центр
    set_cursor_position(center_x, center_y)?;
    Ok(())
}

pub fn mouse_left_down() -> MouseResult<()> {
    unsafe {
        // Флаг MOUSEEVENTF_LEFTDOWN сообщает системе о нажатии левой кнопки
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    }
    Ok(())
}

/// Отпустить левую кнопку мыши в текущей позиции
pub fn mouse_left_up() -> MouseResult<()> {
    unsafe {
        // Флаг MOUSEEVENTF_LEFTUP сообщает системе об отжатии левой кнопки
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
    }
    Ok(())
}

/// Быстрый клик левой кнопкой мыши (нажатие + отпускание)
pub fn mouse_left_click() -> MouseResult<()> {
    mouse_left_down()?;
    // Небольшая пауза для реалистичности
    std::thread::sleep(std::time::Duration::from_millis(15));
    mouse_left_up()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_cursor_position() {
        let result = get_cursor_position();
        assert!(result.is_ok());
        
        let (x, y) = result.unwrap();
        println!("Current cursor position: ({}, {})", x, y);
    }
    
    #[test]
    fn test_screen_size() {
        let result = get_screen_size();
        assert!(result.is_ok());
        
        let (width, height) = result.unwrap();
        println!("Screen size: {}x{}", width, height);
        assert!(width > 0);
        assert!(height > 0);
    }
    
    #[test]
    fn test_validate_coordinates() {
        // Должно пройти для корректных координат
        assert!(validate_coordinates(100, 100).is_ok());
        
        // Должно упасть для негативных координат
        assert!(validate_coordinates(-1, 100).is_err());
        assert!(validate_coordinates(100, -1).is_err());
    }
    
    #[test]
    fn test_human_like_path_generation() {
        let settings = HumanLikeMovementSettings::default();
        let path = generate_human_like_path((0, 0), (100, 100), &settings);
        
        assert!(path.len() >= 2);
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[path.len() - 1], (100, 100));
    }
}