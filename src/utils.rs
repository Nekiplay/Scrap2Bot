use std::error::Error;
use std::io;
use std::io::Write;
use std::process::Command;

use winapi::shared::windef::HWND;
use winapi::shared::minwindef::BOOL;
use winapi::um::winuser::{
    FindWindowW, GetWindowRect, SetWindowPos, SWP_NOZORDER, SWP_NOMOVE,
};
use winapi::shared::windef::RECT;
use std::os::windows::ffi::OsStrExt;
use std::ffi::OsStr;
use std::ptr::null_mut;

fn to_wstring(s: &str) -> Vec<u16> {
    OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
}

pub fn check_and_suggest_window_size(
    window_title: &str,
    recommended_width: i32,
    recommended_height: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Поиск окна по заголовку
    let hwnd: HWND = unsafe { FindWindowW(null_mut(), to_wstring(window_title).as_ptr()) };
    if hwnd.is_null() {
        return Err(format!("Не найдено окно с заголовком '{}'", window_title).into());
    }

    // Получение текущих размеров окна
    let mut rect: RECT = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    let result: BOOL = unsafe { GetWindowRect(hwnd, &mut rect) };
    if result == 0 {
        return Err("Не удалось получить размер окна".into());
    }
    let current_width = rect.right - rect.left;
    let current_height = rect.bottom - rect.top;

    const TOLERANCE: i32 = 5;
    let width_diff = (current_width - recommended_width).abs();
    let height_diff = (current_height - recommended_height).abs();

    if width_diff > TOLERANCE || height_diff > TOLERANCE {
        println!("Текущий размер окна: {}x{}", current_width, current_height);
        println!(
            "Рекомендуемый размер окна: {}x{} (±{}px допуск)",
            recommended_width, recommended_height, TOLERANCE
        );
        if width_diff > TOLERANCE {
            println!("Разница по ширине: {}px (допуск {}px)", width_diff, TOLERANCE);
        }
        if height_diff > TOLERANCE {
            println!("Разница по высоте: {}px (допуск {}px)", height_diff, TOLERANCE);
        }

        print!("Хотите изменить размер окна на рекомендованный? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if input.trim().eq_ignore_ascii_case("y") {
            unsafe {
                SetWindowPos(
                    hwnd,
                    null_mut(),
                    0,
                    0,
                    recommended_width,
                    recommended_height,
                    SWP_NOZORDER | SWP_NOMOVE,
                );
            }
            println!(
                "Размер окна изменен на {}x{}. Пожалуйста, перезапустите программу.",
                recommended_width, recommended_height
            );
        } else {
            println!("Продолжаем с текущим размером окна. Результаты могут быть менее точными.");
        }
    }

    Ok(())
}

pub fn clear_screen() -> Result<(), Box<dyn std::error::Error>> {
    print!("\x1B[2J\x1B[3J\x1B[H");
    io::stdout().flush().map_err(|e| {
        opencv::Error::new(
            opencv::core::StsError,
            format!("Failed to flush stdout: {}", e),
        )
    })?;
    Ok(())
}

use winapi::shared::windef::POINT;
use winapi::um::winuser::GetCursorPos;

pub fn get_current_mouse_position() -> Result<(i32, i32), Box<dyn Error>> {
    let mut point = POINT { x: 0, y: 0 };
    let success = unsafe { GetCursorPos(&mut point) };
    if success == 0 {
        return Err("Не удалось получить позицию мыши".into());
    }
    Ok((point.x, point.y))
}

pub fn extract_barrel_number(name: &str) -> Option<u32> {
    // Remove any file extensions first
    let clean_name = name.split('.').next().unwrap_or(name);

    // Try to find the last sequence of digits in the name
    let mut number_str = String::new();
    let mut found_digits = false;

    for c in clean_name.chars().rev() {
        if c.is_ascii_digit() {
            number_str.insert(0, c);
            found_digits = true;
        } else if found_digits {
            // Stop when we hit non-digit after digits
            break;
        }
    }

    if !number_str.is_empty() {
        return number_str.parse().ok();
    }

    // Fallback: try splitting by spaces and take last token
    if let Some(last_word) = clean_name.split_whitespace().last() {
        return last_word.parse().ok();
    }

    None
}
