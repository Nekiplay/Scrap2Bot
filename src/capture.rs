use opencv::boxed_ref::BoxedRef;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::mem;
use std::process::Command;
use std::ptr;
use winapi::ctypes::c_void;
use winapi::shared::minwindef::{DWORD, UINT};
use winapi::shared::windef::{HBITMAP, HDC, HWND, RECT};
use winapi::um::wingdi::{
    BITMAPINFO, BITMAPINFOHEADER, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDIBits, SRCCOPY, SelectObject,
};
use winapi::um::winuser::{
    FindWindowA, GetClientRect, GetDC, GetDesktopWindow, GetWindowDC, GetWindowRect, PrintWindow,
    ReleaseCapture, ReleaseDC, SetCapture,
};

#[derive(Debug)]
pub enum WindowsCaptureError {
    WindowNotFound(String),
    CaptureFailed(String),
    WinApiError(String),
    ImageProcessing(String),
}

impl fmt::Display for WindowsCaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowsCaptureError::WindowNotFound(title) => write!(f, "Window not found: {}", title),
            WindowsCaptureError::CaptureFailed(msg) => write!(f, "Capture failed: {}", msg),
            WindowsCaptureError::WinApiError(msg) => write!(f, "WinAPI error: {}", msg),
            WindowsCaptureError::ImageProcessing(msg) => {
                write!(f, "ImageProcessing error: {}", msg)
            }
        }
    }
}

impl Error for WindowsCaptureError {}

pub type WindowsCaptureResult<T> = std::result::Result<T, WindowsCaptureError>;

/// Структура для хранения данных изображения
pub struct CapturedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Находит окно по заголовку
pub fn find_window_by_title(title: &str) -> WindowsCaptureResult<HWND> {
    let c_title = CString::new(title)
        .map_err(|e| WindowsCaptureError::WinApiError(format!("Invalid title string: {}", e)))?;

    let hwnd = unsafe { FindWindowA(ptr::null(), c_title.as_ptr()) };

    if hwnd.is_null() {
        return Err(WindowsCaptureError::WindowNotFound(title.to_string()));
    }

    Ok(hwnd)
}

/// Получает размеры окна
pub fn get_window_dimensions(hwnd: HWND) -> WindowsCaptureResult<(i32, i32, i32, i32)> {
    let mut rect: RECT = unsafe { mem::zeroed() };

    let result = unsafe { GetWindowRect(hwnd, &mut rect) };

    if result == 0 {
        return Err(WindowsCaptureError::WinApiError(
            "Failed to get window rect".to_string(),
        ));
    }

    Ok((
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top,
    ))
}

/// Основная функция захвата окна
pub fn capture_window(hwnd: HWND) -> WindowsCaptureResult<CapturedImage> {
    let (_, _, width, height) = get_window_dimensions(hwnd)?;

    if width <= 0 || height <= 0 {
        return Err(WindowsCaptureError::CaptureFailed(
            "Invalid window dimensions".to_string(),
        ));
    }

    unsafe {
        // Получаем DC окна
        let window_dc = GetDC(hwnd);
        if window_dc.is_null() {
            return Err(WindowsCaptureError::WinApiError(
                "Failed to get window DC".to_string(),
            ));
        }

        // Создаем совместимый DC
        let mem_dc = CreateCompatibleDC(window_dc);
        if mem_dc.is_null() {
            ReleaseDC(hwnd, window_dc);
            return Err(WindowsCaptureError::WinApiError(
                "Failed to create compatible DC".to_string(),
            ));
        }

        // Создаем совместимый bitmap
        let bitmap = CreateCompatibleBitmap(window_dc, width, height);
        if bitmap.is_null() {
            DeleteDC(mem_dc);
            ReleaseDC(hwnd, window_dc);
            return Err(WindowsCaptureError::WinApiError(
                "Failed to create compatible bitmap".to_string(),
            ));
        }

        // Выбираем bitmap в memory DC
        let old_bitmap = SelectObject(mem_dc, bitmap as *mut c_void);

        // Копируем содержимое окна в bitmap
        let bit_result = BitBlt(mem_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY);

        if bit_result == 0 {
            // Пробуем альтернативный метод с PrintWindow
            let print_result = PrintWindow(hwnd, mem_dc, 0);
            if print_result == 0 {
                SelectObject(mem_dc, old_bitmap);
                DeleteObject(bitmap as *mut c_void);
                DeleteDC(mem_dc);
                ReleaseDC(hwnd, window_dc);
                return Err(WindowsCaptureError::CaptureFailed(
                    "Both BitBlt and PrintWindow failed".to_string(),
                ));
            }
        }

        // Получаем пиксельные данные
        let mut bitmap_info: BITMAPINFO = mem::zeroed();
        bitmap_info.bmiHeader = BITMAPINFOHEADER {
            biSize: mem::size_of::<BITMAPINFOHEADER>() as DWORD,
            biWidth: width,
            biHeight: -height, // Отрицательное значение для top-down bitmap
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0, // BI_RGB
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        };

        let buffer_size = (width * height * 4) as usize;
        let mut pixels: Vec<u8> = vec![0; buffer_size];

        let result = GetDIBits(
            mem_dc,
            bitmap,
            0,
            height as UINT,
            pixels.as_mut_ptr() as *mut c_void,
            &mut bitmap_info,
            DIB_RGB_COLORS,
        );

        // Очистка ресурсов
        SelectObject(mem_dc, old_bitmap);
        DeleteObject(bitmap as *mut c_void);
        DeleteDC(mem_dc);
        ReleaseDC(hwnd, window_dc);

        if result == 0 {
            return Err(WindowsCaptureError::WinApiError(
                "Failed to get DIB bits".to_string(),
            ));
        }

        // Конвертируем BGRA в RGBA
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2); // Меняем B и R местами
        }

        Ok(CapturedImage {
            width: width as u32,
            height: height as u32,
            pixels,
        })
    }
}

/// Захватывает окно по заголовку
pub fn capture_window_by_title(window_title: &str) -> WindowsCaptureResult<CapturedImage> {
    let hwnd = find_window_by_title(window_title)?;
    capture_window(hwnd)
}

/// Получает размер экрана
pub fn get_screen_dimensions() -> WindowsCaptureResult<(i32, i32)> {
    unsafe {
        let desktop_hwnd = GetDesktopWindow();
        let (_, _, width, height) = get_window_dimensions(desktop_hwnd)?;
        Ok((width, height))
    }
}

/// Захватывает весь экран
pub fn capture_screen() -> WindowsCaptureResult<CapturedImage> {
    unsafe {
        let desktop_hwnd = GetDesktopWindow();
        capture_window(desktop_hwnd)
    }
}

/// Захватывает область экрана
pub fn capture_screen_area(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> WindowsCaptureResult<CapturedImage> {
    if width <= 0 || height <= 0 {
        return Err(WindowsCaptureError::CaptureFailed(
            "Invalid area dimensions".to_string(),
        ));
    }

    unsafe {
        // Получаем DC рабочего стола
        let desktop_dc = GetDC(ptr::null_mut());
        if desktop_dc.is_null() {
            return Err(WindowsCaptureError::WinApiError(
                "Failed to get desktop DC".to_string(),
            ));
        }

        // Создаем совместимый DC
        let mem_dc = CreateCompatibleDC(desktop_dc);
        if mem_dc.is_null() {
            ReleaseDC(ptr::null_mut(), desktop_dc);
            return Err(WindowsCaptureError::WinApiError(
                "Failed to create compatible DC".to_string(),
            ));
        }

        // Создаем совместимый bitmap
        let bitmap = CreateCompatibleBitmap(desktop_dc, width, height);
        if bitmap.is_null() {
            DeleteDC(mem_dc);
            ReleaseDC(ptr::null_mut(), desktop_dc);
            return Err(WindowsCaptureError::WinApiError(
                "Failed to create compatible bitmap".to_string(),
            ));
        }

        // Выбираем bitmap в memory DC
        let old_bitmap = SelectObject(mem_dc, bitmap as *mut c_void);

        // Копируем область экрана в bitmap
        let result = BitBlt(mem_dc, 0, 0, width, height, desktop_dc, x, y, SRCCOPY);

        if result == 0 {
            SelectObject(mem_dc, old_bitmap);
            DeleteObject(bitmap as *mut c_void);
            DeleteDC(mem_dc);
            ReleaseDC(ptr::null_mut(), desktop_dc);
            return Err(WindowsCaptureError::CaptureFailed(
                "BitBlt failed".to_string(),
            ));
        }

        // Получаем пиксельные данные
        let mut bitmap_info: BITMAPINFO = mem::zeroed();
        bitmap_info.bmiHeader = BITMAPINFOHEADER {
            biSize: mem::size_of::<BITMAPINFOHEADER>() as DWORD,
            biWidth: width,
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        };

        let buffer_size = (width * height * 4) as usize;
        let mut pixels: Vec<u8> = vec![0; buffer_size];

        let dib_result = GetDIBits(
            mem_dc,
            bitmap,
            0,
            height as UINT,
            pixels.as_mut_ptr() as *mut c_void,
            &mut bitmap_info,
            DIB_RGB_COLORS,
        );

        // Очистка ресурсов
        SelectObject(mem_dc, old_bitmap);
        DeleteObject(bitmap as *mut c_void);
        DeleteDC(mem_dc);
        ReleaseDC(ptr::null_mut(), desktop_dc);

        if dib_result == 0 {
            return Err(WindowsCaptureError::WinApiError(
                "Failed to get DIB bits".to_string(),
            ));
        }

        // Конвертируем BGRA в RGBA
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }

        Ok(CapturedImage {
            width: width as u32,
            height: height as u32,
            pixels,
        })
    }
}

pub fn get_window_size(window_title: &str) -> WindowsCaptureResult<(i32, i32, i32, i32)> {
    let hwnd = find_window_by_title(window_title)?;
    let (x, y, width, height) = get_window_dimensions(hwnd)?;
    Ok((x, y, width, height))
}

use opencv::core::{CV_8UC4, Mat, MatTraitConst};
use opencv::imgcodecs;

pub fn save_as_png(image: &CapturedImage, filename: &str) -> Result<(), Box<dyn Error>> {
    // Копируем и конвертируем RGBA в BGRA (для OpenCV)
    let mut bgra_pixels = image.pixels.clone();
    for chunk in bgra_pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    // Промежуточная переменная для продления жизни Mat
    let mat_tmp = Mat::from_slice(&bgra_pixels)?;

    // Изменяем форму матрицы с учетом количества каналов и высоты
    let mat = mat_tmp.reshape(4, image.height as i32)?;

    // Сохраняем изображение в файл PNG
    imgcodecs::imwrite(filename, &mat, &opencv::core::Vector::new())?;

    Ok(())
}
