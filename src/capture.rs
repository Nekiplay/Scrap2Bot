use std::error::Error;
use std::fmt;
use std::process::Command;

#[derive(Debug)]
pub enum AppError {
    OpenCV(opencv::Error),
    IO(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    WindowNotFound(String),
    ScrotFailed(String),
    ImageProcessing(String),
    SettingsError(String),
    X11Connect(x11rb::errors::ConnectError),
    X11Error(Box<dyn std::error::Error>),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::OpenCV(e) => write!(f, "OpenCV error: {}", e),
            AppError::IO(e) => write!(f, "IO error: {}", e),
            AppError::Utf8(e) => write!(f, "UTF-8 conversion error: {}", e),
            AppError::WindowNotFound(title) => write!(f, "Window not found: {}", title),
            AppError::ScrotFailed(msg) => write!(f, "Scrot failed: {}", msg),
            AppError::ImageProcessing(msg) => write!(f, "Image processing error: {}", msg),
            AppError::SettingsError(msg) => write!(f, "Settings error: {}", msg),
            AppError::X11Connect(msg) => write!(f, "X11 connect error: {}", msg),
            AppError::X11Error(msg) => write!(f, "X11 error: {}", msg),
        }
    }
}

impl Error for AppError {}

impl From<opencv::Error> for AppError {
    fn from(e: opencv::Error) -> Self {
        AppError::OpenCV(e)
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::IO(e)
    }
}

impl From<std::string::FromUtf8Error> for AppError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        AppError::Utf8(e)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::SettingsError(e.to_string())
    }
}

impl From<x11rb::errors::ConnectError> for AppError {
    fn from(e: x11rb::errors::ConnectError) -> Self {
        AppError::X11Connect(e)
    }
}

impl From<Box<dyn std::error::Error>> for AppError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        AppError::X11Error(e)
    }
}

pub type AppResult<T> = std::result::Result<T, AppError>;

pub fn capture_window_by_title(window_title: &str, output: &str) -> AppResult<(i32, i32)> {
    let geometry = Command::new("xwininfo")
        .args(&["-name", window_title])
        .output()?;

    if !geometry.status.success() {
        return Err(AppError::WindowNotFound(format!(
            "Window '{}' not found",
            window_title
        )));
    }

    let geometry_output = String::from_utf8(geometry.stdout)?;

    let parse_value = |s: &str| -> AppResult<i32> {
        geometry_output
            .lines()
            .find(|l| l.trim().starts_with(s))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().split(' ').next())
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| {
                AppError::WindowNotFound(format!("Could not parse {} from xwininfo output", s))
            })
    };

    let x = parse_value("Absolute upper-left X")?;
    let y = parse_value("Absolute upper-left Y")?;
    let width = parse_value("Width")?;
    let height = parse_value("Height")?;

    let geometry_str = format!("{}x{}+{}+{}", width, height, x, y);
    let output_file = format!("{}", output.replace(" ", "_"));

    let status = Command::new("maim")
        .args(&["-g", &geometry_str, &output_file])
        .status()?;

    if !status.success() {
        let status_import = Command::new("import")
            .args(&[
                "-window",
                &format!("0x{:x}", parse_window_id(&geometry_output)?),
                &output_file,
            ])
            .status();

        if status_import.is_err() || !status_import.unwrap().success() {
            return Err(AppError::ScrotFailed(format!(
                "Failed to capture window area with both maim and import"
            )));
        }
    }

    if !std::path::Path::new(&output_file).exists() {
        return Err(AppError::ScrotFailed("Output file not created".to_string()));
    }

    Ok((x, y))
}

pub fn get_window_size(window_title: &str) -> AppResult<(i32, i32)> {
    let output = Command::new("xwininfo")
        .args(&["-name", window_title])
        .output()?;

    if !output.status.success() {
        return Err(AppError::WindowNotFound(format!(
            "Window '{}' not found",
            window_title
        )));
    }

    let output_str = String::from_utf8(output.stdout)?;

    let width = output_str
        .lines()
        .find(|l| l.trim().starts_with("Width"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().split(' ').next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| AppError::WindowNotFound("Could not parse width".to_string()))?;

    let height = output_str
        .lines()
        .find(|l| l.trim().starts_with("Height"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().split(' ').next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| AppError::WindowNotFound("Could not parse height".to_string()))?;

    Ok((width, height))
}

pub fn is_cursor_in_window(
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

pub fn parse_window_id(geometry_output: &str) -> AppResult<u32> {
    geometry_output
        .lines()
        .find(|l| l.contains("Window id:"))
        .and_then(|l| l.split_whitespace().nth(3))
        .and_then(|id| {
            if id.starts_with("0x") {
                u32::from_str_radix(&id[2..], 16).ok()
            } else {
                id.parse().ok()
            }
        })
        .ok_or_else(|| AppError::WindowNotFound("Could not parse window ID".to_string()))
}
