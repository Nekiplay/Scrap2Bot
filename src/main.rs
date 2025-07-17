use opencv::{
    core::{self, Mat, Point, Rect, Scalar},
    imgcodecs::{self, IMREAD_COLOR},
    imgproc::{
        self, resize, threshold, LineTypes, FILLED, INTER_LINEAR, THRESH_BINARY, TM_CCOEFF_NORMED
    },
    prelude::*,
    Result as OpenCVResult,
};
use std::{collections::HashMap, sync::Arc, time::Instant};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fmt,
    fs,
    process::Command,
    time::Duration,
    thread,
};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::KeyButMask;

#[derive(Debug, Deserialize, Serialize)]
struct Settings {
    window_title: String,
    templates: Vec<TemplateSettings>,
}

#[derive(Debug, Deserialize, Serialize)]
struct TemplateSettings {
    name: String,
    path: String,
    threshold: f64,
    min_distance: f32,
    red: f32,
    green: f32,
    blue: f32
}

#[derive(Debug)]
enum AppError {
    OpenCV(opencv::Error),
    IO(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    WindowNotFound(String),
    CaptureFailed(String),
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
            AppError::CaptureFailed(msg) => write!(f, "Capture failed: {}", msg),
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

type AppResult<T> = std::result::Result<T, AppError>;

#[derive(Clone)]
struct ObjectTemplate {
    name: String,
    template: Mat,
    threshold: f64,
    min_distance: f32,
    red: f32,
    green: f32,
    blue: f32,
}

impl ObjectTemplate {
    fn new(name: &str, template_path: &str, threshold: f64, min_distance: f32, red: f32, green: f32, blue: f32) -> OpenCVResult<Self> {
        let template = imgcodecs::imread(template_path, IMREAD_COLOR)?;
        Ok(Self {
            name: name.to_string(),
            template,
            threshold,
            min_distance,
            red,
            green,
            blue
        })
    }
}

#[derive(Debug, Clone)]
struct DetectionResult {
    object_name: String,
    location: Point,
    confidence: f64,
}

struct ObjectDetector {
    templates: Vec<Arc<ObjectTemplate>>,
    scale_factor: f64,
}

impl ObjectDetector {
    fn new(scale_factor: f64) -> Self {
        Self { 
            templates: Vec::new(),
            scale_factor,
        }
    }

    fn add_template(&mut self, name: &str, template_path: &str, threshold: f64, min_distance: f32, red: f32, green: f32, blue: f32) -> OpenCVResult<()> {
        let mut template = imgcodecs::imread(template_path, IMREAD_COLOR)?;
        
        // Масштабируем шаблон для ускорения поиска
        if self.scale_factor != 1.0 {
            let mut resized = Mat::default();
            resize(
                &template,
                &mut resized,
                core::Size::new(0, 0),
                self.scale_factor,
                self.scale_factor,
                INTER_LINEAR,
            )?;
            template = resized;
        }

        let template = Arc::new(ObjectTemplate {
            name: name.to_string(),
            template,
            threshold,
            min_distance: min_distance * (self.scale_factor as f32),
            red,
            green,
            blue,
        });
        
        self.templates.push(template);
        Ok(())
    }

    fn detect_objects(&self, image: &Mat) -> OpenCVResult<Vec<DetectionResult>> {
        let start_time = Instant::now();
        
        // Масштабируем изображение для ускорения поиска
        let mut resized = Mat::default();
        if self.scale_factor != 1.0 {
            resize(
                image,
                &mut resized,
                core::Size::new(0, 0),
                self.scale_factor,
                self.scale_factor,
                INTER_LINEAR,
            )?;
        }
        let image = if self.scale_factor != 1.0 { &resized } else { image };

        let mut handles = Vec::with_capacity(self.templates.len());
        
        // Запускаем поиск для каждого шаблона в отдельном потоке
        for template in &self.templates {
            let template = template.clone();
            let image = image.clone();
            
            handles.push(thread::spawn(move || {
                let mut results = Vec::new();
                
                let mut result_mat = Mat::default();
                imgproc::match_template(
                    &image,
                    &template.template,
                    &mut result_mat,
                    TM_CCOEFF_NORMED,
                    &Mat::default(),
                )?;

                let mut thresholded = Mat::default();
                threshold(
                    &result_mat,
                    &mut thresholded,
                    template.threshold,
                    1.0,
                    THRESH_BINARY,
                )?;
                
                loop {
                    let mut max_val = 0.0;
                    let mut max_loc = Point::default();
                    
                    core::min_max_loc(
                        &result_mat,
                        None,
                        Some(&mut max_val),
                        None,
                        Some(&mut max_loc),
                        &core::no_array(),
                    )?;

                    if max_val < template.threshold {
                        break;
                    }

                    results.push(DetectionResult {
                        object_name: template.name.clone(),
                        location: max_loc,
                        confidence: max_val,
                    });

                    // Затираем найденную область для поиска следующего совпадения
                    imgproc::rectangle(
                        &mut result_mat,
                        Rect::new(
                            max_loc.x - template.template.cols() / 2,
                            max_loc.y - template.template.rows() / 2,
                            template.template.cols(),
                            template.template.rows(),
                        ),
                        Scalar::all(0.0),
                        FILLED,
                        LineTypes::LINE_8.into(),
                        0,
                    )?;
                }
                
                Ok::<_, opencv::Error>(results)
            }));
        }

        // Собираем результаты из всех потоков
        let mut all_results = Vec::new();
        for handle in handles {
            let mut thread_results = handle.join().unwrap()?;
            all_results.append(&mut thread_results);
        }

        // Фильтруем слишком близкие результаты
        let filtered_results = self.filter_close_detections(all_results);
        
        // Масштабируем координаты обратно
        let mut final_results = Vec::new();
        for result in filtered_results {
            final_results.push(DetectionResult {
                object_name: result.object_name,
                location: Point::new(
                    (result.location.x as f64 / self.scale_factor) as i32,
                    (result.location.y as f64 / self.scale_factor) as i32,
                ),
                confidence: result.confidence,
            });
        }

        println!("Detection took: {:?}", start_time.elapsed());
        Ok(final_results)
    }

    fn filter_close_detections(&self, mut results: Vec<DetectionResult>) -> Vec<DetectionResult> {
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut filtered = Vec::new();
        let mut occupied = Vec::new();
        
        for result in results {
            let too_close = occupied.iter().any(|(center, min_dist): &(Point, f32)| {
                let dx = (center.x - result.location.x) as f32;
                let dy = (center.y - result.location.y) as f32;
                (dx * dx + dy * dy).sqrt() < *min_dist
            });
            
            if !too_close {
                if let Some(template) = self.templates.iter().find(|t| t.name == result.object_name) {
                    occupied.push((result.location, template.min_distance));
                    filtered.push(result);
                }
            }
        }
        
        filtered
    }

    fn draw_detections(&self, image: &mut Mat, detections: &[DetectionResult]) -> OpenCVResult<()> {
        for detection in detections {
            if let Some(template) = self.templates.iter().find(|t| t.name == detection.object_name) {
                // Масштабируем размеры шаблона обратно к оригинальным размерам
                let cols = (template.template.cols() as f64 / self.scale_factor) as i32;
                let rows = (template.template.rows() as f64 / self.scale_factor) as i32;

                let rect = Rect::new(
                    detection.location.x,
                    detection.location.y,
                    cols,
                    rows,
                );

                let color = Scalar::new(
                    template.blue.into(),
                    template.green.into(),
                    template.red.into(),
                    0.0
                );

                imgproc::rectangle(
                    image,
                    rect,
                    color,
                    2,
                    LineTypes::LINE_8.into(),
                    0,
                )?;

                imgproc::put_text(
                    image,
                    &format!("{}: {:.2}", detection.object_name, detection.confidence),
                    Point::new(detection.location.x, detection.location.y - 5),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    LineTypes::LINE_8.into(),
                    false,
                )?;
            }
        }
        Ok(())
    }
}

fn capture_window_by_title(window_title: &str, output: &str) -> AppResult<(i32, i32)> {
    let geometry = Command::new("xwininfo")
        .args(&["-name", window_title])
        .output()?;
    
    if !geometry.status.success() {
        return Err(AppError::WindowNotFound(
            format!("Window '{}' not found", window_title)
        ));
    }

    let geometry_output = String::from_utf8(geometry.stdout)?;
    
    let parse_value = |s: &str| -> AppResult<i32> {
        geometry_output.lines()
            .find(|l| l.trim().starts_with(s))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().split(' ').next())
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| AppError::WindowNotFound(
                format!("Could not parse {} from xwininfo output", s)
            ))
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
                &output_file
            ])
            .status();
        
        if status_import.is_err() || !status_import.unwrap().success() {
            return Err(AppError::ScrotFailed(
                format!("Failed to capture window area with both maim and import")
            ));
        }
    }

    if !std::path::Path::new(&output_file).exists() {
        return Err(AppError::ScrotFailed(
            "Output file not created".to_string()
        ));
    }

    Ok((x, y))
}

fn parse_window_id(geometry_output: &str) -> AppResult<u32> {
    geometry_output.lines()
        .find(|l| l.contains("Window id:"))
        .and_then(|l| l.split_whitespace().nth(3))
        .and_then(|id| if id.starts_with("0x") {
            u32::from_str_radix(&id[2..], 16).ok()
        } else {
            id.parse().ok()
        })
        .ok_or_else(|| AppError::WindowNotFound(
            "Could not parse window ID".to_string()
        ))
}

fn load_settings() -> AppResult<Settings> {
    let settings_content = fs::read_to_string("settings.json")?;
    let settings: Settings = serde_json::from_str(&settings_content)?;
    Ok(settings)
}

fn click_and_drag_with_xdotool(
    window_x: i32,
    window_y: i32,
    from: (i32, i32),
    from_size: (i32, i32),  // Добавляем размеры объекта
    to: (i32, i32),
    to_size: (i32, i32),    // Добавляем размеры объекта
    move_delay_ms: u64
) -> AppResult<()> {
    // Рассчитываем центр начального объекта
    let abs_from_x = window_x + from.0 + from_size.0 / 2;
    let abs_from_y = window_y + from.1 + from_size.1 / 2;
    
    // Рассчитываем центр конечного объекта
    let abs_to_x = window_x + to.0 + to_size.0 / 2;
    let abs_to_y = window_y + to.1 + to_size.1 / 2;

    // 1. Перемещаем курсор в центр начального объекта
    Command::new("xdotool")
        .args(&["mousemove", &abs_from_x.to_string(), &abs_from_y.to_string()])
        .status()?;
    thread::sleep(Duration::from_millis(move_delay_ms));

    // 2. Нажимаем кнопку мыши
    Command::new("xdotool")
        .args(&["mousedown", "1"])
        .status()?;
    thread::sleep(Duration::from_millis(2));

    // 3. Плавно перемещаем курсор в центр конечного объекта
    let steps = 3;
    for step in 0..=steps {
        let x = abs_from_x + (abs_to_x - abs_from_x) * step as i32 / steps as i32;
        let y = abs_from_y + (abs_to_y - abs_from_y) * step as i32 / steps as i32;
        
        Command::new("xdotool")
            .args(&["mousemove", &x.to_string(), &y.to_string()])
            .status()?;
        thread::sleep(Duration::from_millis(move_delay_ms));
    }

    // 4. Отпускаем кнопку мыши
    Command::new("xdotool")
        .args(&["mouseup", "1"])
        .status()?;
    thread::sleep(Duration::from_millis(2));

    Ok(())
}


fn group_detections_by_name(detections: Vec<DetectionResult>) -> HashMap<String, Vec<DetectionResult>> {
    let mut groups = HashMap::new();
    for detection in detections {
        groups.entry(detection.object_name.clone())
            .or_insert_with(Vec::new)
            .push(detection);
    }
    groups
}

fn main() -> AppResult<()> {
    let settings = load_settings()?;
    
    let mut detector = ObjectDetector::new(0.35);
    
    for template_settings in settings.templates {
        detector.add_template(
            &template_settings.name,
            &template_settings.path,
            template_settings.threshold,
            template_settings.min_distance,
            template_settings.red,
            template_settings.green,
            template_settings.blue
        )?;
    }

    let screenshot_path = "screenshot.png";
    let (window_x, window_y) = capture_window_by_title(&settings.window_title, screenshot_path)?;

    let mut image = imgcodecs::imread(screenshot_path, IMREAD_COLOR)
        .map_err(|e| AppError::ImageProcessing(format!("Failed to load screenshot: {}", e)))?;
    
    if image.empty() {
        return Err(AppError::ImageProcessing("Loaded image is empty".to_string()));
    }

    let detections = detector.detect_objects(&image)?;
    
    println!("Found {} objects:", detections.len());
    for detection in &detections {
        println!(
            "- {} at ({}, {}) with confidence {:.2}",
            detection.object_name,
            detection.location.x,
            detection.location.y,
            detection.confidence
        );
    }

    detector.draw_detections(&mut image, &detections)?;
    imgcodecs::imwrite("result.png", &image, &core::Vector::new())?;
    println!("Result saved to result.png");

    let (conn, screen_num) = x11rb::connect(None)?;
    let root = conn.setup().roots[screen_num].root;

    for (name, objects) in group_detections_by_name(detections) {
        let mut filtered_objects: Vec<_> = objects.into_iter()
            .filter(|obj| obj.object_name != "Empty")
            .collect();

        if filtered_objects.len() >= 2 {
            filtered_objects.sort_by(|a, b| a.location.x.cmp(&b.location.x));
            
            let mut connected_objects = Vec::new();
            
            for i in 0..filtered_objects.len() - 1 {
                if connected_objects.contains(&i) {
                    continue;
                }
                
                // Получаем размеры объектов из шаблонов (уже масштабированные обратно)
                let from_template = detector.templates.iter()
                    .find(|t| t.name == filtered_objects[i].object_name)
                    .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;
                
                let to_template = detector.templates.iter()
                    .find(|t| t.name == filtered_objects[i+1].object_name)
                    .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;
                
                // Масштабируем размеры шаблонов обратно к оригинальным размерам
                let from_size = (
                    (from_template.template.cols() as f64 / detector.scale_factor) as i32,
                    (from_template.template.rows() as f64 / detector.scale_factor) as i32
                );
                let to_size = (
                    (to_template.template.cols() as f64 / detector.scale_factor) as i32,
                    (to_template.template.rows() as f64 / detector.scale_factor) as i32
                );
                
                let from = filtered_objects[i].location;
                let to = filtered_objects[i + 1].location;

                click_and_drag_with_xdotool(
                    window_x,
                    window_y,
                    (from.x, from.y), 
                    from_size,
                    (to.x, to.y),
                    to_size,
                    3 // задержка в миллисекундах между шагами перемещения
                )?;
                
                thread::sleep(Duration::from_millis(2));
                
                connected_objects.push(i);
                connected_objects.push(i + 1);
            }
        }
    }

    Ok(())
}