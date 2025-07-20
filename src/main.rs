use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs::{self, IMREAD_COLOR},
    imgproc::{
        self, cvt_color, resize, threshold, LineTypes, COLOR_BGR2GRAY, FILLED, INTER_AREA, INTER_LINEAR, THRESH_BINARY, TM_CCOEFF_NORMED
    },
    prelude::*,
    Result as OpenCVResult,
};
use rayon::prelude::*;
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

#[derive(Debug, Deserialize, Serialize)]
struct Settings {
    window_title: String,
    resolution: f64,
    rescan_delay: u64,
    reference_width: i32,
    reference_height: i32,
    convert_to_grayscale: bool,
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
    blue: f32,
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
    gray_template: Mat,
    threshold: f64,
    min_distance: f32,
    red: f32,
    green: f32,
    blue: f32,
}

impl ObjectTemplate {
    fn new(name: &str, template_path: &str, threshold: f64, min_distance: f32, red: f32, green: f32, blue: f32) -> OpenCVResult<Self> {
        let template = imgcodecs::imread(template_path, IMREAD_COLOR)?;
        let mut gray_template = Mat::default();
        cvt_color(&template, &mut gray_template, COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        
        Ok(Self {
            name: name.to_string(),
            template,
            gray_template,
            threshold,
            min_distance,
            red,
            green,
            blue,
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
    base_scale_factor: f64,
    reference_size: Size,
}

impl ObjectDetector {
    fn new(base_scale_factor: f64, reference_width: i32, reference_height: i32) -> Self {
        Self { 
            templates: Vec::new(),
            base_scale_factor,
            reference_size: Size::new(reference_width, reference_height),
        }
    }

    fn calculate_current_scale(&self, current_size: Size) -> f64 {
        let width_scale = current_size.width as f64 / self.reference_size.width as f64;
        let height_scale = current_size.height as f64 / self.reference_size.height as f64;
        (width_scale + height_scale) / 2.0
    }

    fn add_template(&mut self, name: &str, template_path: &str, threshold: f64, min_distance: f32, red: f32, green: f32, blue: f32) -> OpenCVResult<()> {
        let template = ObjectTemplate::new(name, template_path, threshold, min_distance, red, green, blue)?;
        self.templates.push(Arc::new(template));
        Ok(())
    }

    fn detect_objects_optimized(&mut self, image: &Mat, convert_to_grayscale: bool) -> OpenCVResult<Vec<DetectionResult>> {
        let start_time = Instant::now();
        
        // Подготовка изображения
        let working_image = if convert_to_grayscale {
            let mut gray = Mat::default();
            cvt_color(image, &mut gray, COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
            gray
        } else {
            image.clone()
        };

        // Масштабирование изображения
        let mut resized = Mat::default();
        resize(
            &working_image,
            &mut resized,
            Size::new(0, 0),
            self.base_scale_factor,
            self.base_scale_factor,
            INTER_AREA,
        )?;

        // Параллельное сопоставление шаблонов
        let all_results: Vec<Vec<DetectionResult>> = self.templates
            .par_iter()
            .map(|template| {
                let template_image = if convert_to_grayscale {
                    &template.gray_template
                } else {
                    &template.template
                };

                // Масштабирование шаблона
                let mut scaled_template = Mat::default();
                if resize(
                    template_image,
                    &mut scaled_template,
                    Size::new(0, 0),
                    self.base_scale_factor,
                    self.base_scale_factor,
                    INTER_AREA,
                ).is_err() {
                    return Vec::new();
                }

                let mut result_mat = Mat::default();
                if imgproc::match_template(
                    &resized,
                    &scaled_template,
                    &mut result_mat,
                    TM_CCOEFF_NORMED,
                    &Mat::default(),
                ).is_err() {
                    return Vec::new();
                }

                let mut thresholded = Mat::default();
                if threshold(
                    &result_mat,
                    &mut thresholded,
                    template.threshold,
                    1.0,
                    THRESH_BINARY,
                ).is_err() {
                    return Vec::new();
                }

                let mut mask_8u = Mat::default();
                if thresholded.convert_to(&mut mask_8u, core::CV_8U, 255.0, 0.0).is_err() {
                    return Vec::new();
                }

                let mut local_results = Vec::new();
                let mut max_val = f64::MIN;
                let mut max_loc = Point::default();
                
                loop {
                    if core::min_max_loc(
                        &result_mat,
                        None,
                        Some(&mut max_val),
                        None,
                        Some(&mut max_loc),
                        &mask_8u,
                    ).is_err() {
                        break;
                    }
                    
                    if max_val < template.threshold {
                        break;
                    }
                    
                    local_results.push(DetectionResult {
                        object_name: template.name.clone(),
                        location: Point::new(
                            (max_loc.x as f64 / self.base_scale_factor) as i32,
                            (max_loc.y as f64 / self.base_scale_factor) as i32,
                        ),
                        confidence: max_val,
                    });

                    // Обнуляем найденную область
                    let _ = imgproc::rectangle(
                        &mut result_mat,
                        Rect::new(
                            max_loc.x - scaled_template.cols() / 2,
                            max_loc.y - scaled_template.rows() / 2,
                            scaled_template.cols(),
                            scaled_template.rows(),
                        ),
                        Scalar::all(0.0),
                        FILLED,
                        LineTypes::LINE_8.into(),
                        0,
                    );

                    let _ = imgproc::rectangle(
                        &mut mask_8u,
                        Rect::new(
                            max_loc.x - scaled_template.cols() / 2,
                            max_loc.y - scaled_template.rows() / 2,
                            scaled_template.cols(),
                            scaled_template.rows(),
                        ),
                        Scalar::all(0.0),
                        FILLED,
                        LineTypes::LINE_8.into(),
                        0,
                    );

                    max_val = f64::MIN;
                }
                
                local_results
            })
            .collect();

        println!("Optimized detection took: {:?}", start_time.elapsed());
        
        Ok(self.filter_close_detections(all_results.into_iter().flatten().collect()))
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
                let color = Scalar::new(
                    template.blue.into(),
                    template.green.into(),
                    template.red.into(),
                    0.0
                );

                imgproc::rectangle(
                    image,
                    Rect::new(
                        detection.location.x,
                        detection.location.y,
                        template.template.cols(),
                        template.template.rows(),
                    ),
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
            templates: Vec::new(),
        };
        
        let serialized = serde_json::to_string_pretty(&settings)?;
        fs::write(settings_path, serialized)?;
        
        Ok(settings)
    }
}

fn get_window_size(window_title: &str) -> AppResult<(i32, i32)> {
    let output = Command::new("xwininfo")
        .args(&["-name", window_title])
        .output()?;
    
    if !output.status.success() {
        return Err(AppError::WindowNotFound(
            format!("Window '{}' not found", window_title)
        ));
    }

    let output_str = String::from_utf8(output.stdout)?;
    
    let width = output_str.lines()
        .find(|l| l.trim().starts_with("Width"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().split(' ').next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| AppError::WindowNotFound(
            "Could not parse width".to_string()
        ))?;

    let height = output_str.lines()
        .find(|l| l.trim().starts_with("Height"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().split(' ').next())
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| AppError::WindowNotFound(
            "Could not parse height".to_string()
        ))?;

    Ok((width, height))
}

fn click_and_drag_with_xdotool(
    window_x: i32,
    window_y: i32,
    from: (i32, i32),
    from_size: (i32, i32),
    to: (i32, i32),
    to_size: (i32, i32),
    steps: i32,
    move_delay_ms: u64
) -> AppResult<()> {
    let original_pos = Command::new("xdotool")
        .args(&["getmouselocation", "--shell"])
        .output()?;
    
    let original_pos = String::from_utf8(original_pos.stdout)?;
    let mut x = 0;
    let mut y = 0;
    
    for line in original_pos.lines() {
        if line.starts_with("X=") {
            x = line[2..].parse().unwrap_or(0);
        } else if line.starts_with("Y=") {
            y = line[2..].parse().unwrap_or(0);
        }
    }

    let abs_from_x = window_x + from.0 + from_size.0 / 2;
    let abs_from_y = window_y + from.1 + from_size.1 / 2;
    
    let abs_to_x = window_x + to.0 + to_size.0 / 2;
    let abs_to_y = window_y + to.1 + to_size.1 / 2;

    Command::new("xdotool")
        .args(&["mousemove", &abs_from_x.to_string(), &abs_from_y.to_string()])
        .status()?;
    thread::sleep(Duration::from_millis(move_delay_ms));

    Command::new("xdotool")
        .args(&["mousedown", "1"])
        .status()?;
    thread::sleep(Duration::from_millis(5));

    for step in 0..=steps {
        let x = abs_from_x + (abs_to_x - abs_from_x) * step as i32 / steps as i32;
        let y = abs_from_y + (abs_to_y - abs_from_y) * step as i32 / steps as i32;
        
        Command::new("xdotool")
            .args(&["mousemove", &x.to_string(), &y.to_string()])
            .status()?;
        thread::sleep(Duration::from_millis(move_delay_ms));
    }
    thread::sleep(Duration::from_millis(5));
    Command::new("xdotool")
        .args(&["mouseup", "1"])
        .status()?;
    thread::sleep(Duration::from_millis(5));

    Command::new("xdotool")
        .args(&["mousemove", &x.to_string(), &y.to_string()])
        .status()?;
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

fn display_results_as_table(detections: &[DetectionResult], cols: usize, rows: usize) {
    if detections.is_empty() {
        println!("No objects detected");
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

    // Создаем таблицу
    let mut table: Vec<Vec<Option<(u32)>>> = vec![vec![None; cols]; rows];

    // Заполняем таблицу
    for detection in detections {
        let col = ((detection.location.x - min_x) as f32 / cell_width).round() as usize;
        let row = ((detection.location.y - min_y) as f32 / cell_height).round() as usize;

       let number = detection.object_name
                .chars()
                .filter_map(|c| c.to_digit(10))
                .fold(0, |acc, digit| acc * 10 + digit);

        if row < rows && col < cols {
            table[row][col] = Some(number);
        }
    }

   let cell_width = 4; // Минимальная ширина для "0" и двузначных чисел

    let line_length = cols * (cell_width + 2) + 1;

    // Выводим таблицу
    println!("Detection results ({}x{} grid):", cols, rows);
    println!("{}", "-".repeat(line_length));
    
    for row in table {
        print!("|");
        for cell in row {
            match cell {
                Some((num)) => print!(" {:^3} |", num),
                None => print!(" {:^3} |", "0"), // Заменяем None на "0"
            }
        }
        println!("\n{}", "-".repeat(line_length));
    }
}

fn check_and_suggest_window_size(window_title: &str, recommended_width: i32, recommended_height: i32) -> AppResult<()> {
    let (current_width, current_height) = get_window_size(window_title)?;
    
    // Define tolerance (5 pixels in each direction)
    const TOLERANCE: i32 = 5;
    let width_diff = (current_width - recommended_width).abs();
    let height_diff = (current_height - recommended_height).abs();
    
    if width_diff > TOLERANCE || height_diff > TOLERANCE {
        println!("Current window size: {}x{}", current_width, current_height);
        println!("Recommended window size: {}x{} (with ±{}px tolerance)", 
                recommended_width, recommended_height, TOLERANCE);
        
        // Show exact difference information
        if width_diff > TOLERANCE {
            println!("Width difference: {}px (tolerance: {}px)", width_diff, TOLERANCE);
        }
        if height_diff > TOLERANCE {
            println!("Height difference: {}px (tolerance: {}px)", height_diff, TOLERANCE);
        }
        
        println!("Would you like to resize the window to the recommended size? (y/n)");
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() == "y" {
            Command::new("wmctrl")
                .args(&[
                    "-r", window_title,
                    "-e", &format!("0,-1,-1,{},{}", recommended_width, recommended_height)
                ])
                .status()?;
            println!("Window size changed to {}x{}. Please restart the program.", 
                   recommended_width, recommended_height);
            std::process::exit(0);
        } else {
            println!("Continuing with current window size. Detection results may be less accurate.");
        }
    }
    
    Ok(())
}

use std::env;

fn main() -> AppResult<()> {
    let args: Vec<String> = env::args().collect();
    let infinite_mode = args.iter().any(|arg| arg == "--infinite" || arg == "-i");

    let window_title = "M2006C3MNG";
    let settings = load_or_create_settings(window_title)?;

    check_and_suggest_window_size(&settings.window_title, settings.reference_width, settings.reference_height)?;
    
    let mut detector = ObjectDetector::new(
        settings.resolution,
        settings.reference_width,
        settings.reference_height
    );
    
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

    loop {
        let screenshot_path = "screenshot.png";
        let (window_x, window_y) = capture_window_by_title(&settings.window_title, screenshot_path)?;

        let mut image = imgcodecs::imread(screenshot_path, IMREAD_COLOR)
            .map_err(|e| AppError::ImageProcessing(format!("Failed to load screenshot: {}", e)))?;
        
        if image.empty() {
            return Err(AppError::ImageProcessing("Loaded image is empty".to_string()));
        }

        let detections = detector.detect_objects_optimized(&image, settings.convert_to_grayscale)?;
        display_results_as_table(&detections, 4, 5);

        detector.draw_detections(&mut image, &detections)?;
        imgcodecs::imwrite("result.png", &image, &core::Vector::new())?;
        println!("Result saved to result.png");

        let mut barrels: Vec<_> = detections.into_iter()
            .filter(|d| d.object_name.starts_with("Barrel"))
            .collect();

        // Сортируем бочки по номеру
        barrels.sort_by(|a, b| {
            let num_a = a.object_name.split_whitespace().last().unwrap_or("0").parse().unwrap_or(0);
            let num_b = b.object_name.split_whitespace().last().unwrap_or("0").parse().unwrap_or(0);
            num_a.cmp(&num_b)
        });

        // Создаем карту номеров бочек и их позиций
        let mut barrel_map: HashMap<u32, Vec<(Point, String)>> = HashMap::new();
        for barrel in &barrels {
            let num = barrel.object_name.split_whitespace().last().unwrap_or("0").parse().unwrap_or(0);
            barrel_map.entry(num).or_default().push((barrel.location, barrel.object_name.clone()));
        }

        // Находим все уровни бочек, которые есть в текущем наборе
        let mut available_levels: Vec<u32> = barrel_map.keys().copied().collect();
        available_levels.sort();

        // Обрабатываем слияния начиная с самого младшего уровня
        let mut changed = true;
        while changed {
            changed = false;
            
            for &level in &available_levels {
                // Получаем количество бочек текущего уровня
                let barrel_count = barrel_map.get(&level).map_or(0, |v| v.len());
                
                if barrel_count >= 2 {
                    // Проверяем, есть ли шаблон для бочки следующего уровня
                    let next_level = level + 1;
                    let has_next_level_template = detector.templates.iter()
                        .any(|t| t.name == format!("Barrel {}", next_level));

                    // Сливаем только если есть шаблон для следующего уровня
                    if has_next_level_template {
                        // Получаем данные о бочках (копируем необходимые данные)
                        let first_barrel = barrel_map[&level][0].clone();
                        let second_barrel = barrel_map[&level][1].clone();

                        let from_template = detector.templates.iter()
                            .find(|t| t.name == first_barrel.1)
                            .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;
                        
                        let to_template = detector.templates.iter()
                            .find(|t| t.name == second_barrel.1)
                            .ok_or_else(|| AppError::ImageProcessing("Template not found".to_string()))?;
                        
                        let from_size = (
                            from_template.template.cols(),
                            from_template.template.rows()
                        );
                        let to_size = (
                            to_template.template.cols(),
                            to_template.template.rows()
                        );
                        
                        click_and_drag_with_xdotool(
                            window_x,
                            window_y,
                            (first_barrel.0.x, first_barrel.0.y), 
                            from_size,
                            (second_barrel.0.x, second_barrel.0.y),
                            to_size,
                            5,
                            1
                        )?;
                        
                        thread::sleep(Duration::from_millis(25));

                        // Обновляем карту бочек:
                        // 1. Удаляем две слитые бочки
                        barrel_map.get_mut(&level).unwrap().remove(0);
                        barrel_map.get_mut(&level).unwrap().remove(0);
                        
                        // 2. Добавляем новую бочку следующего уровня
                        barrel_map.entry(next_level).or_default().push((second_barrel.0, format!("Barrel {}", next_level)));

                        // Обновляем список доступных уровней
                        available_levels = barrel_map.keys().copied().collect();
                        available_levels.sort();

                        changed = true;
                        break; // Начинаем проверку снова с младшего уровня
                    }
                }
            }
        }

        // Оригинальная обработка других объектов (магниты и т.д.)
        // ... остальной код обработки ...

        if !infinite_mode {
            break;
        }

        thread::sleep(Duration::from_millis(settings.rescan_delay));
    }

    Ok(())
}