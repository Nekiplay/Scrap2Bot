use opencv::Result as OpenCVResult;
use opencv::core::AlgorithmHint;
use opencv::core::CV_8U;
use opencv::core::Mat;
use opencv::core::Point;
use opencv::core::Rect;
use opencv::core::Scalar;
use opencv::core::Size;
use opencv::core::in_range;
use opencv::core::min_max_loc;
use opencv::imgcodecs;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc;
use opencv::imgproc::COLOR_BGR2GRAY;
use opencv::imgproc::FILLED;
use opencv::imgproc::INTER_AREA;
use opencv::imgproc::LineTypes;
use opencv::imgproc::THRESH_BINARY;
use opencv::imgproc::TM_CCOEFF_NORMED;
use opencv::imgproc::cvt_color;
use opencv::imgproc::resize;
use opencv::imgproc::threshold;
use opencv::prelude::MatTrait;
use opencv::prelude::MatTraitConst;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::io;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct ObjectTemplate {
    pub name: String,
    pub template: Mat,
    pub gray_template: Mat,
    pub threshold: f64,
    pub min_distance: f32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub resolution: Option<f64>,
}

impl ObjectTemplate {
    pub fn new(
        name: &str,
        template_path: &str,
        threshold: f64,
        min_distance: f32,
        red: f32,
        green: f32,
        blue: f32,
        resolution: Option<f64>,
    ) -> OpenCVResult<Self> {
        let mut template = imgcodecs::imread(template_path, IMREAD_COLOR)?;

        // Удаляем цвет (154, 195, 161) из шаблона
        let mut mask = Mat::default();
        in_range(
            &template,
            &Scalar::new(150.0, 190.0, 150.0, 0.0), // Нижняя граница с небольшим запасом
            &Scalar::new(158.0, 200.0, 158.0, 0.0), // Верхняя граница с небольшим запасом
            &mut mask,
        )?;

        // Заменяем цвет на прозрачный (черный)
        let black = Scalar::all(0.0);
        template.set_to(&black, &mask)?;

        let mut gray_template = Mat::default();
        cvt_color(
            &template,
            &mut gray_template,
            COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        Ok(Self {
            name: name.to_string(),
            template,
            gray_template,
            threshold,
            min_distance,
            red,
            green,
            blue,
            resolution,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub object_name: String,
    pub location: Point,
    pub confidence: f64,
}

pub struct ObjectDetector {
    pub templates: Vec<Arc<ObjectTemplate>>,
    pub base_scale_factor: f64,
    pub active_range: (usize, usize), // (start, end) индексы активных шаблонов
    pub empty_template_index: usize,  // Индекс шаблона Empty
    pub cloud_template_index: usize,  // Индекс шаблона Cloud
    pub full_range: bool,
}

impl ObjectDetector {
    pub fn new(base_scale_factor: f64) -> Self {
        Self {
            templates: Vec::new(),
            base_scale_factor,
            active_range: (0, 0), // Будет установлено при добавлении шаблонов
            empty_template_index: 0,
            cloud_template_index: 0,
            full_range: true, // Флаг полного диапазона
        }
    }

    pub fn update_active_range(&mut self, detected_numbers: &[u32]) {
        if detected_numbers.is_empty() {
            // If nothing is detected, use full range but ensure it's within bounds
            self.active_range = (0, self.templates.len().saturating_sub(1));
            self.full_range = true;
            return;
        }

        // If was full range and something found - switch to targeted range
        if self.full_range {
            let min_detected = *detected_numbers.iter().min().unwrap();
            let max_detected = *detected_numbers.iter().max().unwrap();

            // Define new range with buffer
            let new_min = min_detected.saturating_sub(5); // +5 previous levels
            let new_max = max_detected + 8; // +8 next levels

            // Find template indices for new range
            let mut start = 0;
            let mut end = self.templates.len().saturating_sub(1);

            for (i, template) in self.templates.iter().enumerate() {
                if let Some(num) = template
                    .name
                    .split_whitespace()
                    .last()
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    if num >= new_min && num <= new_max {
                        if i < start || start == 0 {
                            start = i;
                        }
                        if i > end || end == self.templates.len().saturating_sub(1) {
                            end = i;
                        }
                    }
                }
            }

            // Ensure the range is valid
            self.active_range = (
                start.min(self.templates.len().saturating_sub(1)),
                end.min(self.templates.len().saturating_sub(1)),
            );
            self.full_range = false;
        } else {
            // If already in targeted range - adjust it smoothly
            let min_detected = *detected_numbers.iter().min().unwrap();
            let max_detected = *detected_numbers.iter().max().unwrap();

            // Current bounds
            let (current_start, current_end) = self.active_range;

            // New bounds with buffer
            let new_min = min_detected.saturating_sub(3) as usize;
            let new_max = (max_detected + 3) as usize;

            // Smooth range expansion
            let new_start = if new_min < current_start {
                new_min
            } else {
                current_start
            };
            let new_end = if new_max > current_end {
                new_max
            } else {
                current_end
            };

            // Ensure the range is valid
            self.active_range = (
                new_start.min(self.templates.len().saturating_sub(1)),
                new_end.min(self.templates.len().saturating_sub(1)),
            );
        }
    }

    pub fn get_active_templates(&self) -> Vec<Arc<ObjectTemplate>> {
        let mut result = Vec::new();

        // Всегда включаем шаблон Empty
        if self.empty_template_index < self.templates.len() {
            result.push(self.templates[self.empty_template_index].clone());
        }
        if self.cloud_template_index < self.templates.len() {
            result.push(self.templates[self.cloud_template_index].clone());
        }

        // Добавляем шаблоны из активного диапазона (исключая Empty, если он уже добавлен)
        let (start, end) = self.active_range;
        for i in start..=end {
            if i < self.templates.len()
                && (i != self.empty_template_index && i != self.cloud_template_index)
            {
                result.push(self.templates[i].clone());
            }
        }

        result
    }

    pub fn add_template(
        &mut self,
        name: &str,
        template_path: &str,
        threshold: f64,
        min_distance: f32,
        red: f32,
        green: f32,
        blue: f32,
        resolution: Option<f64>,
    ) -> OpenCVResult<()> {
        let template = ObjectTemplate::new(
            name,
            template_path,
            threshold,
            min_distance,
            red,
            green,
            blue,
            resolution,
        )?;
        self.templates.push(Arc::new(template));

        self.active_range = (0, self.templates.len() - 1);
        self.full_range = true;

        Ok(())
    }

    pub fn detect_objects_optimized(
        &mut self,
        image: &Mat,
        convert_to_grayscale: bool,
    ) -> OpenCVResult<(Vec<DetectionResult>, u128)> {
        let start_time = Instant::now();

        // Подготовка изображения
        let working_image = if convert_to_grayscale {
            let mut gray = Mat::default();
            cvt_color(
                image,
                &mut gray,
                COLOR_BGR2GRAY,
                0,
                AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;
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
        let active_templates = self.get_active_templates();
        let all_results: Vec<Vec<DetectionResult>> = active_templates
            .par_iter()
            .map(|template| {
                let template_image = if convert_to_grayscale {
                    &template.gray_template
                } else {
                    &template.template
                };

                // Масштабирование шаблона
                let mut scaled_template = Mat::default();
                let scale_factor = template.resolution.unwrap_or(self.base_scale_factor);
                if resize(
                    template_image,
                    &mut scaled_template,
                    Size::new(0, 0),
                    scale_factor,
                    scale_factor,
                    INTER_AREA,
                )
                .is_err()
                {
                    return Vec::new();
                }

                let mut result_mat = Mat::default();
                if imgproc::match_template(
                    &resized,
                    &scaled_template,
                    &mut result_mat,
                    TM_CCOEFF_NORMED,
                    &Mat::default(),
                )
                .is_err()
                {
                    return Vec::new();
                }

                let mut thresholded = Mat::default();
                if threshold(
                    &result_mat,
                    &mut thresholded,
                    template.threshold,
                    1.0,
                    THRESH_BINARY,
                )
                .is_err()
                {
                    return Vec::new();
                }

                let mut mask_8u = Mat::default();
                if thresholded
                    .convert_to(&mut mask_8u, CV_8U, 255.0, 0.0)
                    .is_err()
                {
                    return Vec::new();
                }

                let mut local_results = Vec::new();
                let mut max_val = f64::MIN;
                let mut max_loc = Point::default();

                loop {
                    if min_max_loc(
                        &result_mat,
                        None,
                        Some(&mut max_val),
                        None,
                        Some(&mut max_loc),
                        &mask_8u,
                    )
                    .is_err()
                    {
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
        let elapsed = start_time.elapsed();
        let elapsed_ms = elapsed.as_millis();
        // Очищаем терминал и выводим информацию
        print!("\x1B[2J\x1B[3J\x1B[H");
        io::stdout().flush().map_err(|e| {
            opencv::Error::new(
                opencv::core::StsError,
                format!("Failed to flush stdout: {}", e),
            )
        })?;

        let detected_numbers: Vec<u32> = all_results
            .iter()
            .flatten()
            .filter_map(|d| {
                d.object_name
                    .split_whitespace()
                    .last()
                    .and_then(|s| s.parse().ok())
            })
            .collect();

        self.update_active_range(&detected_numbers);

        Ok((
            self.filter_close_detections(all_results.into_iter().flatten().collect()),
            elapsed_ms,
        ))
    }

    pub fn filter_close_detections(
        &self,
        mut results: Vec<DetectionResult>,
    ) -> Vec<DetectionResult> {
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
                if let Some(template) = self.templates.iter().find(|t| t.name == result.object_name)
                {
                    occupied.push((result.location, template.min_distance));
                    filtered.push(result);
                }
            }
        }

        filtered
    }

    pub fn draw_detections(
        &self,
        image: &mut Mat,
        detections: &[DetectionResult],
    ) -> OpenCVResult<()> {
        for detection in detections {
            if let Some(template) = self
                .templates
                .iter()
                .find(|t| t.name == detection.object_name)
            {
                let color = Scalar::new(
                    template.blue.into(),
                    template.green.into(),
                    template.red.into(),
                    0.0,
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
