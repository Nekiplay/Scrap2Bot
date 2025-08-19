use crate::objectdetector::DetectionResult;
use crate::objectdetector::ObjectTemplate;
use crate::processors::calculate_required_merges;
use std::sync::Arc;

pub fn get_contrast_text_color(bg_r: f32, bg_g: f32, bg_b: f32) -> &'static str {
    // Конвертируем значения RGB в диапазон 0-1
    let r = bg_r / 255.0;
    let g = bg_g / 255.0;
    let b = bg_b / 255.0;

    // Вычисляем относительную яркость по формуле W3C
    let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;

    // Пороговое значение для выбора цвета текста (как в VS Code)
    // VS Code использует порог 0.5, но можно настроить
    if luminance > 0.5 {
        // Для светлых фонов - черный текст
        "\x1b[38;2;0;0;0m" // ANSI escape для черного цвета (#000000)
    } else {
        // Для темных фонов - белый текст
        "\x1b[38;2;255;255;255m" // ANSI escape для белого цвета (#FFFFFF)
    }
}

pub fn draw_cloud(drop_positions: &[usize], is_moving_right: bool, line_length: usize) {
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
}

pub fn display_results_as_table(
    detections: &[DetectionResult],
    cols: usize,
    rows: usize,
    templates: &[Arc<ObjectTemplate>],
    detection_time: usize,
    fps: f64,
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

    // Print table header (исправлено: убрано дублирование)
    print!("╔");
    for c in 0..cols {
        print!("{}", "═".repeat(cell_width));
        if c < cols - 1 {
            print!("╦");
        }
    }
    println!("╗");

    // Print table rows
    for row in 0..rows - 1 {
        // Изменено: rows-1 чтобы последняя строка обрабатывалась отдельно
        print!("║");
        for col in 0..cols {
            if let Some((num, (r, g, b))) = &table[row][col] {
                if *num != 0 {
                    let text_color = get_contrast_text_color(*r, *g, *b);
                    print!(
                        " {}\x1b[48;2;{:.0};{:.0};{:.0}m{:^3}\x1b[0m ",
                        text_color, r, g, b, num
                    );
                } else {
                    print!("{}", empty_cell);
                }
            } else {
                print!("{}", empty_cell);
            }
            print!("║");
        }
        println!();

        // Print row separator
        print!("╠");
        for c in 0..cols {
            print!("{}", "═".repeat(cell_width));
            if c < cols - 1 {
                print!("╬");
            }
        }
        println!("╣");
    }

    // Print last row with FPS (исправлено: убрано дублирование последней строки)
    print!("║");
    for col in 0..cols {
        if let Some((num, (r, g, b))) = &table[rows - 1][col] {
            if *num != 0 {
                let text_color = get_contrast_text_color(*r, *g, *b);
                print!(
                    " {}\x1b[48;2;{:.0};{:.0};{:.0}m{:^3}\x1b[0m ",
                    text_color, r, g, b, num
                );
            } else {
                print!("{}", empty_cell);
            }
        } else {
            print!("{}", empty_cell);
        }
        print!("║");
    }
    println!(" {:.0}fps", fps);

    // Print table footer with detection time
    print!("╚");
    for c in 0..cols {
        print!("{}", "═".repeat(cell_width));
        if c < cols - 1 {
            print!("╩");
        }
    }
    println!("╝ {}ms", detection_time);

    // Statistics section
    let min_w = 5; // Ширина для min_lvl
    let max_w = 5; // Ширина для max_lvl
    let target_w = 5; // Ширина для target
    let merges_w = 10; // Увеличил ширину для merges_remaining 
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
