use crate::capture::AppResult;
use crate::capture::get_window_size;
use std::io;
use std::io::Write;
use std::process::Command;

pub fn check_and_suggest_window_size(
    window_title: &str,
    recommended_width: i32,
    recommended_height: i32,
) -> AppResult<()> {
    let (current_width, current_height) = get_window_size(window_title)?;

    // Define tolerance (5 pixels in each direction)
    const TOLERANCE: i32 = 5;
    let width_diff = (current_width - recommended_width).abs();
    let height_diff = (current_height - recommended_height).abs();

    if width_diff > TOLERANCE || height_diff > TOLERANCE {
        print!(
            "Current window size: {}x{}\n",
            current_width, current_height
        );
        print!(
            "Recommended window size: {}x{} (with ±{}px tolerance)\n",
            recommended_width, recommended_height, TOLERANCE
        );

        // Show exact difference information
        if width_diff > TOLERANCE {
            print!(
                "Width difference: {}px (tolerance: {}px)\n",
                width_diff, TOLERANCE
            );
        }
        if height_diff > TOLERANCE {
            print!(
                "Height difference: {}px (tolerance: {}px)\n",
                height_diff, TOLERANCE
            );
        }

        print!("Would you like to resize the window to the recommended size? (y/n)\n");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if input.trim().to_lowercase() == "y" {
            Command::new("wmctrl")
                .args(&[
                    "-r",
                    window_title,
                    "-e",
                    &format!("0,-1,-1,{},{}", recommended_width, recommended_height),
                ])
                .status()?;
            print!(
                "Window size changed to {}x{}. Please restart the program.\n",
                recommended_width, recommended_height
            );
        } else {
            print!(
                "Continuing with current window size. Detection results may be less accurate.\n"
            );
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

pub fn get_currect_mouse_potision() -> Result<(i32, i32), Box<dyn std::error::Error>> {
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
    Ok((original_x, original_y))
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
