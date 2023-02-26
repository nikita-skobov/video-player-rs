use std::{process::{Command, Stdio}, io::Write};

/// use this program to create the test case video
/// it is 200x200, 30fps, pure color video, encoded losslessly using ffv1 codec (for accurate testing).
/// halfway through it switches from GREEN to RED.
/// You must run `cargo run --example create_video` before running `cargo test`

fn main() {
    let pixel_red = [255, 0, 0];
    let pixel_green = [0, 255, 0];
    let width = 200;
    let height = 200;
    let pixels_per_frame = width * height;
    let resolution_str = format!("{}x{}", width, height);
    let duration_s = 5;
    let fps = 30;
    let num_frames = duration_s * fps;
    let half_frame_index = num_frames / 2;
    let mut video_data = vec![];
    for i in 0..num_frames {
        for _ in 0..pixels_per_frame {
            if i > half_frame_index {
                video_data.extend_from_slice(&pixel_red);
            } else {
                video_data.extend_from_slice(&pixel_green);
            }
        }
    }
    let mut cmd = Command::new("ffmpeg")
        .args(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", &resolution_str, "-framerate", &fps.to_string(),
            "-i", "-",
            "-c:v", "ffv1",
            "testvid.mkv"
        ]).stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn().expect("Failed to spawn ffmpeg command");
    if let Some(stdin) = &mut cmd.stdin {
        stdin.write_all(&video_data).expect("Failed to write to stdin");
    } else {
        eprintln!("Failed to get stdin!");
        std::process::exit(1);
    }
    let output = cmd.wait_with_output().expect("Failed to read stdout");
    println!("{:#?}", output);
}
