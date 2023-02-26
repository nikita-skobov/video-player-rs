use serde::Deserialize;
use std::{process::{Command, Child, Stdio}, io::Read, fmt::Debug, time::Instant};


#[derive(Deserialize, Debug)]
pub struct FFProbeFormatInfo {
    pub duration: String,
    pub size: String,
    pub bit_rate: String,
}

#[derive(Deserialize, Debug)]
pub struct FFProbeStreamInfo {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codec_type: String,
    pub avg_frame_rate: String,
    pub nb_frames: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct FFProbeOutput {
    pub streams: Vec<FFProbeStreamInfo>,
    pub format: FFProbeFormatInfo
}

pub struct FFmpegVideoStream {
    pub child: Child,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub num_frames: usize,
    pub original_path: String,
    pub duration_ms: u128,

    pub current_frame_index: usize,
}

pub struct UsefulFFprobeData {
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub duration: f64,
    pub fps: f64,
}

impl FFProbeOutput {
    pub fn get_useful_data(&self) -> Option<UsefulFFprobeData> {
        let mut data = UsefulFFprobeData {
            width: 0,
            height: 0,
            num_frames: 0,
            duration: 0.0,
            fps: 0.0,
        };
        let mut got_width_height = false;
        let mut got_num_frames = false;
        let mut got_duration = false;
        let mut got_fps = false;
        for stream in self.streams.iter() {
            if stream.codec_type == "video" {
                // try get width/height
                match (stream.width, stream.height) {
                    (Some(width), Some(height)) => {
                        data.width = width;
                        data.height = height;
                        got_width_height = true;
                    }
                    _ => {},
                }
                // try get num frames
                if let Some(nb_frames) = &stream.nb_frames {
                    match nb_frames.parse::<u32>() {
                        Ok(n) => {
                            data.num_frames = n;
                            got_num_frames = true;
                        }
                        _ => {},
                    }
                }
                // try get duration
                match self.format.duration.parse::<f64>() {
                    Ok(d) => {
                        data.duration = d;
                        got_duration = true;
                    }
                    _ => {},
                }
                // try get fps
                let fps_str_opt = &stream.avg_frame_rate.split_once("/");
                if let Some((num_frames, time)) = fps_str_opt {
                    let num_frames = num_frames.parse::<f64>();
                    let time = time.parse::<f64>();
                    match (num_frames, time) {
                        (Ok(n), Ok(t)) => {
                            data.fps = n / t;
                            got_fps = true;
                        }
                        _ => {},
                    }
                }
            }
        }

        // if the stream doesnt have the frame info, we can also try to calculate it from the duration * fps.
        if !got_num_frames && got_duration && got_fps {
            data.num_frames = (data.duration * data.fps) as _;
            got_num_frames = true;
        }
        if got_duration && got_fps && got_num_frames && got_width_height {
            return Some(data);
        }
        None
    }
}

pub fn get_ffprobe_output<S: AsRef<str>>(path: S) -> Result<FFProbeOutput, String> {
    let mut args = [
        "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        "path will go here"
    ];
    let path_str = path.as_ref();
    args[6] = path_str;
    let res = Command::new("ffprobe").args(args).output();
    let output = match res {
        Ok(out) => out,
        Err(e) => {
            return Err(e.to_string());
        }
    };
    if !output.status.success() {
        let err_str = String::from_utf8_lossy(&output.stderr);
        let s = format!("Failed to get ffprobe output from '{:#?}'.\n {}", args, err_str.to_string());
        return Err(s);
    }
    let output_json_str: String = String::from_utf8_lossy(&output.stdout).into();
    let deser_result: Result<FFProbeOutput, _> = serde_json::from_str(&output_json_str);
    match deser_result {
        Ok(o) => {
            Ok(o)
        },
        Err(e) => {
            Err(e.to_string())
        }
    }
}

pub fn get_ffmpeg_video_stream<S: AsRef<str>>(path: S, seek_to: Option<&str>) -> Result<FFmpegVideoStream, String> {
    let original_path: String = path.as_ref().into();
    let video_info = get_ffprobe_output(&path)?;
    let useful_ffprobe_data = video_info.get_useful_data().ok_or_else(|| format!("Failed to find necessary video data for {:?}", path.as_ref()))?;
    let UsefulFFprobeData { width, height, fps, num_frames, duration, .. } = useful_ffprobe_data;
    let mut args = [
        "-ss", "0",
        "-i", "path will go here",
        "-c:v", "rawvideo",
        "-pix_fmt", "rgba",
        "-f", "rawvideo", "-"
    ];
    let path_str = path.as_ref();
    args[3] = path_str;
    if let Some(seek) = seek_to {
        args[1] = seek;
    }
    let res = Command::new("ffmpeg").args(args)
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .stdin(Stdio::null())
        .spawn();
    let child = match res {
        Ok(c) => c,
        Err(e) => {
            return Err(e.to_string());
        }
    };
    Ok(FFmpegVideoStream { child, width, height, fps, original_path, num_frames: num_frames as _, duration_ms: (duration * 1000.0) as _, current_frame_index: 0 })
}

impl FFmpegVideoStream {
    pub fn new<S: AsRef<str>>(path: S) -> Result<Self, String> {
        Ok(get_ffmpeg_video_stream(path, None)?)
    }
    #[inline(always)]
    pub fn get_num_bytes_per_frame(&self) -> usize {
        // we use rgba, so we know each pixel has 4 bytes:
        let bytes_per_pixel = 4;
        let pixels_per_frame = self.width as usize * self.height as usize;
        bytes_per_pixel * pixels_per_frame
    }
    /// returns the frame in a vector where every 4 bytes is a pixel (RGBA).
    /// returns an empty vector if reached the end of the video stream.
    pub fn get_next_frame(&mut self) -> Result<Vec<u8>, String> {
        let num_bytes = self.get_num_bytes_per_frame();
        let stdout = match &mut self.child.stdout {
            Some(s) => s,
            None => return Err("ffmpeg process has no stdout".into()),
        };
        let mut frame = Vec::with_capacity(num_bytes);
        frame.resize(num_bytes, 0);
        let res = stdout.read_exact(&mut frame);
        match res {
            Ok(_) => {},
            Err(e) => {
                match e.kind() {
                    // this shouldnt be an error, but rather we should indicate
                    // to the user that the stream is done.
                    std::io::ErrorKind::UnexpectedEof => {
                        return Ok(vec![]);
                    },
                    _ => {
                        return Err(e.to_string());
                    }
                }
            }
        }
        self.current_frame_index += 1;
        Ok(frame)
    }

    pub fn get_frame_at_ms(&mut self, ms: u128) -> Result<Vec<u8>, String> {
        let frame_index = self.get_frame_index_at_ms(ms);
        self.seek_to_frame_index(frame_index)?;
        self.get_next_frame()
    }

    pub fn get_frame_index_at_ms(&self, ms: u128) -> usize {
        // first, find the percentage of the video at that timestamp
        let percentage = ((ms * 100) as f64) / ((self.duration_ms * 100) as f64);
        // find the frame index at that percentage
        let frame_index = (self.num_frames as f64 * percentage).floor() as usize;
        frame_index
    }

    // if we wish to seek backwards, we must restart the child ffmpeg process,
    // and then seek forward to the desired frame.
    pub fn restart_and_seek_to(&mut self, frame_index: usize) -> Result<(), String> {
        // we tried our best to kill the child, but if we fail
        // ultimately it doesn't matter since we can just replace the child.
        let _ = self.child.kill();
        let seek_str = format!("{:.1}s", (frame_index as f64 / self.fps));
        let new_stream = match get_ffmpeg_video_stream(&self.original_path, Some(&seek_str)) {
            Ok(s) => s,
            Err(e) => {
                return Err(format!("Failed to seek backwards. {e}"));
            }
        };
        // make sure it has all the same properties. if something changed since we last read it,
        // it means we'd fail in a weird way later on:
        if new_stream.duration_ms != self.duration_ms {
            return Err(format!("{}'s duration has changed. Used to be {}, now {}. Aborting because it seems the file has been modified.", self.original_path, self.duration_ms, new_stream.duration_ms));
        }
        if new_stream.fps != self.fps {
            return Err(format!("{}'s fps has changed. Used to be {}, now {}. Aborting because it seems the file has been modified.", self.original_path, self.fps, new_stream.fps));
        }
        if new_stream.height != self.height {
            return Err(format!("{}'s height has changed. Used to be {}, now {}. Aborting because it seems the file has been modified.", self.original_path, self.height, new_stream.height));
        }
        if new_stream.width != self.width {
            return Err(format!("{}'s width has changed. Used to be {}, now {}. Aborting because it seems the file has been modified.", self.original_path, self.width, new_stream.width));
        }
        if new_stream.num_frames != self.num_frames {
            return Err(format!("{}'s frame count has changed. Used to be {}, now {}. Aborting because it seems the file has been modified.", self.original_path, self.num_frames, new_stream.num_frames));
        }
        // otherwise, we're good to just take the child and call it our own.
        self.child = new_stream.child;
        self.current_frame_index = frame_index;
        self.seek_to_frame_index(frame_index)
    }

    pub fn seek_to_frame_index(&mut self, frame_index: usize) -> Result<(), String> {
        let frame_index = if frame_index > self.num_frames {
            self.num_frames
        } else {
            frame_index
        };
        if frame_index < self.current_frame_index {
            return self.restart_and_seek_to(frame_index);
        }
        if frame_index == self.current_frame_index {
            // this is a noop
            return Ok(());
        }
        // // this is safe even if the desired frame index is way past the end of the
        // // video. this is because get_next_frame will just return empty Vec[] if
        // // we reach the end.
        let num_frames_to_skip = frame_index - self.current_frame_index;
        if num_frames_to_skip > 100 {
            return self.restart_and_seek_to(frame_index);
        }
        for _ in 0..num_frames_to_skip {
            let _ = self.get_next_frame()?;
        }
        Ok(())
    }
}

pub struct VideoPlayer<F: FrameObject> {
    pub stream: FFmpegVideoStream,
    pub stream_width: u32,
    pub stream_height: u32,
    pub current_frame: (usize, F),
    pub start_time: Option<Instant>,
    pub current_video_time_ms: u128,
    pub is_paused: bool,
    pub duration_ms: u128,
    pub num_frames: usize,
}

pub trait FrameObject {
    fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Self;
    fn delete(self);
}

impl<F: FrameObject> VideoPlayer<F> {
    pub fn new<S: AsRef<str>>(path: S) -> Result<Self, String> {
        let mut stream = FFmpegVideoStream::new(path)?;

        let frame_data = stream.get_next_frame()?;        
        let current_frame = F::from_raw(stream.width, stream.height, frame_data);
        Ok(Self {
            num_frames: stream.num_frames,
            duration_ms: stream.duration_ms,
            stream_width: stream.width,
            stream_height: stream.height,
            stream: stream,
            current_frame: (0, current_frame),
            start_time: None,
            current_video_time_ms: 0,
            is_paused: false,
        })
    }
    pub fn pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
    pub fn get_frame_index_at_ms(&self, ms: u128) -> usize {
        // first, find the percentage of the video at that timestamp
        let percentage = ((ms * 100) as f64) / ((self.duration_ms * 100) as f64);
        // find the frame index at that percentage
        let frame_index = (self.num_frames as f64 * percentage).floor() as usize;
        frame_index
    }
    pub fn skip_by_ms(&mut self, ms: i128) {
        if ms > 0 {
            self.current_video_time_ms += ms as u128;
        } else {
            let sub_by = (-ms) as u128;
            if self.current_video_time_ms < sub_by {
                self.current_video_time_ms = 0;
            } else {
                self.current_video_time_ms -= (-ms) as u128;
            }
        }
        self.start_time = None;
    }

    /// updates internal state, and returns your frame object for you to draw.
    pub fn update(&mut self) -> Result<&F, String> {
        let elapsed_ms = match self.start_time {
            Some(instant) => {
                if self.is_paused {
                    0
                } else {
                    instant.elapsed().as_millis()
                }
            }
            None => {
                0
            }
        };
        self.start_time = Some(Instant::now());
        self.current_video_time_ms += elapsed_ms;
        if self.current_video_time_ms >= self.duration_ms {
            self.current_video_time_ms = self.duration_ms;
        }
        let desired_frame_index = self.get_frame_index_at_ms(self.current_video_time_ms);
        if desired_frame_index == self.current_frame.0 {
            return Ok(&self.current_frame.1);
        }

        // otherwise we need to seek:
        self.stream.seek_to_frame_index(desired_frame_index)?;
        let frame_data = self.stream.get_next_frame()?;
        if frame_data.is_empty() {
            return Ok(&self.current_frame.1);
        }
        let frame_object = F::from_raw(self.stream.width, self.stream.height, frame_data);
        let old_frame = std::mem::replace(&mut self.current_frame, (desired_frame_index, frame_object));
        old_frame.1.delete();
        Ok(&self.current_frame.1)
    }

    pub fn current_time_ms(&self) -> u128 {
        self.current_video_time_ms
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    const TESTVID: &str = "./testvid.mkv";
    const TESTVID_FPS: usize = 30;
    const TESTVID_DURATION: usize = 5;
    const TESTVID_NUM_FRAMES: usize = TESTVID_FPS * TESTVID_DURATION;
    const TESTVID_SIZE: usize = 200; // width and height
    const TESTVID_PIXELS_PER_FRAME: usize = TESTVID_SIZE * TESTVID_SIZE;
    const BYTES_PER_PIXEL: usize = 4;
    const PIX_RED: &[u8; 4] = &[255, 0, 0, 255];
    const PIX_GREEN: &[u8; 4] = &[0, 255, 0, 255];

    fn assert_pixel_eq(frame: &Vec<u8>, desired_pixel: &[u8; 4]) {
        for pixels in frame.chunks(4) {
            assert_eq!(pixels, desired_pixel);
        }
    }

    fn testvid_exists() {
        let file = std::fs::File::open(TESTVID);
        assert!(file.is_ok(), "Failed to find {} which is needed for test cases. make sure to run 'cargo run --example create_video' if you have not yet", TESTVID);
    }

    #[test]
    fn can_read_ffprobe_info() {
        testvid_exists();
        let ffprobe_info = get_ffprobe_output(TESTVID).expect("failed to get ffprobe output");
        let useful_info = ffprobe_info.get_useful_data().expect("Failed to get useful ffprobe data");
        assert_eq!(useful_info.num_frames as usize, TESTVID_NUM_FRAMES);
    }

    #[test]
    fn can_read_frame() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        let frame = stream.get_next_frame().expect("Failed to read next frame");
        assert_eq!(frame.len(), BYTES_PER_PIXEL * TESTVID_PIXELS_PER_FRAME);
        assert_pixel_eq(&frame, PIX_GREEN);
    }

    #[test]
    fn can_read_all_frames() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        for _ in 0..TESTVID_NUM_FRAMES {
            let frame = stream.get_next_frame().expect("Failed to read next frame");
            assert_eq!(frame.len(), BYTES_PER_PIXEL * TESTVID_PIXELS_PER_FRAME);
        }
        // we read all the frames successfully, now we should confirm that
        // the next frame is empty, ie: we reached end of the stream.
        let frame = stream.get_next_frame().expect("Failed to read next frame");
        assert_eq!(frame.len(), 0);
    }

    #[test]
    fn can_seek() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        // testvid halfway through changes colors. go to the frame before the change first, and confirm its green:
        let half_frame_index = TESTVID_NUM_FRAMES / 2;
        stream.seek_to_frame_index(half_frame_index).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, half_frame_index);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_GREEN);
        assert_eq!(stream.current_frame_index, half_frame_index + 1); // getting the frame advances the current frame index
        // if we read again, it should be the second half of the video where the color changes:
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_RED);
    }

    #[test]
    fn can_seek_with_delay() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        // go roughly to the end of the stream, such that if we delay,
        // will we still be able to read the rest of the data even after the underlying child process is done?
        stream.seek_to_frame_index(TESTVID_NUM_FRAMES - 10).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, TESTVID_NUM_FRAMES - 10);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_RED);

        // now purposefully add a delay to see how it affects the underlying ffmpeg child process.
        std::thread::sleep(std::time::Duration::from_millis(1000));
        // now if we seek again, will it still work?
        stream.seek_to_frame_index(TESTVID_NUM_FRAMES - 2).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, TESTVID_NUM_FRAMES - 2);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_RED);
    }

    #[test]
    fn can_seek_backwards() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        // testvid halfway through changes colors. go to the frame before the change first, and confirm its green:
        let half_frame_index = TESTVID_NUM_FRAMES / 2;
        stream.seek_to_frame_index(half_frame_index).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, half_frame_index);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_GREEN);
        assert_eq!(stream.current_frame_index, half_frame_index + 1); // getting the frame advances the current frame index
        // if we read again, it should be the second half of the video where the color changes:
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_RED);
        // now if we seek back to the halfway point, we should expect to read a green frame:
        stream.seek_to_frame_index(half_frame_index).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, half_frame_index);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_pixel_eq(&frame, PIX_GREEN);
    }

    #[test]
    fn seeking_past_end_of_video_is_valid() {
        testvid_exists();
        let mut stream = FFmpegVideoStream::new(TESTVID).expect("failed to get ffmpeg video stream");
        stream.seek_to_frame_index(TESTVID_NUM_FRAMES + 100).expect("Failed to seek");
        assert_eq!(stream.current_frame_index, TESTVID_NUM_FRAMES);
        let frame = stream.get_next_frame().expect("failed to get next frame");
        assert_eq!(frame.len(), 0);
    }
}
