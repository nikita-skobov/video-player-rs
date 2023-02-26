use macroquad::prelude::*;
use video_reader_rs::*;

pub struct TextureWrapper {
    pub t: Texture2D,
}

impl From<(u32, u32, Vec<u8>)> for TextureWrapper {
    fn from(value: (u32, u32, Vec<u8>)) -> Self {
        let t = Texture2D::from_rgba8(value.0 as _, value.1 as _, &value.2);
        Self { t }
    }
}

impl FrameObject for TextureWrapper {
    fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Self {
        let t = Texture2D::from_rgba8(width as _, height as _, &data);
        Self { t }
    }

    fn delete(self) {
        self.t.delete();
    }
}

struct Timeline {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    duration: f32,
    current_time: f32,
}

impl Timeline {
    pub fn new(x: f32, y: f32, width: f32, height: f32, duration: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            duration,
            current_time: 0.0,
        }
    }

    pub fn set_current_time(&mut self, time: f32) {
        self.current_time = time.min(self.duration).max(0.0);
    }

    /// returns the millisecond time where the user clicked
    pub fn handle_click(&mut self) -> Option<u128> {
        if !is_mouse_button_pressed(MouseButton::Left) {
            return None;
        }
        let (x, y) = mouse_position();
        if y >= self.y && y <= self.y + self.height && x >= self.x && x <= self.x + self.width {
            let progress = (x - self.x) / self.width;
            let current_time = progress * self.duration;
            let current_time_ms = (current_time * 1000.0) as u128;
            return Some(current_time_ms);
        }
        return None;
    }

    pub fn draw(&self) {
        let progress = self.current_time / self.duration;

        // Draw background
        draw_rectangle(self.x, self.y, self.width, self.height, BLACK);

        // Draw progress bar
        let progress_width = progress * self.width;
        draw_rectangle(self.x, self.y, progress_width, self.height, BLUE);

        // Draw current time label
        let current_time_label = format!("{:.1}s", self.current_time);
        let current_time_label_width = measure_text(&current_time_label, None, 14, 1.0).width;
        let current_time_label_x = self.x + progress_width - current_time_label_width;
        let current_time_label_y = self.y + self.height + 4.0;
        draw_text(&current_time_label, current_time_label_x, current_time_label_y, 14.0, WHITE);

        // Draw duration label
        let duration_label = format!("{:.1}s", self.duration);
        let duration_label_width = measure_text(&duration_label, None, 14, 1.0).width;
        let duration_label_x = self.x + self.width - duration_label_width;
        let duration_label_y = self.y + self.height + 4.0;
        draw_text(&duration_label, duration_label_x, duration_label_y, 14.0, WHITE);
    }
}

#[macroquad::main("VideoPlayer")]
async fn main() {
    let filepath = std::env::args().nth(1).expect("Must provide path to a video");
    let mut player = VideoPlayer::<TextureWrapper>::new(filepath).expect("Failed to read video");
    let mut timeline = Timeline::new(10.0, 10.0, 620.0, 40.0, player.duration_ms as f32 / 1000.0);
    loop {
        clear_background(BLACK);
        let screen_width = screen_width();
        let screen_height = screen_height();
        let dest_size = Some(Vec2::new(screen_width, screen_height));
        if let Ok(texture) = player.update() {
            draw_texture_ex(texture.t, 0.0, 0.0, WHITE, DrawTextureParams { dest_size, ..Default::default() })
        }
        if is_key_pressed(KeyCode::Space) {
            player.pause();
        }
        if is_key_pressed(KeyCode::Right) {
            player.skip_by_ms(10000);
        }
        if is_key_pressed(KeyCode::Left) {
            player.skip_by_ms(-10000);
        }
        if let Some(ms) = timeline.handle_click() {
            let ms_diff = ms - player.current_time_ms();
            player.skip_by_ms(ms_diff as _);
        }
        timeline.set_current_time(player.current_time_ms() as f32 / 1000.0);
        timeline.draw();
        next_frame().await;
    }
}
