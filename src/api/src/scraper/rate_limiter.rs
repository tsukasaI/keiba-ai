//! Rate limiter using token bucket algorithm.

use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

/// Token bucket rate limiter
pub struct RateLimiter {
    state: Arc<Mutex<RateLimiterState>>,
}

struct RateLimiterState {
    tokens: f64,
    last_update: Instant,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    min_delay: Duration,
    max_delay: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `requests_per_minute` - Maximum requests per minute
    /// * `min_delay_secs` - Minimum delay between requests
    /// * `max_delay_secs` - Maximum delay between requests
    pub fn new(requests_per_minute: u32, min_delay_secs: f64, max_delay_secs: f64) -> Self {
        let max_tokens = requests_per_minute as f64;
        let refill_rate = requests_per_minute as f64 / 60.0;

        Self {
            state: Arc::new(Mutex::new(RateLimiterState {
                tokens: max_tokens,
                last_update: Instant::now(),
                max_tokens,
                refill_rate,
                min_delay: Duration::from_secs_f64(min_delay_secs),
                max_delay: Duration::from_secs_f64(max_delay_secs),
            })),
        }
    }

    /// Create with default settings (60 req/min, 0.5-1.0s delay)
    pub fn default_limiter() -> Self {
        Self::new(60, 0.5, 1.0)
    }

    /// Acquire a token, waiting if necessary
    pub async fn acquire(&self) {
        let delay = {
            let mut state = self.state.lock().await;

            // Refill tokens
            let now = Instant::now();
            let elapsed = now.duration_since(state.last_update).as_secs_f64();
            state.tokens = (state.tokens + elapsed * state.refill_rate).min(state.max_tokens);
            state.last_update = now;

            // Check if we have tokens
            if state.tokens >= 1.0 {
                state.tokens -= 1.0;
                // Return random delay between min and max
                let delay_range = state.max_delay - state.min_delay;
                let random_factor = rand_delay();
                state.min_delay + delay_range.mul_f64(random_factor)
            } else {
                // Wait for token to become available
                let wait_time = (1.0 - state.tokens) / state.refill_rate;
                state.tokens = 0.0;
                Duration::from_secs_f64(wait_time) + state.min_delay
            }
        };

        tokio::time::sleep(delay).await;
    }
}

/// Generate a pseudo-random delay factor (0.0 - 1.0)
fn rand_delay() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}
