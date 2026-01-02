//! Retry logic with exponential backoff.
//!
//! Provides utilities for retrying failed operations with configurable
//! backoff strategies.

use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Create a config for network operations
    pub fn network() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
        }
    }

    /// Create a config for browser operations
    pub fn browser() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }

    /// Calculate delay for a given attempt
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_ms = self.initial_delay.as_millis() as f64 * self.multiplier.powi(attempt as i32);
        let delay = Duration::from_millis(delay_ms as u64);
        delay.min(self.max_delay)
    }
}

/// Retry an async operation with exponential backoff
pub async fn retry<T, E, F, Fut>(
    config: &RetryConfig,
    operation_name: &str,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut last_error: Option<E> = None;

    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(result) => {
                if attempt > 0 {
                    debug!("{} succeeded after {} retries", operation_name, attempt);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt < config.max_retries {
                    let delay = config.delay_for_attempt(attempt);
                    warn!(
                        "{} failed (attempt {}/{}): {}. Retrying in {:?}...",
                        operation_name,
                        attempt + 1,
                        config.max_retries + 1,
                        e,
                        delay
                    );
                    sleep(delay).await;
                }
                last_error = Some(e);
            }
        }
    }

    Err(last_error.unwrap())
}

/// Retry an async operation that returns anyhow::Result
pub async fn retry_anyhow<T, F, Fut>(
    config: &RetryConfig,
    operation_name: &str,
    operation: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<T>>,
{
    retry(config, operation_name, operation).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_first_try() {
        let config = RetryConfig::default();
        let result: Result<i32, &str> = retry(&config, "test", || async { Ok(42) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            multiplier: 2.0,
        };

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<i32, &str> = retry(&config, "test", || {
            let c = counter_clone.clone();
            async move {
                let attempt = c.fetch_add(1, Ordering::SeqCst);
                if attempt < 2 {
                    Err("temporary failure")
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_failures() {
        let config = RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(100),
            multiplier: 2.0,
        };

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<i32, &str> = retry(&config, "test", || {
            let c = counter_clone.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err("permanent failure")
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Initial + 2 retries
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig {
            max_retries: 5,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_delay_max_cap() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
        };

        // After several attempts, delay should be capped at max_delay
        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(5));
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(5));
    }
}
