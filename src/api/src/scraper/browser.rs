//! Browser automation using chromiumoxide.
//!
//! Provides browser automation with:
//! - DOM readiness detection (faster than fixed timeouts)
//! - Connection pooling (reuse browser instance)
//! - Retry logic for failed page loads

use anyhow::Result;
use chromiumoxide::browser::{Browser as ChromeBrowser, BrowserConfig};
use chromiumoxide::page::Page;
use futures::StreamExt;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tracing::{debug, warn};

use crate::retry::{retry_anyhow, RetryConfig};

/// Browser wrapper for web scraping
pub struct Browser {
    browser: ChromeBrowser,
    handle: tokio::task::JoinHandle<()>,
}

/// Configuration for page loading behavior
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PageLoadConfig {
    /// Maximum time to wait for page load
    pub timeout: Duration,
    /// Minimum wait time after DOM ready
    pub min_wait: Duration,
    /// Whether to wait for network idle
    pub wait_for_network_idle: bool,
}

impl Default for PageLoadConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            min_wait: Duration::from_millis(500),
            wait_for_network_idle: true,
        }
    }
}

impl Browser {
    /// Launch a new headless browser instance
    pub async fn launch() -> Result<Self> {
        Self::launch_with_retry(&RetryConfig::browser()).await
    }

    /// Launch browser with retry logic
    pub async fn launch_with_retry(retry_config: &RetryConfig) -> Result<Self> {
        retry_anyhow(retry_config, "browser launch", || async {
            Self::launch_internal().await
        })
        .await
    }

    /// Internal launch implementation
    async fn launch_internal() -> Result<Self> {
        // Find Chrome executable
        let chrome_path = if cfg!(target_os = "macos") {
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        } else if cfg!(target_os = "windows") {
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        } else {
            "google-chrome"
        };

        let config = BrowserConfig::builder()
            .chrome_executable(chrome_path)
            .no_sandbox()
            .disable_default_args()
            .arg("--headless=new")
            .arg("--disable-gpu")
            .arg("--disable-dev-shm-usage")
            .arg("--disable-software-rasterizer")
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-extensions")
            .arg("--disable-background-networking")
            .arg("--disable-sync")
            .arg("--disable-translate")
            .arg("--mute-audio")
            .window_size(1920, 1080)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build browser config: {}", e))?;

        let (browser, mut handler) = ChromeBrowser::launch(config)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to launch browser: {}", e))?;

        // Spawn handler task - must keep running for browser to work
        let handle = tokio::spawn(async move {
            loop {
                match handler.next().await {
                    Some(Ok(_)) => continue,
                    Some(Err(_)) => continue, // Don't break on errors
                    None => break,
                }
            }
        });

        // Wait for browser to be ready
        sleep(Duration::from_secs(1)).await;

        Ok(Self { browser, handle })
    }

    /// Fetch page content with JavaScript rendering
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        self.fetch_page_with_config(url, &PageLoadConfig::default())
            .await
    }

    /// Fetch page with retry logic (for future use)
    #[allow(dead_code)]
    pub async fn fetch_page_with_retry(
        &self,
        url: &str,
        retry_config: &RetryConfig,
    ) -> Result<String> {
        let url = url.to_string();
        retry_anyhow(retry_config, "page fetch", || {
            let url = url.clone();
            async move { self.fetch_page(&url).await }
        })
        .await
    }

    /// Fetch page content with custom configuration
    pub async fn fetch_page_with_config(&self, url: &str, config: &PageLoadConfig) -> Result<String> {
        let page = self
            .browser
            .new_page(url)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create new page: {}", e))?;

        // Wait for page load with timeout
        let load_result = timeout(config.timeout, self.wait_for_dom_ready(&page, config)).await;

        match load_result {
            Ok(Ok(())) => {
                debug!("Page loaded successfully: {}", url);
            }
            Ok(Err(e)) => {
                warn!("Page load error (continuing anyway): {}", e);
            }
            Err(_) => {
                warn!("Page load timeout after {:?}, continuing with partial content", config.timeout);
            }
        }

        let html = page
            .content()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get page content: {}", e))?;

        // Close the page
        let _ = page.close().await;

        Ok(html)
    }

    /// Wait for DOM to be ready using JavaScript detection
    async fn wait_for_dom_ready(&self, page: &Page, config: &PageLoadConfig) -> Result<()> {
        // Check document.readyState using JavaScript
        let check_ready_script = r#"
            (function() {
                return document.readyState === 'complete' || document.readyState === 'interactive';
            })()
        "#;

        // Poll until ready or timeout
        let poll_interval = Duration::from_millis(100);
        let max_polls = 50; // 5 seconds max polling

        for i in 0..max_polls {
            match page.evaluate(check_ready_script).await {
                Ok(result) => {
                    if let Some(ready) = result.value().and_then(|v| v.as_bool()) {
                        if ready {
                            debug!("DOM ready after {} polls", i + 1);
                            // Additional wait for dynamic content
                            sleep(config.min_wait).await;
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    debug!("readyState check failed (attempt {}): {}", i + 1, e);
                }
            }
            sleep(poll_interval).await;
        }

        // Fallback: wait minimum time
        warn!("DOM readyState polling failed, using fallback wait");
        sleep(config.min_wait).await;
        Ok(())
    }

    /// Close the browser
    pub async fn close(mut self) -> Result<()> {
        let _ = self.browser.close().await;
        self.handle.abort();
        Ok(())
    }
}
