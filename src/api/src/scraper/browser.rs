//! Browser automation using chromiumoxide.

use anyhow::Result;
use chromiumoxide::browser::{Browser as ChromeBrowser, BrowserConfig};
use chromiumoxide::page::Page;
use futures::StreamExt;

/// Browser wrapper for web scraping
pub struct Browser {
    browser: ChromeBrowser,
    handle: tokio::task::JoinHandle<()>,
}

impl Browser {
    /// Launch a new headless browser instance
    pub async fn launch() -> Result<Self> {
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

        // Wait for browser to be ready (reduced from 2s for performance)
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        Ok(Self { browser, handle })
    }

    /// Fetch page content with JavaScript rendering
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        let page = self
            .browser
            .new_page(url)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create new page: {}", e))?;

        // Wait for page load
        Self::wait_for_load(&page).await?;

        let html = page
            .content()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get page content: {}", e))?;

        // Close the page
        let _ = page.close().await;

        Ok(html)
    }

    /// Wait for page to finish loading
    async fn wait_for_load(_page: &Page) -> Result<()> {
        // Wait for network idle or timeout (reduced from 2s for performance)
        tokio::time::sleep(tokio::time::Duration::from_millis(1500)).await;
        Ok(())
    }

    /// Close the browser
    pub async fn close(mut self) -> Result<()> {
        let _ = self.browser.close().await;
        self.handle.abort();
        Ok(())
    }
}
