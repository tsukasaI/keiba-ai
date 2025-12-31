//! File-based cache with TTL support.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::PathBuf;

/// Cache entry with timestamp
#[derive(Serialize, Deserialize)]
struct CacheEntry<T> {
    data: T,
    cached_at: DateTime<Utc>,
}

/// Cache categories with different TTLs
#[derive(Debug, Clone, Copy)]
pub enum CacheCategory {
    RaceCard, // 24 hours
    Horse,    // 7 days
    Jockey,   // 7 days
    Trainer,  // 7 days
}

impl CacheCategory {
    /// Get TTL duration
    pub fn ttl(&self) -> Duration {
        match self {
            CacheCategory::RaceCard => Duration::hours(24),
            CacheCategory::Horse => Duration::hours(24 * 7),
            CacheCategory::Jockey => Duration::hours(24 * 7),
            CacheCategory::Trainer => Duration::hours(24 * 7),
        }
    }

    /// Get directory name for this category
    pub fn dir_name(&self) -> &str {
        match self {
            CacheCategory::RaceCard => "race_card",
            CacheCategory::Horse => "horse",
            CacheCategory::Jockey => "jockey",
            CacheCategory::Trainer => "trainer",
        }
    }
}

/// File-based cache
pub struct Cache {
    base_dir: PathBuf,
}

impl Cache {
    /// Create a new cache with the given base directory
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Create cache with default directory
    pub fn default_cache() -> Self {
        let base_dir = PathBuf::from("data/cache/scraper");
        Self::new(base_dir)
    }

    /// Get cache directory for a category
    fn category_dir(&self, category: CacheCategory) -> PathBuf {
        self.base_dir.join(category.dir_name())
    }

    /// Get cache file path for a key
    fn cache_path(&self, category: CacheCategory, key: &str) -> PathBuf {
        self.category_dir(category).join(format!("{}.json", key))
    }

    /// Get cached data if valid
    pub fn get<T: DeserializeOwned>(&self, category: CacheCategory, key: &str) -> Option<T> {
        let path = self.cache_path(category, key);

        if !path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&path).ok()?;
        let entry: CacheEntry<T> = serde_json::from_str(&content).ok()?;

        // Check if expired
        let elapsed = Utc::now() - entry.cached_at;
        if elapsed > category.ttl() {
            // Remove expired cache
            let _ = std::fs::remove_file(&path);
            return None;
        }

        Some(entry.data)
    }

    /// Set cache data
    pub fn set<T: Serialize>(&self, category: CacheCategory, key: &str, data: &T) -> Result<()> {
        let dir = self.category_dir(category);
        std::fs::create_dir_all(&dir)?;

        let entry = CacheEntry {
            data,
            cached_at: Utc::now(),
        };

        let path = self.cache_path(category, key);
        let content = serde_json::to_string_pretty(&entry)?;
        std::fs::write(&path, content)?;

        Ok(())
    }

    /// Clear cache for a category
    #[allow(dead_code)]
    pub fn clear(&self, category: CacheCategory) -> Result<()> {
        let dir = self.category_dir(category);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Clear all cache
    #[allow(dead_code)]
    pub fn clear_all(&self) -> Result<()> {
        if self.base_dir.exists() {
            std::fs::remove_dir_all(&self.base_dir)?;
        }
        Ok(())
    }
}
