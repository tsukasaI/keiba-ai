"""
Kaggleからデータセットをダウンロード

前提条件:
1. Kaggle APIのインストール: pip install kaggle
2. API認証情報の設定: ~/.kaggle/kaggle.json
   - Kaggleアカウント設定からAPIトークンを取得
   - chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_CONFIG, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    """Kaggle API認証情報の確認"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error(
            "Kaggle API credentials not found.\n"
            "1. Go to https://www.kaggle.com/settings\n"
            "2. Click 'Create New API Token'\n"
            "3. Save kaggle.json to ~/.kaggle/kaggle.json\n"
            "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )
        return False
    
    # パーミッションチェック（Unix系のみ）
    if os.name != "nt":
        mode = os.stat(kaggle_json).st_mode & 0o777
        if mode != 0o600:
            logger.warning(
                f"Kaggle credentials have insecure permissions: {oct(mode)}\n"
                "Run: chmod 600 ~/.kaggle/kaggle.json"
            )
    
    return True


def download_dataset(dataset_name: str, output_dir: Path) -> bool:
    """
    Kaggleデータセットをダウンロード
    
    Args:
        dataset_name: データセット名（例: "takamotoki/jra-horse-racing-dataset"）
        output_dir: 出力ディレクトリ
        
    Returns:
        成功した場合True
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=False
        )
        
        # ZIPファイルを解凍
        zip_files = list(output_dir.glob("*.zip"))
        for zip_file in zip_files:
            logger.info(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(output_dir)
            zip_file.unlink()  # ZIPファイルを削除
        
        logger.info(f"Dataset downloaded to: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def list_downloaded_files(data_dir: Path) -> None:
    """ダウンロードしたファイル一覧を表示"""
    if not data_dir.exists():
        logger.info("No data directory found")
        return
    
    logger.info("Downloaded files:")
    for f in sorted(data_dir.glob("**/*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  {f.relative_to(data_dir)} ({size_mb:.1f} MB)")


def main():
    """メイン処理"""
    if not check_kaggle_credentials():
        logger.info("\n--- Manual Download Alternative ---")
        logger.info("1. Go to: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset")
        logger.info("2. Click 'Download' button")
        logger.info(f"3. Extract to: {RAW_DATA_DIR}")
        sys.exit(1)
    
    dataset_name = DATA_CONFIG["kaggle_dataset"]
    success = download_dataset(dataset_name, RAW_DATA_DIR)
    
    if success:
        list_downloaded_files(RAW_DATA_DIR)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
