# 競馬AI予測システム (JRA)

JRA（日本中央競馬会）の馬単（2連単）予測を行うAIシステム。
期待値ベースの賭け戦略で回収率向上を目指す。

## プロジェクト構成

```
keiba-ai/
├── config/
│   └── settings.py          # 設定ファイル
├── data/
│   ├── raw/                  # 生データ（Kaggle CSV）
│   └── processed/            # 処理済みデータ
├── src/
│   ├── data_collection/      # データ取得
│   │   └── download_kaggle.py
│   ├── preprocessing/        # 特徴量エンジニアリング
│   ├── models/               # 予測モデル（Phase 2）
│   └── api/                  # 推論API（Phase 4）
├── notebooks/                # Jupyter探索用
├── requirements.txt
├── CLAUDE.md                 # Claude Code用指示書
└── README.md
```

## セットアップ

```bash
# 仮想環境作成（uv使用）
uv venv
source .venv/bin/activate

# 依存パッケージインストール
uv pip install -r requirements.txt
```

## データ取得

### Kaggle APIを使用する場合

```bash
# 1. Kaggle API認証設定
#    https://www.kaggle.com/settings からAPIトークンを取得
#    ~/.kaggle/kaggle.json に保存
chmod 600 ~/.kaggle/kaggle.json

# 2. ダウンロード実行
uv run python src/data_collection/download_kaggle.py
```

### 手動ダウンロードの場合

1. https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset にアクセス
2. 「Download」ボタンをクリック
3. ZIPを解凍して `data/raw/` に配置

## データソース

- **Kaggle JRA Dataset**: 1986〜2021年のJRAレースデータ
  - レース結果、オッズ、ラップタイム、コーナー通過順位
  - 本プロジェクトでは2019〜2021年を使用

## 開発フェーズ

- [x] Phase 1: データ収集 & 探索
- [ ] Phase 2: モデル構築
- [ ] Phase 3: バックテスト
- [ ] Phase 4: 推論API & UI

## 特徴量

| カテゴリ | 特徴量 |
|----------|--------|
| 基本情報 | 馬齢、性別、馬体重、斤量、枠番 |
| 騎手・調教師 | 勝率、連対率 |
| レース条件 | 距離、芝/ダート、馬場状態、競馬場 |
| 過去成績 | 直近5走の着順、勝率 |
| 脚質 | 逃げ/先行/差し/追込（コーナー通過順から算出） |
| 血統 | 父、母父の適性 |

※調教データは有料データ（JRA-VAN）導入時に追加予定

## 戦略

**期待値ベース**
```
期待値 = 予測的中確率 × オッズ
期待値 > 1.0 の買い目のみ購入
```

## 競艇プロジェクトとの違い

| 観点 | 競艇 | 競馬 |
|------|------|------|
| 出走数 | 6艇固定 | 8〜18頭（可変） |
| データ | 公式無料提供 | Kaggle or 有料（JRA-VAN） |
| 重要要素 | モーター、スタート | 血統、騎手、調教 |

## 拡張予定

- 券種拡張（三連単、三連複、ワイド）
- 地方競馬（NAR）対応
- JRA-VAN連携（リアルタイム予測）
