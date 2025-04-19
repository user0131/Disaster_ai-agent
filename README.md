# 避難所NPC対話システム

このプログラムは避難所運営における対話シミュレーションを行うものです。避難者が避難所に来訪した際の対応を行政職員（プレイヤー役）とNPCとの間で議論する形式で実行します。

## セットアップ

### 仮想環境(.venv)での実行方法

1. 仮想環境を作成・有効化します：

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化（Windows）
.venv\Scripts\activate

# 仮想環境の有効化（macOS/Linux）
source .venv/bin/activate
```

2. 必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

3. `.env`ファイルを編集して、OpenAI APIキーを設定します：

```
OPENAI_API_KEY=your_api_key_here
```

4. データファイルが正しく配置されていることを確認します：

- `hinanzyo_events_evacuee.csv` - 避難者情報のCSVファイル
- `npc_profiles.json` - NPCの性格設定などが記述されたJSONファイル

### グローバル環境での実行方法

1. 必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

2. `.env`ファイルを編集して、OpenAI APIキーを設定します

## 実行方法

以下のコマンドでプログラムを実行します：

```bash
python npc.py
```

## 使い方

1. プログラムを実行すると、避難者が順番に来訪します
2. 各避難者の情報が表示されたら、行政職員（剛史君）として発言を入力します
3. NPCが応答し、議論が行われます
4. 最終的な配置方針を入力すると、次の避難者に進みます

## 注意事項

- OpenAI APIの使用には料金が発生する場合があります
- プログラムの実行にはインターネット接続が必要です
