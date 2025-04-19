#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import time
import re
import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# --- 環境変数の読み込み ---
load_dotenv()
client = OpenAI()

# --- NPCプロファイルの読み込み（1人分） ---
with open("npc_profiles.json", encoding="utf-8") as f:
    npc_list = json.load(f)
    single_npc = npc_list[0]

# --- 備考から状況（全壊など）を抽出する関数 ---
def extract_situation(text):
    keywords = ["全壊", "半壊", "一部損壊", "全焼", "無被害", "被害なし"]
    for word in keywords:
        if word in text:
            return word
    return "不明"

# --- 避難者本人の自己紹介文を生成（より具体的な悩みを含める） ---
def evacuee_profile_to_statement(evacuee):
    # 状況に基づく具体的な悩みを設定
    situation = evacuee["状況"]
    memo = evacuee["構成"]
    specific_concerns = generate_specific_concerns(evacuee)
    
    # 自己紹介のテンプレート
    intro = f"{evacuee['名前']}と申します。{evacuee['構成']}で避難してきました。"
    
    # 状況別の具体的な表現
    situation_expression = ""
    if situation == "全壊":
        situation_expression = "家が全壊してしまい、何もかも失ってしまいました。"
    elif situation == "半壊":
        situation_expression = "家が半壊で、今後どうしたらいいのか不安です。"
    elif situation == "全焼":
        situation_expression = "火事で家が全焼してしまい、手元には何も残っていません。"
    elif situation == "一部損壊":
        situation_expression = "家は一部損壊したものの、修理が必要で戻れません。"
    else:
        situation_expression = f"状況は{situation}です。"
    
    # 一番の悩みを選択
    primary_concern = specific_concerns[0] if specific_concerns else "今後どうしたらいいか不安です。"
    
    # 自己紹介文の組み立て
    statement = f"{intro}{situation_expression} {primary_concern} できれば家族で安心して過ごせる方法を相談したいです。"
    
    return statement

# --- 避難者の具体的な悩みを生成 ---
def generate_specific_concerns(evacuee):
    situation = evacuee["状況"]
    memo = evacuee["構成"]
    specific_concerns = []
    
    # 状況別の具体的な悩み
    if situation == "全壊":
        specific_concerns.append("生活再建の見通しが立たず、不安で夜も眠れません。")
        if "子" in memo or "息子" in memo or "娘" in memo:
            specific_concerns.append("子どもが精神的に不安定になっており、心配です。")
        specific_concerns.append("仮設住宅の申請方法や入居時期について知りたいです。")
        specific_concerns.append("家の再建資金をどう工面すればいいのか分かりません。")
    elif situation == "半壊":
        specific_concerns.append("家の修理費用の目処が立たず、困っています。")
        specific_concerns.append("修理期間中の住まいをどうすればいいか悩んでいます。")
        specific_concerns.append("支援金の申請手続きが複雑で理解できません。")
    elif situation == "全焼":
        specific_concerns.append("思い出の品も含めてすべて失い、気持ちの整理がつきません。")
        specific_concerns.append("保険の申請手続きについて相談したいです。")
        specific_concerns.append("当面の生活必需品が全くない状態です。")
    elif situation == "一部損壊":
        specific_concerns.append("修理費用の補助があるのか知りたいです。")
        specific_concerns.append("修理が完了するまでどのくらいかかるのか不安です。")
    
    # 家族構成による具体的な悩み
    if "子" in memo or "幼児" in memo or "赤ちゃん" in memo or "息子" in memo or "娘" in memo:
        if "幼児" in memo or "赤ちゃん" in memo:
            specific_concerns.append("小さな子どもがいるので、プライバシーが保たれる場所が必要です。")
            specific_concerns.append("赤ちゃんのミルクや離乳食の準備に困っています。")
        else:
            specific_concerns.append("子どもの学校や勉強の問題が心配です。")
            specific_concerns.append("子どもが精神的に不安定で、落ち着けるスペースが欲しいです。")
    
    if "高齢" in memo or "老人" in memo or "父" in memo or "母" in memo:
        specific_concerns.append("高齢の家族がいるので、バリアフリーの環境が必要です。")
        specific_concerns.append("持病の薬が少なくなってきて心配です。")
        specific_concerns.append("避難所での生活が長引くと体調を崩しそうで不安です。")
    
    if "障害" in memo:
        specific_concerns.append("障害がある家族のケアをどうすればいいか困っています。")
        specific_concerns.append("バリアフリー環境が整った避難先を探しています。")
    
    if "ペット" in memo or "犬" in memo or "猫" in memo:
        specific_concerns.append("ペットと一緒に避難できる場所を探しています。")
        specific_concerns.append("ペットのストレスが心配で、どうケアすればいいかわかりません。")
    
    # 心理的な悩み（共通）
    psychological_concerns = [
        "不安で夜もよく眠れず、精神的に疲れています。",
        "先行きが見えず、どうしたらいいのか途方に暮れています。",
        "家族全員の心のケアが必要だと感じています。"
    ]
    
    # 具体的な状況から選択するランダム要素（同じ名前なら同じ悩みを持つように）
    name = evacuee["名前"]
    random.seed(name)  # 名前を基にシード値を設定
    
    # 状況別と家族構成別の悩みからランダムに選択
    random.shuffle(specific_concerns)
    if specific_concerns:
        # 重複を除去
        unique_concerns = []
        for concern in specific_concerns:
            if concern not in unique_concerns:
                unique_concerns.append(concern)
        
        selected_concerns = unique_concerns[:3]  # 最大3つの具体的悩み
        # 心理的な悩みを1つ追加
        selected_concerns.append(random.choice(psychological_concerns))
        return selected_concerns
    else:
        return [random.choice(psychological_concerns)]

# --- CSVから避難者データ読み込み ---
def load_evacuees():
    df = pd.read_csv("hinanzyo_events_evacuee.csv", encoding="utf-8", dtype=str)
    evacuees_df = df[df["タイプ"] == "避難者"].dropna(subset=["名前", "地区", "備考"])
    
    evacuees = []
    for _, row in evacuees_df.iterrows():
        備考 = str(row["備考"]).strip()
        # 性格タイプ列が存在する場合は読み込む
        性格タイプ = str(row["性格タイプ"]) if "性格タイプ" in df.columns and pd.notna(row.get("性格タイプ", "")) else ""
        evacuees.append({
            "名前": str(row["名前"]).strip(),
            "構成": 備考,
            "状況": extract_situation(備考),
            "地区": str(row["地区"]).strip() if pd.notna(row["地区"]) else "",
            "番号": str(row["番号"]).strip() if pd.notna(row["番号"]) else "",
            "世帯番号": str(row["世帯番号"]).strip() if pd.notna(row["世帯番号"]) else "",
            "性格タイプ": 性格タイプ
        })
    
    # 重複を除去（同じ名前の避難者は最新のデータを使用）
    unique_evacuees = {}
    for evacuee in evacuees:
        unique_evacuees[evacuee["名前"]] = evacuee
    
    return list(unique_evacuees.values())

# --- 避難者データの読み込み ---
evacuees = load_evacuees()

# --- ユーザーの発言をIBIS要素に分類（バックグラウンドで実行） ---
def classify_statement(statement):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                IBIS構造における以下の3つの要素のいずれかに分類してください：
                1. 課題(Issue): 解決すべき問題や疑問。通常「?」で終わる疑問文の形式。
                2. 意見(Position): 課題に対する解決策や主張。
                3. 論拠(Argument): 意見を支持する(Pro)または反対する(Con)理由や根拠。
                
                分類結果は「課題」「意見」「論拠(Pro)」「論拠(Con)」のいずれかのみを返してください。
                """},
                {"role": "user", "content": f"以下の発言を分類してください: '{statement}'"}
            ],
            temperature=0.3
        )
        classification = response.choices[0].message.content.strip()
        if "課題" in classification:
            return "課題"
        elif "意見" in classification:
            return "意見"
        elif "論拠" in classification:
            if "Pro" in classification or "支持" in classification:
                return "論拠(Pro)"
            else:
                return "論拠(Con)"
        else:
            return "不明"
    except Exception as e:
        print(f"分類エラー: {e}")
        return "不明"

# --- 避難者の状況に基づく懸念点や問題点を特定する ---
def identify_evacuee_concerns(evacuee):
    situation = evacuee["状況"]
    memo = evacuee["構成"]
    
    concerns = []
    
    # 状況に基づく懸念点
    if situation == "全壊":
        concerns.append("長期的な避難生活への備え")
        concerns.append("生活再建支援の必要性")
    elif situation == "半壊":
        concerns.append("一時的な避難と自宅修繕の両立")
        concerns.append("必要な支援物資の確保")
    elif situation == "一部損壊":
        concerns.append("自宅修繕の見通し")
    elif situation == "全焼":
        concerns.append("当面の生活必需品の確保")
        concerns.append("心理的なケアの必要性")
    
    # 家族構成に基づく懸念点
    if "子" in memo or "幼児" in memo or "赤ちゃん" in memo:
        concerns.append("子どもの教育・遊び場の確保")
        concerns.append("子育て環境の整備")
    if "高齢" in memo or "老人" in memo:
        concerns.append("高齢者の健康管理")
        concerns.append("バリアフリー環境の必要性")
    if "障害" in memo or "介護" in memo:
        concerns.append("介護・支援体制の構築")
    if "妊婦" in memo or "妊娠" in memo:
        concerns.append("母子の健康管理体制")
    
    # 地域に基づく懸念点
    if evacuee["地区"]:
        concerns.append(f"{evacuee['地区']}地区の復興見通し")
    
    # 基本的な懸念点
    concerns.append("プライバシーの確保")
    concerns.append("コミュニティ形成と情報共有")
    concerns.append("健康・衛生管理")
    
    return concerns[:3]  # 上位3つの懸念点を返す

# --- 避難者の背景情報を深堀りして生成する ---
def generate_evacuee_background(evacuee):
    memo = evacuee["構成"]
    situation = evacuee["状況"]
    district = evacuee["地区"] if evacuee["地区"] else "不明"
    
    # 家族構成の解析
    family_members = []
    if "本人" in memo:
        family_members.append("本人")
    if "妻" in memo:
        family_members.append("妻")
    if "夫" in memo or "だんな" in memo:
        family_members.append("夫")
    if "子" in memo:
        # 子供の情報を抽出
        child_pattern = r"子\（(\d+)歳\）"
        child_matches = re.findall(child_pattern, memo)
        if child_matches:
            for age in child_matches:
                family_members.append(f"{age}歳の子ども")
        else:
            family_members.append("子ども")
    if "父" in memo:
        # 父親の年齢を抽出
        father_pattern = r"父\（(\d+)歳\）"
        father_match = re.search(father_pattern, memo)
        if father_match:
            family_members.append(f"{father_match.group(1)}歳の父親")
        else:
            family_members.append("父親")
    if "母" in memo:
        # 母親の年齢を抽出
        mother_pattern = r"母\（(\d+)歳\）"
        mother_match = re.search(mother_pattern, memo)
        if mother_match:
            family_members.append(f"{mother_match.group(1)}歳の母親")
        else:
            family_members.append("母親")
    
    # 特殊な状況の抽出
    special_conditions = []
    if "認知症" in memo:
        special_conditions.append("認知症を抱えている家族がいる")
    if "介護" in memo:
        special_conditions.append("介護が必要な家族がいる")
    if "障害" in memo:
        special_conditions.append("障害を持つ家族がいる")
    if "妊娠" in memo:
        pregnancy_pattern = r"妊娠(\d+)ヶ月"
        pregnancy_match = re.search(pregnancy_pattern, memo)
        if pregnancy_match:
            special_conditions.append(f"妊娠{pregnancy_match.group(1)}ヶ月の家族がいる")
        else:
            special_conditions.append("妊娠中の家族がいる")
    if "アレルギー" in memo:
        special_conditions.append("食物アレルギーがある家族がいる")
    if "糖尿病" in memo or "心臓病" in memo or "高血圧" in memo:
        special_conditions.append("持病を抱えている家族がいる")
    
    # ペットの情報
    pets = []
    pet_patterns = [r"(犬|猫|ウサギ|ハムスター|インコ|カナリア)を連れ", r"(秋田犬|柴犬|マルチーズ|ダックスフント)を連れ"]
    for pattern in pet_patterns:
        pet_match = re.search(pattern, memo)
        if pet_match:
            pets.append(pet_match.group(1))
    
    # 車での避難
    has_car = "車" in memo
    
    # 国籍・言語
    nationality = "日本"
    if "ブラジル人" in memo:
        nationality = "ブラジル"
    elif "インド人" in memo:
        nationality = "インド"
    elif "アメリカ人" in memo:
        nationality = "アメリカ"
    
    # 特殊な経緯
    special_circumstances = []
    if "行方不明" in memo:
        special_circumstances.append("家族が行方不明")
    if "死亡" in memo:
        special_circumstances.append("家族を亡くした")
    if "入院" in memo:
        special_circumstances.append("家族が入院中")
    if "旅行" in memo:
        special_circumstances.append("旅行中に被災")
    
    # 心理状態の推定
    psychological_state = []
    if situation == "全壊":
        psychological_state.extend([
            "家を失った喪失感と悲しみ",
            "将来への不安と恐怖",
            "生活再建への強いストレス"
        ])
    elif situation == "半壊":
        psychological_state.extend([
            "住居の部分的喪失による不安",
            "復旧への見通しが立たない焦り"
        ])
    elif situation == "全焼":
        psychological_state.extend([
            "火災によるトラウマの可能性",
            "思い出の品々を失った喪失感"
        ])
    
    if "行方不明" in memo:
        psychological_state.append("家族の安否を案じる強い不安")
    if "死亡" in memo:
        psychological_state.append("家族の死による強い悲嘆と喪失感")
    
    if "子" in memo:
        psychological_state.append("子どもの将来や教育に対する不安")
    
    background = {
        "family_members": family_members,
        "special_conditions": special_conditions,
        "pets": pets,
        "has_car": has_car,
        "nationality": nationality,
        "special_circumstances": special_circumstances,
        "district": district,
        "situation": situation,
        "psychological_state": psychological_state
    }
    
    return background

# --- 避難者の性格や特徴を生成する ---
def generate_evacuee_personality(evacuee):
    name = evacuee["名前"]
    memo = evacuee["構成"]
    situation = evacuee["状況"]
    
    # CSVに設定された性格タイプがあればそれを使用
    predefined_type = evacuee.get("性格タイプ", "")
    if predefined_type and predefined_type in ["外交的", "内向的", "感情的", "論理的", "楽観的", "悲観的", 
                                             "社交的", "慎重", "几帳面", "大雑把", "温厚", "短気"]:
        personality_type = predefined_type
    else:
        # ランダムシード値を名前から設定（同じ名前なら同じ性格になる）
        first_char = name[0] if name else "あ"
        random.seed(first_char + name[-1] if len(name) > 1 else "")
        
        # 性格タイプ（主要な性格分類）
        personality_types = [
            "外交的", "内向的", "感情的", "論理的", "楽観的", "悲観的", 
            "社交的", "慎重", "几帳面", "大雑把", "温厚", "短気"
        ]
        
        # 個人の名前に基づいて確定的に性格タイプを決定
        personality_type = personality_types[sum(ord(c) for c in name) % len(personality_types)]
    
    # 基本的な性格特性（複数の特性を持つ）
    personality_traits = []
    personality_traits.append(personality_type)  # メインの性格タイプを追加
    
    # 避難者の名前に基づいた追加の性格特性
    basic_traits = [
        "穏やか", "心配性", "頑固", "柔軟", "積極的", "消極的", 
        "冷静", "感情豊か", "用心深い", "大胆", "協調的", "独立的",
        "親切", "厳格", "粘り強い", "気まぐれ", "思いやりがある", "自己主張が強い"
    ]
    
    # 2-3個の追加特性をランダムに選択（名前による一貫性を保持）
    additional_traits_count = random.randint(2, 3)
    random.shuffle(basic_traits)
    personality_traits.extend(basic_traits[:additional_traits_count])
    
    # 状況に基づく性格
    if situation == "全壊" or situation == "全焼":
        stress_traits = ["不安を抱えている", "将来に悲観的", "助けを求めている", "強い精神力で耐えている"]
        personality_traits.append(random.choice(stress_traits))
    
    # メモの内容から推測される性格
    if "子" in memo:
        parent_traits = ["子煩悩", "責任感が強い", "家族思い"]
        personality_traits.append(random.choice(parent_traits))
    
    if "高齢" in memo or "老人" in memo:
        if random.random() < 0.5:
            personality_traits.append("伝統を重んじる")
        else:
            personality_traits.append("経験から学んだ知恵がある")
    
    if "ひきこもり" in memo:
        personality_traits.append("社会的な交流に不安を感じている")
    
    # 会話スタイル設定
    verbosity = determine_verbosity(personality_type)
    emotional_expressiveness = determine_emotional_expressiveness(personality_type)
    detail_level = determine_detail_level(personality_type)
    
    # コミュニケーションスタイル
    communication_styles = [
        "丁寧な話し方", "感情表現が豊か", "簡潔に話す", "詳細に説明する", 
        "ユーモアを交えて話す", "真面目な口調", "遠回しな表現を使う", "率直に話す"
    ]
    communication_style = random.choice(communication_styles)
    
    return {
        "traits": personality_traits,
        "communication_style": communication_style,
        "type": personality_type,
        "verbosity": verbosity,
        "emotional_expressiveness": emotional_expressiveness,
        "detail_level": detail_level
    }

# --- 性格タイプから会話の冗長性（長さ）を決定 ---
def determine_verbosity(personality_type):
    # 冗長性スコア (1-10)
    verbosity_mapping = {
        "外交的": 8,
        "内向的": 3,
        "感情的": 7,
        "論理的": 5,
        "楽観的": 7,
        "悲観的": 6,
        "社交的": 9,
        "慎重": 4,
        "几帳面": 6,
        "大雑把": 5,
        "温厚": 5,
        "短気": 3
    }
    # デフォルト値は中間の5
    return verbosity_mapping.get(personality_type, 5)

# --- 性格タイプから感情表現の強さを決定 ---
def determine_emotional_expressiveness(personality_type):
    # 感情表現スコア (1-10)
    expressiveness_mapping = {
        "外交的": 7,
        "内向的": 3,
        "感情的": 9,
        "論理的": 2,
        "楽観的": 6,
        "悲観的": 5,
        "社交的": 8,
        "慎重": 4,
        "几帳面": 4,
        "大雑把": 6,
        "温厚": 5,
        "短気": 8
    }
    # デフォルト値は中間の5
    return expressiveness_mapping.get(personality_type, 5)

# --- 性格タイプから詳細さのレベルを決定 ---
def determine_detail_level(personality_type):
    # 詳細さスコア (1-10)
    detail_mapping = {
        "外交的": 5,
        "内向的": 4,
        "感情的": 3,
        "論理的": 8,
        "楽観的": 4,
        "悲観的": 6,
        "社交的": 5,
        "慎重": 7,
        "几帳面": 9,
        "大雑把": 2,
        "温厚": 5,
        "短気": 3
    }
    # デフォルト値は中間の5
    return detail_mapping.get(personality_type, 5)

# --- 避難者のプロファイルを作成する ---
def create_evacuee_profile(evacuee):
    name = evacuee["名前"]
    background = generate_evacuee_background(evacuee)
    personality = generate_evacuee_personality(evacuee)
    concerns = identify_evacuee_concerns(evacuee)
    
    profile = {
        "name": name,
        "background": background,
        "personality": personality,
        "concerns": concerns,
        "raw_data": evacuee
    }
    
    return profile

# --- 避難者NPC（住民）の発言を生成する ---
def generate_evacuee_response(evacuee_profile, conversation_log, trigger_type="normal"):
    evacuee = evacuee_profile["raw_data"]
    name = evacuee_profile["name"]
    background = evacuee_profile["background"]
    personality = evacuee_profile["personality"]
    concerns = evacuee_profile["concerns"]
    
    # 性格に基づく会話スタイルを取得
    verbosity = personality.get("verbosity", 5)  # 冗長性/会話の長さ (1-10)
    emotional_expressiveness = personality.get("emotional_expressiveness", 5)  # 感情表現の強さ (1-10)
    detail_level = personality.get("detail_level", 5)  # 詳細さのレベル (1-10)
    personality_type = personality.get("type", "一般的")
    
    # 最後の発言（通常は剛史君の発言）を取得
    last_message = ""
    for msg in reversed(conversation_log):
        if msg["speaker"] == "剛史君":
            last_message = msg["content"]
            break
    
    # 曖昧な応答パターンを検出
    vague_responses = [
        "大丈夫です", "わかりました", "了解しました", "後でやります", "後ほど", 
        "また来ます", "考えておきます", "検討します", "承知しました", "分かりました",
        "お願いします", "任せます", "大丈夫", "了解", "承知"
    ]
    
    # ユーザーが曖昧な返答をしているかチェック
    is_vague_response = any(phrase in last_message for phrase in vague_responses)
    
    # 会話終了に向かう応答が必要かどうか確認
    if is_vague_response:
        # 曖昧な応答に対して会話を終了させる方向の返答
        closing_responses = [
            f"ありがとうございます。また何かあればご相談します。",
            f"わかりました。ありがとうございます。",
            f"そうですか。それでは失礼します。ありがとうございました。",
            f"ご対応ありがとうございます。少し安心しました。",
            f"了解しました。ではまた改めて相談させてください。"
        ]
        # 避難者の名前を基にした固定のシード値を使用して一貫性のある応答を選択
        random.seed(name)
        return random.choice(closing_responses)
     # 状況別の具体的な悩み
    # 会話の長さを設定（文字数を制限するため全体的に下げる）
    if verbosity >= 8:  # 非常に話好き
        max_tokens_value = 100  # 約150-180字程度
        min_sentences = 2
    elif verbosity >= 6:  # 話好き
        max_tokens_value = 80   # 約120-150字程度
        min_sentences = 2
    elif verbosity >= 4:  # 普通
        max_tokens_value = 60   # 約90-120字程度
        min_sentences = 1
    else:  # 寡黙
        max_tokens_value = 40   # 約60-80字程度
        min_sentences = 1
    
    # 背景情報のテキスト化
    family_text = "、".join(background["family_members"]) if background["family_members"] else "不明"
    conditions_text = "、".join(background["special_conditions"]) if background["special_conditions"] else "なし"
    pets_text = "、".join(background["pets"]) if background["pets"] else "なし"
    special_text = "、".join(background["special_circumstances"]) if background["special_circumstances"] else "なし"
    personality_text = "、".join(personality["traits"]) if personality["traits"] else "普通"
    
    # 心理状態の情報
    psychological_text = "、".join(background.get("psychological_state", [])) if background.get("psychological_state") else "一般的な不安"
    
    # 状況の深刻さに基づく感情表現の調整
    emotion_level = 0
    if background["situation"] == "全壊" or background["situation"] == "全焼":
        emotion_level = 3  # 非常に高い
    elif background["situation"] == "半壊":
        emotion_level = 2  # 高い
    elif background["situation"] == "一部損壊":
        emotion_level = 1  # 中程度
    
    if background.get("special_circumstances") and any(("死亡" in circ or "行方不明" in circ) for circ in background.get("special_circumstances", [])):
        emotion_level += 1  # さらに高める
    
    # 感情表現の強さと性格タイプを考慮した感情表現の指示
    emotion_level = min(emotion_level + (emotional_expressiveness - 5) // 2, 4)  # 性格による感情表現の調整（-2〜+2）
    
    emotion_instruction = ""
    if emotion_level >= 3:
        # 非常に感情表現が強い場合
        if personality_type in ["感情的", "社交的", "外交的"]:
            emotion_instruction = "声が震える、言葉に詰まるなど、感情表現を簡潔に示してください。"
        elif personality_type in ["内向的", "慎重", "論理的"]:
            emotion_instruction = "内面の動揺を抑えきれず、時折言葉に詰まったり声が小さくなる様子を簡潔に表現してください。"
        else:
            emotion_instruction = "感情表現を簡潔に示し、精神的苦痛や喪失感を端的に表現してください。"
    elif emotion_level == 2:
        if personality_type in ["楽観的", "温厚"]:
            emotion_instruction = "不安を感じつつも前向きな姿勢を簡潔に示してください。"
        else:
            emotion_instruction = "不安や心配を声のトーンに表しつつ、短く簡潔に話してください。"
    elif emotion_level == 1:
        emotion_instruction = "軽い不安や懸念を表現しつつも、簡潔に話してください。"
    else:
        emotion_instruction = "冷静に状況に対応しようとする様子を簡潔に示してください。"
    
    # 詳細さのレベルに応じた指示
    if detail_level >= 8:  # 非常に詳細
        detail_instruction = "状況や感情の要点だけを伝えてください。細かい説明は避けてください。"
    elif detail_level >= 5:  # 普通の詳細さ
        detail_instruction = "状況や感情を簡潔にまとめ、要点のみを伝えてください。"
    else:  # あまり詳細でない
        detail_instruction = "最小限の言葉で状況を伝えてください。"
    
    # 避難者の状況に応じたプロンプト作成
    evacuee_prompt = f"""あなたは避難所に避難してきた「{name}」です。以下の特徴と背景情報に基づいて会話してください。

【基本情報】
・名前: {name}
・地区: {background["district"]}
・被害状況: {background["situation"]}
・家族構成: {family_text}
・特別な状況: {conditions_text}
・ペット: {pets_text}
・特殊な事情: {special_text}
・性格: {personality_text}
・性格タイプ: {personality_type}
・会話スタイル: {personality["communication_style"]}

【心理状態】
{psychological_text}

【主な懸念点】
{', '.join(concerns)}

【応答指示】
【重要】状況が混雑した避難所であることを考慮し、必ず簡潔に話してください。200字以内、できれば100字程度に収めてください。
あなたは{personality_type}タイプの性格で、{min_sentences}文程度の文で話してください。{emotion_instruction}
{detail_instruction}
被災による不安や困惑などの感情を必要最小限の言葉で表現してください。多くの避難者がいる状況を想定し、重要な情報だけを端的に伝えてください。
"""

    if trigger_type == "question":
        evacuee_prompt += "質問には直接的かつ簡潔に答え、必要な情報だけを伝えてください。"
    elif trigger_type == "solution":
        evacuee_prompt += "提案に対する反応は簡潔に、要点のみ伝えてください。"
    elif trigger_type == "first_contact":
        evacuee_prompt += "初めての対応では緊急ニーズのみを簡潔に伝えてください。"
    else:
        evacuee_prompt += "会話の流れに沿って簡潔に反応し、長い説明は避けてください。"
    
    # 会話履歴をテキスト化
    log_text = "\n".join([f"{entry['speaker']}：{entry['content']}" for entry in conversation_log[-5:]])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": evacuee_prompt},
                {"role": "user", "content": f"これまでの会話：\n{log_text}\n\nあなた（{name}）の発言："}
            ],
            temperature=0.7,  # わずかに下げて安定させる
            max_tokens=max_tokens_value
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(rf"^{name}[:：]\s*", "", content)
        return content.strip()
    except Exception as e:
        return f"（すみません...）"

# --- 避難者が発言すべきかを判断する関数 ---
def determine_if_evacuee_should_speak(conversation_log, threshold=0.4):
    # 最新の3つの発言を分析
    recent_logs = conversation_log[-3:] if len(conversation_log) >= 3 else conversation_log
    
    # 避難者が直近で発言している場合は発言しない
    for log in recent_logs:
        if log["speaker"] != "剛史君" and log["speaker"] != single_npc["name"] and log["speaker"] != "system":
            return False
    
    # 最新の発言が質問や呼びかけの場合、避難者が応答する可能性が高い
    if recent_logs:
        last_msg = recent_logs[-1]["content"]
        # 質問の特徴を検出
        if "?" in last_msg or "？" in last_msg or "いかがですか" in last_msg or "どうですか" in last_msg or "思いますか" in last_msg:
            return True
        # 避難者への直接的な呼びかけ
        if "さん、" in last_msg or "さんは" in last_msg or "どう思われますか" in last_msg:
            return True
    
    # 会話の停滞感を検出
    if len(conversation_log) >= 4:
        # 最近のやり取りが職員間のみの場合、避難者から発言する確率を上げる
        staff_only = True
        for log in conversation_log[-4:]:
            if log["speaker"] != "剛史君" and log["speaker"] != single_npc["name"] and log["speaker"] != "system":
                staff_only = False
                break
        if staff_only:
            return random.random() < 0.6  # 60%の確率で避難者が割り込む
    
    # 基本確率で判断
    return random.random() < threshold

# --- 避難者の満足度と心理状態を追跡するクラスを追加 ---
class EvacueeMentalState:
    def __init__(self, name, initial_concerns, situation="不明", family_composition=""):
        self.name = name
        self.concerns = {concern: {"resolved": False, "satisfaction": 0} for concern in initial_concerns}
        # overall_satisfactionを使わないように修正
        self.overall_satisfaction = 0  # この行を追加
        
        # 被害状況に基づく初期値設定
        if situation == "全壊" or situation == "全焼":
            self.comfort_level = 2.0  # 非常に低い快適度
            self.trust_level = 4.0    # やや低い信頼度
        elif situation == "半壊":
            self.comfort_level = 3.0  # 低い快適度
            self.trust_level = 4.5    # やや低い信頼度
        elif situation == "一部損壊":
            self.comfort_level = 4.0  # 中程度の快適度
            self.trust_level = 5.0    # 中程度の信頼度
        else:
            self.comfort_level = 5.0  # デフォルト値
            self.trust_level = 5.0    # デフォルト値
        
        # 家族構成に基づく追加の調整
        if "子" in family_composition or "幼児" in family_composition or "赤ちゃん" in family_composition:
            self.comfort_level -= 0.5  # 子どもがいる場合は快適度が低下
        if "高齢" in family_composition or "老人" in family_composition:
            self.comfort_level -= 0.5  # 高齢者がいる場合は快適度が低下
        if "障害" in family_composition:
            self.comfort_level -= 1.0  # 障害者がいる場合は快適度がさらに低下
        if "行方不明" in family_composition or "死亡" in family_composition:
            self.comfort_level -= 1.5  # 家族を失った場合は大幅に快適度が低下
            self.trust_level -= 1.0    # 信頼度も低下
            
        # 快適度と信頼度の範囲を制限（0〜10）
        self.comfort_level = max(0, min(10, self.comfort_level))
        self.trust_level = max(0, min(10, self.trust_level))
        
        self.conversation_history = []
    
    def add_conversation(self, speaker, content):
        self.conversation_history.append({"speaker": speaker, "content": content})
    
    def update_concern_status(self, concern, resolved=False, satisfaction_delta=0):
        if concern in self.concerns:
            self.concerns[concern]["resolved"] = resolved
            self.concerns[concern]["satisfaction"] += satisfaction_delta
            # 満足度は0-10の範囲に制限
            self.concerns[concern]["satisfaction"] = min(10, max(0, self.concerns[concern]["satisfaction"]))
    
    def calculate_overall_satisfaction(self):
        if not self.concerns:
            return 0
        total = sum(concern["satisfaction"] for concern in self.concerns.values())
        # overall_satisfactionを更新しつつ、値も返す
        self.overall_satisfaction = total / len(self.concerns)
        return self.overall_satisfaction
    
    def update_comfort_level(self, delta):
        self.comfort_level += delta
        # 快適度は0-10の範囲に制限
        self.comfort_level = min(10, max(0, self.comfort_level))
    
    def update_trust_level(self, delta):
        self.trust_level += delta
        # 信頼度は0-10の範囲に制限
        self.trust_level = min(10, max(0, self.trust_level))
    
    def get_status_summary(self):
        resolved_concerns = sum(1 for concern in self.concerns.values() if concern["resolved"])
        # overall_satisfactionを直接使用
        return {
            "name": self.name,
            "resolved_concerns": resolved_concerns,
            "total_concerns": len(self.concerns),
            "overall_satisfaction": self.overall_satisfaction,
            "comfort_level": self.comfort_level,
            "trust_level": self.trust_level
        }

# --- 避難者の発言から心理状態を分析する関数 ---
def analyze_evacuee_response(evacuee_mental_state, response, conversation_log):
    try:
        # 対応する避難者の懸念事項
        concern_keywords = {
            "長期的な避難生活": ["長期", "避難生活", "いつまで", "今後"],
            "生活再建": ["再建", "支援金", "補償", "今後の生活"],
            "子どもの教育": ["子供", "子ども", "教育", "学校", "勉強"],
            "健康管理": ["健康", "医療", "病院", "薬", "持病"],
            "プライバシー": ["プライバシー", "個人空間", "落ち着ける"],
            "心理的なケア": ["不安", "恐怖", "ストレス", "眠れない", "落ち込む"],
            "必要な支援物資": ["物資", "食料", "水", "衣類", "毛布"],
            "情報不足": ["情報", "わからない", "知りたい", "連絡"],
            "バリアフリー": ["階段", "車椅子", "移動", "バリアフリー"],
            "ペット": ["ペット", "犬", "猫"]
        }
        
        # 感情や満足度の変化を示す単語
        positive_words = ["ありがとう", "助かる", "嬉しい", "安心", "良かった", "感謝"]
        negative_words = ["不安", "心配", "困る", "辛い", "疲れた", "悲しい"]
        
        # 応答の感情分析
        positive_count = sum(1 for word in positive_words if word in response)
        negative_count = sum(1 for word in negative_words if word in response)
        
        # 信頼度・快適度の更新
        sentiment_score = positive_count - negative_count
        evacuee_mental_state.update_comfort_level(sentiment_score * 0.5)
        evacuee_mental_state.update_trust_level(sentiment_score * 0.3)
        
        # 最近の会話から関連する懸念事項を特定
        recent_messages = conversation_log[-3:]
        recent_conversation = " ".join([msg["content"] for msg in recent_messages])
        
        # 具体的な悩みと一般的な懸念事項の両方に対応
        for concern in evacuee_mental_state.concerns.keys():
            # キーワードリストを作成（キーワードが見つからない場合は単語を分割して使用）
            if concern in concern_keywords:
                keywords = concern_keywords[concern]
            else:
                # 具体的な悩みの文章から単語を抽出してキーワードとして使用
                keywords = [word for word in concern.split() if len(word) > 1]
            
            # 会話に懸念事項のキーワードが含まれているか確認
            keyword_match = any(keyword in recent_conversation for keyword in keywords)
            
            # 懸念事項が言及され、ポジティブな反応があれば満足度を上げる
            if keyword_match:
                if positive_count > negative_count:
                    evacuee_mental_state.update_concern_status(
                        concern, 
                        resolved=positive_count > 1, 
                        satisfaction_delta=sentiment_score
                    )
                elif negative_count > 0:
                    # ネガティブな反応があれば満足度を下げる
                    evacuee_mental_state.update_concern_status(
                        concern, 
                        satisfaction_delta=-0.5
                    )
        
        # 全体的な満足度を再計算して更新
        evacuee_mental_state.calculate_overall_satisfaction()
        
        return evacuee_mental_state.get_status_summary()
    
    except Exception as e:
        print(f"心理状態分析エラー: {e}")
        return None

# --- NPCが避難者の心理状態を考慮して応答を生成 ---
def generate_npc_response(npc, evacuee, conversation_log, last_statement_type, evacuee_mental_state=None):
    evacuee_info = f"名前：{evacuee['名前']}\n構成：{evacuee['構成']}\n状況：{evacuee['状況']}\n地区：{evacuee['地区']}"
    
    # 避難者の懸念点を特定
    concerns = identify_evacuee_concerns(evacuee)
    concerns_text = "\n".join([f"- {concern}" for concern in concerns])
    
    # 心理状態情報を追加
    mental_state_info = ""
    if evacuee_mental_state:
        status = evacuee_mental_state.get_status_summary()
        solved_concerns = []
        unsolved_concerns = []
        
        for concern, status_data in evacuee_mental_state.concerns.items():
            if status_data["resolved"]:
                solved_concerns.append(concern)
            else:
                unsolved_concerns.append(concern)
        
        # overall_satisfactionをステータスから直接取得する
        satisfaction = status.get('overall_satisfaction', 0.0)
        
        mental_state_info = f"""
【心理状態】
・全体的な満足度: {satisfaction:.1f}/10
・快適度: {status['comfort_level']:.1f}/10
・信頼度: {status['trust_level']:.1f}/10
・解決済みの懸念: {', '.join(solved_concerns) if solved_concerns else 'なし'}
・未解決の懸念: {', '.join(unsolved_concerns) if unsolved_concerns else 'なし'}
"""
    
    # IBISに基づいた応答指示
    ibis_instruction = ""
    if last_statement_type == "課題":
        ibis_instruction = "避難者の課題に対して、剛史君が解決策を提案できるようなヒントを提供してください。"
    elif last_statement_type == "意見":
        ibis_instruction = "避難者の意見に対して、剛史君が適切に対応できるようなヒントを提供してください。"
    elif last_statement_type.startswith("論拠"):
        ibis_instruction = "避難者の論拠に対して、剛史君が深く理解できるようなヒントを提供してください。"
    else:
        ibis_instruction = "避難者の発言に対して、剛史君が適切に対応できるようなヒントを提供してください。"
    
    # 避難者が次に発言すべきかの判断（常にfalseにする - 会話の流れを変更するため）
    should_evacuee_speak = False
    
    prompt = f"""あなたは避難所運営のファシリテーター役の「{npc['name']}」です。
役割：避難者と行政職員（剛史君）の間の会話を促進するファシリテーター、性格：{npc.get('personality', '穏やか')}、話し方：{npc.get('style', '丁寧')}。

あなたの役割は、避難者の発言を受けて剛史君に対応を促すことです。直接的な解決策を提案するのではなく、
剛史君が適切な対応ができるよう、ヒントを提供してください。

【避難者情報】
{evacuee_info}

【避難者の主な懸念点】
{concerns_text}

{mental_state_info}

【応答指示】
{ibis_instruction}

【重要】あなたの発言は「（ヒント：〜）」という形式で行ってください。
例：「（ヒント：避難者は子どもの教育について心配しています。具体的な支援策を伝えると安心するかもしれません）」
必ず極めて簡潔に話してください。長い説明は避け、1文程度の簡潔なヒントを提供してください。
ヒントの内容は、避難者の懸念や感情に基づいた、剛史君が対応するための示唆にしてください。
"""
    
    log_text = "\n".join([f"{entry['speaker']}：{entry['content']}" for entry in conversation_log])
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": log_text}
            ],
            temperature=0.7,
            max_tokens=80  # トークン数を減らして短い応答を促す
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(rf"^{npc['name']}[:：]\s*", "", content)
        
        # ヒント形式になっていなければ修正
        if not content.startswith("（ヒント：") and not content.startswith("(ヒント:"):
            content = f"（ヒント：{content}）"
        elif not content.endswith("）") and not content.endswith(")"):
            content = f"{content}）"
            
        return content.strip(), should_evacuee_speak
    except Exception as e:
        return f"（ヒント：避難者の話を聞いてみましょう）", False

def evaluate_evacuee_satisfaction(conversation_log, evacuee_mental_state=None):
    # evacuee_mental_stateがNoneの場合は早期リターン
    if evacuee_mental_state is None:
        return "避難者の心理状態データが利用できないため、満足度を評価できません。"
        
    # 会話履歴と心理状態から避難者の満足度を評価するプロンプトを作成
    recent_log = conversation_log[-min(10, len(conversation_log)):]
    log_text = "\n".join([f"{entry['speaker']}：{entry['content']}" for entry in recent_log])
    
    status = evacuee_mental_state.get_status_summary()
    
    # 未解決の懸念事項を収集
    unsolved_concerns = []
    for concern, data in evacuee_mental_state.concerns.items():
        if not data.get("resolved", False):
            unsolved_concerns.append(concern)
    
    # 関連キーワードに基づいて懸念事項をグループ化
    grouped_concerns = {}
    for concern in evacuee_mental_state.concerns.keys():
        if "生活再建" in concern or "支援金" in concern or "再建" in concern:
            category = "生活再建・支援"
        elif "子ども" in concern or "子供" in concern or "教育" in concern:
            category = "子育て・教育"
        elif "心理" in concern or "不安" in concern or "ストレス" in concern:
            category = "心理的ケア"
        elif "医療" in concern or "健康" in concern:
            category = "健康・医療"
        else:
            category = "その他の懸念"
            
        if category not in grouped_concerns:
            grouped_concerns[category] = []
        grouped_concerns[category].append(concern)
    
    # グループ化された懸念事項をテキストに変換
    concerns_text = ""
    for category, concerns_list in grouped_concerns.items():
        concerns_text += f"\n● {category}:\n"
        for concern in concerns_list:
            concerns_text += f"  - {concern}\n"
    
    prompt = f"""
あなたは避難所職員です。以下の会話から、避難者の悩みに適切に対応できたかどうかを分析してください。

【会話履歴】
{log_text}

【避難者の懸念事項】{concerns_text}

【指示】
各懸念事項について、会話の中で「適切に対応できた」か「対応できなかった」かを二択で評価してください。
評価は以下の簡易なフォーマットで行ってください：

【評価結果】
1. [対応○/対応×] 懸念事項の内容
2. [対応○/対応×] 懸念事項の内容
...

「対応○」は適切に対応できた場合、「対応×」は対応できなかった場合に使用してください。
「適切に対応できた」とは、その懸念に対して具体的な情報提供や解決策の提案があり、避難者が満足している場合です。
「対応できなかった」とは、その懸念について触れなかったか、具体的な対応がなかった場合です。

最後に、全体的な対応についての簡潔なコメントを2〜3文で付け加えてください。
評価コメントは完結した文章にしてください。
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=350  # トークン数を増やして全文表示できるようにする
        )
        analysis = response.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        print(f"分析エラー: {e}")
        return f"分析中にエラーが発生しました: {e}"

# --- NPCプロファイルを読み込む（シンプル版） ---
def load_npc_profiles():
    # 既存のNPCプロファイルを読み込む
    try:
        with open("npc_profiles.json", encoding="utf-8") as f:
            npc_list = json.load(f)
        
        # 佐藤さんだけを使用
        for npc in npc_list:
            if npc["name"] == "佐藤さん":
                return [npc]
        
        # 佐藤さんが見つからない場合はデフォルト値を返す
        return [{"name": "佐藤さん", "role": "避難所運営を実施する自治体職員", "personality": "穏やかで落ち着いた性格", "style": "敬語を使い、論理的に話す"}]
    
    except Exception as e:
        print(f"NPCプロファイル読み込みエラー: {e}")
        # エラー時はデフォルト値を返す
        return [{"name": "佐藤さん", "role": "避難所運営を実施する自治体職員", "personality": "穏やかで落ち着いた性格", "style": "敬語を使い、論理的に話す"}]

def main():
    # --- 会話開始 ---
    print("【避難所サポートシミュレーション】")
    print("あなたは避難所担当の行政職員「剛史君」です。避難者の不安を解消し、心理的なケアを行う役割を担っています。")
    print("佐藤さんは会話のファシリテーターとして、あなたに適切なヒントを提供します。避難者の悩みを聞き取り、解決策を提案しましょう。\n")
    print("【操作方法】")
    print("・会話に応答する場合は、メッセージを入力してEnterを押してください。")
    print("・会話を終了する場合は、何も入力せずにEnterを押すか、「会話終了」と入力してください。")
    print("・会話を終了すると、避難者の満足度評価が表示されます。\n")

    # NPCプロファイルを読み込み（佐藤さんのみ）
    npc_list = load_npc_profiles()
    # グローバル変数を更新
    global single_npc
    if npc_list and len(npc_list) > 0:
        single_npc = npc_list[0]
    
    # 避難者の心理状態管理リスト
    evacuee_mental_states = {}

    for evacuee_raw in evacuees:
        print(f"\n【イベント】避難者「{evacuee_raw['名前']}」さんが来ました（地区：{evacuee_raw['地区']}）\n")

        # 避難者プロファイルの作成と具体的な悩みの生成
        evacuee_profile = create_evacuee_profile(evacuee_raw)
        specific_concerns = generate_specific_concerns(evacuee_raw)
        
        # 避難者の懸念点を特定して心理状態オブジェクトを初期化
        concerns = identify_evacuee_concerns(evacuee_raw)
        # 具体的な悩みも追加（文字列化して重複を避ける）
        all_concerns = concerns.copy()
        for concern in specific_concerns:
            if concern not in all_concerns:
                all_concerns.append(concern)
        
        # 心理状態オブジェクトの初期化（状況と家族構成を反映）
        evacuee_mental_state = EvacueeMentalState(
            evacuee_raw["名前"], 
            all_concerns, 
            situation=evacuee_raw["状況"],
            family_composition=evacuee_raw["構成"]
        )
        evacuee_mental_states[evacuee_raw["名前"]] = evacuee_mental_state
        
        # 会話ログの初期化
        conversation_log = [{"speaker": "system", "content": f"避難者『{evacuee_raw['名前']}』来訪イベント"}]

        # 避難者本人の自己紹介（より具体的な悩みを含む）
        evacuee_speech = evacuee_profile_to_statement(evacuee_raw)
        conversation_log.append({"speaker": evacuee_raw["名前"], "content": evacuee_speech})
        evacuee_mental_state.add_conversation(evacuee_raw["名前"], evacuee_speech)
        print(f"{evacuee_raw['名前']}：{evacuee_speech}")

        # 具体的な悩みを表示
        print("\n[避難者の抱える具体的な悩み（内部情報）]")
        for i, concern in enumerate(specific_concerns, 1):
            print(f"{i}. {concern}")
        print("")

        # 佐藤さんからのヒント（最初のみ）
        initial_concern_text = "\n".join([f"- {concern}" for concern in all_concerns])
        
        initial_prompt = f"""あなたは避難所運営のファシリテーター役の「{single_npc['name']}」です。
避難者「{evacuee_raw['名前']}」が来訪し、自己紹介をしました。
状況：{evacuee_raw['状況']}、構成：{evacuee_raw['構成']}、地区：{evacuee_raw['地区']}

【避難者の具体的な悩み】
{initial_concern_text}

避難者の自己紹介：「{evacuee_speech}」

【重要】
あなたの役割は剛史君へのヒント提供です。
「（ヒント：〜）」という形式で、剛史君が適切に対応するための短い助言を提供してください。
剛史君に避難者の悩みを聞き出すような応対を促すヒントを1文で簡潔に伝えてください。
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": initial_prompt}
                ],
                temperature=0.7,
                max_tokens=80
            )
            initial_hint = response.choices[0].message.content.strip()
            
            # ヒント形式になっていなければ修正
            if not initial_hint.startswith("（ヒント：") and not initial_hint.startswith("(ヒント:"):
                initial_hint = f"（ヒント：{initial_hint}）"
            elif not initial_hint.endswith("）") and not initial_hint.endswith(")"):
                initial_hint = f"{initial_hint}）"
                
            conversation_log.append({"speaker": single_npc["name"], "content": initial_hint})
            evacuee_mental_state.add_conversation(single_npc["name"], initial_hint)
            print(f"{single_npc['name']}：{initial_hint}")
            
            # 剛史君の最初の発言を促す
            print("\n剛史君、会話に参加してください")
            print("返答を入力してください。何も入力せずEnterを押すと会話をスキップします。")
            first_input = input("剛史君: ").strip()
            if first_input:
                msg = {"speaker": "剛史君", "content": first_input}
                conversation_log.append(msg)
                evacuee_mental_state.add_conversation("剛史君", first_input)
                
                # 剛史君の発言を分類
                statement_type = classify_statement(first_input)
                
                # 会話ループ - 議論終了まで継続
                continue_discussion = True
                while continue_discussion:
                    # 剛史君の発言後、避難者が応答する
                    # 剛史君が質問したかどうかを検出
                    contains_question = any(q in first_input for q in ["?", "？", "いかがですか", "どうですか", "教えていただけますか", "思いますか", "ありますか", "でしょうか", "ください"])
                    direct_address = f"{evacuee_raw['名前']}さん" in first_input
                    
                    # 避難者の応答を生成
                    trigger_type = "question" if contains_question else "normal"
                    evacuee_response = generate_evacuee_response(evacuee_profile, conversation_log, trigger_type)
                    conversation_log.append({"speaker": evacuee_raw["名前"], "content": evacuee_response})
                    evacuee_mental_state.add_conversation(evacuee_raw["名前"], evacuee_response)
                    print(f"{evacuee_raw['名前']}：{evacuee_response}")
                    
                    # 避難者の心理状態を分析・更新
                    analyze_evacuee_response(evacuee_mental_state, evacuee_response, conversation_log)
                    
                    # 避難者の発言を分類
                    evacuee_statement_type = classify_statement(evacuee_response)
                    
                    # 佐藤さんのヒントを生成
                    hint, _ = generate_npc_response(single_npc, evacuee_raw, conversation_log, evacuee_statement_type, evacuee_mental_state)
                    conversation_log.append({"speaker": single_npc["name"], "content": hint})
                    evacuee_mental_state.add_conversation(single_npc["name"], hint)
                    print(f"{single_npc['name']}：{hint}")
                    
                    # ユーザーの選択（発言継続か会話終了か）
                    print("\n会話に応答する場合は返答を入力してください。")
                    print("何も入力せずEnterを押すと会話を終了し、避難者の満足度を評価します。")
                    user_input = input("剛史君: ").strip()
                    
                    if not user_input:
                        continue_discussion = False
                        print("\n会話を終了します。")
                    elif user_input.lower() == "会話終了":
                        continue_discussion = False
                        print("\n会話を終了します。")
                    else:
                        # 剛史君の応答を記録
                        msg = {"speaker": "剛史君", "content": user_input}
                        conversation_log.append(msg)
                        evacuee_mental_state.add_conversation("剛史君", user_input)
                        
                        # 剛史君の発言を更新
                        first_input = user_input
                
                # 会話の結論と避難者の満足度評価
                satisfaction_analysis = evaluate_evacuee_satisfaction(conversation_log, evacuee_mental_state)
                print("\n【避難者対応評価】")
                print(satisfaction_analysis)
                
                # 簡易情報のみ表示
                print(f"\n避難者: {evacuee_raw['名前']}（{evacuee_raw['地区']}、{evacuee_raw['状況']}）")
                print("\n------------------------------\n")
            else:
                print("（剛史君からの発言がありません）")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

if __name__ == "__main__":
    main() 