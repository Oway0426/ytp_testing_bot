# project.py

import discord
from discord.ext import commands
import sqlite3
import datetime
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # 如果不需要定時更新模型，可以省略
import config  # 包含 TOKEN 與 PREFIX
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. 基本設定與全域變數
# -------------------------------

# 模擬產品菜單
menu = {
    "漢堡": 100,
    "薯條": 50,
    "可樂": 30,
    "沙拉": 70,
    "炸雞": 120
}

# 初始化 Discord Bot（啟用 message_content intent）
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=config.PREFIX, intents=intents)

# 全域推薦模型（初始為 None，稍後載入或訓練）
model = None

# APScheduler，用來定時更新模型（可選）
scheduler = AsyncIOScheduler()


# -------------------------------
# 2. 數據庫相關功能（隱式反饋）
# -------------------------------

def init_db():
    """
    初始化資料庫，建立 orders 表（若尚不存在）。
    使用 user_id 與 item 作為主鍵，隱式反饋存儲在 implicit_value 欄位中。
    """
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            user_id INTEGER,
            item TEXT,
            implicit_value INTEGER,
            timestamp TEXT,
            PRIMARY KEY (user_id, item)
        )
    """)
    conn.commit()
    conn.close()

def record_order(user_id, item):
    """
    當用戶點餐時，記錄隱式反饋：
      - 若該用戶對該產品尚無記錄，則插入新行，implicit_value 設為 1。
      - 若已存在，則將 implicit_value 累加 1，同時更新 timestamp。
    """
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    current_time = datetime.datetime.now().isoformat()
    
    # 檢查是否已有記錄
    cursor.execute("SELECT implicit_value FROM orders WHERE user_id=? AND item=?", (user_id, item))
    result = cursor.fetchone()
    
    if result is None:
        cursor.execute("""
            INSERT INTO orders (user_id, item, implicit_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_id, item, 1, current_time))
    else:
        new_value = result[0] + 1
        cursor.execute("""
            UPDATE orders
            SET implicit_value=?, timestamp=?
            WHERE user_id=? AND item=?
        """, (new_value, current_time, user_id, item))
    
    conn.commit()
    conn.close()

# -------------------------------
# 3. 模型訓練與推薦功能
# -------------------------------

def fetch_data():
    """
    從 orders 資料表中讀取數據，返回一個 DataFrame。
    欄位包括：user_id, item, implicit_value
    """
    conn = sqlite3.connect("orders.db")
    df = pd.read_sql_query("SELECT user_id, item, implicit_value FROM orders", conn)
    conn.close()
    return df

def train_model():
    """
    使用 Surprise 套件訓練 SVD 模型。
    隱式反饋數據的值會視為評分，模型訓練後會計算 RMSE 並儲存模型。
    """
    df = fetch_data()
    if df.empty:
        print("尚無數據進行訓練。")
        return None

    # 設定評分範圍：若所有隱式反饋皆為 1，則設為 1~5
    max_rating = df['implicit_value'].max() if df['implicit_value'].max() > 1 else 5
    reader = Reader(rating_scale=(1, max_rating))
    
    # 轉換成 Surprise 資料集
    data = Dataset.load_from_df(df[['user_id', 'item', 'implicit_value']], reader)
    
    # 切分訓練集與測試集
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # 使用 SVD 模型進行訓練
    algo = SVD()
    algo.fit(trainset)
    
    # 在測試集上預測並計算 RMSE
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"模型訓練完成。RMSE: {rmse}")
    
    # 儲存模型
    with open("recommend_model.pkl", "wb") as f:
        pickle.dump(algo, f)
    
    return algo

def load_model():
    """
    載入模型；若模型不存在則重新訓練。
    """
    global model
    try:
        with open("recommend_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("成功載入模型。")
    except FileNotFoundError:
        print("找不到模型檔案，開始訓練新模型...")
        model = train_model()
    return model

def get_recommendations(user_id, top_n=3):
    """
    根據模型預測使用者對各產品的評分，返回預測分數最高的品項清單。
    """
    if model is None:
        return "模型尚未建立。"
    
    recommendations = {}
    for product in menu.keys():
        # 使用模型預測 user_id 對每個 product 的評分
        pred = model.predict(user_id, product).est
        recommendations[product] = pred
    
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    top_recs = sorted_recs[:top_n]
    return top_recs

# -------------------------------
# 4. APScheduler 定時更新模型（可選）
# -------------------------------

def update_model_job():
    """
    定時任務：重新訓練模型並更新全域變數 model。
    """
    global model
    print("開始更新推薦模型...")
    new_model = train_model()
    if new_model:
        model = new_model
        print("模型更新完成！")

# 設定每日凌晨 0:00 更新模型（根據需求調整）
scheduler.add_job(update_model_job, 'cron', hour=0, minute=0)


# -------------------------------
# 5. Discord Bot 指令整合
# -------------------------------

@bot.event
async def on_ready():
    print(f"Bot {bot.user} 已上線！")
    init_db()           # 初始化資料庫（建立表格）
    load_model()        # 載入或訓練模型
    scheduler.start()   # 啟動 APScheduler（必須在事件循環中啟動）

@bot.command()
async def 菜單(ctx):
    """
    顯示產品菜單。
    """
    menu_text = "\n".join([f"{item}: {price} 元" for item, price in menu.items()])
    await ctx.send(f"**菜單**:\n{menu_text}")

@bot.command()
async def 點餐(ctx, *, item: str):
    """
    點餐指令：記錄用戶點餐的隱式反饋。
    """
    if item not in menu:
        await ctx.send(f"❌ 找不到產品 **{item}**，請檢查菜單。")
        return
    record_order(ctx.author.id, item)
    await ctx.send(f"✅ {ctx.author.name} 點了一份 **{item}**！")
    # 若希望每次點餐後更新模型（依需求決定，通常不建議頻繁更新）
    # update_model_job()

@bot.command()
async def 推薦(ctx):
    """
    根據歷史資料與模型預測，提供個性化推薦品項。
    """
    recs = get_recommendations(ctx.author.id)
    if isinstance(recs, str):
        await ctx.send(recs)
        return
    rec_text = "\n".join([f"{prod}: 預測評分 {score:.2f}" for prod, score in recs])
    await ctx.send(f"親愛的 {ctx.author.name}，我們推薦您試試以下品項：\n{rec_text}")

# -------------------------------
# 6. 主程式執行
# -------------------------------

if __name__ == "__main__":
    bot.run(config.TOKEN)
