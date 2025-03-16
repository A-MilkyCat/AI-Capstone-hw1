import csv
import google.generativeai as genai
from googleapiclient.discovery import build
import os
import time
# Setting the API key from the environment variable
YOUTUBE_API_KEY = os.getenv('yt_key')
GEMINI_API_KEY = os.getenv("GEMINI_KEY")


genai.configure(api_key=GEMINI_API_KEY)

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_video_info(video_id):
    """ 從 YouTube API 獲取影片長度與觀看次數 """
    request = youtube.videos().list(
        part="contentDetails,statistics,snippet",
        id=video_id
    )
    response = request.execute()

    if "items" not in response or not response["items"]:
        return None

    item = response["items"][0]
    
    # 解析影片長度 (ISO 8601 格式轉換成秒)
    duration_str = item["contentDetails"]["duration"]
    duration = parse_iso8601_duration(duration_str)

    # 取得觀看次數
    view_count = int(item["statistics"].get("viewCount", 0))
    like_count = int(item["statistics"].get("likeCount", 0))
    comment_count = int(item["statistics"].get("commentCount", 0))

    # 取得說明欄內容
    description = item["snippet"].get("description", "")

    title = item["snippet"]["title"]
    like_rate = round(like_count / view_count * 100, 2) if view_count > 0 else 0
    comment_rate = round(comment_count / view_count * 100, 2) if view_count > 0 else 0
    return duration, view_count, description, title, like_rate, comment_rate

def parse_iso8601_duration(duration):
    """ 解析 YouTube 影片長度 (ISO 8601 -> 秒) """
    import isodate
    duration_seconds = isodate.parse_duration(duration).total_seconds()
    return int(duration_seconds)

def classify_video(description):
    """ 使用 Gemini API 根據影片說明欄分類 """
    model = genai.GenerativeModel("gemini-1.5-pro-002")
    prompt = f"Please classify the Netflix series based on the following video description into the most suitable genre. Choose only one category:\n\
              1. Drama\n\
              2. Comedy\n\
              3. Action/Adventure\n\
              4. Sci-Fi/Fantasy\n\
              5. Horror/Thriller\n\
              6. Documentary\n\
              7. Animation\n\
              Video description:\n{description}\n\
              Please reply with a number (1~7) only, without any additional content."

    response = model.generate_content(prompt)
    return response.text.strip()

def getcategory(category):
    if category == "1":
        return "Drama"
    elif category == "2":
        return "Comedy"
    elif category == "3":
        return "Action/Adventure"
    elif category == "4":
        return "Sci-Fi/Fantasy"
    elif category == "5":
        return "Horror/Thriller"
    elif category == "6":
        return "Documentary"
    elif category == "7":
        return "Animation"
    else:
        return "Unknown"
    
def save_to_csv(video_id, duration, view_count, category, title, likerate, commentrate):
    """ 儲存結果到 CSV 檔案 """
    filename = "netflix_trailers.csv"
    header = ["Video ID", "Title", "Duration", "Views", "Category", "Category ID", "Like Rate", "Comment Rate"]
    
    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            existing_data = f.readlines()
    except FileNotFoundError:
        existing_data = []

    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not existing_data:
            writer.writerow(header)
        writer.writerow([video_id, title, duration, view_count, getcategory(category), category, likerate, commentrate])

def read_video_ids(filename="video.csv"):
    video_ids = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                video_ids.append(row[0])
    return video_ids

video_ids = read_video_ids()
t = 1
for video_id in video_ids[301:]:
    video_data = get_video_info(video_id)
    if video_data:
        duration, view_count, description, title, likerate, commentrate = video_data
        category = classify_video(description)
        save_to_csv(video_id, duration, view_count, category, title, likerate, commentrate)
        print(f"已儲存：{video_id}, 長度: {duration} 秒, 觀看次數: {view_count}, 分類: {getcategory(category)}, 標題: {title}")
    else:
        print(f"無法取得影片資訊：{video_id}")
    time.sleep(50)
