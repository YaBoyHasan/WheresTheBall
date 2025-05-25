import requests, os, re, sqlite3, time
from datetime import datetime
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from app.models.user import db
from app.models.botbcomp import BotbComp
from config import Config
from app.utils.preprocess import preprocess_and_save

def extract_coordinates(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    judges_checkbox = soup.find('input', {'id': 'judged_checkbox'})
    winner_checkbox = soup.find('input', {'id': 'winner_checkbox'})
    image_tag = soup.find('a', {'class': 'view_image_trigger'})

    judges_coords = None
    winner_coords = None
    if judges_checkbox:
        judges_coords = re.search(r'X (\d+) Y (\d+)', judges_checkbox.get('data-label', ''))
    if winner_checkbox:
        winner_coords = re.search(r'X (\d+) Y (\d+)', winner_checkbox.get('data-label', ''))

    return {
        'judges_x': judges_coords.group(1) if judges_coords else None,
        'judges_y': judges_coords.group(2) if judges_coords else None,
        'winner_x': winner_coords.group(1) if winner_coords else None,
        'winner_y': winner_coords.group(2) if winner_coords else None,
        'image_guid': image_tag.get('data-competition_picture_guid') if image_tag else None
    }

def slugify_url(url):
    return url.rstrip('/').split('/')[-1]

def download_image(guid, filename):
    url = f"https://www.botb.com/umbraco/botb/spottheball/getcompetitionpicture/?competitionpictureguid={guid}&size=RESULT_FULL"
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            if img.size == (4416, 3336):
                os.makedirs(Config.IMAGES_FOLDER, exist_ok=True)
                path = os.path.join(Config.IMAGES_FOLDER, filename)
                img.save(path)
                return img.size
            else:
                return img.size
        else:
            return None
    except Exception:
        return None

def insert_into_db(data):
    for row in data:
        if not is_comp_in_db(row['CompUrl']):
            comp = BotbComp(
                CompUrl=row['CompUrl'],
                JudgesX=row['JudgesX'],
                JudgesY=row['JudgesY'],
                ImageFileName=row['ImageFilename']
            )
            db.session.add(comp)
    db.session.commit()

def is_comp_in_db(comp_url):
    return BotbComp.query.filter_by(CompUrl=comp_url).first() is not None

def process_competition(url):
    if is_comp_in_db(url):
        print(f"Skipping already processed comp: {url}")
        return None

    print(f"Processing competition URL: {url}")
    try:
        html_resp = requests.get(url)
        if html_resp.status_code != 200:
            print(f"Failed to fetch competition page: {url} with status code {html_resp.status_code}")
            return None
    except Exception as e:
        print(f"Request failed for {url}: {e}")
        return None

    coords = extract_coordinates(html_resp.text)
    if coords['judges_x'] and coords['judges_y'] and coords['image_guid']:
        slug = slugify_url(url)
        filename = f"{slug}.jpg"
        img_size = download_image(coords['image_guid'], filename)
        if img_size:
            if img_size != (4416, 3336):
                print(f"Image size {img_size} differs from expected. Stopping scraper.")
                return "STOP"
            return {
                'CompUrl': url,
                'JudgesX': int(coords['judges_x']),
                'JudgesY': int(coords['judges_y']),
                'ImageFilename': filename
            }
        else:
            print(f"Image download failed for GUID: {coords['image_guid']}")
    else:
        print(f"Missing coordinates or image guid for {url}")
    return None

def fetch_and_store_comp_data():
    print("Fetching BOTB competition data...")
    try:
        r = requests.get("https://www.botb.com/umbraco/surface/WinnersSurface/GetWinners")
        if r.status_code != 200:
            print(f"Failed to fetch BOTB data, status code: {r.status_code}")
            return

        json_data = r.json()
        comps = [f"https://www.botb.com{w['Url']}" for w in json_data['data']['winnerList']
                 if w['CompetitionPictureTitle'] == 'Dream Car']

        print(f"Found {len(comps)} 'Dream Car' competitions")

        results = []
        for url in comps:
            if is_comp_in_db(url):
                print(f"Competition already in DB: {url}. Stopping scraper.")
                break

            time.sleep(0.2)
            res = process_competition(url)
            if res == "STOP":
                print("Stopping further scraping due to image size mismatch.")
                break
            if res:
                results.append(res)

        if results:
            insert_into_db(results)
            print(f"Inserted {len(results)} new comps.")
            preprocess_and_save()
        else:
            print("No valid competitions to insert.")
    except Exception as e:
        print(f"Scheduler error: {e}")
