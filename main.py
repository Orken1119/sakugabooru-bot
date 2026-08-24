import requests
import tweepy
import schedule
import random
import time
import os
import bs4
from bs4 import BeautifulSoup
from pybooru import Moebooru
from dotenv import load_dotenv
from audio_helper import fetch_and_add_audio
from music_helper import add_ost_to_video
from config import connect_api

load_dotenv()

siteurl = 'https://www.sakugabooru.com/post/show/'
header = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,video/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}
client = Moebooru(site_url='https://www.sakugabooru.com')

# Only pick posts the sakugabooru community rated as exceptional
SCORE_FILTER = "score:>100"


def grab_post_metadata(posturl):
    """
    Fetches the Sakugabooru post page ONCE and extracts artist name and anime name
    in a single HTTP request.
    Returns (artist_str, anime_str).
    """
    artist_str = "Unknown"
    anime_str = "Unknown"
    try:
        r = requests.get(posturl, headers=header)
        print("grab_post_metadata status:", r.status_code)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')

            artists_found = [
                a.text for div in soup.find_all(class_="tag-type-artist")
                for a in div.find_all('a')
                if a.text and a.text != '?'
            ]
            if artists_found:
                artist_str = ", ".join(artists_found)

            anime_found = [
                a.text for div in soup.find_all(class_="tag-type-copyright")
                for a in div.find_all('a')
                if a.text and a.text != '?'
            ]
            if anime_found:
                anime_str = ", ".join(anime_found)

    except Exception as e:
        print("grab_post_metadata error:", e)

    print("Artist:", artist_str)
    print("Anime:", anime_str)
    return artist_str, anime_str


def filetypechecker(boorurl):
    if boorurl.find('/'):
        if ".mp4" in (boorurl.rsplit('/', 1)[1]):
            return True
        else:
            return False


def boorurandom(retries=0):
    """
    Bug #7 Fix: added retry counter to prevent infinite recursion when
    Sakugabooru keeps returning non-MP4 batches. Gives up after 10 retries.
    """
    print("Hello!")

    # Bug #7 Fix: was `return boorurandom()` with no limit — stack overflow risk
    if retries > 10:
        print("[!] Max retries reached — could not find an MP4 post. Giving up.")
        return None

    try:
        files = client.post_list(tags=f"order:random -western {SCORE_FILTER}")
        mp4_files = [f for f in files if filetypechecker(f.get('file_url', ''))]
        if not mp4_files:
            print(f"No mp4 files found in this batch, retrying... (attempt {retries + 1}/10)")
            return boorurandom(retries + 1)  # Bug #7 Fix: pass counter

        choice = random.choice(mp4_files)
        boorurl = choice['file_url']
        tags = choice['tags']
        posturl = siteurl + "{0}".format(choice['id'])
        source_url = choice.get('source', '')  # e.g. 'https://youtube.com/...' or '#06 (BD)'

        animatorname, animename = grab_post_metadata(posturl)
        time.sleep(5)

        os.makedirs("sakugabooru-video-files", exist_ok=True)
        data = requests.get(boorurl, headers=header)
        print("data:", data.status_code)
        print("source:", source_url)
        video_path = "sakugabooru-video-files/{}.mp4".format(choice['id'])
        with open(video_path, 'wb') as file:
            file.write(data.content)

        raw_video = fetch_and_add_audio(video_path, animename, source_url=source_url)

        # Add harmonically matched anime OST with fade in/out
        final_video = add_ost_to_video(raw_video, animename)
        if not final_video:
            final_video = raw_video  # fallback: clip without music

        # Format tweet text — proper sakuga credit format
        animator_credit = f"Key Animation: {animatorname}" if animatorname != "Unknown" else ""
        tweet_text = (
            f"{animator_credit}\n"
            f"Anime: {animename.split(',')[0].strip().title()}\n"
            f"\n"
            f"#sakuga #anime #keyanimation"
        ).strip()

        params = "Tweet Text:\n{}\n\nPost URL: {}\nFinal Video: {}\n".format(
            tweet_text, posturl, final_video
        )
        print("Extracted Metadata:\n" + params)

        # time.sleep(5)
        # mediapost(params)

    except Exception as e:
        print("Main() Error:", e)


# def mediapost(params):
#     try:
#         api = connect_api()
#         file_path = []
#         directory_name = 'sakugabooru-video-files'
#         media_list = filter(lambda x: os.path.isfile(os.path.join(directory_name, x)), os.listdir(directory_name))
#         media_list = sorted(media_list, key=lambda x: os.path.getmtime(os.path.join(directory_name, x)), reverse=True)
#
#         for media in media_list:
#             file_path.append(os.path.join(directory_name, media))
#         media = file_path[0]
#
#         print(media)
#         upload_media = api.media_upload(media, media_category='tweet_video')
#         api.update_status(status=params, media_ids=[upload_media.media_id_string])
#
#     except Exception as e:
#         print("Mediapost() Error:", e)


if __name__ == '__main__':
    boorurandom()
    # schedule.every(45).minutes.do(boorurandom)
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)