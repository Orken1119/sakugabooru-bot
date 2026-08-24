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
from config import connect_api

load_dotenv()

siteurl='https://www.sakugabooru.com/post/show/'
header = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,video/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}
client = Moebooru(site_url='https://www.sakugabooru.com')
files = client.post_list(tags="order:random -western")

def boorurandom():
        print("Hello!")
        try:
            files = client.post_list(tags="order:random -western") #Random Post 
            mp4_files = [f for f in files if filetypechecker(f.get('file_url', ''))]
            if not mp4_files:
                print("No mp4 files found in this batch, retrying...")
                return boorurandom()

            choice = random.choice(mp4_files) #Select 1 Random MP4 Post
            boorurl=choice['file_url'] #File URL
            tags = choice['tags'] #Post Tags

            posturl = siteurl+"{0}".format(choice['id']) #POST URL from SakugaBooru

            animatorname=artistgrabber(posturl)
            animename=animegrabber(posturl)
            time.sleep(5)
            
            os.makedirs("sakugabooru-video-files", exist_ok=True)
            data = requests.get(boorurl,headers=header)
            print("data:",data.status_code)
            video_path = "sakugabooru-video-files/{}".format(choice['id']) + ".mp4"
            with open(video_path, 'wb') as file: 
                file.write(data.content)
            
            raw_video = fetch_and_add_audio(video_path, animename)

            params="Animator Name: {}\nListed Anime Name: {}\nTags: {}\nPost URL: {}\nRaw Audio Video: {}\n".format(animatorname,animename,tags,posturl,raw_video)
            print("Extracted Metadata:\n" + params)
                
                # time.sleep(5)
                # mediapost(params)

        except Exception as e:
            print("Main() Error:",e)


def artistgrabber(posturl):
    artiststr = "Unknown"
    try:
        r = requests.get(posturl, headers=header)
        print("artistgrabber status:", r.status_code)
        if r.status_code == 200:
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            artists_found = [a.text for div in soup.find_all(class_="tag-type-artist") for a in div.find_all('a') if a.text and a.text != '?']
            if artists_found:
                artiststr = ", ".join(artists_found)
    except Exception as e:
        print("artistgrabber error:", e)
    print("Artist:", artiststr)
    return artiststr

def animegrabber(posturl):
    animestr = "Unknown"
    try:
        r = requests.get(posturl, headers=header)
        print("animegrabber status:", r.status_code)
        if r.status_code == 200:
            soup = bs4.BeautifulSoup(r.text, 'html.parser')
            anime_found = [a.text for div in soup.find_all(class_="tag-type-copyright") for a in div.find_all('a') if a.text and a.text != '?']
            if anime_found:
                animestr = ", ".join(anime_found)
    except Exception as e:
        print("animegrabber error:", e)
    print("Anime:", animestr)
    return animestr

def filetypechecker(boorurl):
    if boorurl.find('/'):
            if ".mp4" in (boorurl.rsplit('/',1)[1]):
                return True
            else:
                return False

# def mediapost(params):
#     try:
#         api=connect_api()
#         file_path=[]
#         directory_name='sakugabooru-video-files' #Customize Directory
#         media_list=filter(lambda x: os.path.isfile(os.path.join(directory_name,x)),os.listdir(directory_name))
#         media_list=sorted(media_list,key=lambda x: os.path.getmtime(os.path.join(directory_name,x)),reverse=True)
# 
#         for media in media_list:
#             file_path.append(os.path.join(directory_name,media))       
#         media=file_path[0]
# 
#         print(media)
#         upload_media=api.media_upload(media, media_category='tweet_video')
#         api.update_status(status=params, media_ids=[upload_media.media_id_string])
# 
#     except Exception as e:
#         print("Mediapost() Error:",e)
        
if __name__ == '__main__':
    boorurandom()
    # schedule.every(45).minutes.do(boorurandom)
    # 
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)