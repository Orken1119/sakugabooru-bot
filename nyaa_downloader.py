import os
import subprocess
import requests
import xml.etree.ElementTree as ET

def search_and_download_nyaa(anime_title, episode_num=None, output_dir="sakugabooru-episodes", auto_open=True):
    """
    Automated Nyaa.si Torrent Search & Auto-Launcher:
    1. Searches Nyaa.si RSS feed sorted by most downloads (s=downloads&o=desc).
    2. Downloads the top .torrent file into output_dir.
    3. Auto-launches torrent client via xdg-open if auto_open is True.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    parts = [p.strip() for p in anime_title.replace('_', ' ').replace('-', ' ').split(',') if p.strip()]
    candidate_titles = [p for p in reversed(parts) if "series" not in p.lower()]
    if not candidate_titles:
        candidate_titles = parts if parts else [anime_title]

    clean_title = candidate_titles[0]
    
    if episode_num:
        query = f"{clean_title} {episode_num:02d} 1080p" if isinstance(episode_num, int) else f"{clean_title} {episode_num} 1080p"
    else:
        query = f"{clean_title} 1080p"

    rss_url = f"https://nyaa.si/?page=rss&q={requests.utils.quote(query)}&c=1_2&f=0&s=downloads&o=desc"

    print(f"[*] Searching Nyaa.si (Most Downloaded): '{query}'...")
    try:
        res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        if res.status_code == 200:
            root = ET.fromstring(res.text)
            items = root.findall('.//item')
            if items:
                top_item = items[0]
                torrent_title = top_item.find('title').text
                torrent_link = top_item.find('link').text

                torrent_res = requests.get(torrent_link, timeout=15)
                if torrent_res.status_code == 200:
                    clean_filename = "".join(c for c in torrent_title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                    torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                    with open(torrent_filepath, 'wb') as f:
                        f.write(torrent_res.content)

                    print(f"[+] Nyaa Torrent Saved: {torrent_filepath}")

                    # Automatically launch user's default torrent client (qBittorrent / Transmission)
                    if auto_open:
                        try:
                            print(f"[*] Auto-opening torrent in client: {torrent_filepath}")
                            subprocess.Popen(['xdg-open', torrent_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            pass

                    return torrent_filepath
            else:
                print(f"[!] No torrent results found on Nyaa for: '{query}'")
    except Exception as e:
        print("[!] Nyaa search error:", e)

    return None

if __name__ == "__main__":
    import sys
    search_query = sys.argv[1] if len(sys.argv) > 1 else "Demon Slayer 01 1080p"
    search_and_download_nyaa(search_query)
