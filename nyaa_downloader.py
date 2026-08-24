import os
import subprocess
import requests
import xml.etree.ElementTree as ET

def search_and_download_nyaa(anime_title, episode_num=None, output_dir="sakugabooru-episodes", auto_open=True):
    """
    Bulletproof Nyaa.si Torrent Search & Auto-Launcher:
    1. Generates single-episode queries and excludes season batches if episode_num is specified.
    2. Downloads the top single-episode .torrent file into output_dir.
    3. Auto-launches torrent client via xdg-open.
    """
    os.makedirs(output_dir, exist_ok=True)

    parts = [p.strip() for p in anime_title.replace('_', ' ').replace('-', ' ').split(',') if p.strip()]
    candidate_titles = [p for p in reversed(parts) if "series" not in p.lower()]
    if not candidate_titles:
        candidate_titles = parts if parts else [anime_title]

    clean_title = candidate_titles[0]
    ep_str = f"{episode_num:02d}" if isinstance(episode_num, int) else (str(episode_num) if episode_num else "")

    # Build fallback query list
    queries = []
    if ep_str:
        queries.append(f"{clean_title} E{ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str}")
        queries.append(f"{clean_title}")
    else:
        queries.append(f"{clean_title} 1080p")
        queries.append(f"{clean_title}")

    domains = ['https://nyaa.si', 'https://nyaa.land']

    for query in queries:
        for domain in domains:
            rss_url = f"{domain}/?page=rss&q={requests.utils.quote(query)}&c=1_2&f=0&s=downloads&o=desc"
            print(f"[*] Querying Nyaa Search ({domain}): '{query}'...")
            try:
                res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
                if res.status_code == 200:
                    root = ET.fromstring(res.text)
                    items = root.findall('.//item')
                    if items:
                        selected_item = None

                        # If episode_num is specified, avoid season batches (Batch, Complete, S01-S12)
                        if ep_str:
                            for item in items:
                                title_text = item.find('title').text.lower()
                                if not any(b in title_text for b in ['batch', 'complete series', 'season 1', 'season 2', '01-']):
                                    selected_item = item
                                    break

                        if not selected_item:
                            selected_item = items[0]

                        torrent_title = selected_item.find('title').text
                        torrent_link = selected_item.find('link').text

                        torrent_res = requests.get(torrent_link, timeout=15)
                        if torrent_res.status_code == 200:
                            clean_filename = "".join(c for c in torrent_title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                            torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                            with open(torrent_filepath, 'wb') as f:
                                f.write(torrent_res.content)

                            print(f"[+] Single Episode Nyaa Torrent Found & Saved: {torrent_filepath}")

                            if auto_open:
                                try:
                                    print(f"[*] Auto-opening torrent in client: {torrent_filepath}")
                                    subprocess.Popen(['xdg-open', torrent_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                except Exception:
                                    pass

                            return torrent_filepath
            except Exception as e:
                print(f"[!] Query attempt '{query}' via {domain} notice:", e)

    print(f"[!] No single episode torrent results found for: '{anime_title}'.")
    return None

if __name__ == "__main__":
    import sys
    search_query = sys.argv[1] if len(sys.argv) > 1 else "Demon Slayer 06"
    search_and_download_nyaa(search_query, episode_num=6)
