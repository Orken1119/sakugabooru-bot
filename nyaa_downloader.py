import os
import subprocess
import requests
import xml.etree.ElementTree as ET

def search_and_download_nyaa(anime_title, episode_num=None, output_dir="sakugabooru-episodes", auto_open=True):
    """
    Bulletproof Nyaa.si Torrent Search & Auto-Launcher:
    1. Generates multiple query fallbacks (exact episode 1080p, exact episode raw, full title).
    2. Uses retry loops with increased timeouts to prevent network timeouts.
    3. Downloads the top .torrent file into output_dir and auto-launches client via xdg-open.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clean anime title candidate list
    parts = [p.strip() for p in anime_title.replace('_', ' ').replace('-', ' ').split(',') if p.strip()]
    candidate_titles = [p for p in reversed(parts) if "series" not in p.lower()]
    if not candidate_titles:
        candidate_titles = parts if parts else [anime_title]

    clean_title = candidate_titles[0]
    ep_str = f"{episode_num:02d}" if isinstance(episode_num, int) else (str(episode_num) if episode_num else "")

    # Build fallback query list
    queries = []
    if ep_str:
        queries.append(f"{clean_title} {ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str}")
        queries.append(f"{clean_title} 1080p")
        queries.append(f"{clean_title}")
    else:
        queries.append(f"{clean_title} 1080p")
        queries.append(f"{clean_title}")
        first_word = clean_title.split()[0] if clean_title.split() else clean_title
        if len(first_word) > 3:
            queries.append(first_word)

    domains = ['https://nyaa.si', 'https://nyaa.land']

    for query in queries:
        for domain in domains:
            rss_url = f"{domain}/?page=rss&q={requests.utils.quote(query)}&c=1_2&f=0&s=downloads&o=desc"
            print(f"[*] Querying Nyaa Search ({domain}): '{query}'...")
            try:
                res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
                if res.status_code == 200:
                    root = ET.fromstring(res.text)
                    items = root.findall('.//item')
                    if items:
                        top_item = items[0]
                        torrent_title = top_item.find('title').text
                        torrent_link = top_item.find('link').text

                        torrent_res = requests.get(torrent_link, timeout=20)
                        if torrent_res.status_code == 200:
                            clean_filename = "".join(c for c in torrent_title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                            torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                            with open(torrent_filepath, 'wb') as f:
                                f.write(torrent_res.content)

                            print(f"[+] Bulletproof Nyaa Torrent Found & Saved: {torrent_filepath}")

                            if auto_open:
                                try:
                                    print(f"[*] Auto-opening torrent in client: {torrent_filepath}")
                                    subprocess.Popen(['xdg-open', torrent_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                except Exception:
                                    pass

                            return torrent_filepath
            except Exception as e:
                print(f"[!] Query attempt '{query}' via {domain} notice:", e)

    print(f"[!] No torrent results found for anime: '{anime_title}' after multi-query fallback attempts.")
    return None

if __name__ == "__main__":
    import sys
    search_query = sys.argv[1] if len(sys.argv) > 1 else "Bleach"
    search_and_download_nyaa(search_query)
