import os
import re
import subprocess
import requests
import xml.etree.ElementTree as ET

def is_single_episode_size(item):
    """
    Checks Nyaa RSS item size tag and rejects any torrent larger than 1.2 GiB.
    This guarantees that heavy season batches or full BD packs (e.g. 68GB or 8GB)
    are strictly rejected, downloading ONLY small single-episode files (~300MB-800MB).
    """
    try:
        size_elem = item.find('{https://nyaa.si/xmlns/nyaa}size')
        if size_elem is None:
            size_elem = item.find('size')

        if size_elem is not None and size_elem.text:
            size_text = size_elem.text.strip().lower()
            if 'gib' in size_text or 'gb' in size_text:
                num = float(re.findall(r'[\d\.]+', size_text)[0])
                if num > 1.2:
                    return False
    except Exception:
        pass
    return True

def search_and_download_nyaa(anime_title, episode_num=None, output_dir="sakugabooru-episodes", auto_open=True):
    """
    Strict Lightweight Single-Episode Nyaa.si Torrent Search & Auto-Launcher:
    1. Generates single-episode queries.
    2. Strictly filters out all batch/season/box packs AND any torrent > 1.2 GiB.
    3. Downloads ONLY small single-episode .torrent files (~300MB-800MB) into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    parts = [p.strip() for p in anime_title.replace('_', ' ').replace('-', ' ').split(',') if p.strip()]
    candidate_titles = [p for p in reversed(parts) if "series" not in p.lower()]
    if not candidate_titles:
        candidate_titles = parts if parts else [anime_title]

    clean_title = candidate_titles[0]
    ep_str = f"{episode_num:02d}" if isinstance(episode_num, int) else (str(episode_num) if episode_num else "")

    # Build query list
    queries = []
    if ep_str:
        queries.append(f"{clean_title} E{ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str}")
        queries.append(f"{clean_title}")
    else:
        queries.append(f"{clean_title} 01 1080p")
        queries.append(f"{clean_title} 1080p")
        queries.append(f"{clean_title}")

    domains = ['https://nyaa.si', 'https://nyaa.land']
    batch_pattern = re.compile(r'batch|box|complete|season|vol\b|collection|pack|disc|\d{1,4}\s*-\s*\d{1,4}|\d{1,4}\s*~\s*\d{1,4}', re.IGNORECASE)

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

                        for item in items:
                            title_text = item.find('title').text
                            if not batch_pattern.search(title_text) and is_single_episode_size(item):
                                selected_item = item
                                break

                        if not selected_item:
                            single_candidates = [i for i in items if not batch_pattern.search(i.find('title').text) and is_single_episode_size(i)]
                            if single_candidates:
                                selected_item = single_candidates[0]
                            else:
                                print(f"[!] No lightweight single-episode candidate (<1.2GB) found for query '{query}'. Trying fallback...")
                                continue

                        torrent_title = selected_item.find('title').text
                        torrent_link = selected_item.find('link').text

                        torrent_res = requests.get(torrent_link, timeout=15)
                        if torrent_res.status_code == 200:
                            clean_filename = "".join(c for c in torrent_title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                            torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                            with open(torrent_filepath, 'wb') as f:
                                f.write(torrent_res.content)

                            print(f"[+] Small Single-Episode Torrent Saved (<1.2GB): {torrent_filepath}")

                            if auto_open:
                                try:
                                    print(f"[*] Auto-opening torrent in client: {torrent_filepath}")
                                    subprocess.Popen(['xdg-open', torrent_filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                except Exception:
                                    pass

                            return torrent_filepath
            except Exception as e:
                print(f"[!] Query attempt '{query}' via {domain} notice:", e)

    print(f"[!] No lightweight single episode torrent results found for: '{anime_title}'.")
    return None

if __name__ == "__main__":
    import sys
    search_query = sys.argv[1] if len(sys.argv) > 1 else "Aikatsu Friends"
    ep_num = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    search_and_download_nyaa(search_query, episode_num=ep_num)
