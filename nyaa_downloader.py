import os
import re
import subprocess
import requests
import xml.etree.ElementTree as ET


def is_single_episode_size(item):
    """
    Checks Nyaa RSS item size tag.
    REJECTS by default on missing tag or parse error (fail-safe).
    Accepts only files between 50 MiB and 1.2 GiB (single-episode range).
    """
    try:
        size_elem = item.find('{https://nyaa.si/xmlns/nyaa}size')
        if size_elem is None:
            size_elem = item.find('size')

        # Bug #1 Fix: if tag is missing, REJECT (was: silently accept)
        if size_elem is None or not size_elem.text:
            print("[!] Size tag missing from RSS item — rejecting to be safe.")
            return False

        size_text = size_elem.text.strip().lower()
        nums = re.findall(r'[\d\.]+', size_text)
        if not nums:
            return False

        num = float(nums[0])

        if 'gib' in size_text or 'gb' in size_text:
            # reject anything over 1.2 GiB or under 0.05 GiB (50MB) — too small to be a real episode
            return 0.05 <= num <= 1.2

        if 'mib' in size_text or 'mb' in size_text:
            # accept 50 MB – 1228 MB
            return 50 <= num <= 1228

        # Unknown unit — reject
        return False

    except Exception as e:
        print(f"[!] Size parse error — rejecting item: {e}")
        return False  # Bug #1 Fix: was returning True on any exception


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

    # Build query list — most specific first
    queries = []
    if ep_str:
        queries.append(f"{clean_title} E{ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str} 1080p")
        queries.append(f"{clean_title} {ep_str}")
    else:
        # Bug #3 handled upstream: we should never reach here without an episode number
        # But if we do, try episode 1 only — never bare title (causes batch downloads)
        queries.append(f"{clean_title} E01 1080p")
        queries.append(f"{clean_title} 01 1080p")
        queries.append(f"{clean_title} 01")

    # Sort by seeders (not downloads) to avoid stale batch packs dominating results
    domains = ['https://nyaa.si', 'https://nyaa.land']
    batch_pattern = re.compile(
        r'batch|box|complete|season|vol\b|collection|pack|disc|bd\s*box|'
        r'\d{1,4}\s*-\s*\d{1,4}|\d{1,4}\s*~\s*\d{1,4}',
        re.IGNORECASE
    )

    for query in queries:
        for domain in domains:
            # Sort by seeders instead of downloads to get fresh single episodes
            rss_url = f"{domain}/?page=rss&q={requests.utils.quote(query)}&c=1_2&f=0&s=seeders&o=desc"
            print(f"[*] Querying Nyaa ({domain}): '{query}'...")
            try:
                res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
                if res.status_code != 200:
                    continue

                root = ET.fromstring(res.text)
                items = root.findall('.//item')
                if not items:
                    print(f"[!] No results for query: '{query}'")
                    continue

                for item in items:
                    title_elem = item.find('title')
                    if title_elem is None:
                        continue
                    title_text = title_elem.text or ""

                    if batch_pattern.search(title_text):
                        print(f"    [skip batch] {title_text}")
                        continue

                    if not is_single_episode_size(item):
                        print(f"    [skip size]  {title_text}")
                        continue

                    # Passed all filters — download torrent
                    torrent_link = item.find('link').text
                    print(f"[+] Selected: {title_text}")

                    torrent_res = requests.get(torrent_link, timeout=15)
                    if torrent_res.status_code == 200:
                        clean_filename = "".join(c for c in title_text if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                        torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                        with open(torrent_filepath, 'wb') as f:
                            f.write(torrent_res.content)

                        print(f"[+] Torrent saved: {torrent_filepath}")

                        if auto_open:
                            try:
                                subprocess.Popen(
                                    ['xdg-open', torrent_filepath],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                )
                            except Exception:
                                pass

                        return torrent_filepath

                print(f"[!] No valid single-episode candidate found for: '{query}'")

            except Exception as e:
                print(f"[!] Query '{query}' via {domain} failed: {e}")

    print(f"[!] All queries exhausted — no single episode found for: '{anime_title}'")
    return None


if __name__ == "__main__":
    import sys
    search_query = sys.argv[1] if len(sys.argv) > 1 else "Aikatsu Friends"
    ep_num = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
    print(f"[*] Searching: '{search_query}' Episode {ep_num}")
    search_and_download_nyaa(search_query, episode_num=ep_num)
