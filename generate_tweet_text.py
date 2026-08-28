"""
Tweet Text Generator for Sakuga & Genga Posts
---------------------------------------------
Generates formatted Twitter / X post copy following the exact template:

Animator: [Artist_1 & Artist_2] / [Anime Title]

[Custom Commentary or Koma-uchi note]

#[custom_tags] #falsememory #原画 #xsheet #コマ打ち
"""

import os
from typing import List, Optional, Dict, Any


def make_tweet_text(
    animator: str,
    anime_title: str,
    commentary: str = "",
    custom_tags: Optional[List[str]] = None
) -> str:
    """
    Generates single formatted tweet text.

    Parameters:
    -----------
    animator : str
        Name of Key Animator(s) (e.g. 'Taichi Hattori (服部聰司)').
    anime_title : str
        Name of Anime series (e.g. 'Witch Hat Atelier').
    commentary : str
        Optional custom commentary or Koma-uchi note.
    custom_tags : list[str]
        Optional list of hashtags (e.g. ['#WitchHatAtelier', '#TaichiHattori']).

    Returns:
    --------
    str : Formatted tweet text block.
    """
    if custom_tags is None:
        custom_tags = []

    mandatory_tags = ["#falsememory", "#原画", "#xsheet", "#コマ打ち"]

    # Ensure hashtags have '#' prefix
    cleaned_custom = [t if t.startswith("#") else f"#{t}" for t in custom_tags]
    all_tags = " ".join(cleaned_custom + mandatory_tags)

    header_line = f"Animator: {animator} / {anime_title}"

    if commentary.strip():
        return f"{header_line}\n\n{commentary.strip()}\n\n{all_tags}"
    else:
        return f"{header_line}\n\n{all_tags}"


def generate_tweets_for_selected_cuts() -> List[Dict[str, Any]]:
    """
    Generates tweet texts for all 5 selected cuts in sakugabooru-video-files/selected_cuts/.
    """
    cuts_metadata = [
        {
            "id": "01",
            "file": "01_witch_hat_atelier_taichi_hattori_309653.mp4",
            "animator": "Taichi Hattori (服部聰司)",
            "anime": "Witch Hat Atelier",
            "commentary": "Fluid Magic Particle & Wind FX Pencil Test. Analyzed with Digital X-Sheet HUD.",
            "custom_tags": ["#WitchHatAtelier", "#TaichiHattori"]
        },
        {
            "id": "02",
            "file": "02_jujutsu_kaisen_s3_daniel_kim_313851.mp4",
            "animator": "Daniel Kim",
            "anime": "Jujutsu Kaisen Season 3",
            "commentary": "Culling Game Combat & Impact Smears Draft. Analyzed with Digital X-Sheet HUD.",
            "custom_tags": ["#JujutsuKaisen", "#DanielKim"]
        },
        {
            "id": "03",
            "file": "03_one_piece_shin_kashiwaguma_311346.mp4",
            "animator": "Shin Kashiwaguma (柏熊信)",
            "anime": "One Piece",
            "commentary": "Climax Impact Frame Line Art & High-Density Body Dynamics.",
            "custom_tags": ["#OnePiece", "#ShinKashiwaguma"]
        },
        {
            "id": "04",
            "file": "04_blue_archive_cm_311659.mp4",
            "animator": "Ichibombu",
            "anime": "Blue Archive CM",
            "commentary": "Character Hair Movement Physics & Fluid Webgen Line-Art Draft.",
            "custom_tags": ["#BlueArchive", "#Ichibombu"]
        },
        {
            "id": "05",
            "file": "05_bleach_genga_314048.mp4",
            "animator": "Bleach Key Animation Team",
            "anime": "Bleach: TYBW",
            "commentary": "High-Impact Sword Battle Genga Reel. Analyzed with Digital X-Sheet HUD.",
            "custom_tags": ["#Bleach", "#BLEACH_TYBW"]
        }
    ]

    results = []
    for cut in cuts_metadata:
        tweet_text = make_tweet_text(
            animator=cut["animator"],
            anime_title=cut["anime"],
            commentary=cut["commentary"],
            custom_tags=cut["custom_tags"]
        )
        results.append({
            "id": cut["id"],
            "file": cut["file"],
            "tweet_text": tweet_text
        })
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("            TWITTER / X POST COPY GENERATOR             ")
    print("=" * 70 + "\n")

    tweets = generate_tweets_for_selected_cuts()
    for item in tweets:
        print(f"📌 [{item['id']}] File: {item['file']}")
        print("-" * 50)
        print(item["tweet_text"])
        print("-" * 50 + "\n")
