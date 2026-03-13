"""
Mock Social Listening API.
Generates realistic social media posts for 3 brands across 5 topic buckets each.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict

BRANDS = {
    "zephyr_energy": {
        "name": "Zephyr Energy",
        "topics": [
            ["deep work", "focus session", "flow state", "cognitive performance"],
            ["morning routine", "5am club", "daily ritual", "wake up early"],
            ["afternoon slump", "energy crash", "3pm dip", "tired at work"],
            ["productivity hack", "time blocking", "pomodoro", "GTD method"],
            ["clean energy", "no crash", "L-theanine", "natural caffeine"],
        ],
        "platforms": ["twitter", "reddit", "instagram", "tiktok"],
    },
    "brighten_home": {
        "name": "Brighten Home",
        "topics": [
            ["plant-based cleaner", "non-toxic cleaning", "eco friendly", "safe ingredients"],
            ["pet safe cleaning", "dog friendly", "cat owner", "animal safe products"],
            ["apartment cleaning", "small space", "studio cleaning", "renter tips"],
            ["refill station", "zero waste", "sustainable living", "reduce plastic"],
            ["clean with me", "cleaning routine", "sunday reset", "deep clean"],
        ],
        "platforms": ["tiktok", "instagram", "reddit", "twitter"],
    },
    "trailblaze": {
        "name": "TrailBlaze",
        "topics": [
            ["solo hiking", "hiking alone", "trail safety", "women hiking"],
            ["beginner hiker", "first trail", "easy hikes", "newbie outdoors"],
            ["trail snacks", "hiking food", "backpacking meals", "protein on trail"],
            ["national park", "best trails", "trail recommendation", "weekend hike"],
            ["hiking with dogs", "pet friendly trail", "dog hike", "trail dog"],
        ],
        "platforms": ["instagram", "tiktok", "reddit", "twitter"],
    },
}

POST_TEMPLATES = [
    "just tried {topic} and it completely changed my routine",
    "anyone else obsessed with {topic} lately?",
    "unpopular opinion: {topic} is actually overrated",
    "day 7 of {topic} — here's what I noticed",
    "sharing my experience with {topic} so you don't have to learn the hard way",
    "the {topic} community is surprisingly helpful",
    "{topic} honestly saved my mornings",
    "why I finally started taking {topic} seriously",
    "tested {topic} for 30 days. here are my honest thoughts",
    "can we talk about how underrated {topic} is?",
]


def generate_posts(seed: int = 42) -> List[Dict]:
    """
    Generate 150 mock social posts: 50 per brand, 10 per topic bucket.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of post dicts with schema:
        {id, brand_id, platform, text, engagement:{likes,reposts,replies}, created_at}
    """
    random.seed(seed)
    posts = []
    now = datetime(2025, 1, 15, 12, 0, 0)

    for brand_id, brand_data in BRANDS.items():
        for topic_idx, topic_keywords in enumerate(brand_data["topics"]):
            for i in range(10):  # 10 posts per topic bucket = 50 per brand
                keyword = random.choice(topic_keywords)
                template = random.choice(POST_TEMPLATES)
                text = template.format(topic=keyword)

                platform = random.choice(brand_data["platforms"])
                age_hours = random.uniform(0, 48)
                created_at = now - timedelta(hours=age_hours)

                # Engagement: topic_idx 0-1 get higher engagement (trending topics)
                base_likes = 500 if topic_idx <= 1 else 100
                likes = int(random.gauss(base_likes, base_likes * 0.3))
                reposts = int(likes * random.uniform(0.1, 0.3))
                replies = int(likes * random.uniform(0.05, 0.15))

                posts.append({
                    "id": f"{brand_id[:3]}_{topic_idx}_{i:02d}",
                    "brand_id": brand_id,
                    "platform": platform,
                    "text": text,
                    "engagement": {
                        "likes": max(0, likes),
                        "reposts": max(0, reposts),
                        "replies": max(0, replies),
                    },
                    "created_at": created_at.isoformat(),
                    "topic_bucket": topic_idx,  # internal, used by clusterer
                })

    random.shuffle(posts)
    return posts
