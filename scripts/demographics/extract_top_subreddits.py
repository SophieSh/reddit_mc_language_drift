"""Most popular subreddits among the initial CD cohort.

Ranks subreddits two ways so we can pick the metric:
  - n_posts: total posts in that subreddit
  - n_users: distinct cohort users who ever posted there
"""
import pandas as pd

COHORT = "data/submission/processed/users_database_CD_20260329T064620.csv"
POSTS  = "data/interim/posts_all_users_preprocessed_no_anchors_20260517T104734.csv"

users = set(pd.read_csv(COHORT, usecols=["user"]).user)
posts = pd.read_csv(POSTS, usecols=["author", "subreddit"])
posts = posts[posts.author.isin(users)]

tab = pd.DataFrame({
    "n_posts": posts.subreddit.value_counts(),
    "n_users": posts.groupby("subreddit").author.nunique(),
})

print(f"cohort users with posts: {posts.author.nunique()}   subreddits: {len(tab)}   posts: {len(posts)}\n")
print("=== Top 20 by POSTS ===")
print(tab.sort_values("n_posts", ascending=False).head(20).to_string())
print("\n=== Top 20 by USERS ===")
print(tab.sort_values("n_users", ascending=False).head(20).to_string())

tab.sort_values("n_posts", ascending=False).to_csv("outputs/top_subreddits.csv")
print("\nSaved outputs/top_subreddits.csv")
