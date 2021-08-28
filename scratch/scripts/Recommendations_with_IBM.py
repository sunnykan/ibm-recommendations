import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ordered_set import OrderedSet

df = pd.read_csv("../data/user-item-interactions.csv")
df_content = pd.read_csv("../data/articles_community.csv")

del df["Unnamed: 0"]
del df_content["Unnamed: 0"]

df["article_id"] = df.article_id.astype("str")

print(df.email.value_counts().head())
plt.hist(df.email.value_counts().values)

df_content.drop_duplicates(subset=["article_id"], inplace=True)


def email_mapper():
    df_copy = df.copy()
    email_info = df_copy.email.unique()
    email_user_id_dict = dict(zip(email_info, range(1, len(email_info) + 1)))
    df_copy["user_id"] = df_copy.email.apply(lambda x: email_user_id_dict[x])
    df_copy.drop(["email"], inplace=True, axis=1)
    return df_copy


df = email_mapper()

df = df[["user_id", "article_id", "title"]]


def get_user_item_mat(df):
    """TO DO"""
    num_users = df.iloc[:, 0].nunique()
    num_items = df.iloc[:, 1].nunique()
    # user_item_mat = np.empty((num_users, num_items))
    # user_item_mat.fill(np.nan)

    user_item_mat = np.full((num_users, num_items), fill_value=0)

    user_id_lookup = dict(zip(df.iloc[:, 0].unique(), range(num_users)))
    item_id_lookup = dict(zip(df.iloc[:, 1].unique(), range(num_items)))

    users_keys = user_id_lookup.keys()
    items_keys = item_id_lookup.keys()

    for idx, row in df.iterrows():
        user_item_mat[user_id_lookup[row[0]], item_id_lookup[row[1]]] = 1
    return user_item_mat, users_keys, items_keys


def get_top_articles(n, df=df):
    """
TO DO
    """
    top_articles = df.title.value_counts().sort_values(ascending=False)

    return top_articles.index[
        :n
    ].to_list()  # Return the top article titles from df (not df_content)


def get_top_article_ids(n, df=df):
    """
TO DO    
    """
    top_articles = df.article_id.value_counts().sort_values(ascending=False)

    return top_articles.index[:n].to_list()  # Return the top article ids


def find_similar_users(user_id, user_item=user_item):
    """
TO DO
    """
    # compute similarity of each user to the provided user
    similarity_scores_user_id = np.dot(user_item[user_id - 1], np.transpose(user_item))
    ranked_users = np.argsort(similarity_scores_user_id)[::-1]
    ranked_users = np.delete(ranked_users, np.argwhere(ranked_users == user_id - 1))
    most_similar_users = list(ranked_users + 1)
    return most_similar_users  # return a list of the users in order from most to least similar


def get_article_names(article_ids, df=df):
    """
TO DO
    """

    df_copy = df.copy()
    df_copy.set_index(["article_id"], inplace=True)
    article_names = df_copy.title[article_ids].unique().tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    """
TO DO
    """
    # Your code here
    df1 = df.copy()
    df1.set_index(["user_id"], inplace=True)
    article_ids = df1.article_id[user_id]
    article_names = df1.title[user_id]

    if not isinstance(article_ids, (str)):
        article_ids = list(set(article_ids))
        article_names = list(set(article_names))
    else:
        article_ids = [article_ids]
        article_names = [article_names]

    return article_ids, article_names  # return the ids and names


def user_user_recs(user_id, m=10):
    """
TO DO
    """
    recs_set = OrderedSet()

    articles_seen_user_id, _ = get_user_articles(user_id, user_item=user_item)
    similar_users = find_similar_users(user_id, user_item=user_item)

    for user in similar_users:
        article_ids, _ = get_user_articles(user, user_item=user_item)
        article_ids = np.setdiff1d(article_ids, articles_seen_user_id)
        recs_set.update(article_ids)
        recs = list(recs_set)
        if len(recs_set) >= m:
            recs = recs[:m]
            break
    return recs  # return your recommendations for this user_id


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    """
TO DO
    """
    # compute similarity of each user to the provided user
    similarity_scores_user_id = np.dot(user_item[user_id - 1], np.transpose(user_item))
    ranked_users = np.argsort(similarity_scores_user_id)[::-1]
    ranked_users = np.delete(ranked_users, np.argwhere(ranked_users == user_id - 1))
    ranked_similarities = similarity_scores_user_id[ranked_users]
    user_interactions = np.sum(user_item, axis=1)
    user_interactions = user_interactions[ranked_users]
    ranked_users = ranked_users + 1  # adjust for correct user_id number

    cols = ["neighbor_id", "similarity", "num_interactions"]
    data = [ranked_users, ranked_similarities, user_interactions]
    neighbors_df = pd.DataFrame.from_dict(dict(zip(cols, data)))
    neighbors_df.sort_values(by=["similarity", "num_interactions"], ascending=False)

    return neighbors_df  # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    """
TO DO
    """
    recs_set = OrderedSet()

    num_items = df.iloc[:, 1].nunique()
    id_item_lookup = dict(zip(range(num_items), df.iloc[:, 1].unique()))

    article_interactions = np.sum(user_item, axis=0)
    article_interactions_idx_ranked = np.argsort(article_interactions)[::-1]
    article_interactions_ids_ranked = [
        id_item_lookup[article_idx] for article_idx in article_interactions_idx_ranked
    ]

    articles_seen_user_id, _ = get_user_articles(user_id, user_item=user_item)
    similar_users = get_top_sorted_users(user_id, df=df.copy(), user_item=user_item)[
        "neighbor_id"
    ].values

    for user in similar_users:
        article_ids, _ = get_user_articles(user, user_item=user_item)
        article_ids = np.setdiff1d(article_ids, articles_seen_user_id)
        num_articles_found = len(article_ids)

        found = 0
        user_article_interactions_ranked = []
        for article_id in article_interactions_ids_ranked:
            if article_id in article_ids:
                user_article_interactions_ranked.append(article_id)
                found += 1
                if found >= num_articles_found:
                    break

        recs_set.update(user_article_interactions_ranked)
        recs_ids = list(recs_set)
        if len(recs_set) >= m:
            recs_ids = recs_ids[:m]
            break
    recs_names = get_article_names(recs_ids, df=df)
    return recs_ids, recs_names  # return your recommendations for this user_id


user_item, _, _ = get_user_item_mat(df)

a = article_ids_idx
b = np.argsort(article_interactions)[::-1]
sorter = np.argsort(b)
sorter[np.searchsorted(b, a, sorter=sorter)]