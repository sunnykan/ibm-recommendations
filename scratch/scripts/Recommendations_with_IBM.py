import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ordered_set import OrderedSet


def email_mapper():
    df_copy = df.copy()
    email_info = df_copy.email.unique()
    email_user_id_dict = dict(zip(email_info, range(1, len(email_info) + 1)))
    df_copy["user_id"] = df_copy.email.apply(lambda x: email_user_id_dict[x])
    df_copy.drop(["email"], inplace=True, axis=1)
    return df_copy


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


def get_top_articles(n, df):
    """
TO DO
    """
    top_articles = df.title.value_counts().sort_values(ascending=False)

    return top_articles.index[
        :n
    ].to_list()  # Return the top article titles from df (not df_content)


def get_top_article_ids(n, df):
    """
TO DO    
    """
    top_articles = df.article_id.value_counts().sort_values(ascending=False)

    return top_articles.index[:n].to_list()  # Return the top article ids


def find_similar_users(user_id, user_item):
    """
TO DO
    """
    # compute similarity of each user to the provided user
    similarity_scores_user_id = np.dot(user_item[user_id - 1], np.transpose(user_item))
    ranked_users = np.argsort(similarity_scores_user_id)[::-1]
    ranked_users = np.delete(ranked_users, np.argwhere(ranked_users == user_id - 1))
    most_similar_users = list(ranked_users + 1)
    return most_similar_users  # return a list of the users in order from most to least similar


def get_article_names(article_ids, df):
    """
TO DO
    """

    df_copy = df.copy()
    df_copy.set_index(["article_id"], inplace=True)
    article_names = df_copy.title[article_ids].unique().tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item):
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


def get_top_sorted_users(user_id, df, user_item):
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


def evaluate_accuracy(df_train, df_test):
    (
        user_item_train,
        user_item_test,
        test_user_ids,
        test_article_ids,
    ) = create_test_train_user_item(df_train, df_test)

    train_user_ids = df_train.user_id.unique()
    train_article_ids = df_train.article_id.unique()

    num_users = df_train.user_id.nunique()
    num_articles = df_train.article_id.nunique()

    user_id_lookup = dict(zip(df_train.user_id.unique(), range(num_users)))
    article_id_lookup = dict(zip(df_train.article_id.unique(), range(num_users)))

    num_latent_features = np.arange(10, 701, 20)
    sum_errors_k = []

    num_latent_features = np.arange(10, 700 + 10, 20)
    for k in num_latent_features:
        print(k)
        u_train, s_train, vt_train = np.linalg.svd(user_item_train)

        u_k = u_train[:, :k]
        vt_k = vt_train[:k, :]
        s_k = np.zeros((k, k))
        s_k[:k, :k] = np.diag(s_train[:k])
        predictions = np.dot(np.dot(u_k, s_k), vt_k)

        predictions_user_ids = np.intersect1d(test_user_ids, train_user_ids)
        predictions_article_ids = np.intersect1d(test_article_ids, train_article_ids)

        predictions_user_ids_idx = [
            user_id_lookup[user_id] for user_id in predictions_user_ids
        ]
        predictions_article_ids_idx = [
            article_id_lookup[article_id] for article_id in predictions_article_ids
        ]

        predictions_temp = predictions[:, predictions_article_ids_idx]
        predictions_test = predictions_temp[predictions_user_ids_idx, :]

        user_item_temp = user_item_train[:, predictions_article_ids_idx]
        user_item_train_actual = user_item_temp[predictions_user_ids_idx, :]

        errors = np.subtract(user_item_train_actual, predictions_test)
        errors_total = np.sum(np.sum(np.abs(errors)))
        sum_errors_k.append(errors_total)
        mean_errors_k = np.array(sum_errors_k) / (
            predictions_test.shape[0] * predictions_test.shape[1]
        )

    accuracy_k = 1 - mean_errors_k
    return accuracy_k


def create_test_train_user_item(df_train, df_test):
    user_item_train, _, _ = get_user_item_mat(df_train)
    user_item_test, _, _ = get_user_item_mat(df_test)

    test_user_ids = df_test.user_id.unique()
    test_article_ids = df_test.article_id.unique()

    return user_item_train, user_item_test, test_user_ids, test_article_ids


if __name__ == "__main__":
    df = pd.read_csv("../data/user-item-interactions.csv")
    df_content = pd.read_csv("../data/articles_community.csv")

    del df["Unnamed: 0"]
    del df_content["Unnamed: 0"]

    df["article_id"] = df.article_id.astype("str")

    df = email_mapper()
    df = df[["user_id", "article_id", "title"]]

    # print(df.email.value_counts().head())
    # plt.hist(df.email.value_counts().values)

    df_content.drop_duplicates(subset=["article_id"], inplace=True)

    user_item, _, _ = get_user_item_mat(df)
    rec_ids, rec_names = user_user_recs_part2(20, 10)
    user1_most_sim = find_similar_users(user_id=1, user_item=user_item)[0]
    user131_10th_sim = find_similar_users(user_id=131, user_item=user_item)[9]

    new_user = "0.0"
    new_user_recs = get_top_article_ids(10, df=df)

    # SVD
    u, s, vt = np.linalg.svd(user_item)
    print(u.shape, s.shape, vt.shape)

    df_train = df.head(40000)
    df_test = df.tail(5993)
    accuracy_k = evaluate_accuracy(df_train, df_test)

    plt.plot(num_latent_features, accuracy_k)
    plt.xlabel("Number of Latent Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Latent Features")

