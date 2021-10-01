import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple


def email_mapper():
    """Map the user email to a user_id column

    :return: Dataframe with a new user_id column 
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    email_info = df_copy.email.unique()
    email_user_id_dict = dict(zip(email_info, range(1, len(email_info) + 1)))
    df_copy["user_id"] = df_copy.email.apply(lambda x: email_user_id_dict[x])
    df_copy.drop(["email"], inplace=True, axis=1)

    return df_copy


def create_user_item_matrix(df):
    """Returns a dataframe with user ids as rows and articles ids as columns.
    Each cell may be 1 (interaction) or 0 (no interaction).

    :param df: Dataframe with data on users and articles, defaults to df
    :type df: pd.DataFrame, optional
    :return: Dataframe with user-article interactions
    :rtype: pd.DataFrame
    """
    num_users = df.iloc[:, 0].nunique()
    num_items = df.iloc[:, 1].nunique()
    users_keys = df.iloc[:, 0].unique()
    items_keys = df.iloc[:, 1].unique()

    user_item_mat = np.full((num_users, num_items), fill_value=0)

    user_id_lookup = dict(zip(users_keys, range(num_users)))
    item_id_lookup = dict(zip(items_keys, range(num_items)))

    for idx, row in df.iterrows():
        user_item_mat[user_id_lookup[row[0]], item_id_lookup[row[1]]] = 1

    user_item_df = pd.DataFrame(
        data=user_item_mat, index=users_keys, columns=items_keys
    )

    return user_item_df


def get_top_articles(n, df):
    """Returns top 'n' article titles ordered in descending order by most interactions

    :param n: The number of articles to return
    :type n: int
    :param df: Dataframe with data on users and articles, defaults to df
    :type df: pd.DataFrame, optional
    :return: A list of the top 'n' article titles
    :rtype: List[str]
    """
    top_articles = df.title.value_counts().sort_values(ascending=False)

    return list(top_articles.index[:n])


def get_top_article_ids(n, df):
    """Returns top 'n' article ids ordered in descending order by most interactions

    :param n: The number of articles to return
    :type n: int
    :param df: Dataframe with data on users and articles, defaults to df
    :type df: pd.DataFrame, optional
    :return: A list of the top 'n' article ids
    :rtype: List[str]
    """
    top_articles = df.article_id.value_counts().sort_values(ascending=False)

    return list(top_articles.index[:n])


def _get_similar_users_ranked(user_id, user_item):
    """Helper function that finds users most similar to user_id and orders them
    from hight to low on similarity

    :param user_id: user_id
    :type user_id: int
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :return: Tuple of similarity scores and corresponding rank ordered indices of 
    scores from high to low
    :rtype: List[np.ndarray, np.ndarray]
    """
    similarity_scores_user_id = np.dot(
        user_item.loc[user_id, :], np.transpose(user_item)
    )
    ranked_users_idx = np.argsort(similarity_scores_user_id)[::-1]
    delete_self_idx = np.where(user_item.index == user_id)[0][0]
    ranked_users_idx = np.delete(
        ranked_users_idx, np.where(ranked_users_idx == delete_self_idx)
    )

    return similarity_scores_user_id, ranked_users_idx


def find_similar_users(user_id, user_item):
    """Finds the users most similar to a given user.

    :param user_id: user_id
    :type user_id: int
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :return: List of article ids of most similar users ordered by similarity
    :rtype: List[int]
    """
    _, ranked_users_idx = _get_similar_users_ranked(user_id, user_item)
    most_similar_users = user_item.index[ranked_users_idx]

    return most_similar_users


def get_article_names(article_ids, df):
    """Get list of article names associated with list of article ids

    :param article_ids: List of article ids
    :type article_ids: List[str]
    :param df: Dataframe with data on users and articles, defaults to df
    :type df: pd.DataFrame, optional
    :return: List of article names associated with article ids
    :rtype: List[str]
    """
    df_copy = df.copy()
    df_copy.set_index(["article_id"], inplace=True)
    article_names = df_copy.title[article_ids].unique().tolist()

    return article_names


def get_user_articles(user_id, user_item):
    """Finds the ids and associated titles of articles that a given user has viewed.

    :param user_id: user id
    :type user_id: int
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :return: Tuple of lists of article ids and corresponding article titles
    :rtype: Tuple[List[str], List[str]]
    """
    df1 = df.copy()
    df1.set_index(["user_id"], inplace=True)
    article_ids = df1.article_id[user_id]
    article_names = df1.title[user_id]

    # If a user has interacted with only one article, then the article_ids variable
    # will be a string value not a series.
    if not isinstance(article_ids, (str)):
        article_ids = list(set(article_ids))
        article_names = list(set(article_names))
    else:
        article_ids = [article_ids]
        article_names = [article_names]

    return article_ids, article_names


def user_user_recs(user_id, user_item, m=10):
    """Find m recommendations for a given user. Starting with the user most similar to user_id,
    finds articles not seen by user_id until m recommendations are found. If users are
    equally close then a user is chosen arbitrarily. If a given user returns fewer than m
    recommendations, then the remaining ones are added arbitrarily.

    :param user_id: user_id
    :type user_id: int
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :param m: Number of recommendations desired, defaults to 10
    :type m: int, optional
    :return: List of m recommendations - article ids - for the user
    :rtype: List[int]
    """
    recs_set = set()

    articles_seen_user_id, _ = get_user_articles(user_id, user_item)
    similar_users = find_similar_users(user_id, user_item)

    for user in similar_users:
        article_ids, _ = get_user_articles(user, user_item)
        article_ids = np.setdiff1d(article_ids, articles_seen_user_id)
        recs_set.update(article_ids)
        recs = list(recs_set)
        if len(recs_set) >= m:
            recs = recs[:m]
            break

    return recs


def get_top_sorted_users(user_id, user_item, df):
    """Calculates the similarity of users (neighbors) to the requested user along with the
    number of articles viewed. The neighbors are sorted in descending order of similarity and
    number of articles viewed such that closest neighbors appear at the top. In cases where 
    neighbors are equally close, those with more articles viewed appear first.

    :param user_id: user id
    :type user_id: int
    :param df: Dataframe with data on users and articles, defaults to df
    :type df: pd.DataFrame, optional
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :return: A dataframe with neighbors of user_id sorted first by similarity and then by the 
    number of articles viewed by the neighbor in descending order
    :rtype: pd.DataFrame
    """
    similarity_scores_user_id, ranked_users_idx = _get_similar_users_ranked(
        user_id, user_item
    )
    ranked_similarities = similarity_scores_user_id[ranked_users_idx]
    ranked_users = user_item.index[ranked_users_idx]
    user_interactions = df.user_id.value_counts()
    user_interactions.drop(labels=[user_id], inplace=True)

    cols = ["neighbor_id", "similarity", "num_interactions"]
    data = [ranked_users, ranked_similarities, user_interactions]
    neighbors_df = pd.DataFrame.from_dict(dict(zip(cols, data)))
    neighbors_df.sort_values(
        by=["similarity", "num_interactions"], ascending=[False, False], inplace=True
    )

    return neighbors_df


def user_user_recs_part2(user_id, user_item, df, m=10):
    """Find m recommendations for a given user. Starting with the closest user to user_id,
    finds articles not seen by user_id until m recommendations are found. If users are
    equally close then users are chosen in descending order of the number of interactions. 
    Articles are added based on the descending order of total article interactions.

    :param user_id: user_id
    :type user_id: int
    :param user_item: A dataframe with users as rows and articles as columns; each cell
    is a 1 (interaction) or 0 (no interaction), defaults to user_item
    :type user_item: pd.DataFrame, optional
    :param m: Number of recommendations desired, defaults to 10
    :type m: int, optional
    :return: Tuple with lists of article ids and corresponding article titles - the lists are
    of size m
    :rtype: Tuple[List[str], List[str]]
    """
    recs_set = set()

    articles_seen_user_id, _ = get_user_articles(user_id, user_item)
    similar_users = get_top_sorted_users(user_id, user_item, df)["neighbor_id"].values

    article_interactions = df.article_id.value_counts()

    for user in similar_users:
        article_ids, _ = get_user_articles(user, user_item)
        article_ids = np.setdiff1d(article_ids, articles_seen_user_id)
        user_interactions = article_interactions[article_ids]
        user_interactions_ranked = user_interactions.sort_values(ascending=False)

        recs_set.update(user_interactions_ranked)
        recs_ids = list(recs_set)
        if len(recs_set) >= m:
            recs_ids = recs_ids[:m]
            break
    recs_names = get_article_names(recs_ids, df)

    return recs_ids, recs_names


def create_test_and_train_user_item(df, train_size=40000):
    """Creates a matrix of users and items for the training and test data where each
    user is a row and each article is a column. A cell with value of 1 represents an interaction
    while 0 represents an absence of one.

    :param df: Data set with , defaults to df
    :type df: pd.DataFrame, optional
    :param train_size: Size of the training set, defaults to 40000
    :type train_size: int, optional
    :return: A tuple consisting of user-item dataframes for the training and test data, the user ids
    for the test data, and the article ids for the test data
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
    """

    df_train = df.head(train_size)
    df_test = df.tail(df.shape[0] - train_size)

    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)

    test_user_ids = df_test.user_id.unique()
    test_article_ids = df_test.article_id.unique()

    return user_item_train, user_item_test, test_user_ids, test_article_ids


def evaluate_accuracy(num_latent_features, df, train_size=40000):
    """Estimate the mean error in the train and test sets where error is the difference 
    between the actual value and the predicted value for a given number of latent features. 
    Calculate accuracy for train and test set

    :param num_latent_features: A vector with a range of values for the number of features to 
    use to generate the predictions matrix. The maximum is the number of columns in the user
    item matrix
    :type num_latent_features: np.ndarray
    :param train_size: Size of the training set, defaults to 40000
    :type train_size: int, optional
    :return: a tuple with list of accuracy values for the train set and the test set
    :rtype: Tuple[List[float], List[float]]
    """

    (
        user_item_train,
        user_item_test,
        test_user_ids,
        test_article_ids,
    ) = create_test_and_train_user_item(df, train_size=train_size)

    train_user_ids = user_item_train.index
    train_article_ids = user_item_train.columns

    users_in_common = train_user_ids.intersection(test_user_ids)
    articles_in_common = train_article_ids.intersection(test_article_ids)

    test_users_idx = np.where(train_user_ids.isin(users_in_common))[0]
    test_articles_idx = np.where(train_article_ids.isin(articles_in_common))[0]

    errors_train_k, errors_test_k = [], []
    u, s, vt = np.linalg.svd(user_item_train)

    for k in num_latent_features:
        s_train, u_train, vt_train = (
            np.diag(s[:k]),
            u[:, :k],
            vt[:k, :],
        )

        s_test = s_train
        u_test = u_train[test_users_idx, :k]
        vt_test = vt_train[:k, test_articles_idx]

        user_item_train_est = np.around(np.dot(np.dot(u_train, s_train), vt_train))
        user_item_train_act = user_item_train.values

        user_item_test_est = np.around(np.dot(np.dot(u_test, s_test), vt_test))
        user_item_test_act = user_item_test.loc[
            users_in_common, test_article_ids
        ].values

        train_errors = np.subtract(user_item_train_act, user_item_train_est)
        test_errors = np.subtract(user_item_test_act, user_item_test_est)

        sum_errors_train = np.sum(np.abs(train_errors))
        sum_errors_test = np.sum(np.abs(test_errors))

        errors_train_k.append(sum_errors_train)
        errors_test_k.append(sum_errors_test)

    mean_errors_train_k = np.array(errors_train_k) / (
        user_item_train_act.shape[0] * user_item_train_act.shape[1]
    )
    mean_errors_test_k = np.array(errors_test_k) / (
        user_item_test_act.shape[0] * user_item_test_act.shape[1]
    )

    accuracy_train_k = 1 - mean_errors_train_k
    accuracy_test_k = 1 - mean_errors_test_k

    return accuracy_train_k, accuracy_test_k


if __name__ == "__main__":
    df = pd.read_csv("data/user-item-interactions.csv")
    df_content = pd.read_csv("data/articles_community.csv")

    del df["Unnamed: 0"]
    del df_content["Unnamed: 0"]

    df["article_id"] = df.article_id.astype("str")

    df = email_mapper()
    df = df[["user_id", "article_id", "title"]]

    df_content.drop_duplicates(subset=["article_id"], inplace=True)

    user_item = create_user_item_matrix(df)

    print("User-User based collaborative filtering.")
    rec_names = get_article_names(user_user_recs(20, user_item, m=10), df)
    print("The top 10 recommendations for user 20 are the following article names:")
    print(rec_names)
    print("----------------------------------------")

    print("User-User based collaborative filtering.", end=" ")
    print(
        "If users are equally close, then those with greater article interactions are preferred."
    )
    rec_ids, rec_names = user_user_recs_part2(20, user_item, df, m=10)
    print("The top 10 recommendations for user 20 are the following article names:")
    print(rec_names)
    print("----------------------------------------")

    print("Matrix factorization")
    print(
        "Evaluate the effect of the number of latent features on accuracy in train and test data."
    )
    num_latent_features = np.arange(10, 720, 10)
    accuracy_train_k, accuracy_test_k = evaluate_accuracy(num_latent_features, df)

    plt.plot(num_latent_features, accuracy_train_k)
    plt.plot(num_latent_features, accuracy_test_k)
    plt.xlabel("Number of Latent Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Latent Features")
    plt.show()
