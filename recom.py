import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from operator import itemgetter
df = pd.read_csv('ratings.csv')

# this is bound wide of size of book and user
books_size = 10
users_size = 10
book_dict = {}
users_dict = {}


def make_subset_of_dataset():
    """ this function make subset of original database by this order first book then user """
    subset_of_df = df.loc[df['book_id'] < books_size]
    subset_of_df = subset_of_df.loc[subset_of_df['user_id'] < users_size]
    df_temp = subset_of_df['rating'].isin(range(4, 6))
    df_truncated = subset_of_df[df_temp]
    return df_truncated


def draw_dataset(subset_of_df):
    """drawing the data base """
    fig, ax = plt.subplots(figsize=(15, 7))
    subset_of_df.groupby(['book_id', 'rating']).count().sort_values(['user_id'], ascending=False)['user_id'].unstack().plot(ax=ax)
    plt.show()


def make_adjacency_matrix(subset_of_df):
    """make matrix"""
    users_id_size = subset_of_df.user_id.nunique()
    users = subset_of_df.user_id.unique()
    global users_dict
    users_dict = {v: index for index, v in np.ndenumerate(users)}
    book_id_size = subset_of_df.book_id.nunique()
    books = subset_of_df.book_id.unique()
    global book_dict
    book_dict = {v: index for index, v in np.ndenumerate(books)}

    matrix = np.zeros((users_id_size + book_id_size, users_id_size + book_id_size), dtype=int)
    for index, row in subset_of_df.iterrows():
        matrix[users_dict[row['user_id']], (book_dict[row['book_id']][0] + users_id_size,)] = 1
        matrix[(book_dict[row['book_id']][0] + users_id_size,), users_dict[row['user_id']]] = 1
    # print(book_dict, "book_dic")
    # print(users_dict, "user_dic")
    return matrix

def information_data():
    """information"""
    g1 = df.groupby(['rating'])
    print(df.groupby(['rating']).describe())
    print('')
    print(g1.count(), g1.var())
    print(g1.groups)
    print(df.value_counts())
    print(df[df].var())
    print(df[df].mean())

# in this part we try to extract data from data set with the user, book, rating, and relatively among them
df_user_temp = df['user_id'].isin(range(0, users_size))
df_book_temp = df['book_id'].isin(range(0, books_size))

df_user_temp22 = df[df_user_temp]
df_book_temp22 = df[df_book_temp]

df_user_temp2 = df_user_temp22['rating'].isin(range(4, 6))
df_book_temp2 = df_book_temp22['rating'].isin(range(4, 6))

df_book = df_book_temp22[df_book_temp2]
df_user = df_user_temp22[df_user_temp2]


# ========================== evaluation code ================================

matrix = make_adjacency_matrix(make_subset_of_dataset())
g = nx.from_numpy_matrix(matrix)
color = []
for index in users_dict:
    color.append('b')
for index in book_dict:
    color.append('r')
# double distance between all nodes
graph_pos = nx.shell_layout(g, scale=10)
plt.figure(figsize=(64, 60))
nx.draw(g, with_labels=True, node_color=color, pos=graph_pos)
# plt.savefig("plot.png", dpi=1000)
plt.show()


size = matrix.shape
# print(matrix)
book_size_real = book_dict.__len__()
user_size_real = users_dict.__len__()
edge_size = np.count_nonzero(matrix)
edge_del_size = temp = int(edge_size / 10)
del_edge = []
# for i in range(edge_del_size):
#     del_edge.append((np.random.randint(user_size_real, user_size_real+book_size_real),
#                      np.random.randint(user_size_real, user_size_real+book_size_real)))
#
# for item in del_edge:
#     print((item[1], item[0]))
#     print(item)
#     print(matrix[item[1], item[0]])
# this function remove amount of edge_del_size edge from matrix
while temp:
    a = np.random.randint(0, user_size_real)
    b = np.random.randint(user_size_real, user_size_real+book_size_real)
    if matrix[a, b] == 1:
        temp -= 1
        matrix.itemset((a, b), 0)
        matrix.itemset((b, a), 0)
        del_edge.append((a, b))

print(del_edge)
print('del_edge')
# this code show all tuple
# print('=========book=======')
# with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#     print(df_book.describe())
#     print(df_book.sort_values(by='rating'))
# print('=========user=======')
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(df_user.describe())
    # print(df_user.sort_values(by='rating'))


# make_adjacency_matrix2(df_user_temp22)
# print(make_subset_of_dataset())
# print(df_user_temp22)
# print(make_subset_of_dataset())

# g = nx.from_numpy_matrix(matrix)
# color = []
# for index in users_dict:
#     color.append('b')
# for index in book_dict:
#     color.append('r')
# # double distance between all nodes
# graph_pos = nx.shell_layout(g, scale=10)
# plt.figure(figsize=(64, 64))
# nx.draw(g, with_labels=True, node_color=color, pos=graph_pos)
# plt.savefig("plot.png", dpi=1000)
# plt.show()

# draw_dataset(make_subset_of_dataset())
# # ===============================
# preds = nx.resource_allocation_index(g)
# for u, v, p in preds:
#     if (u, v) in del_edge:
#       print('(%d, %d) -> %.8f' % (u, v, p))
# # ===============================
#
# preds = nx.jaccard_coefficient(g)
# for u, v, p in preds:
#     if (u, v) in del_edge:
#       print('(%d, %d) -> %.8f' % (u, v, p))
#
# preds = nx.adamic_adar_index(g)
# for u, v, p in preds:
#     if (u, v) in del_edge:
#       print('(%d, %d) -> %.8f' % (u, v, p))

predi_pro = []
p_without_useless = []
up = 0
down = 0
preds= temp2 = nx.preferential_attachment(g)
# print(matrix)
# print(user_size_real, book_size_real, 'real')
for u, v, p in preds:
    if v in range(user_size_real) and u in range(user_size_real, user_size_real + book_size_real ):
        # down = down + p
        p_without_useless.append((v, u, p))
        # print(u, v, p)
    if v in range(user_size_real, user_size_real + book_size_real) and u in range(user_size_real):
        # down = down + p
        p_without_useless.append((v, u, p))
        # print(u, v, p)

# print(p_without_useless[2])
for u1, v1, p1 in p_without_useless:
    if (v1, u1) in del_edge:
        up = up + p1
        # print(v1,u1,p1)
    # print(p1, v1, u1)

a = sorted(p_without_useless, key=itemgetter(2), reverse=True)
# print(a)
for i in range(edge_del_size):
    down = down + a[i][2]

print(down)
print(up)
print(up/down)
print("============= mutliply ==================")
print(matrix)
print("============= matrix ==================")
matrix_2 = matrix.dot(matrix)
print(matrix_2)
print("============= matrix 2 ==================")
matrix_3 = matrix_2.dot(matrix)
print(matrix_3)
print("============= matrix 3 ==================")
matrix_4 = matrix_3.dot(matrix)
print(matrix_4)
