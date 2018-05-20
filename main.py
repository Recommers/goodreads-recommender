import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

df = pd.read_csv('/home/aliiz/Desktop/recommender/goodbooks-10k/ratings.csv')

books_size = 100
df100 = df.loc[df['book_id'] < books_size]


fig, ax = plt.subplots(figsize=(15, 7))
df100.groupby(['book_id', 'rating']).count().sort_values(['user_id'], ascending=False)['user_id'].unstack().plot(ax=ax)
# plt.show()

newdf = df100.groupby(['book_id', 'user_id'])


users_id_size = df100.user_id.nunique()
users = df100.user_id.unique()

users_dict = {v: index for index, v in np.ndenumerate(users)}

matrix = np.zeros((users_id_size, 5 * books_size))

start_time = datetime.now()

for index, row in df100.iterrows():
    matrix[users_dict[row['user_id']], row['book_id'] + (row['rating'] - 1) * books_size] = 1

time_elapsed = datetime.now() - start_time

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


# print(matrix)



















