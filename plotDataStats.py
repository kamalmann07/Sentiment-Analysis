import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Reviews.csv')
import numpy as np

col_names =  ['Rating', 'Good' , 'Bad']
my_df  = pd.DataFrame(columns = col_names)

ratings = [1, 2, 3, 4, 5]

for rating in ratings:
    df_good = data[(data.Score == rating) & ((data['Text'].str.lower()).str.contains("good"))]
    count_Good = df_good.shape[0]
    df_bad = data[(data.Score == rating) & ((data['Text'].str.lower()).str.contains("bad"))]
    count_bad = df_bad.shape[0]
    my_df.loc[rating-1, 'Good'], my_df.loc[rating - 1, 'Rating'], my_df.loc[rating - 1, 'Bad'] = count_Good, rating, count_bad
# print (my_df)

# pd.options.display.float_format = '{:,.0f}'.format

ax = my_df[['Good','Bad']].plot(kind='bar', title = "Key word count per Rating across Dataset", figsize=(10, 10), legend=True, fontsize=12)
# ax.set_xticks(np.arange(-1,6,1))
# ax.set_xlim([0, 5])

ax.set_xlabel("Review Rating", fontsize=12)
ax.set_ylabel("Word Count", fontsize=12)
# plt.ylim(ymin=0)
ax.set_xticklabels(my_df.Rating)
plt.show()

