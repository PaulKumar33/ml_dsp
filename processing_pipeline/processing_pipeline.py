import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_excel("./20210620_tx2rx_screening.xlsx", "status")
#set the label
df['label'] = df['rxbist_ber'] <= 1e-4

#run correlation
df_correlation = df.drop(columns=["label"])
v = df_correlation.std() > 0.3
df_correlation = df_correlation.loc[:, v.reindex(df_correlation.columns, axis=1, fill_value=False)]
corrs = df_correlation.corr()
corrs_bist = abs(corrs['rxbist_ber'])

corrs_bist = corrs_bist.sort_values(ascending=False).drop(['rxbist_ber'])
print(corrs_bist)

#take the first bunch of correlation values
top_correlations_index = list(corrs_bist.index)[0:20]
df_top_corrs = df[top_correlations_index]
df_top_corrs['label'] = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(df_top_corrs.drop(columns=["label"]),
                                                    df_top_corrs["label"],
                                                    test_size=0.3, random_state=0)


cols = X_train.columns
scaler = RobustScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)
X_train, X_test = pd.DataFrame(X_train, columns=[cols]), pd.DataFrame(X_test, columns=[cols])

rfc = RandomForestClassifier(random_state=0, n_estimators=100)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)

print("After 100 trees, the accuracy is: {}".format(accuracy_score(Y_test, y_pred)))
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(list(feature_scores))

f, ax = plt.subplots(figsize=(30, 24))
fig_df = pd.DataFrame(dict(scores=list(feature_scores), labels=
                          list(cols)))
ax = sns.barplot(data=fig_df, x='scores', y='labels')
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(list(cols))
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()


