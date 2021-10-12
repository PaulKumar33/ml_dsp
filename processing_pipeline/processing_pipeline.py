import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
from patsy import dmatrices

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_blobs


df = pd.read_excel("./20210620_tx2rx_screening.xlsx", "status")
#set the label
df['label'] = df['rxbist_ber'] <= 1e-4
df = df.drop(df[df.rxbist_ber > 1E-1].index)

print(df.head())

###first do an correlation regression curious at the results

df_reg = df.drop(columns=['label'])
Xtr, XTr, Ytr, YTr = train_test_split(df_reg.drop(columns=["rxbist_ber"]), df_reg['rxbist_ber'],
                                      test_size=0.3, random_state=0)

#now that we have our test data, lets do a regression and find stat signif.
f, pval = f_regression(Xtr, Ytr)
results = pd.DataFrame(dict(label=list(Xtr.columns), significance = pval))
results = results.sort_values(by='significance', ascending=True)

print("The results of correlation regression")
print(results)
labels = list(results['label'])
p=len(labels)
denrogram = sch.dendrogram(sch.linkage(Xtr, method="ward"),p=p,
               truncate_mode="lastp",no_plot=True)
print(denrogram['leaves'])
temp = {denrogram["leaves"][ii]: labels[ii] for ii in range(len(denrogram["leaves"]))}
def llf(x):
    return "{}".format(temp[x])

sch.dendrogram(sch.linkage(Xtr, method="ward"),
               p=p,
               truncate_mode="lastp",
               leaf_label_func=llf,
               leaf_rotation=60,
               leaf_font_size=12.,
               show_contracted=True)

#plt.show()

#run correlation
df_correlation = df.drop(columns=["label"])
print(df_correlation)
v = df_correlation.std() > 0.0000001
df_correlation = df_correlation.loc[:, v.reindex(df_correlation.columns, axis=1, fill_value=False)]

corrs = df_correlation.corr()
corrs_bist = abs(corrs['rxbist_ber'])

corrs_bist = corrs_bist.sort_values(ascending=False).drop(['rxbist_ber'])

#take the first bunch of correlation values
top_correlations_index = list(corrs_bist.index)[0:20]
df_top_corrs = df[top_correlations_index]
print("top correlations")
print(df_top_corrs)
#now for each of the top correlations
r,c = 5 ,4
fig, ax = plt.subplots(r,c)
fig.set_figheight(15)
fig.set_figwidth(15)


def get_line(p, x):
    x_min, x_max = np.min(x), np.max(x)
    x_arr = np.linspace(x_min, x_max, len(x))
    l = [x_arr[i]*p[1]+p[0] for i in range(len(x_arr))]
    return ([x_arr[i]*p[1]+p[0] for i in range(len(x_arr))], x_arr)

n = list(df_top_corrs.columns)
#build the regresion string
regress_str = ""
for i in range(len(n)):
    if(i == len(n)-1):
        regress_str+=n[i]
        break
    regress_str += "{} + ".format(n[i])

regress_str = "{0} ~ ".format("rxbist_ber")+regress_str
for i in range(r):
    for j in range(c):
        #X, Y = df_top_corrs[n[i * 5 + j]], df["rxbist_ber"]
        if(i*5 + j >= len(n)):
            continue
        ax[i,j].scatter(df_top_corrs[n[i*5 + j]], np.log10(df["rxbist_ber"]))

        #fit linear regression
        t = df_top_corrs
        t["rxbist_ber"] = np.log10(df["rxbist_ber"])
        Y, X = dmatrices(regress_str, data=t, return_type='dataframe')
        ols = sm.OLS(Y,sm.add_constant(X[n[i*5 + j]]))
        res = ols.fit()

        y_hat, x_arr = get_line(list(res.params), list(X[n[i*5 + j]]))
        ax[i,j].plot(x_arr, y_hat, c='red')
        ax[i, j].set_title("{0} vs {1}".format(n[i * 5 + j], "rxbist_ber"))

        #now let so do a smoothed scatter
        x_min, x_max = np.min(X[n[5*i+j]]), np.max(X[n[5*i+j]])
        divisor = (x_max - x_min)/5
        x_smooth, y_smooth = [], []

        for k in range(5):
            top, bot = x_min+(k+1)*divisor, x_min+k*(divisor)
            #cond = X.loc[(X[n[5*i+j]]>=b) & (X[n[5*i+j]]<=t)]
            cond = t[(t[n[5*i+j]]>=bot)&(t[n[5*i+j]]<=top)]
            Y_pts = np.mean(cond["rxbist_ber"])
            x_smooth.append((bot+top)/2)
            y_smooth.append(Y_pts)
        ax[i, j].scatter(x_smooth, y_smooth, c='orange')


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


