import pandas as pd
from sklearn.linear_model import LinearRegression
df_sales = pd.read_csv('data/sales.csv', header=1)
print(df_sales.columns)
df_country = df_sales[['Country ']]
df_pop_sales = df_sales[['Pop (millions)', 'Computer Sales']]

df_sales = df_sales.drop(columns=['Country ', 'Pop (millions)', 'Computer Sales'])

print(df_sales)
print(df_sales.dtypes)

X = df_sales.drop(columns=['Sales Capita'])
X['GNP per head'] = X['GNP per head'].replace('[\$\,\.]',"",regex=True).astype(float)

y = df_sales[['Sales Capita']]
y['Sales Capita'] = y['Sales Capita'].replace('[\$\,\.]',"",regex=True).astype(float)








def train_lr(X, y, threshold):
    lr = LinearRegression()
    lr.fit(X,y)

    print(f'R2 is {lr.score(X, y)}')
    print(f'Intercept is {lr.intercept_}')
    print(f'Coef is {lr.coef_}')

    y_pred = lr.predict(X)

    df_y_pred = pd.DataFrame(y_pred, columns = ['Sales Pred'])
    print(type (df_y_pred))
    y['Sales Pred'] = df_y_pred['Sales Pred']
    y['Residuals'] = y['Sales Pred'] - y['Sales Capita']

    y['Std Res'] = y['Residuals']/pd.DataFrame.std(y['Residuals'] )
    index_remove = y.loc[abs(y['Std Res']) >= threshold].index

    #while index_remove !=[]:
    #    y = y.drop(index_remove)
    #    y['Std Res'] = y['Residuals']/pd.DataFrame.std(y['Residuals'] )
    #    index_remove = y.loc[abs(y['Std Res']) >= threshold].index


    y = y.drop(columns = ['Sales Pred', 'Residuals', 'Std Res'])
    X = X.drop(index_remove)
    print(y)




train_lr(X, y, 2.0)
train_lr.fit(X, y)



