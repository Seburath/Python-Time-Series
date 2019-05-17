print('INIT!')
x = open('ile.csv')
x = list(x)
for i in x:
    y = i.split(',')
    if y[0] == 'Colombia ': 
        y[14] = y[14][:2]
        print(y)
        
import pandas as pd
import matplotlib.pyplot as plt
df  = pd.read_csv("ile.csv")
df.plot()  # plots all columns against index
df.plot(kind='scatter',x='x',y='y') # scatter plot
df.plot(kind='density')  # estimate density function
# df.plot(kind='hist')  # histogram
df.show()
