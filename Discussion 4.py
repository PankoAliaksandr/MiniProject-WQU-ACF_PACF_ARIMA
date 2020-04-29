# Import libraries
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA

# Download data
df = pd.read_csv('Nikkei.csv')
df.set_index('Date', inplace = True)

# Plot original data
df.plot(legend=False, title='Original Data')
pyplot.show()

# Apply Augmented Dickey-Fuller Test to original data
adf_result_before_differencing = adfuller(df['Value'])
print 'Results of ADF test for original data:'
print('ADF Statistic: %f', adf_result_before_differencing[0])
print('p-value: %f', adf_result_before_differencing[1])
if adf_result_before_differencing[1] >= 0.05:
    print 'Fail to reject the null hypothesis (H0) at 5 % level of \
    significance. The data has a unit root and is non-stationary'
else:
    print 'Reject the null hypothesis (H0) at 5 % level of \
    significance. The data does not have a unit root and is stationary'
    
# Plot ACF
tsaplots.plot_acf(df['Value'], lags=50)
pyplot.show()

# Plot PACF
tsaplots.plot_pacf(df['Value'], lags=50)
pyplot.show()    

# Implement ARIMA(1,0,1)
model = ARIMA(df['Value'], order=(1, 0, 1))
model_fit = model.fit()
print model_fit.summary()

