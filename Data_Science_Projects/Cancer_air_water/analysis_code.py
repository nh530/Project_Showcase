import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,AgglomerativeClustering, DBSCAN
from sklearn import preprocessing
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage  
from pandas.plotting import scatter_matrix
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression


# Have to specify dtype because FIPS State and FIPS County should be strings.
# If this is not done, then python thinks its integers, which removes leading
# 0's. Note, this is for windows os.  
air = pd.read_csv('cleanAirData.csv', dtype={
        'FIPS State':object, 'FIPS County':object})
water = pd.read_csv('water_clean.csv')
cancer = pd.read_csv('cleaned_cancer.csv')
# add leading 0's to FIPS
cancer['FIPS'] = cancer['FIPS'].apply(lambda x: '{0:0>5}'.format(x))
cancer['FIPS'] = cancer['FIPS'].astype(object)
water['FIPS'] = water['FIPS'].apply(lambda x: '{0:0>5}'.format(x))
water['FIPS'] = water['FIPS'].astype(object)
# Creating a single FIPS Column.
air['FIPS'] = air['FIPS State'] + air['FIPS County']
air = air.drop(['FIPS State', 'FIPS County'], axis=1)
air['FIPS'] = air['FIPS'].str.replace("=", "")
air['FIPS'] = air['FIPS'].str.replace("\"", "")

    
##This function creates a histogram showing the frequency of max AQI values 
# for 2011-2015
def AirQualityHist(data):
    name= "Maximum AQI Values in the US by Year"
    data['Max AQI'].hist(by=data['Year'])
    plt.suptitle(
            "Histograms by year for the Maximum AQI in each county of the US")
    plt.savefig(name)

# This function creates a histogram showing the frequency of water quality 
# levels
def waterQualityHist(result):
    name = "Number of counties at different water quality levels"
    result['Value'].hist(by=result['Year'])
    plt.suptitle("Counties at different water quality levels")
    plt.savefig(name)

def cancerHist(df):
    name = "Number of counties at different water quality levels"
    df['Age-Adjusted Incidence Rate(†) - cases per 100,000'].hist(
            by=df['Category'])
    plt.suptitle("Cancer incidence rates in US Counties")
    plt.savefig(name)

waterQualityHist(water) 
AirQualityHist(air)
cancerHist(cancer)


# clustering analysis
# average water value to match the average annual cancer count
avg_value = water.groupby('FIPS', as_index=False)['Value'].mean()
# subset all type cancer
cancer_alltype = cancer.loc[cancer['Category'] == 'All']
# merge chosen columns in cancer and water avg value
cancer_water = pd.merge(cancer_alltype[['FIPS','Average Annual Count']], 
                        avg_value,on='FIPS')
### cancer data has more fips than water/avg_value
###a = cancer_alltype['FIPS']
###b = avg_value['FIPS']
#list_1 = ["a", "b", "c", "d", "e"]
#list_2 = ["a", "f", "c", "m"] 
#morecancerfips = np.setdiff1d(b,a)
avg_gooddaynorm = air.groupby('FIPS', as_index=False)['Good Days_Norm'].mean()
#c = cancer_water['FIPS']
#d = avg_gooddaynorm['FIPS']
#list_1 = ["a", "b", "c", "d", "e"]
#list_2 = ["a", "f", "c", "m"] 
#morecancerfips = np.setdiff1d(c,d)
cancer_water_air = pd.merge(cancer_water,avg_gooddaynorm,on='FIPS')

# Hierarchical Clustering
def getDendrogram(cancer_water_air):
    # Remove missing data 
    cancer_water_air = cancer_water_air.dropna()
    # normalize data
    mycancer_water_air=pd.concat([cancer_water_air['Average Annual Count'], 
                cancer_water_air['Value'], cancer_water_air['Good Days_Norm']], 
               axis=1, keys=['Average Annual Count','Value','Good Days_Norm' ])
    x = mycancer_water_air.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    normalizedDataFrame.columns = ['Average Annual Count','Value',
                                   'Good Days_Norm']
    # generate the linkage matrix
    Z = linkage(normalizedDataFrame, 'ward')
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

def Hier_clustering(cancer_water_air,k):
    # Remove missing data 
    cancer_water_air = cancer_water_air.dropna()
    # normalize data
    mycancer_water_air=pd.concat([cancer_water_air['Average Annual Count'], 
                cancer_water_air['Value'], cancer_water_air['Good Days_Norm']], 
              axis=1, keys=['Average Annual Count','Value','Good Days_Norm' ])
    x = mycancer_water_air.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    normalizedDataFrame.columns = ['Average Annual Count','Value',
                                   'Good Days_Norm']
    
    model = AgglomerativeClustering(n_clusters=k, affinity = 'euclidean', 
                                    linkage = 'ward')
    clust_labels1 = model.fit_predict(normalizedDataFrame)
    agglomerative = pd.DataFrame(clust_labels1)
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, clust_labels1)
    print("For n_clusters =", k, "The average silhouette_score is :", 
          silhouette_avg)
    #centroids = model.cluster_centers_
    pprint(dict(pd.Series(clust_labels1).value_counts()))
    #pprint(centroids)
    
    # plot the clustering in 3D graph
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(normalizedDataFrame['Value'], 
                         normalizedDataFrame['Good Days_Norm'], 
                         normalizedDataFrame['Average Annual Count'],
                         c=agglomerative[0],s=80,alpha=0.8)
    ax.set_title('Hierarchical Clustering')
    ax.set_xlabel('arsenic in water')
    ax.set_ylabel('rates of good air quality days per year')
    ax.set_zlabel('cancer rate')
    ax.set_zlim3d(0.0,0.3)
    ax.set_ylim3d(0.2,1.2)
    plt.colorbar(scatter)
    

#####
#K means Clustering 
#####
def KMeans_cluster(cancer_water_air,k):
    # Remove missing data 
    cancer_water_air = cancer_water_air.dropna()
    # normalize data
    mycancer_water_air=pd.concat([cancer_water_air['Average Annual Count'], 
                cancer_water_air['Value'], cancer_water_air['Good Days_Norm']], 
               axis=1, keys=['Average Annual Count','Value','Good Days_Norm' ])
    x = mycancer_water_air.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    normalizedDataFrame.columns = ['Average Annual Count', 'Value', 
                                   'Good Days_Norm']
    
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)

    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", 
          silhouette_avg)
    centroids = kmeans.cluster_centers_
    pprint(dict(pd.Series(cluster_labels).value_counts()))
    pprint(centroids)
    
    # plot the clustering in 3D graph
    kmeans = pd.DataFrame(cluster_labels)
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(normalizedDataFrame['Value'], 
                         normalizedDataFrame['Good Days_Norm'], 
                         normalizedDataFrame['Average Annual Count'],
                         c=kmeans[0],s=80,alpha=0.8)
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('arsenic in water')
    ax.set_ylabel('rates of good air quality days per year')
    ax.set_zlabel('cancer rate')
    ax.set_zlim3d(0.0,0.3)
    ax.set_ylim3d(0.2,1.2)
    plt.colorbar(scatter)

# dbscan clustering
def DBSCAN_clustering(cancer_water_air):
    # Remove missing data 
    cancer_water_air = cancer_water_air.dropna()
    # normalize data
    mycancer_water_air=pd.concat([cancer_water_air['Average Annual Count'], 
                cancer_water_air['Value'], cancer_water_air['Good Days_Norm']], 
              axis=1, keys=['Average Annual Count','Value','Good Days_Norm' ])
    x = mycancer_water_air.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    normalizedDataFrame.columns = ['Average Annual Count',
                                   'Value', 'Good Days_Norm']
    
    
    model = DBSCAN(eps=0.05, metric='euclidean', min_samples=5)
    clust_labels2 = model.fit_predict(normalizedDataFrame)
    dbscan = pd.DataFrame(clust_labels2)
    
    
    
    # Determine if the clustering is good
    silhouette_avg = silhouette_score(normalizedDataFrame, clust_labels2)
    print("The average silhouette_score is :", silhouette_avg)
    #centroids = model.cluster_centers_
    pprint(dict(pd.Series(clust_labels2).value_counts()))
    #pprint(centroids)
    
    # plot the clustering in 3D graph
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(normalizedDataFrame['Value'],normalizedDataFrame[
            'Good Days_Norm'],normalizedDataFrame['Average Annual Count'],
                         c=dbscan[0],s=80,alpha=0.8)
    ax.set_title('DBSCAN Clustering')
    ax.set_xlabel('arsenic in water')
    ax.set_ylabel('rates of good air quality days per year')
    ax.set_zlabel('cancer rate')
    ax.set_zlim3d(0.0,0.3)
    ax.set_ylim3d(0.2,1.2)
    plt.colorbar(scatter)
    
    
## call functions
getDendrogram(cancer_water_air)
Hier_clustering(cancer_water_air,4)
KMeans_cluster(cancer_water_air,4)
DBSCAN_clustering(cancer_water_air)

# CORRELATION
# Calculate 5-year averages for Input data.  Merging cancer data to air
# and water reduced the number of data entries.  The counties in each data
# set does not seem to be overlapping.  
median = air.groupby('FIPS', as_index=False)['Median AQI'].mean()
pm = air.groupby('FIPS', as_index=False)['Days PM2.5_Norm'].mean()
so2 = air.groupby('FIPS', as_index=False)['Days SO2_Norm'].mean()
co = air.groupby('FIPS', as_index=False)['Days CO_Norm'].mean()
no2 = air.groupby('FIPS', as_index=False)['Days NO2_Norm'].mean()
ozone = air.groupby('FIPS', as_index=False)['Days Ozone_Norm'].mean()
ar = water.groupby('FIPS', as_index=False)['Value'].mean()
#data = pd.merge(median, pm, on='FIPS')
data = pd.merge(median, ar, on='FIPS')
data = pd.merge(data, ozone, on='FIPS')
data = pd.merge(data, pm, on='FIPS')
#data = pd.merge(data, so2, on='FIPS')
#data = pd.merge(data, co, on='FIPS')
#data = pd.merge(data, no2, on='FIPS')
#data = pd.merge(data, ar, on='FIPS')

# Class label.
labels = cancer[['FIPS', 'Incidence Quant']][cancer['Category']=='All']

data = pd.merge(data, labels, on='FIPS')
data = data.drop('FIPS', axis=1)
data = data.dropna()

scatter_matrix(data)
plt.show()
# calculate the correlation matrix
corr = data.corr() # contains the matrix of correlation coefficients
# plot the heatmap - creates a colorful matrix 
plt.figure()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            cmap='GnBu')
plt.show()


# hypothesis 1(t-test & linear regression)
def hypo1(cancer, water):
    
    cancerSubset =cancer[cancer['Group']=='Total']    
    # create aa new data frame named cancer_new
    cancer_new = pd.DataFrame(columns = ["FIPS", "cancer_incidence"]) 
    cancer_new['FIPS']=cancerSubset['FIPS']
    cancer_new['cancer_incidence']=cancerSubset[
            'Age-Adjusted Incidence Rate(†) - cases per 100,000']
    cancer_new=cancer_new.dropna()  # drop cancer_incidence=NaN


    result_prep = water.groupby(['FIPS','County','State'])['Value'].mean()
    result_prep = result_prep.to_frame().reset_index()
    
    # Combine tables by FIPS
    water_cancer = pd.merge(result_prep, cancer_new,on="FIPS",how='inner')
    water_cancer['cancer_incidence']=pd.to_numeric(water_cancer[
            'cancer_incidence'])
    water_cancer['Value']=pd.to_numeric(water_cancer['Value'])
    del water_cancer['FIPS']
    del water_cancer['County']
    del water_cancer['State']
    #Now, here is a table 'water_cancer' that only contain
    # the value of arsenic in water and the cancer incident
    
    # 1.t-test and linear regression
    # H0: counties with arsenic at 'non-detected' level have the same cancer 
    # rate with arsenic detected 
    water_cancer1=water_cancer[water_cancer['Value']<=1]
    water_cancer2=water_cancer[water_cancer['Value']>1]
    # cancer incedence of water at 'non-detected' level
    cancer_gw=water_cancer1['cancer_incidence']
    # cancer incedence of water at 'non-detected' level
    cancer_bw=water_cancer2['cancer_incidence']
    
    p_value=stats.ttest_ind(cancer_gw,cancer_bw, equal_var = False)
    print(p_value)
    # Here, the pvalue=0.000283, which is smaller than 0.05. 
    # So counties with arsenic at 'non-detected' level have different cancer 
    # rate with arsenic at 'detectable' level 
    
    
    # 2.Linear Regression: y=ax+b, where the x= result['value'] and 
    # y=cancer['cancer_incidence']
    x= water_cancer.iloc[:, :-1].values   # x is the arsenic_content
    y = water_cancer.iloc[:, 1].values   #y is cancer incidence
    reg = LinearRegression().fit(x, y) # fit the model
    
    print('intercept is ',reg.intercept_)
    print('coefficient is ',reg.coef_)
    #the intercept in the regression is 456.85 and the slope is -4.91
    # so the linear regression is cancer_incidence=-4.91*arsenic_content+456.85
    
    plt.scatter(x, y, facecolor='None', edgecolor='k', alpha=0.3)
    plt.title("prediction of cancer incidence from water quality")
    plt.xlabel("arsenic content (µg/L)")
    plt.ylabel("cancer incidence(/100,000)")

    plt.plot(x, reg.predict(x.reshape(-1,1)), color='red')
    plt.show()

hypo1(cancer, water)


# This function creates a histogram showing the frequency of cancer incidence
# by county/group
def cancerHist(result):
    name= "Cancer Incidence Rates by Group Histogram"
    result['Age-Adjusted Incidence Rate(†) - cases per 100,000'].hist(
                                            by=result['Group'])
    plt.suptitle("Number of counties at different Cancer Incidence Rates")
    plt.savefig(name)

cancerHist(cancer)










