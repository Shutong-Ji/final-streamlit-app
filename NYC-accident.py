from distutils.command.config import config
from tkinter.ttk import Style
import matplotlib.dates as mdate
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
from folium import plugins
import matplotlib as mpl
import numpy as np
from sklearn.cluster import KMeans


st.set_page_config( page_title="Traffic Accidents Analysis",page_icon="car",layout="wide")
st.title(':car: Traffic accident analysis in New York City, January-August 2020')
st.markdown('#### Tips for filitering widgets: ')
st.markdown('##### Borough Selector can be used for Heatmap, Scatter Plot, Location of accident Map;')
st.markdown('##### Crash Time Slider can be used for Scatter Plot, Location of accident Map;')
st.markdown('##### Month text_input can be used for Scatter Plot, Location of accident Map (by inputing 1-8);')
st.markdown('##### Map Selectbox can choose the type of map: Location of accident Map or Thermal Map')


# 读取文件 删掉没有用的列
plt.style.use('seaborn')
df = pd.read_csv('NYC Accidents 2020.csv')
df = df.sample(5000, random_state=200)
df = df.drop(['ZIP CODE', 'CROSS STREET NAME', 'OFF STREET NAME', 'NUMBER OF MOTORIST INJURED', 'CONTRIBUTING FACTOR VEHICLE 2', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
             'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'], axis=1)
df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"}, inplace=True)

# 处理空值
df['lat'].fillna(0, inplace=True)
df['lon'].fillna(0, inplace=True)
df['BOROUGH'].fillna('UNKOWN', inplace=True)
df['ON STREET NAME'].fillna('UNKOWN', inplace=True)
df['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified', inplace=True)

# 读取数据和数据预处理
df1 = pd.read_csv('NYC Accidents 2020.csv')

df1 = df1.drop(['ZIP CODE', 'CROSS STREET NAME', 'OFF STREET NAME', 'NUMBER OF MOTORIST INJURED', 'CONTRIBUTING FACTOR VEHICLE 2', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
               'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'], axis=1)

df1.LONGITUDE.fillna(0, inplace=True)
df1.LATITUDE.fillna(0, inplace=True)
df1.BOROUGH.fillna('UNKOWN', inplace=True)

# 创建一个名为 hour 的列， 代表事故发生的时间（1-24小时）


def get_hour(TIME):
    hour = TIME[0:2]
    hour = int(hour)
    return hour


df['hour'] = df['CRASH TIME'].apply(get_hour)

# 展示事故发生最多的时间
hour_statistics = df['hour'].value_counts()
sorted_hour_statistics = hour_statistics.sort_index()
max_rate = 0
time = 0
for i in range(0, 24):
    if df.hour.value_counts(normalize=True).sort_index()[i] > max_rate:
        max_rate = df.hour.value_counts(normalize=True).sort_index()[i]
        time = i

# 事故发生时间的折线图
st.subheader('Line Chart Of The Clock And Accidents:')
_, yv = np.meshgrid(np.linspace(0, 1, 210), np.linspace(0, 1, 90))
hour = [i for i in range(24)]
xlims = hour
fig, ax = plt.subplots()
ax.plot(hour, sorted_hour_statistics, 'r-', label='Accidents', linewidth=2)
extent = [xlims[0], xlims[23], min(sorted_hour_statistics), 360]
ax.imshow(yv, cmap=mpl.cm.Blues, origin='lower',
          alpha=0.45, aspect='auto', extent=extent)
ax.fill_between(hour, sorted_hour_statistics, 360, color='white')
plt.yticks(color='gray')
plt.xticks(color='gray', rotation=15)
fontdict = {"family": "Times New Roman", 'size': 12,
            'color': 'gray'} 
plt.title("Line Chart Of The Clock And Accidents:", fontdict=fontdict)
plt.xlabel("hour", fontdict=fontdict)
plt.ylabel("account value", fontdict=fontdict)
plt.ylim(0, 360)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('lightgray')
plt.xticks([i for i in range(24)])
plt.tick_params(left='off')
plt.tick_params(which='major', direction='out',
                width=0.2, length=5)  # in, out or inout
plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.3)
plt.grid(axis='x', color='lightgray', linestyle='-', linewidth=0.1)
plt.legend(loc='best', fontsize=12, frameon=False, ncol=1)
st.markdown('##### It can be found from the chart below that the number of car accidents is most likely to occur at 5 P.M., that is likely that drivers were distracted during the 5 P.M. rush hour.')
st.pyplot(fig)



# 事故发生区域图
st.subheader('Bar Chart Of The Borough Accidents:')
a = df.BOROUGH.value_counts(normalize=True).drop('UNKOWN')
fig, ax = plt.subplots()
ax = a.plot.bar()
plt.xticks(rotation=0)
st.markdown('##### The bar chart shows that, excluding accidents of unknown location, Brooklyn has the largest proportion of accidents, while Staten Island has the least. ')
st.pyplot(fig)

# 筛选事故发生的区
borough_filter = st.sidebar.multiselect(
    'Borough Selector',
    df.BOROUGH.unique(),
    df.BOROUGH.unique()
)
df = df[df.BOROUGH.isin(borough_filter)]

# 创建一个事故时间与事故街区的热图
st.subheader('HeatMap Of Accidents per Borough and Hour:')
accidents_hour_pt = df.pivot_table(
    index='BOROUGH', columns='hour', aggfunc='size')
accidents_hour_pt = accidents_hour_pt.apply(
    lambda x: x / accidents_hour_pt.max(axis=1))
fig, ax = plt.subplots(figsize=(15, 5))
plt.title('Average Number of Accidents per Borough and Hour', fontsize=14)
xLabel = range(0, 24)
yLabel = ['BRONX', 'BROOKLYN', 'MANHATTAN',
          'QUEENS', 'STATEN ISLAND', 'UNKNOWN']
ax.set_yticks(range(len(yLabel)))
ax.set_yticklabels(yLabel)
ax.set_xticks(range(len(xLabel)))
ax.set_xticklabels(xLabel)
im = ax.imshow(accidents_hour_pt, cmap=plt.cm.hot_r)
plt.colorbar(im)
st.markdown('##### The heatmap shows the relationship between time and borough.')
st.pyplot(fig)

# slider 用于过滤发生事故的时间
df = df[df.lon <= -21]
time_filter = st.sidebar.slider('Crash Time:', 0, 23, 12)
df = df[df.hour == time_filter]

# 创建一个month列，为事故发生的月份


def get_month(Month):
    month = Month[5:7]
    month = int(month)
    return month


df['month'] = df['CRASH DATE'].apply(get_month)

# 在左边创建一个表单， 用于输入发生的月份
form = st.sidebar.form("month_form")
month_filter = form.text_input(
    'Enter the number of month(enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")

# 过滤发生事故的月份
if month_filter != 'ALL':
    df = df[df.month == int(month_filter)]

# 5000个样本事故原因图
df2 = pd.read_csv('NYC Accidents 2020.csv')
df2 = df2.sample(5000, random_state=180)
df2 = df2.drop(['ZIP CODE', 'CROSS STREET NAME', 'OFF STREET NAME', 'NUMBER OF MOTORIST INJURED', 'CONTRIBUTING FACTOR VEHICLE 2', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
               'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5'], axis=1)
df2['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified', inplace=True)
a = df2['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().drop('Unspecified')
a = a.head(20)
a = a.tolist()
b = list(reversed(a))
c = ['Brakes Defective', 'Oversized Vehicle', 'Aggressive Driving/Road Rage', 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion', 'View Obstructed/Limited', 'Pavement Slippery', 'Alcohol Involvement', 'Reaction to Uninvolved Vehicle', 'Driver Inexperience',
     'Turning Improperly', 'Traffic Control Disregarded', 'Unsafe Lane Changing', 'Unsafe Speed', 'Other Vehicular', 'Passing Too Closely', 'Passing or Lane Usage Improper', 'Backing Unsafely', 'Failure to Yield Right-of-Way', 'Following Too Closely', 'Driver Inattention/Distraction']
fig, ax = plt.subplots()
plt.barh(c, b, height=0.5)
st.subheader('Bar Chart of the cause of the accident in 5,000 samples:')
st.markdown('##### As can be clearly seen from the bar chart, the main cause of accidents is distraction while driving.')
st.pyplot(fig)

# 一月发生事故的散点图, 描绘一月发生事故的大致地点与事故原因
st.subheader('Scatter Plot of the cause of the accident in 5,000 samples:')
df_1 = df
plt.figure(figsize=(7.5, 10))
sns.scatterplot(x='lon', y='lat', hue='CONTRIBUTING FACTOR VEHICLE 1',
                data=df_1, style='CONTRIBUTING FACTOR VEHICLE 1')
plt.title("Month Accidents")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('##### By taking longitude and latitude as the coordinate axis, we first made the general distribution of accident locations. Different colored dots represent different causes of accidents. It can be roughly seen from the scatter plot that most accidents are caused by distraction while driving. ')
st.pyplot()


# 用folium 规定地图的中心和缩放比例
mmap = folium.Map(location=[40.8, -73.05], zoom_start=10.2)
marker_cluster = plugins.MarkerCluster().add_to(mmap)

# 输入每个点的经纬度信息 画出第一张地图
for i in range(len(df)):
    lat = df.iloc[i][3]
    lon = df.iloc[i][4]
    color = '#FF4500'
    popu_text = """Lat : {}<br>
                Lon : {}<br>
                ON STREET NAME : {}<br>
                BOROUGH : {}<br>"""
    popu_text = popu_text.format(
        df['lat'].iloc[i], df['lon'].iloc[i], df['ON STREET NAME'].iloc[i], df['BOROUGH'].iloc[i])
    popup = folium.Popup(popu_text, min_width=200, max_width=300)
    folium.Marker(location=[lat, lon], popup=popup, radius=3,
                  color=color, fill=True).add_to(marker_cluster)

df1 = df1[~df1.isin([0])].dropna(axis=0)
x = df1[['LATITUDE', 'LONGITUDE']]
mod = KMeans(n_clusters=20, random_state=9)
y_pre = mod.fit_predict(x)
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = ['Lon', 'Lat', 'Cluster Number']
r = r.sort_values(by='Cluster Number', ascending=False)
r = r.head(9)
for i in range(len(r)):
    lat = r.iloc[i][0]
    lon = r.iloc[i][1]
    popu_text = """Lat : {}<br>
                Lon : {}<br>
                Cluster Number : {}<br>"""
    popu_text = popu_text.format(
        r['Lat'].iloc[i], r['Lon'].iloc[i], r['Cluster Number'].iloc[i])
    popup = folium.Popup(popu_text, min_width=200, max_width=300)
    folium.Marker(location=[lat, lon], popup=popup, icon=folium.Icon(
        color='red', icon='info-sign'), fill=True).add_to(mmap)


# 画事故发生的数量地图
location = df1['LOCATION'].value_counts()
count_loc = pd.DataFrame({'LOCATION': location.index, 'ValueCount': location})
count_loc.index = range(len(location))
loc = df1.groupby('LOCATION').first()
new_loc = loc.loc[:, ['LATITUDE', 'LONGITUDE', 'ON STREET NAME', 'BOROUGH']]
the_loc = pd.merge(count_loc, new_loc, on='LOCATION')
the_loc.drop(the_loc.index[1], inplace=True)
nmap = folium.Map(location=[40.75, -73.5], zoom_start=11)

for i in range(1000):
    lat = the_loc.iloc[i][2]
    lon = the_loc.iloc[i][3]
    radius = the_loc['ValueCount'].iloc[i] / 3
    if the_loc['ValueCount'].iloc[i] > 15:
        color = '#FF4500'
    else:
        color = '#008080'
    popu_text = """Lat : {}<br>
                Lon : {}<br>
                ON STREET NAME : {}<br>
                BOROUGH : {}<br>
                INCIDENTS : {}<br>"""
    popu_text = popu_text.format(the_loc['LATITUDE'].iloc[i], the_loc['LONGITUDE'].iloc[i],
                                 the_loc['ON STREET NAME'].iloc[i], the_loc['BOROUGH'].iloc[i], the_loc['ValueCount'].iloc[i])
    popup = folium.Popup(popu_text, min_width=200, max_width=300)
    folium.CircleMarker(location=[lat, lon], popup=popup,
                        radius=radius, color=color, fill=True).add_to(nmap)


# 热成像图
the_loc.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"}, inplace=True)
the_loc.lon.fillna(0, inplace=True)
the_loc.lat.fillna(0, inplace=True)
HeatMap(data=the_loc[['lat', 'lon']], radius=17).add_to(nmap)

# 选择地图
selectbox = st.sidebar.selectbox('Please choose the map type you want', [
                                 'Location of accident Map', 'Thermal Map'])
if selectbox == 'Location of accident Map':
    st.subheader('Location of accident Map')
    st.markdown('##### This map shows that locations of 5,000 samples where accidents occurred at different times. The red markers show the top ten clusters with the largest number after K-Means clustering.')
    st.markdown('##### When you zoom in the map,  you can click the blue marker to get the accident information. You can also click the red marker to get the cluster information. ')
    st.data = st_folium(mmap, width=2000)
else:
    st.subheader('Thermal Map:')
    st.markdown('##### Value-Count the location of the accident for all 70,000+ samples.')
    st.markdown('##### Radius is used as a marker for the location of accidents. The more accidents occur in the same location, the larger radius marker will be.')
    st.markdown('##### If the number of accidents in the same place is greater than 15, the radius marker will be marked as red, and the points less than 15 will be marked as green. ')
    st.markdown('##### We chose the top 1000 data  with the most accidents in the same location for the map.')
    st.markdown('##### you can click the radius marker to get the accident information.')

    st.data = st_folium(nmap, width=2000)
