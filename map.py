import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


from sklearn.metrics import adjusted_rand_score



# read csv
community_df = pd.read_csv('C:/Users/Bill/Desktop/499country/community_detection_results.csv')

countries_shapefile = gpd.read_file('C:/Users/Bill/Desktop/499country/ne_50m_admin_0_countries.shp')

# merge
gdf = countries_shapefile.merge(community_df, left_on='ISO_A3', right_on='Country')
print(gdf)

def plot_community_map(gdf, community_col, cmap='plasma', title='Community Map'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    gdf.plot(column=community_col, cmap=cmap, linewidth=0.8, edgecolor='0.8', legend=True, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()

#louvain
plot_community_map(gdf, 'Louvain_Community', title='Louvain Community Map')

#leiden
plot_community_map(gdf, 'Leiden_Community', title='Leiden Community Map')



def plot_comparison_map(comparison_df, real_life_col, community_col, title='Comparison Map'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    comparison_df.plot(column=real_life_col, cmap='plasma', linewidth=0.8, edgecolor='0.8', legend=True, ax=ax1)
    ax1.set_title('Real Life Relations')

    comparison_df.plot(column=community_col, cmap='plasma', linewidth=0.8, edgecolor='0.8', legend=True, ax=ax2)
    ax2.set_title('Community Detection Results')

    for ax in [ax1, ax2]:
        ax.set_axis_off()

    plt.suptitle(title)
    plt.show()

# read world bank data
trade_data = pd.read_csv('C:/Users/Bill/Desktop/499country/API_TG.VAL.TOTL.GD.ZS_DS2_en_csv_v2_4901999.csv', skiprows=4)

# keep 2019
trade_data = trade_data[['Country Code', '2019']]

# rename
trade_data.rename(columns={'Country Code': 'ISO_A3', '2019': 'Trade'}, inplace=True)

# group
trade_data['Trade_Group'] = pd.qcut(trade_data['Trade'], 4, labels=False)


comparison_df = gdf.merge(trade_data, on='ISO_A3')


plot_comparison_map(comparison_df, 'Trade_Group', 'Louvain_Community', title='Comparison with Louvain Community Results (Trade)')


plot_comparison_map(comparison_df, 'Trade_Group', 'Leiden_Community', title='Comparison with Leiden Community Results (Trade)')






comparison_df_clean = comparison_df.dropna(subset=['Trade_Group', 'Louvain_Community', 'Leiden_Community'])


louvain_ari = adjusted_rand_score(comparison_df_clean['Trade_Group'], comparison_df_clean['Louvain_Community'])
leiden_ari = adjusted_rand_score(comparison_df_clean['Trade_Group'], comparison_df_clean['Leiden_Community'])


plot_comparison_map(comparison_df_clean, 'Trade_Group', 'Louvain_Community', title=f'Comparison with Louvain Community Results (Trade) - ARI: {louvain_ari:.2f}')

plot_comparison_map(comparison_df_clean, 'Trade_Group', 'Leiden_Community', title=f'Comparison with Leiden Community Results (Trade) - ARI: {leiden_ari:.2f}')

