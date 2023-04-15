import pandas as pd
import matplotlib as plt

trade_data = pd.read_csv('API_TG.VAL.TOTL.GD.ZS_DS2_en_csv_v2_3527745.csv', skiprows=4)

# 2019 data
trade_data = trade_data[['Country Code', '2019']]

# rename
trade_data.rename(columns={'Country Code': 'ISO_A3', '2019': 'Trade'}, inplace=True)

# 
trade_data['Trade_Group'] = pd.qcut(trade_data['Trade'], 4, labels=False)

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

comparison_df = gdf.merge(trade_data, on='ISO_A3')


plot_comparison_map(comparison_df, 'Trade_Group', 'Louvain_Community', title='Comparison with Louvain Community Results (Trade)')


plot_comparison_map(comparison_df, 'Trade_Group', 'Leiden_Community', title='Comparison with Leiden Community Results (Trade)')

