import pandas as pd
import re
import numpy as np
import os
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

#######################################################
# CHATGPT3 referenced for data preprocessing and plotting
#######################################################

#directory_path = r"C:\Users\gohar\OneDrive - Georgia Institute of Technology\Project ISYE 6740"
#os.chdir(directory_path)





pd.set_option('display.max_columns', 55)
random.seed(2)
np.random.seed(2)


allowed_positions = ['WR', 'RB', 'TE', 'QB']

file_paths = [
    r"data\FantasyPros_2016_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2017_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2018_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2019_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2020_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2021_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2022_Overall_ADP_Rankings.csv",
    r"data\FantasyPros_2023_Overall_ADP_Rankings.csv",
]

# dictionary to store DataFrames
adp = {}

# Load each file into a DataFrame
for file_path in file_paths:
    # Extract the year from the file name
    year = file_path.split("_")[1]
    # Read CSV into DataFrame and store in the dictionary
    adp[year] = pd.read_csv(file_path)
    adp[year].drop(columns=["Rank", 'Team', 'Bye'], inplace=True)
    adp[year]["POS"] = adp[year]["POS"].astype(str).str[:2]
for key,value in adp.items():
    adp[key]['Year'] = key


excel_file_path = "C:\\Users\\gohar\\Downloads\\adpFantasty.xlsx"
# Export the DataFrame to Excel
#adp['2017'].to_excel(excel_file_path, index=False)

file_paths_points = [
    r"data\FantasyPros_Fantasy_Football_Points_PPR.csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (1).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (2).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (3).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (4).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (5).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (6).csv",
    r"data\FantasyPros_Fantasy_Football_Points_PPR (7).csv",
]

# dictionary to store DataFrames
points = {}
# Load each file into a DataFrame and name it accordingly
for i, file_path_points in enumerate(file_paths_points):
    # Extract the year from the file name
    year = 2016 + i
    # Read CSV into DataFrame and store in the dictionary
    points[f"{year}-total-points"] = pd.read_csv(file_path_points)

# Specify the columns to keep
selected_columns = ['Player', 'Pos', 'AVG', 'TTL', 'Year']
for key,value in points.items():
    points[key]['Year'] = key[:4]
    points[key] = points[key][selected_columns]
    points[key].columns = ['Player', 'Pos', 'AVG', 'Total', 'Year']

excel_file_path = "C:\\Users\\gohar\\Downloads\\pointsFantasty.xlsx"
# Export the DataFrame to Excel
#points["2017-total-points"].to_excel(excel_file_path, index=False)

file_paths_stats = [
    r"data\FantasyPros_Fantasy_Football_Statistics_RB.csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR.csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE.csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB.csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (1).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (1).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (1).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (1).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (2).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (2).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (2).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (2).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (3).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (3).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (3).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (3).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (4).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (4).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (4).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (4).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (5).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (5).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (5).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (5).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_QB (6).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_RB (6).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_WR (6).csv",
    r"data\FantasyPros_Fantasy_Football_Statistics_TE (6).csv",
]

# dictionary to store DataFrames
stats = {}

# Load each file into a DataFrame and name it accordingly
for i, file_path_stats in enumerate(file_paths_stats):
    # Determine the year based on the set of four files
    year = 2016 + (i // 4)
    # Determine the position from the file name
    position = file_path_stats.split("_")[-1].split(".")[0].split(" (")[0]
    # Read CSV into DataFrame and store in the dictionary
    df = pd.read_csv(file_path_stats)
    df['Position'] = position
    df.drop(columns=['Rank',"ROST"], inplace=True)
    if position == "RB":
        df.columns = ['Player', 'Rushing_ATT', 'Rushing_YDS', 'Rushing_Y/A', 'Rushing_LG',
        'Rushing_20+', 'Rushing_TD', 'REC', 'TGT', 'Receiving_YDS', 'Receiving_Y/R',
        'Receiving_TD','FL', 'G', 'FPTS', 'FPTS/G','Position']
    elif position in ["WR","TE"]:
        df.columns = ['Player', 'REC', 'TGT', 'Receiving_YDS', 'Receiving_Y/R',
        'Receiving_LG', 'Receiving_20+', 'Receiving_TD',
        'Rushing_ATT', 'Rushing_YDS', 'Rushing_TD',
        'FL', 'G', 'FPTS', 'FPTS/G','Position']
    else:
        df.columns = ['Player', 'CMP', 'ATT', 'CMP_%', 'Passing_YDS', 'Passing_Y/A',
    'Passing_TD', 'INT', 'SACKS', 'Rushing_ATT', 'Rushing_YDS', 'Rushing_TD',
    'FL', 'G', 'FPTS', 'FPTS/G','Position']
    df['Player'] = df['Player'].astype(str).str[:-5]
    stats[f"{year}-{position}"] = df


# Create a list to store the merged DataFrames
merged_stats = []
# Iterate through the years
for year in range(2016, 2023):
    # Create keys for the DataFrames in dfs_stats
    rb_key = f"{year}-RB"
    wr_key = f"{year}-WR"
    te_key = f"{year}-TE"
    qb_key = f"{year}-QB"
    # Merge DataFrames for each position
    # Create a list of DataFrames to concatenate
    dfs_to_concat = [
        stats[rb_key],
        stats[wr_key],
        stats[te_key],
        stats[qb_key]]
    # Use concat to append DataFrames along the index
    merged_df = pd.concat(dfs_to_concat, axis=0, join='outer', sort=False)
    merged_df['Year'] = year
    # Append the merged DataFrame to the list
    merged_stats.append(merged_df)
excel_file_path = "C:\\Users\\gohar\\Downloads\\statsFantasty.xlsx"
# Export the DataFrame to Excel
#merged_stats[0].to_excel(excel_file_path, index=False)



base_path = r'data\{}_top300.txt'
# Initialize lists to store data
all_dfs = []
# Iterate over the years (e.g., from 2017 to 2022)
for year in range(2017, 2024):
    file_path = base_path.format(year)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store data
    overall_ranking = []
    position_rank = []
    names = []
    positions = []
    # Loop through each line and extract information using regular expressions
    for line in lines:
        matches = re.findall(r'(\d+)\. \((\w+)(\d+)\) (.*?), \w+ \$(\d+) (\d+)', line)

        for match in matches:
            overall, position, position_ranking, name, _, _ = match
            overall_ranking.append(int(overall))
            names.append(name)
            positions.append(position[0:2])
        matches = re.findall(r'\(([A-Za-z]*)(\d+)\)', line)
        for match in matches:
            position_rank.append(int(match[1]))
    # Create a DataFrame for the current year
    df = pd.DataFrame({
        'Pre-season_Overall_Rank': overall_ranking,
        'Pre-season_Position_Rank': position_rank,
        'Player': names,
        'Year': year,
        'Position': positions
    })
    # Append the DataFrame to the list
    all_dfs.append(df)


espnrank = pd.concat(all_dfs, ignore_index=True)
# Filter rows based on allowed positions
espnrank  = espnrank[espnrank['Position'].isin(allowed_positions)]

# Dictionary mapping old names to new names
name_mapping = {
    "Melvin Gordon": "Melvin Gordon III",
    "Willie Snead": "Willie Snead IV",
    "Todd Gurley": "Todd Gurley II",
    "Mohamed Sanu": "Mohamed Sanu Sr.",
    "Will Fuller V": "William Fuller V",
    "Paul Richardson": "Paul Richardson Jr.",
    "Wayne Gallman": "Wayne Gallman Jr.",
    "De'Angelo Henderson": "De'Angelo Henderson Sr.",
    "Ty Montgomery": "Ty Montgomery II",
    "Robby Anderson": "NA",
    "Allen Robinson": "Allen Robinson II",
    "Phillip Dorsett": "Phillip Dorsett II",
    "Mark Ingram": "Mark Ingram II",
    "Malcolm Mitchell": "NA",
    "Rob Kelley": "NA",
    "D.J. Chark": "DJ Chark Jr.",
    "Keelan Cole": "Keelan Cole Sr.",
    "Mitchell Trubisky": "Mitch Trubisky",
    "Patrick Mahomes": "Patrick Mahomes II",
    "John Kelly": "John Kelly Jr.",
    "John Ross III": "John Ross",
    "Duke Johnson": "Duke Johnson Jr.",
    "Dwayne Haskins Jr.": "Dwayne Haskins",
    "Mecole Hardman": "Mecole Hardman Jr.",
    "Chris Herndon": "Chris Herndon IV",
    "Scotty Miller": "Scott Miller",
    "Jakeem Grant": "Jakeem Grant Sr.",
    "Robby Anderson": "Robert Davis",
    "Gabriel Davis": "Gabe Davis",
    "Josh Palmer": "Joshua Palmer",
    "D'Wayne Eskridge": "Dee Eskridge",
    "Ken Walker III": "Kenneth Walker III",
    "Robbie Anderson": "Robbie Chosen",
    "Cedrick Wilson": "Cedrick Wilson Jr.",
    "Nathaniel Dell": "Neal Sterling"
}
espnrank["Player"] = espnrank["Player"].replace(name_mapping)

excel_file_path = "C:\\Users\\gohar\\Downloads\\espnrankFantasty.xlsx"
# Export the DataFrame to Excel
#espnrank.to_excel(excel_file_path, index=False)


years = list(range(2016, 2024))
# Create a list of DataFrames from the dictionary values
dfs_to_concat = [adp[str(year)] for year in years]
# Concatenate DataFrames along rows
merged_df = pd.concat(dfs_to_concat, ignore_index=True)
merged_df.rename(columns={'POS': 'Position'}, inplace=True)

# Filter rows based on allowed positions
merged_df = merged_df[merged_df['Position'].isin(allowed_positions)]
excel_file_path = "C:\\Users\\gohar\\Downloads\\adpFantasty.xlsx"
# Export the DataFrame to Excel
#merged_df.to_excel(excel_file_path, index=False)
adp = merged_df

keys = list(points.keys())
# Create a list of DataFrames from the dictionary values
dfs_to_concat = [points[key] for key in keys]
# Concatenate DataFrames along rows
merged_df = pd.concat(dfs_to_concat, ignore_index=True)
merged_df.rename(columns={'Pos': 'Position'}, inplace=True)

# Filter rows based on allowed positions
merged_df = merged_df[merged_df['Position'].isin(allowed_positions)]
excel_file_path = "C:\\Users\\gohar\\Downloads\\pointsFantasty.xlsx"
#merged_df.to_excel(excel_file_path, index=False)
points = merged_df
merged_df = pd.concat(merged_stats, ignore_index=True)
# Filter rows based on allowed positions
merged_df = merged_df[merged_df['Position'].isin(allowed_positions)]
excel_file_path = "C:\\Users\\gohar\\Downloads\\statsFantasty.xlsx"
#merged_df.to_excel(excel_file_path, index=False)
stats = merged_df
# accidentally captured duplicate information
columns_to_drop = ['FPTS', 'FPTS/G','SACKS']
stats.drop(columns=columns_to_drop, inplace=True)


# Trim spaces for 'adp' DataFrame
adp = adp.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
# Trim spaces for 'points' DataFrame
points = points.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
# Trim spaces for 'stats' DataFrame
stats = stats.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
# Trim spaces for 'espnrank' DataFrame
espnrank = espnrank.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

points['Year'] = points['Year'].astype(int)
stats['Year'] = stats['Year'].astype(int)
espnrank['Year'] = espnrank['Year'].astype(int)
adp['Year'] = adp['Year'].astype(int)
adp.rename(columns={'AVG': 'AVG_ADP'}, inplace=True)

# Adjust the "Year" column in points for the first join (last year's points)
previous_year_points = points.copy()
previous_year_points['Year'] = previous_year_points['Year'] + 1
# First join with last year's points
merged_df = pd.merge(points, previous_year_points, left_on=['Player', 'Position', 'Year'], right_on=['Player', 'Position', 'Year'], how='left', suffixes=('','_Previous_Year'))
merged_df = merged_df[merged_df['Year'] > 2016] # 2017 will be our starting year for analysis
merged_df.drop(columns=['AVG'], inplace=True) # We dont want to predict AVG points. We want total points
# Attach this year's adp
merged_df = pd.merge(merged_df, adp, left_on=['Player', 'Position', 'Year'], right_on=['Player', 'Position', 'Year'], how='left',suffixes=('',''))
# attach last years adp
adp['Year'] +=1
merged_df = pd.merge(merged_df, adp, left_on=['Player', 'Position', 'Year'], right_on=['Player', 'Position', 'Year'], how='left', suffixes=('','_Previous_Year'))
adp['Year'] -=1
# Attach rank
merged_df = pd.merge(merged_df, espnrank, left_on=['Player', 'Position', 'Year'], right_on=['Player', 'Position', 'Year'], how='left')
# Attach Stats
stats['Year'] +=1
merged_df = pd.merge(merged_df, stats, left_on=['Player', 'Position', 'Year'], right_on=['Player', 'Position', 'Year'], how='left')
stats['Year'] -=1

# Handle missing values
columns_to_replace = ['AVG_Previous_Year', 'Total_Previous_Year']
merged_df['Rookie'] = np.where(merged_df['Total_Previous_Year'].isna(), 1, 0)
merged_df[columns_to_replace] = merged_df[columns_to_replace].fillna(0)
merged_df['Undrafted'] = np.where(merged_df['AVG_ADP'].isna(), 1, 0)
merged_df['AVG_ADP'].fillna(1000, inplace=True)
columns_to_replace = ['ESPN', 'Sleeper', 'NFL', 'RTSports', 'FFC']
for column in columns_to_replace:
    merged_df[column].fillna(merged_df['AVG_ADP'], inplace=True)


# previous year's na adp should be rookie if player did not play last year. otherwise not ranked
merged_df['PreviousYearUndrafted'] = np.where(
    (merged_df['AVG_ADP_Previous_Year'].isna()) & (merged_df['Rookie'] == 0),
    1,
    0
)

merged_df['AVG_ADP_Previous_Year'].fillna(1000, inplace=True)
columns_to_replace_na = ['ESPN_Previous_Year', 'Sleeper_Previous_Year', 'NFL_Previous_Year', 'RTSports_Previous_Year', 'FFC_Previous_Year']
# Deal with na values
for column in columns_to_replace_na:
    merged_df[column].fillna(merged_df['AVG_ADP_Previous_Year'], inplace=True)

columns_to_replace_na = ['Pre-season_Overall_Rank', 'Pre-season_Position_Rank']
merged_df['Unranked'] = np.where(merged_df['Pre-season_Overall_Rank'].isna(), 1, 0)
merged_df[columns_to_replace_na] = merged_df[columns_to_replace_na].fillna(400)

columns_to_replace = [
    'Rushing_ATT', 'Rushing_YDS', 'Rushing_Y/A', 'Rushing_LG', 'Rushing_20+',
    'Rushing_TD', 'REC', 'TGT', 'Receiving_YDS', 'Receiving_Y/R', 'Receiving_TD',
    'FL', 'G', 'Receiving_LG', 'Receiving_20+', 'CMP', 'ATT', 'CMP_%', 'Passing_YDS',
    'Passing_Y/A', 'Passing_TD', 'INT']
merged_df[columns_to_replace] = merged_df[columns_to_replace].fillna(0)

columns_to_check_duplicates = ["Player", "Position", "Year"]
merged_df = merged_df.drop_duplicates(subset=columns_to_check_duplicates, keep="first")
# create binary variables for the "Position" column
position_dummies = pd.get_dummies(merged_df['Position'], prefix='Position')
# Concatenate the dummy variables with the original DataFrame
merged_df = pd.concat([merged_df, position_dummies], axis=1)
# Drop the original "Position" column since it's now represented by the dummy variables
merged_df = merged_df.drop('Position', axis=1)
print(merged_df.isna().sum())
print(merged_df.shape)
merged_df = merged_df.loc[~((merged_df['Pre-season_Overall_Rank'] == 400) & (merged_df['Total'] < 40))]
print(merged_df.shape)

# List of columns to convert to float
columns_to_convert = [
    'Total', 'Year', 'AVG_Previous_Year', 'Total_Previous_Year', 'ESPN', 'Sleeper',
    'NFL', 'RTSports', 'FFC', 'AVG_ADP', 'ESPN_Previous_Year', 'Sleeper_Previous_Year',
    'NFL_Previous_Year', 'RTSports_Previous_Year', 'FFC_Previous_Year',
    'AVG_ADP_Previous_Year', 'Pre-season_Overall_Rank', 'Pre-season_Position_Rank',
    'Rushing_ATT', 'Rushing_YDS', 'Rushing_Y/A', 'Rushing_LG', 'Rushing_20+',
    'Rushing_TD', 'REC', 'TGT', 'Receiving_YDS', 'Receiving_Y/R', 'Receiving_TD',
    'FL', 'G', 'Receiving_LG', 'Receiving_20+', 'CMP', 'ATT', 'CMP_%',
    'Passing_YDS', 'Passing_Y/A', 'Passing_TD', 'INT','Passing_YDS'
]
# Convert specified columns to float
for column in columns_to_convert:
    merged_df[column] = pd.to_numeric(merged_df[column].replace({'[^0-9.]': ''}, regex=True))

# List of boolean columns to convert to integer
boolean_columns_to_convert = ['Position_QB', 'Position_RB', 'Position_TE', 'Position_WR']
# Convert boolean columns to integer
merged_df[boolean_columns_to_convert] = merged_df[boolean_columns_to_convert].astype(int)
excel_file_path = "C:\\Users\\gohar\\Downloads\\FinalProduct.xlsx"
#merged_df.to_excel(excel_file_path, index=False)


# Train and Testing data
random_years = merged_df['Year'].unique()
selected_years = np.random.choice(random_years, size=320, replace=True) #size=320
Test = pd.DataFrame()
merged_df.sort_values(by='Total', ascending=False, inplace=True)
for finish, year in enumerate(selected_years):
    year_data = merged_df[merged_df['Year'] == year]
    selected_players = year_data.iloc[finish].to_frame().T
    Test = pd.concat([Test, selected_players], axis=0)
    Test = Test.astype(merged_df.dtypes)
test_indices = Test.index
Train = merged_df.drop(test_indices)
print(merged_df.shape)
describe = merged_df.describe()
print(merged_df.dtypes)

# Calculate the correlation matrix
correlation_matrix = merged_df.iloc[:, 1:].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation')
#plt.show()

median_total = merged_df['Total'].median()
merged_df['Above_Median_Total'] = np.where(merged_df['Total'] > median_total, 1, 0)

columns_for_boxplots = [
    'Total_Previous_Year', 'AVG_ADP', 'AVG_ADP_Previous_Year',
    'Pre-season_Overall_Rank', 'Pre-season_Position_Rank',
    'Rookie', 'Undrafted', 'Unranked', 'Position_QB', 'Position_RB',
    'Position_TE', 'Position_WR'
]

for column in columns_for_boxplots:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Above_Median_Total', y=column, data=merged_df)
    plt.title(f'{column}')
    #plt.show()

positions_for_barplot = ['Position_QB', 'Position_RB', 'Position_TE', 'Position_WR']
plt.figure(figsize=(10, 6))
for position in positions_for_barplot:
    position_data = merged_df[merged_df[position] == 1]
    sns.barplot(x=[position], y=position_data['Total'].mean(), label=position)
plt.xlabel('Position')
plt.ylabel('Average Points')
plt.title('Average Points by Position')
plt.legend()
#plt.show()
plt.figure(figsize=(10, 6))
for position in positions_for_barplot:
    sns.barplot(x=[position], y=merged_df[position].sum())
plt.xlabel('Position')
plt.ylabel('Count')
plt.title('Count of Each Position')
#plt.show()
plt.figure(figsize=(12, 6))
sns.barplot(x='Year', y='Total', data=merged_df)
plt.xlabel('Year')
plt.ylabel('Average of Total')
plt.title('Bar Plot of Average Total by Year')
#plt.show()
# KDE of Total
sns.kdeplot(data=merged_df, x='Total', fill=True, common_norm=False)
plt.xlabel('Total')
plt.ylabel('Density')
plt.title('Kernel Density Estimation (KDE) Plot for Total')
#plt.show()
# List of position columns
position_columns = ['Position_QB', 'Position_RB', 'Position_TE', 'Position_WR']

# Create a new column 'Position' indicating the player's position
merged_df['Position'] = merged_df[position_columns].idxmax(axis=1).apply(lambda x: x[9:])
# Create a FacetGrid with Seaborn
g = sns.FacetGrid(merged_df, hue='Position', palette='tab10', height=6)
g.map(sns.kdeplot, 'Total', fill=True, common_norm=False)
g.set(xlabel='Total', ylabel='Density')
g.fig.suptitle('KDE Plot for Total by Position')
g.add_legend(title='Position')
#plt.show()
print(Train.columns)
Train.set_index('Player', inplace=True)
Test.set_index('Player', inplace=True)
YTrain = Train['Total']
Train = Train.drop(columns=['Total','Year'])
YTest = Test['Total']
Test = Test.drop(columns=['Total','Year'])


###################
##### MODELING ######
###################

# remove quote marks to model without these columns
'''
col = ['ESPN', 'Sleeper', 'NFL', 'RTSports', 'FFC',
'ESPN_Previous_Year', 'Sleeper_Previous_Year', 'NFL_Previous_Year',
'RTSports_Previous_Year', 'FFC_Previous_Year',
'Rushing_Y/A', 'Rushing_LG', 'Rushing_20+',
'Receiving_Y/R','FL', 'G', 'Receiving_LG', 'Receiving_20+', 'CMP_%','Passing_Y/A', 'INT']

Test = Test.drop(columns=col)
Train = Train.drop(columns=col)
'''

# Scale data
features = Train.columns
scaler = StandardScaler()
Train = scaler.fit_transform(Train)
Test = scaler.transform(Test)

print(YTest.head())
# DF to store model results
results = pd.DataFrame(columns=['Model', 'MSE',"MAD", 'R-squared','MSE_Top_200'])
# Models to try
models = [
    LinearRegression(),RandomForestRegressor(random_state=2),GradientBoostingRegressor(random_state=2),
    #SVR(kernel='linear'),
    SVR(kernel='poly'),SVR(kernel='rbf'),SVR(kernel='sigmoid'),XGBRegressor(),
    KNeighborsRegressor(n_neighbors=8),Lasso(),Ridge()
]
excel_file_path = "C:\\Users\\gohar\\Downloads\\predictions.xlsx"

# Fit data with each model
for model in models:
    print(model)
    model.fit(Train, YTrain)
    predictions = model.predict(Test)
    mse = mean_squared_error(YTest, predictions)
    x = predictions - YTest
    #x.to_excel(excel_file_path, index=False)
    mse200 = mean_squared_error(YTest[0:200], predictions[0:200])
    r2 = r2_score(YTest, predictions)
    results = pd.concat([results, pd.DataFrame([{'Model': str(model), 'MSE': mse,'MAD': np.mean(np.abs(x)), 'R-squared': r2,'MSE_Top_200': mse200}])])
    if isinstance(model, (LinearRegression, Ridge)):
        print(pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}))


print(results)


#https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning referenced for parameter tuning
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }
def objective(space):
    model = XGBRegressor(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']),eval_metric="rmse",early_stopping_rounds=10)
    evaluation = [(Train, YTrain), (Test, YTest)]
    model.fit(Train, YTrain,
        eval_set=evaluation,
        verbose=False)

    predictions = model.predict(Test)
    mse = mean_squared_error(YTest, predictions)
    print(f'mse: {mse}')
    return {'loss': mse, 'status': STATUS_OK}

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = Trials())

print("The best hyperparameters include:")
print(best_hyperparams)
best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
best_hyperparams['min_child_weight'] = int(best_hyperparams['min_child_weight'])

# XGBoost regression with the best parameters
model = XGBRegressor(**best_hyperparams)
model.fit(Train, YTrain)
predictions = model.predict(Test)
mse = mean_squared_error(YTest, predictions)
mse200 = mean_squared_error(YTest[0:200], predictions[0:200])
print(f'mse: {mse}')
print(f'mse200: {mse200}')
