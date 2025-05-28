import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff

st.set_page_config(page_title="Bird Species Analysis", layout="wide")
st.title("🦅 Interactive Bird Species Observation Analysis")
st.markdown("### Forest and Grassland Ecosystem Dashboard")

# Load data
forest=pd.read_csv('c:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Bird Species Observation Analysis in Forest and Grassland Ecosystem\\forest.csv')
grass=pd.read_csv('c:\\Users\\Sai Krishnan\\OneDrive\\Desktop\\ML Program\\Bird Species Observation Analysis in Forest and Grassland Ecosystem\\Grass_Land.csv')

def preprocess_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Convert time columns
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%H:%M:%S").dt.hour
    df["End_Time"] = pd.to_datetime(df["End_Time"], format="%H:%M:%S").dt.hour
    
    return df

forest = preprocess_data(forest)
grass = preprocess_data(grass)

# Add location type for combined analysis
forest['Location_Type'] = 'Forest'
grass['Location_Type'] = 'Grassland'
combined_df = pd.concat([forest, grass], ignore_index=True)

# Sidebar filters
st.sidebar.header("🔍 Interactive Filters")
selected_location = st.sidebar.multiselect(
    "Select Location Type", 
    options=['Forest', 'Grassland'], 
    default=['Forest', 'Grassland']
)

selected_seasons = st.sidebar.multiselect(
    "Select Seasons", 
    options=['Spring', 'Summer', 'Fall', 'Winter'], 
    default=['Spring', 'Summer', 'Fall', 'Winter']
)

# Filter data based on selections
filtered_df = combined_df[
    (combined_df['Location_Type'].isin(selected_location)) &
    (combined_df['Season'].isin(selected_seasons))
]

# Create tabs
tabs = st.tabs([
    "Overview Dashboard",
    "Temporal Analysis", 
    "Spatial Analysis",
    "Species Analysis",
    " Environmental Conditions",
    "Distance & Behavior",
    "Observer Trends",
    "Conservation Insights"
])

# Overview Dashboard
with tabs[0]:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Species", 
            filtered_df['Scientific_Name'].nunique(),
            delta=f"{filtered_df['Scientific_Name'].nunique() - combined_df['Scientific_Name'].nunique()//2} vs avg"
        )
    
    with col2:
        st.metric(
            "Total Observations", 
            len(filtered_df),
            delta=f"{len(filtered_df) - len(combined_df)//2} vs avg"
        )
    
    with col3:
        st.metric(
            "Active Observers", 
            filtered_df['Observer'].nunique()
        )
    
    with col4:
        st.metric(
            "Unique Locations", 
            filtered_df['Plot_Name'].nunique()
        )
    
    # High-level visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Location distribution pie chart
        location_counts = filtered_df["Location_Type"].value_counts()
        fig_pie = px.pie(
            values=location_counts.values, 
            names=location_counts.index,
            title="Observation Distribution by Location",
            color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Seasonal distribution
        season_counts = filtered_df['Season'].value_counts()
        fig_bar = px.bar(
            x=season_counts.index, 
            y=season_counts.values,
            title="Seasonal Observation Patterns",
            labels={'x': 'Season', 'y': 'Number of Observations'},
            color=season_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    

# 1. Temporal Analysis
with tabs[1]:
    st.header("Temporal Patterns Analysis")
    
    # Forest seasonal trends
    col1, col2 = st.columns(2)

    with col1:
        forest_season = forest['Season'].value_counts()
        fig_forest_season = px.bar(
            x=forest_season.index,
            y=forest_season.values,
            title="Forest Seasonal Distribution",
            labels={'x': 'Season', 'y': 'Number of Observations'},
            color=forest_season.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_forest_season, use_container_width=True)

    with col2:
        grass_season = grass['Season'].value_counts()
        fig_grass_season = px.bar(
            x=grass_season.index,
            y=grass_season.values,
            title="Grassland Seasonal Distribution",
            labels={'x': 'Season', 'y': 'Number of Observations'},
            color=grass_season.values,
            color_continuous_scale='YlOrBr'
        )
        st.plotly_chart(fig_grass_season, use_container_width=True)
    
    
    
    # Time of day analysis
    st.subheader("Daily Activity Patterns")
    
    # Create hourly distribution
    forest_hourly = forest.groupby('Start_Time').size()
    grass_hourly = grass.groupby('Start_Time').size()
    
    fig_hourly = go.Figure()
    
    fig_hourly.add_trace(go.Scatter(
        x=forest_hourly.index,
        y=forest_hourly.values,
        mode='lines+markers',
        name='Forest',
        line=dict(color='forestgreen', width=3),
        marker=dict(size=8)
    ))
    
    fig_hourly.add_trace(go.Scatter(
        x=grass_hourly.index,
        y=grass_hourly.values,
        mode='lines+markers',
        name='Grassland',
        line=dict(color='goldenrod', width=3),
        marker=dict(size=8)
    ))
    
    fig_hourly.update_layout(
        title="Bird Activity Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Observations",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)

# 2. Spatial Analysis
with tabs[2]:
    st.header("Spatial Distribution Analysis")
    
    # High-activity zones
    st.subheader("High-Activity Zones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top forest plots
        forest_plots = forest['Plot_Name'].value_counts().head(10)
        fig_forest_plots = px.bar(
            x=forest_plots.values,
            y=forest_plots.index,
            orientation='h',
            title=" Top Forest Observation Sites",
            labels={'x': 'Number of Observations', 'y': 'Plot Name'},
            color=forest_plots.values,
            color_continuous_scale='Greens'
        )
        fig_forest_plots.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_forest_plots, use_container_width=True)
    
    with col2:
        # Top grassland plots
        grass_plots = grass['Plot_Name'].value_counts().head(10)
        fig_grass_plots = px.bar(
            x=grass_plots.values,
            y=grass_plots.index,
            orientation='h',
            title="🌾 Top Grassland Observation Sites",
            labels={'x': 'Number of Observations', 'y': 'Plot Name'},
            color=grass_plots.values,
            color_continuous_scale='YlOrBr'
        )
        fig_grass_plots.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_grass_plots, use_container_width=True)
    
    # Plot diversity analysis
    st.subheader(" Location Diversity Analysis")
    
    plot_diversity = filtered_df.groupby('Plot_Name').agg({
        'Scientific_Name': 'nunique',
        'Common_Name': 'count'
    }).rename(columns={'Scientific_Name': 'Species_Count', 'Common_Name': 'Total_Observations'})
    
    fig_diversity = px.scatter(
        plot_diversity,
        x='Total_Observations',
        y='Species_Count',
        title=" Plot Diversity: Species Count vs Total Observations",
        labels={'Total_Observations': 'Total Observations', 'Species_Count': 'Unique Species'},
        hover_name=plot_diversity.index,
        size='Species_Count',
        color='Species_Count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig_diversity, use_container_width=True)

# 3. Species Analysis
with tabs[3]:
    st.header(" Species Diversity & Behavior Analysis")
    
    # Species selection for detailed analysis
    top_species = filtered_df['Common_Name'].value_counts().head(20).index.tolist()
    selected_species = st.multiselect(
        "Select Species for Detailed Analysis",
        options=top_species,
        default=top_species[:5]
    )
    
    if selected_species:
        species_df = filtered_df[filtered_df['Common_Name'].isin(selected_species)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Species abundance by location
            species_location = species_df.groupby(['Common_Name', 'Location_Type']).size().reset_index(name='Count')
            
            fig_species_loc = px.bar(
                species_location,
                x='Common_Name',
                y='Count',
                color='Location_Type',
                title=" Species Distribution by Location",
                labels={'Count': 'Number of Observations'},
                color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
            )
            fig_species_loc.update_xaxes(tickangle=45)
            st.plotly_chart(fig_species_loc, use_container_width=True)
        
        with col2:
            # Species seasonal patterns
            species_season = species_df.groupby(['Common_Name', 'Season']).size().reset_index(name='Count')
            
            fig_species_season = px.bar(
                species_season,
                x='Common_Name',
                y='Count',
                color='Season',
                title=" Species Seasonal Activity",
                labels={'Count': 'Number of Observations'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_species_season.update_xaxes(tickangle=45)
            st.plotly_chart(fig_species_season, use_container_width=True)
    
    # Activity patterns
    st.subheader(" Activity Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ID Method distribution
        id_method_data = filtered_df.groupby(['Location_Type', 'ID_Method']).size().reset_index(name='Count')
        
        fig_id_method = px.sunburst(
            id_method_data,
            path=['Location_Type', 'ID_Method'],
            values='Count',
            title="🔍 Identification Method Distribution"
        )
        st.plotly_chart(fig_id_method, use_container_width=True)
    
    with col2:
        # Sex ratio analysis
        sex_data = filtered_df.groupby(['Location_Type', 'Sex']).size().reset_index(name='Count')
        
        fig_sex = px.bar(
            sex_data,
            x='Location_Type',
            y='Count',
            color='Sex',
            title="⚥ Sex Distribution Analysis",
            color_discrete_map={'Male': 'blue', 'Female': 'pink', 'Unknown': 'gray'}
        )
        st.plotly_chart(fig_sex, use_container_width=True)

# 4. Environmental Conditions
with tabs[4]:
    st.header("Environmental Impact Analysis")
    
    # Weather correlation analysis
    st.subheader("Weather Correlation Matrix")
    
    # Create correlation data
    env_df = filtered_df.copy()
    label_enc = LabelEncoder()
    
    # Encode categorical variables for correlation
    categorical_cols = ['Sky', 'Wind', 'Common_Name']
    for col in categorical_cols:
        if col in env_df.columns:
            env_df[f'{col}_encoded'] = label_enc.fit_transform(env_df[col].astype(str))
    
    # Select numeric columns for correlation
    numeric_cols = ['Temperature', 'Humidity', 'Sky_encoded', 'Wind_encoded']
    numeric_cols = [col for col in numeric_cols if col in env_df.columns]
    
    if len(numeric_cols) >= 2:
        corr_matrix = env_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Environmental Factors Correlation",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Environmental distributions
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature distribution
        fig_temp = px.histogram(
            filtered_df,
            x='Temperature',
            color='Location_Type',
            title="Temperature Distribution",
            marginal="box",
            color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Humidity distribution
        fig_humidity = px.histogram(
            filtered_df,
            x='Humidity',
            color='Location_Type',
            title="💧 Humidity Distribution",
            marginal="box",
            color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
        )
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Disturbance effects
    st.subheader("Disturbance Impact Analysis")
    
    disturbance_data = filtered_df.groupby(['Location_Type', 'Disturbance']).size().reset_index(name='Count')
    
    fig_disturbance = px.bar(
        disturbance_data,
        x='Disturbance',
        y='Count',
        color='Location_Type',
        title="Environmental Disturbance Effects",
        color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
    )
    fig_disturbance.update_xaxes(tickangle=45)
    st.plotly_chart(fig_disturbance, use_container_width=True)

# 5. Distance & Behavior
with tabs[5]:
    st.header("Distance & Behavioral Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance distribution
        distance_data = filtered_df.groupby(['Location_Type', 'Distance']).size().reset_index(name='Count')
        
        fig_distance = px.bar(
            distance_data,
            x='Distance',
            y='Count',
            color='Location_Type',
            title="Observation Distance Distribution",
            color_discrete_map={'Forest':'#2E8B57', 'Grassland': '#DAA520'}
        )
        fig_distance.update_xaxes(tickangle=45)
        st.plotly_chart(fig_distance, use_container_width=True)
    
    with col2:
        # Flyover patterns
        flyover_data = filtered_df.groupby(['Location_Type', 'Flyover_Observed']).size().reset_index(name='Count')
        
        fig_flyover = px.pie(
            flyover_data,
            values='Count',
            names='Flyover_Observed',
            title="Flyover Observation Patterns",
            facet_col='Location_Type'
        )
        st.plotly_chart(fig_flyover, use_container_width=True)
    
    # Behavior correlation with distance
    st.subheader("Distance vs Species Behavior")
    
    behavior_distance = filtered_df.groupby(['Distance', 'Common_Name']).size().reset_index(name='Count')
    top_species_behavior = behavior_distance.groupby('Common_Name')['Count'].sum().nlargest(10).index
    behavior_distance_filtered = behavior_distance[behavior_distance['Common_Name'].isin(top_species_behavior)]

    fig_behavior = px.bar(
        behavior_distance_filtered,
        x='Distance',
        y='Count',
        color='Common_Name',
        barmode='group',
        title="Species Observation Patterns by Distance",
        hover_data=['Common_Name']
    )
    st.plotly_chart(fig_behavior, use_container_width=True)

# 6. Observer Trends
with tabs[6]:
    st.header("👥 Observer Analysis & Bias Detection")
    
    # Observer productivity
    observer_stats = filtered_df.groupby('Observer').agg({
        'Common_Name': 'count',
        'Scientific_Name': 'nunique'
    }).rename(columns={'Common_Name': 'Total_Observations', 'Scientific_Name': 'Unique_Species'}).sort_values('Total_Observations', ascending=False)
    
    fig_observer = px.bar(
        observer_stats,
        x=observer_stats.index,
        y='Total_Observations',
        title="Observer Productivity Analysis",
        labels={'x': 'Observer', 'Total_Observations': 'Total Observations'},
        color='Unique_Species',
        color_continuous_scale='Viridis'
    )
    fig_observer.update_xaxes(tickangle=45)
    st.plotly_chart(fig_observer, use_container_width=True)
    
    # Observer bias for top species
    st.subheader("Observer Species Bias")
    
    top_species_obs = filtered_df['Common_Name'].value_counts().head(10).index
    observer_species = filtered_df[filtered_df['Common_Name'].isin(top_species_obs)]
    observer_species_data = observer_species.groupby(['Observer', 'Common_Name']).size().reset_index(name='Count')
    
    fig_bias = px.bar(
        observer_species_data,
        x='Observer',
        y='Count',
        color='Common_Name',
        title="Observer Bias: Top Species Distribution",
        labels={'Count': 'Number of Observations'}
    )
    fig_bias.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bias, use_container_width=True)
    
    # Visit frequency impact
    st.subheader("Visit Frequency Impact")
    
    visit_impact = filtered_df.groupby(['Visit', 'Location_Type'])['Scientific_Name'].nunique().reset_index(name='Unique_Species')
    
    fig_visit = px.line(
        visit_impact,
        x='Visit',
        y='Unique_Species',
        color='Location_Type',
        title=" Species Diversity vs Visit Frequency",
        markers=True,
        color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
    )
    st.plotly_chart(fig_visit, use_container_width=True)

# 7. Conservation Insights
with tabs[7]:
    st.header(" Conservation Status & Priorities")
    
    # Conservation status overview
    col1, col2 = st.columns(2)
    
    with col1:
        # PIF Watchlist status
        watchlist_data = filtered_df.groupby(['Location_Type', 'PIF_Watchlist_Status']).size().reset_index(name='Count')
        
        fig_watchlist = px.bar(
            watchlist_data,
            x='PIF_Watchlist_Status',
            y='Count',
            color='Location_Type',
            title=" PIF Watchlist Status Distribution",
            color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
        )
        fig_watchlist.update_xaxes(tickangle=45)
        st.plotly_chart(fig_watchlist, use_container_width=True)
    
    with col2:
        # Regional stewardship status
        stewardship_data = filtered_df.groupby(['Location_Type', 'Regional_Stewardship_Status']).size().reset_index(name='Count')
        
        fig_stewardship = px.bar(
            stewardship_data,
            x='Regional_Stewardship_Status',
            y='Count',
            color='Location_Type',
            title="Regional Stewardship Status",
            color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
        )
        fig_stewardship.update_xaxes(tickangle=45)
        st.plotly_chart(fig_stewardship, use_container_width=True)
    
    # At-risk species identification
    st.subheader("🚨 Conservation Priority Species")
    
    # Filter for species with concerning conservation status
    priority_species = filtered_df[
        (filtered_df['PIF_Watchlist_Status'].notna()) & 
        (filtered_df['PIF_Watchlist_Status'] != 'Not on PIF Watchlist')
    ]
    
    if not priority_species.empty:
        priority_summary = priority_species.groupby(['Common_Name', 'PIF_Watchlist_Status', 'Location_Type']).size().reset_index(name='Observations')
        
        # Priority species table
        st.subheader(" Priority Species Summary")
        priority_table = priority_summary.groupby('Common_Name').agg({
            'Observations': 'sum',
            'PIF_Watchlist_Status': 'first',
            'Location_Type': lambda x: ', '.join(x.unique())
        }).sort_values('Observations', ascending=False)
        
        st.dataframe(priority_table, use_container_width=True)
    
    else:
        st.info("No species with special conservation status found in the current selection.")
    
    # AOU Code patterns
    st.subheader("Species Code Analysis")
    
    aou_data = filtered_df.groupby(['AOU_Code', 'Common_Name', 'Location_Type']).size().reset_index(name='Count')
    top_aou = aou_data.groupby('AOU_Code')['Count'].sum().nlargest(20).index
    aou_filtered = aou_data[aou_data['AOU_Code'].isin(top_aou)]
    
    fig_aou = px.bar(
        aou_filtered,
        x='AOU_Code',
        y='Count',
        color='Location_Type',
        title="Top 20 Species by AOU Code",
        hover_data=['Common_Name'],
        color_discrete_map={'Forest': '#2E8B57', 'Grassland': '#DAA520'}
    )
    fig_aou.update_xaxes(tickangle=45)
    st.plotly_chart(fig_aou, use_container_width=True)

# Summary insights
st.header("Key Insights Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("High-Activity Zones")
    top_plot = filtered_df['Plot_Name'].value_counts().index[0] if not filtered_df.empty else "N/A"
    st.write(f"**Most Active Location:** {top_plot}")
    
    peak_season = filtered_df['Season'].value_counts().index[0] if not filtered_df.empty else "N/A"
    st.write(f"**Peak Season:** {peak_season}")

with col2:
    st.subheader("Environmental Influence")
    if 'Temperature' in filtered_df.columns:
        avg_temp = filtered_df['Temperature'].mean()
        st.write(f"**Average Temperature:** {avg_temp:.1f}°")
    
    if 'Humidity' in filtered_df.columns:
        avg_humidity = filtered_df['Humidity'].mean()
        st.write(f"**Average Humidity:** {avg_humidity:.1f}%")

with col3:
    st.subheader("Conservation Status")
    priority_count = len(filtered_df[
        (filtered_df['PIF_Watchlist_Status'].notna()) & 
        (filtered_df['PIF_Watchlist_Status'] != 'Not on PIF Watchlist')
    ]['Common_Name'].unique()) if 'PIF_Watchlist_Status' in filtered_df.columns else 0
    
    st.write(f"**Priority Species:** {priority_count}")
    
    total_species = filtered_df['Scientific_Name'].nunique()
    st.write(f"**Species Diversity:** {total_species}")