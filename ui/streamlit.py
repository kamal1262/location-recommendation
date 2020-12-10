import pickle
import folium
import numpy as np
import pandas as pd
import streamlit as st

import re
import os
import sys
import gzip
import math
import json
import pickle
import datetime
import itertools

import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.functional as F
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import Any
from collections import Counter
from scipy.spatial import distance
from typing import Dict, List, Tuple
from streamlit_folium import folium_static

sys.path.append(os.getcwd())
from src.config import MODEL_PATH
from src.utils.logger import logger
from src.ml.skipgram import SkipGram
from src.utils.io_utils import save_model
from src.ml.data_loader import Sequences, SequencesDataset


class Utils:
    """
    Utils class to handle procecssing be it before or after
    """
    
    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_pickle(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved to pickle: { path } !")


dict_idx2loc = Utils.load_pickle("models/idx2loc.p")
dict_loc2idx = Utils.load_pickle("models/loc2idx.p")
dict_loc2name = Utils.load_pickle("data/loc2name.p")

dict_idx2name = {}
for x in dict_idx2loc.keys():
    dict_idx2name[x] = dict_loc2name[dict_idx2loc[x]]

dict_name2idx = {f"{ v['display_name'] } - { v['location_type'] }": k for k, v in dict_idx2name.items()}

bld_list = []
for building in list(dict_name2idx.keys()):
    if building.endswith("BUILDING_NAME"):
        bld_list.append(building)

ct_list = []
for city in list(dict_name2idx.keys()):
    if city.endswith("CITY"):
        ct_list.append(city)

state_list = []
for state in list(dict_name2idx.keys()):
    if state.endswith("STATE"):
        state_list.append(state)


class Loc2Vec:
    
    vocab_size = 14699
    embedding_dims = 128
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skipgram = SkipGram(vocab_size, embedding_dims).to(self.device)
        self.load_model("models/skipgram_epoch_24_2020-11-16-1439.pt")
        self.skipgram.eval()

        self.emb_vec = self.get_embeddings()
    
    def load_model(self, model_path):
        print(f"Loading model from: { model_path } ...")
        self.skipgram.load_state_dict(torch.load(model_path, map_location=self.device))
    
    @staticmethod
    def get_embeddings():
        return np.load("models/embeddings.npy")
    
    @staticmethod
    def closest_node(loc_type, index, nodes, top_n=5, filter_type=False):
        node = nodes[index]
        closest_index = distance.cdist([node], nodes, "cosine")[0]
        result = zip(range(len(closest_index)), closest_index)
        result = sorted(result, key=lambda x: x[1])
        
        location_src = dict_loc2name[dict_idx2loc[index]]
        # print(f"Finding location near to: { dict_idx2loc[index] } ({location_src['map_location_3']}, {location_src['map_location_2']}, {location_src['map_location_1']}) - is a {location_src['location_type']} \n")
        
        recommended = []
        if not filter_type:
            cnt = 1
            for idx, dist in result[1:top_n+1]:
                location = dict_loc2name[dict_idx2loc[idx]]
                l1 = location["map_location_1"]
                l2 = location["map_location_2"]
                l3 = location["map_location_3"]

                # print(f"{cnt} ==> {dict_idx2loc[idx]} - {l3}, {l2}, {l1}, (score: {dist})")
                recommended.append(f"[{dict_idx2loc[idx]}] - {l3}, {l2}, {l1}, (score: {dist})")
                cnt += 1
        
        if filter_type:
            cnt_loc_type = 0
            idx_loc_type = 0
            while cnt_loc_type < top_n:
                idx, dist = result[idx_loc_type]
                location = dict_loc2name[dict_idx2loc[idx]]
                l1 = f"- {location['map_location_1']} " if (loc_type == "STATE") or (loc_type == "CITY") or (loc_type == "BUILDING_NAME") else ""
                l2 = f"- {location['map_location_2']} " if (loc_type == "CITY") or (loc_type == "BUILDING_NAME") else ""
                l3 = f"- {location['map_location_3']} " if (loc_type == "BUILDING_NAME") else ""

                if location["location_type"] == loc_type:
                    # print(f"{cnt_loc_type} ==> {dict_idx2loc[idx]} - {l3}, {l2}, {l1}, (score: {dist})")
                    recommended.append(f"[{dict_idx2loc[idx]}] {l3}{l2}{l1} ") # (score: {dist})
                    cnt_loc_type += 1
                idx_loc_type += 1
        
        return recommended
            

def display_name_mapping(x):
    display = None
    if x["location_type"] == "BUILDING_NAME":
        display = x["bld_display_name"]
    if x["location_type"] == "STREET_NAME":
        display = x["str_display_name"]
    if x["location_type"] == "COUNTRY":
        display = x["country_display_name"]
    if x["location_type"] == "REGION":
        display = x["region_display_name"]
    if x["location_type"] == "STATE":
        display = x["state_display_name"]
    if x["location_type"] == "DISTRICT":
        display = x["district_display_name"]
    if x["location_type"] == "DIVISION":
        display = x["div_display_name"]
    if x["location_type"] == "CITY":
        display = x["city_display_name"]
    if x["location_type"] == "POST_CODE":
        display = x["postcode_display_name"]
    
    if not display:
        display = x["location_name"]
        
    return display

def load_location_data():
    location_table = pq.read_table("data/my_locations_db.parquet")
    location_db_df = location_table.to_pandas()
    location_db_df["display_name"] = location_db_df.apply(display_name_mapping, axis=1)
    location_db_df["map_location_1"] = location_db_df["state_display_name"]
    location_db_df["map_location_2"] = location_db_df["city_display_name"]
    location_db_df["map_location_3"] = location_db_df["bld_display_name"]
    location_db_df["point"] = location_db_df["geo_coordinate"].str.replace(r"POINT","").str.strip("\(").str.strip("\)").str.split()
    location_db_df["latitude"] = location_db_df["point"].apply(lambda x: x[1] if x else None).astype(float)
    location_db_df["longitude"] = location_db_df["point"].apply(lambda x: x[0] if x else None).astype(float)
    return location_db_df

if __name__ == '__main__':
    # Adding selected location to recent_searches
    # with open("recent_searches.json", "r") as fr:
    #     recent_searches = list(set(json.load(fr)["recent"]))
    
    emb_vec = Loc2Vec.get_embeddings()
    
    data_load_state = st.text('Loading data...')
    loc_data = load_location_data()
    data_load_state.text('Loading data... Done!')
    
    st.header('Location data')
    cols_default = ["global_id", "legacy_id", "location_type", "location_name", "latitude", "longitude", "map_location_1", "map_location_2", "map_location_3"]
    cols_selection = st.multiselect("Columns", loc_data.columns.tolist(), default=cols_default)
    
    protype_default = ["BUILDING_NAME", "STATE", "CITY"]
    protype_selection = st.multiselect("Location Type", loc_data.location_type.unique().tolist(), default=protype_default)
    
    filtered_df = loc_data[loc_data["location_type"].isin(protype_selection)][cols_selection]
    st.write(filtered_df)
    
    my_range = (pd.notnull(filtered_df.latitude)) & \
                        (filtered_df.longitude >= 100.0) & \
                        (filtered_df.longitude <= 120.0) & \
                        (filtered_df.latitude >= 0.0) & (filtered_df.latitude <= 6.5)
    st.map(filtered_df[my_range])

    st.header('Location Recommendations')

    protype_select_default = ["BUILDING_NAME", "CITY", "STATE"]
    protype_select_selection = st.selectbox("Location Type (User search)", protype_select_default)

    locs = []
    if protype_select_selection == "BUILDING_NAME":
        locs = sorted(bld_list)
    elif protype_select_selection == "CITY":
        locs = sorted(ct_list)
    elif protype_select_selection == "STATE":
        locs = sorted(state_list)
        
    location_select_selection = st.selectbox("Locations", locs)
    
    # Show selected in maps
    selected_global_id = dict_idx2loc[dict_name2idx[location_select_selection]]
    selected_df = loc_data[loc_data["global_id"] == selected_global_id]
    lat, lng = (selected_df[["latitude", "longitude"]].values[0])
    
    if (not math.isnan(float(lat)) and not math.isnan(float(lng))):
        m = folium.Map(location=[lat, lng], zoom_start=12)
        folium.Marker(
            [lat, lng], popup=location_select_selection
        ).add_to(m)
        folium_static(m)
    
    st.subheader('Popular Area/City')
    popular_cities = ["Puchong", "Petaling Jaya", "KLCC", "Mont Kiara", 
                      "Shah Alam", "Cheras", "Petaling Jaya", 
                      "Subang Jaya", "Seri Kembangan", "Cyberjaya", "Bukit Jalil", 
                      "Kajang", "Kepong", "Ampang", "Klang", 
                      "Jalan Klang Lama (Old Klang Road)", 
                      "Rawang", "Kota Damansara", "Kota Kemuning", 
                      "Setapak", "Semenyih", 
                      "Ara Damansara", "Setapak", "Desa ParkCity", 
                      "Bangsar", "Setia Alam", "Damansara Perdana", "Bandar Kinrara"]
    popular_exclude_selection = st.multiselect("User clicked on these PDP (to be exclude)", popular_cities, default=[])
    filtered_popular_cities = [n for n in popular_cities if n not in popular_exclude_selection][:10]
    popular_txt = "".join([ f"{idx + 1}. {city} \n" for idx, city in enumerate(filtered_popular_cities) ])
    st.markdown(popular_txt)
    
    st.subheader('Area/City Recommendation')
    recommended_city = Loc2Vec.closest_node("CITY", dict_name2idx[location_select_selection], emb_vec, top_n=11, filter_type=True)
    recommended_city_txt = "".join([ f"{idx + 1}. {city} \n" for idx, city in enumerate(recommended_city) ])
    st.markdown(recommended_city_txt)
    recommend_city_list_id = [re.search(r"\[([A-Za-z0-9_]+)\]", s).group(1) for s in recommended_city]
    
    # Show recommendation in map
    if (not math.isnan(float(lat)) and not math.isnan(float(lng))):
        m = folium.Map(location=[lat, lng], zoom_start=12)
        folium.Marker(
            [lat, lng], popup=location_select_selection,
            icon=folium.Icon(color='blue')
        ).add_to(m)
    
    for city_id in recommend_city_list_id:
        selected_df = loc_data[(loc_data["global_id"] == city_id)]
        tlat, tlng = (selected_df[["latitude", "longitude"]].values[0])
        city, state = selected_df[["map_location_2", "map_location_1"]].values[0]
        if (not math.isnan(float(tlat)) and not math.isnan(float(tlng))) and (city_id != selected_global_id):
            if not (('m' in locals()) or ('m' in globals())):
                m = folium.Map(location=[tlat, tlng], zoom_start=12)
            folium.Marker(
                [tlat, tlng], popup=f"{city}, {state}",
                icon=folium.Icon(color='green')
            ).add_to(m)
    
    if (('m' in locals()) or ('m' in globals())):
        folium_static(m)

    
    st.subheader('Building Recommendation')
    recommended_bld = Loc2Vec.closest_node("BUILDING_NAME", dict_name2idx[location_select_selection], emb_vec, top_n=11, filter_type=True)
    recommended_bld_txt = "".join([ f"{idx + 1}. {bld} \n" for idx, bld in enumerate(recommended_bld) ])
    st.markdown(recommended_bld_txt)
    recommend_bld_list_id = [re.search(r"\[([A-Za-z0-9_]+)\]", s).group(1) for s in recommended_bld]
    
    # Show recommendation in map
    if (not math.isnan(float(lat)) and not math.isnan(float(lng))):
        m = folium.Map(location=[lat, lng], zoom_start=12)
        folium.Marker(
            [lat, lng], popup=location_select_selection,
            icon=folium.Icon(color='blue')
        ).add_to(m)
    
    for bld_id in recommend_bld_list_id:
        selected_df = loc_data[(loc_data["global_id"] == bld_id)]
        tlat, tlng = (selected_df[["latitude", "longitude"]].values[0])
        bld, city, state = selected_df[["map_location_3", "map_location_2", "map_location_1"]].values[0]
        if (not math.isnan(float(tlat)) and not math.isnan(float(tlng))) and (bld_id != selected_global_id):
            if not (('m' in locals()) or ('m' in globals())):
                m = folium.Map(location=[tlat, tlng], zoom_start=12)
            folium.Marker(
                [tlat, tlng], popup=f"{bld}, {city}, {state}",
                icon=folium.Icon(color='green')
            ).add_to(m)
    
    if (('m' in locals()) or ('m' in globals())):
        folium_static(m)
    