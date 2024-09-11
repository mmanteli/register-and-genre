import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import os
import glob
from datasets import concatenate_datasets
#from transformers import AutoTokenizer
import sys
import argparse
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    # Required argument: --path
    parser.add_argument('--path', type=str, required=True, help='A string path to data')
    # Optional argument: --cmap with default value
    parser.add_argument('--cmap', type=str, default='Set3', help='Color map, defaults to "Set3"')
    # Optional argument: --focus with choices and default value
    parser.add_argument('--focus', type=str, choices=['HI','IN','IP','ID','LY','NA','OP','SP', 'MT'], default=None, help='Focus area, choices are ["NA", "IN"]')
    # Optional flag: --use_log
    parser.add_argument('--use_log', type=int, default=0, help='If >0, scale plot thickness')
    return parser.parse_args()

# Modified from 
# https://github.com/TurkuNLP/pytorch-registerlabeling/blob/main/src/tools/plot_sankey_xgenre_mapping.py

"""
from ..labels import (
    map_full_names,
    labels_structure,
    map_xgenre,
    map_childless_upper_to_other,
)
from ..data import get_dataset
"""

# Set3 = ihan hyvä
# colormap_name="Set3"
# palette = sns.color_palette(colormap_name, n_colors=9).as_hex()[1:]
# palette2 = sns.color_palette("PuBu", n_colors=12).as_hex()[3:]
# template = "plotly_white"


# Label hierarchy with the "other" categories and self-references
# This is needed to map the labels to the correct X-GENRE category  => same here, we want upper categories only?
labels_all_hierarchy_with_other = {
    #"MT": ["MT"],
    "LY": ["LY"],
    "SP": ["SP", "it", "os"],
    "ID": ["ID"],
    "NA": ["NA", "ne", "sr", "nb", "on"],
    "HI": ["HI", "re", "oh"],
    "IN": ["IN", "en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["OP", "rv", "ob", "rs", "av", "oo"],
    "IP": ["IP", "ds", "ed", "oe"],
}

reverse_lookup = {}
for main_label, sublabels in labels_all_hierarchy_with_other.items():
    for sublabel in sublabels:
        reverse_lookup[sublabel] = main_label
print(reverse_lookup)

def plot_sankey(left_to_center, center_to_right):
    
    def extract_nodes_links(left_to_center, center_to_right):
        left_nodes = list(left_to_center.keys())
        center_nodes = sorted(list(set([node for center in left_to_center.values() for node in center.keys()])))
        right_nodes = sorted(list(set([node for right in center_to_right.values() for node in right.keys()])))
    
        nodes = left_nodes + center_nodes + right_nodes
    
        # Create the links
        links = {
            'source': [],
            'target': [],
            'value': [],
            'color': []
        }

        def yield_opacity(x):
            return str(x/5.0 + 0.01)
                
        # Left to center links
        for left, centers in left_to_center.items():
            values = list(centers.values())
            max_value = max([i for i in values])
            print("MAX", max_value)
            for center, value in centers.items():
                links['source'].append(nodes.index(left))
                links['target'].append(nodes.index(center))
                links['value'].append(value)
                opacity = yield_opacity(value/max_value)
                color = f'rgba(0, 0, 0, {opacity})' if value != 1 else 'rgba(0, 0, 0, 0)'
                links['color'].append(color)
    
        # Center to right links
        for center, rights in center_to_right.items():
            values = list(rights.values())
            max_value = max([i for i in values])
            print("MAX", max_value)
            for right, value in rights.items():
                links['source'].append(nodes.index(center))
                links['target'].append(nodes.index(right))
                links['value'].append(value)
                opacity = yield_opacity(value/max_value)
                color = f'rgba(0, 0, 0, {opacity})' if value != 1 else 'rgba(0, 0, 0, 0)'
                links['color'].append(color)

        return nodes, links
    
    nodes, links = extract_nodes_links(left_to_center, center_to_right)
    print("nodes = ",nodes)
    print("links = ",links)

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            color="grey",
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    ))
    
    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=14
        ),
        title_text=""
    )
    fig.write_image("SCALED_COLOR.png")


def plot_sankey_old(left_to_center, center_to_right):
    
    def extract_nodes_links(left_to_center, center_to_right):
        left_nodes = list(left_to_center.keys())
        center_nodes = sorted(list(set([node for center in left_to_center.values() for node in center.keys()])))
        right_nodes = sorted(list(set([node for right in center_to_right.values() for node in right.keys()])))
    
        nodes = left_nodes + center_nodes + right_nodes
    
        # Create the links
        links = {
            'source': [],
            'target': [],
            'value': [],
            'color': []
        }


        # accidentally deleted the better version of this;
        # e.g. calculate the max opacity for the largest register--genre pair per register
        # and then normalize with it and return x/4+0.05 etc.
        def yield_opacity(x):
            if x < 10:
                return "0.0"
            elif 10 <= x < 100:
                return "0.05"
            elif 100 <= x < 1000:
                return "0.1"
            elif 1000 <= x < 5000:
                return "0.3"
            else:
                return "0.6"
                
        # Left to center links
        for left, centers in left_to_center.items():
            for center, value in centers.items():
                links['source'].append(nodes.index(left))
                links['target'].append(nodes.index(center))
                links['value'].append(value)
                links['color'].append(['rgba(0, 0, 0, 0)' if value == 1 else 'rgba(211, 211, 211,'+ yield_opacity(value) +')'][0])
    
        # Center to right links
        for center, rights in center_to_right.items():
            for right, value in rights.items():
                links['source'].append(nodes.index(center))
                links['target'].append(nodes.index(right))
                links['value'].append(value)
                links['color'].append(['rgba(0, 0, 0, 0)' if value == 1 else 'rgba(211, 211, 211,'+ yield_opacity(value) +')'][0])
                #print(f"color added {yield_color(value)}")
        return nodes, links
    
    nodes, links = extract_nodes_links(left_to_center, center_to_right)

    # These are printed for jupyter plotting
    # the jupyter notebook has somehow vanished :(
    # it uses the same code, but the scaling and font work better on jupyter.
    #print("nodes =", nodes)
    #print("links =", links)


    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            color="grey",
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links["color"]
        )
    ))
    
    fig.update_layout(font_family="Times New Roman", title_text="", font_size=14)
    fig.write_image("SCALED_COLOR.png")


def read_data(path):

    # Initialize an empty list to store DataFrames
    dfs = []

    # Walk through the directory
    for subdir, _, _ in os.walk(path):
        # Find all .tsv files in the current directory
        for file in glob.glob(os.path.join(subdir, '*.tsv')):
            # Read the file into a DataFrame
            df = pd.read_csv(file, sep='\t')
            print(f'Read {file} succesfully.', flush=True)
            # Append the DataFrame to the list
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Now `combined_df` contains data from all .tsv files
    print("All data read.", flush=True)
    return combined_df

def separate_sub_labels_from_incorrect_main_labels(df):
    new_sublabels = []
    for index, d in df.iterrows():
        #print(type(d["subregister_prediction"]))
        #print(d["register_prediction"], d["original_register"])
        if type(d["subregister_prediction"]) == str:
            if d["subregister_prediction"] not in labels_all_hierarchy_with_other[d["register_prediction"]]:
                new_sublabels.append(np.nan)   # remove erronious => makes LY ID (and MT) be dropped!
                continue
                #e.g. NA + ob can be an error OR from NA, OP, ob
                #print(reverse_lookup[d["subregister_prediction"]], " in ", d["original_register"])
                if reverse_lookup[d["subregister_prediction"]] in d["original_register"]:    
                    # which case; is this from NA OP ob, if is, remove ob from NA row
                    #d["subregister_prediction"] = np.nan     # modify by removing, else keep erronious NA+ob
                    #df.at[index, "subregister_prediction"] = np.nan
                    new_sublabels.append(np.nan)
                else:
                    new_sublabels.append(d["subregister_prediction"])
            else:
                new_sublabels.append(d["subregister_prediction"])
        else:
                new_sublabels.append(d["subregister_prediction"])
    df["subregister_prediction"] = new_sublabels

    return df


def run(options, n_colors=20):

    palette = sns.color_palette(options.cmap, n_colors=n_colors).as_hex()[1:]
    print(f'Reading in {options.path}...')
    #df = pd.read_csv(options.path, delimiter="\t")
    #df = df.drop(columns="text")
    df = read_data(options.path)
    # Convert to main labels
    #df["register_prediction"] = df["register_prediction"].apply(
    #    lambda labels: map_childless_upper_to_other(labels.split())
    #)

    # this one first so we do not modify the register prediction yet
    df["original_register"] = df["register_prediction"].apply(lambda x: eval(x))     ### to check later if label like ["NA", "OP", "ob"] is divided correctly
    df["subregister_prediction"] = df["register_prediction"].apply(lambda x:  [i for i in eval(x) if i.islower()])
    df["register_prediction"] = df["register_prediction"].apply(lambda x: [i for i in eval(x) if i in labels_all_hierarchy_with_other.keys()])
    
    df["genre_prediction"] = df["genre_prediction"].apply(lambda x: eval(x))
    df = df.explode("register_prediction").explode("subregister_prediction")
    df = df[df['register_prediction'].notna()]
    #print(df[["register_prediction","subregister_prediction","original_register"]].head(n = 20))
    df = separate_sub_labels_from_incorrect_main_labels(df)
    #print(df[["register_prediction","subregister_prediction","original_register"]].head(n = 20))
    df = df.explode("genre_prediction")
    
    #data = get_ordered_data(df, "register_prediction")

    df.sort_values(["register_prediction", "genre_prediction"], inplace = True)
    # Calculate the label distributions
    data = (
        df.groupby(["register_prediction", "genre_prediction"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    df.sort_values(["subregister_prediction", "register_prediction"], inplace = True)
    data2 = (
        df.groupby(["subregister_prediction", "register_prediction"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    
    new_data2 = {}
    for k,v in data2.items():
        new_data2[k] = v
        for reg in labels_all_hierarchy_with_other.keys():
            print(reg, v)
            if reg not in v.keys():
                new_data2[k][reg] = 1   # REMOVED LATER
    print("data_reg2reg = ", data2)
    print("data_reg2gen = ", data)
    print("Data grouping complete")

    plot_sankey(new_data2, data)
    exit()


if __name__ == "__main__":
    #path = sys.argv[1]
    #focus = sys.argv[2]
    #use_log = int(sys.argv[4]) if len(sys.argv) > 2 else 0
    #c_map = sys.argv[3] if len(sys.argv) > 3 else "Set3"
    options = parse_arguments()
    print(options)
    run(options)
