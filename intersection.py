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
#colormap_name="Set3"
#palette = sns.color_palette(colormap_name, n_colors=9).as_hex()[1:]
#palette2 = sns.color_palette("PuBu", n_colors=12).as_hex()[3:]
#template = "plotly_white"


# Label hierarchy with the "other" categories and self-references
# This is needed to map the labels to the correct X-GENRE category  => same here, we want upper categories only?
labels_all_hierarchy_with_other = {
    "MT": ["MT"],
    "LY": ["LY"],
    "SP": ["SP", "it", "os"],
    "ID": ["ID"],
    "NA": ["NA", "ne", "sr", "nb", "on"],
    "HI": ["HI", "re", "oh"],
    "IN": ["IN", "en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["OP", "rv", "ob", "rs", "av", "oo"],
    "IP": ["IP", "ds", "ed", "oe"],
}


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
    df["register_prediction"] = df["register_prediction"].apply(lambda x: [i for i in eval(x) if i in labels_all_hierarchy_with_other.keys()])
    df["genre_prediction"] = df["genre_prediction"].apply(lambda x: eval(x))
    df = df.explode("register_prediction").explode("genre_prediction")
    #data = get_ordered_data(df, "register_prediction")

    df.sort_values(["register_prediction", "genre_prediction"], inplace = True)
    # Calculate the label distributions
    data = (
        df.groupby(["register_prediction", "genre_prediction"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    print("Data grouping complete")
    num_total = len(df)
    reg_labels = data.keys()
    gen_labels = data["HI"].keys()
    num_examples_reg = {k:v for k,v in zip(reg_labels, [len(df[df.register_prediction == label]) for label in reg_labels])}
    num_examples_gen = {k:v for k,v in zip(gen_labels, [len(df[df.genre_prediction == label]) for label in gen_labels])}
    

    # Extract the nodes and links
    sources = []
    targets = []
    values = []
    colors=[]

    register_names = {"HI":"How-to Instructions",
                      "IN": "Inform. Description",
                      "ID": "Interact. Discussion",
                      "IP": "Inform. Persuasion",
                      "LY":"Lyrical",
                      "NA":"Narrative",
                      "OP":"Opinion",
                      "SP":"Spoken",
                      "MT":"Machine transl."
                     }
    
    # Create a mapping of node names to indices
    #node_labels = list(data.keys()) + list(
    #    set(k for sub_dict in data.values() for k in sub_dict)
    #)
    node_labels = [register_names[r] for r in list(reg_labels)]+list(gen_labels)
    print([i%(n_colors-1) for i, n in enumerate(node_labels)])
    node_colors = [palette[i] for i, n in enumerate(node_labels)]
    node_indices = {label: idx for idx, label in enumerate(node_labels)}
    node_x = [0 if x in num_examples_reg else 1 for x in node_labels]
    node_reg = [num_examples_reg[i]/num_total for i in reg_labels]
    node_gen = [num_examples_gen[i]/num_total for i in gen_labels]
    print(node_gen)
    node_y = (np.cumsum(np.array(node_reg)-node_reg[0])).tolist() + [0] + (np.cumsum(np.array(node_gen))).tolist()[:-1]
    #for i,j in zip(num_examples_gen,(np.cumsum(np.array(node_gen)-node_gen[0])).tolist()):
    #    print(i, ";", j)
    #exit()

    for label, ind, x,y in zip(node_labels, node_indices, node_x, node_y):
        print(f'{label} ({ind}): \t {x} \t\t {y}')
    #print(node_labels)
    #print(reg_labels)
    #print(gen_labels)
    #print(node_y)

    # FOCUS: reorder to make focus last
    #desired_order_list = [k for k,v in data.items()]
    #desired_order_list.remove(focus)
    #desired_order_list.append(focus)
    #data = {k: data[k] for k in desired_order_list}

    
    # Populate the source, target, and value lists
    i = 0
    for main_key, sub_dict in data.items():
        print(main_key)
        main_key = register_names[main_key]
        for sub_key, value in sub_dict.items():
            sources.append(node_indices[main_key])
            targets.append(node_indices[sub_key])
            if options.use_log > 0:
                values.append(np.log(value) / np.log(options.use_log))
                #values.append(np.log2(value))
            else:
                values.append(value)
            if options.focus in register_names and main_key == register_names[options.focus]:
                colors.append('rgba(0, 0, 0, 1.0)')
            else:
                colors.append('rgba(211, 211, 211, 0.5)')
        i+=1

    
    # Define the Sankey diagram
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15, thickness=20, color=node_colors, line=dict(color="white", width=0.5), label=node_labels,# x= node_x, y=node_y,
            ),
            link=dict(source=sources, target=targets, value=values, color=colors),
        )
    )
    
    # Update layout
    #fig.update_layout(title_text="\tIntersection of predicted register and genre labels", font_size=30)
    fig.update_layout(
    font_size=30,
    autosize=False,
    width=1200,
    height=650,
)
    
    # Show the plot
    address = "plots/"
    if options.focus is not None:
        address += "focus_plots/fig_"+str(options.focus)+"_"
    address+="reg_oscar_log-"+str(options.use_log)+".png"
    folder = "/".join(address.split("/")[0:-1])+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f'Saving to {address}')
    #fig.write_image("plots/focus_plots/fig_NEW_Focus_"+focus+"_register_oscar_grey_correct_thresholds"+str(use_log)+"_"+str(colormap_name)+".png")
    fig.write_image(address)
    
    
if __name__ == "__main__":
    #path = sys.argv[1]
    #focus = sys.argv[2]
    #use_log = int(sys.argv[4]) if len(sys.argv) > 2 else 0
    #c_map = sys.argv[3] if len(sys.argv) > 3 else "Set3"
    options = parse_arguments()
    print(options)
    run(options)