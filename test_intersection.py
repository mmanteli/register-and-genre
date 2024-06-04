import plotly.graph_objects as go

# Given dictionary
data = {
    "A": {"a": 5, "b": 7, "c": 5},
    "B": {"a": 7, "b": 1, "c": 1},
    "C": {"a": 14, "b": 2, "c": 15},
}

# Extract the nodes and links
sources = []
targets = []
values = []

# Create a mapping of node names to indices
node_labels = list(data.keys()) + list(
    set(k for sub_dict in data.values() for k in sub_dict)
)
node_indices = {label: idx for idx, label in enumerate(node_labels)}

# Populate the source, target, and value lists
for main_key, sub_dict in data.items():
    for sub_key, value in sub_dict.items():
        sources.append(node_indices[main_key])
        targets.append(node_indices[sub_key])
        values.append(value)

# Define the Sankey diagram
fig = go.Figure(
    go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5), label=node_labels
        ),
        link=dict(source=sources, target=targets, value=values),
    )
)

# Update layout
fig.update_layout(title_text="Simple Sankey Diagram", font_size=10)

# Show the plot
fig.write_image("results/fig.png")