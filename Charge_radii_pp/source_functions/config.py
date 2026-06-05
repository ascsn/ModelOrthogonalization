""" 
Will come back later
"""

models = [
    'ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS',
    'SKP', 'SLY4', 'SV', 'UNEDF0', 'UNEDF1'
]

fy_added_models = models + [
    'Fy_IVP', 'Fy_Dr'   # or the exact column name you decide to use
]

colors = [
    "#1f77b4", # Vivid blue
    "#8f623b", # Bright orange
    "#2ca02c", # Rich green
    "#d62728", # Strong red
    "#9467bd", # Deep purple
    "#8c564b", # Brownish-pink
    "#e377c2", # Pink
    "#7f7f7f", # Medium gray
    "#bcbd22", # Lime green
    "#17becf", # Cyan
    "#2A2EAC", # Dark blue
    "#637939", # Olive green
    "#8c6d31", # Bronze
    "#843c39", # Dark red
    "#ad494a", # Reddish brown
    "#d6616b", # Soft red
    "#e7ba52", # Golden yellow
    "#7b4173", # Dark purple
    "#a55194", # Mauve
    "#ce6dbd", # Light purple
]

colors_sets = [
    "#ff7f0e",

    "#1f77b4",

    "#2ca02c",
    "#d62728",
    
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    '#1f77b4',  # muted blue
    'r',  
] 

markers = [
    "o",  # Circle
    "^",  # Triangle up
    "s",  # Square
    "P",  # Plus (filled)
    "*",  # Star
    "X",  # X (filled)
    "D",  # Diamond
    "H",  # Hexagon
]


colors_models = colors


Selected_element=50
Selected_element_name="Sn"

color_train=colors_sets[9]
color_validation='orange'
color_test=colors_sets[3]

marker_train='s'
marker_validation='*'
marker_test='o'

size_train=30
size_validation=80
size_test=35

alpha_train=0.8
alpha_validation=0.9
alpha_test=0.4

heterogeneous_data_type = ['BE', 'ChRad', 'CPn', 'CPp', 'PEn', 'PEp', 'QDB2n', 'QDB2p', 'QDB4n', 'QDB4p', 'MRadN', 'MRadP']
num_properties = 2

DEFAULT_SPLIT_STYLE = {
    'train': {'color': colors_sets[9], 'marker': 's', 'alpha': 0.8},
    'validation': {'color': colors_sets[1], 'marker': '*', 'alpha': 0.9},
    'test': {'color': colors_sets[3], 'marker': 'o', 'alpha': 0.4},
}



DEFAULT_RANDOM_SEED = 142858