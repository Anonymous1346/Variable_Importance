import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import imgkit
import os

nb_features = 8

# modified from http://cssmenumaker.com/br/blog/stylish-css-tables-tutorial
css = """
<style type=\"text/css\">
table {
color: #333;
font-family: Helvetica, Arial, sans-serif;
width: 1560px;
font-size: 250%;
font-strech:condensed;
border-collapse:
collapse; 
border-spacing: 0;
}
td, th {
border: 1px solid transparent; /* No more visible border */
# height: 100px;
# width: 150px;
}
th {
background: #DFDFDF; /* Darken header a bit */
font-weight: bold;
word-wrap: break-word;
max-width: 250px;
}
td {
background: #FAFAFA;
text-align: center;
word-wrap: break-word;
max-width: 250px;
}
table tr:nth-child(odd) td{
background-color: white;
}
table tr td:nth-child(1) {
    text-align: left;
}
</style>
"""

pd.reset_option('display.float_format', silent=True)
from collections import OrderedDict
from ukbb_variables import (brain_dmri_fa, brain_dmri_icvf,
                            brain_dmri_isovf, brain_dmri_l1,
                            brain_dmri_l2, brain_dmri_l3,
                            brain_dmri_md, brain_dmri_mo,
                            brain_dmri_od, brain_smri_plus,
                            earlylife, fluid_intelligence, brain_smri,
                            education, lifestyle, mental_health)

codes = OrderedDict(list(brain_dmri_fa.items()) + list(brain_dmri_icvf.items()) + \
                    list(brain_dmri_isovf.items()) + list(brain_dmri_l1.items()) + \
                    list(brain_dmri_l2.items()) + list(brain_dmri_l3.items()) + \
                    list(brain_dmri_md.items()) + list(brain_dmri_mo.items()) + \
                    list(brain_dmri_od.items()) + list(brain_smri_plus.items()) + \
                    list(earlylife.items()) + list(fluid_intelligence.items()) + \
                    list(brain_smri.items()) + list(education.items()) + \
                    list(lifestyle.items()) + list(mental_health.items()))

brain_image = OrderedDict(list(brain_dmri_fa.items()) + list(brain_dmri_icvf.items()) + \
                          list(brain_dmri_isovf.items()) + list(brain_dmri_l1.items()) + \
                          list(brain_dmri_l2.items()) + list(brain_dmri_l3.items()) + \
                          list(brain_dmri_md.items()) + list(brain_dmri_mo.items()) + \
                          list(brain_dmri_od.items()) + list(brain_smri_plus.items())).values()

def rename_numbers(x):
    if x.isdigit():
        return "rfMRI_" + x
    else:
        return x

def color_blue_red(val):
    if val in codes.values() or (val.split('_')[0] == 'rfMRI'):
        color = 'red' if ((val in brain_image) or (val.split('_')[0] == 'rfMRI')) else 'blue'
    else:
        color = 'black'
    return 'color: %s' % color

def conv_label(x, data):
    if len(x.split("-")) > 1:
        split_x = x.split("_")
        if len(split_x) > 1:
            return data[split_x[0]] + ":" + split_x[1]
        else:
            return data[split_x[0]]
    else:
        return x

df_1 = pd.read_csv("results_permfitdnn_intelligence_withscore_cross.csv")
df_1['variable'] = df_1['variable'].apply(lambda x: rename_numbers(x))
df_1['importance'] = df_1['importance'].apply(lambda x: "{:.2e}".format(x))
df_1['p_value'] = df_1['p_value'].apply(lambda x: "{:.2e}".format(x))
df_1 = df_1.astype({'p_value': 'str'})
res_df_1 = [conv_label(i, codes) for i in df_1.iloc[:nb_features, 0]]


df_2 = pd.read_csv("results_cpidnn_intelligence_withscore_cross.csv")
df_2['variable'] = df_2['variable'].apply(lambda x: rename_numbers(x))
df_2['importance'] = df_2['importance'].apply(lambda x: "{:.2e}".format(x))
df_2['p_value'] = df_2['p_value'].apply(lambda x: "{:.2e}".format(x))
df_2 = df_2.astype({'p_value': 'str'})
res_df_2 = [conv_label(i, codes) for i in df_2.iloc[:nb_features, 0]]

data_res = pd.DataFrame({('Permfit-DNN', 'Variable'):res_df_1,
    ('Permfit-DNN', 'Importance'):df_1.iloc[:nb_features, 1],
    ('Permfit-DNN', 'p_value'):df_1.iloc[:nb_features, 2],
    ('CPI-DNN', 'Variable'):res_df_2,
    ('CPI-DNN', 'Importance'):df_2.iloc[:nb_features, 1],
    ('CPI-DNN', 'p_value'):df_2.iloc[:nb_features, 2]})
data_res.columns = pd.MultiIndex.from_tuples(data_res.columns)

text_file = open("filename.html", "a")
text_file.write(css)
df = data_res.iloc[:, :3].style.applymap(color_blue_red).hide(axis='index')
text_file.write(df.to_html())
text_file.close()
imgkitoptions = {"format": "png"}
imgkit.from_file("filename.html", "permfit_intelligence_cross.png", options=imgkitoptions)
os.remove("filename.html")

text_file = open("filename.html", "a")
text_file.write(css)
df = data_res.iloc[:,3:].style.applymap(color_blue_red).hide(axis='index')
text_file.write(df.to_html())
text_file.close()
imgkitoptions = {"format": "png"}
imgkit.from_file("filename.html", "cpi_intelligence_cross.png", options=imgkitoptions)
os.remove("filename.html")
