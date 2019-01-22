import os
import pandas as pd
import bs4

def createViz(data,folder='main'):

	# Save data and visualisation
	visualisation_path = 'visualisations/{}/'.format(folder)

	if not os.path.exists(visualisation_path):
		os.makedirs(visualisation_path)

	# Load and create main.html file
	with open("visualisations/main/main.html") as inf:
		txt = inf.read()
		soup = bs4.BeautifulSoup(txt, "html.parser")

	with open("{}index.html".format(visualisation_path), "w") as outf:
		outf.write(str(soup))

	data.to_csv('{}data.csv'.format(visualisation_path),index_label='text_reference')

