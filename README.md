created by Markus Krarup & Valdemar Schultz 14.05.2025

This Github repository contains all relevant data and python code in form of Jupyter Nottebooks.

Notebooks
 1. data_prep_cleaning.ipynb 		       --> contains  all the data prepartion and cleaning done. 
 2. Exploratory_data_analysis.ipynb		 --> containins all the Exploratory Data Analysis Done
 3. traditional_modelling.ipynb        --> containins python code used for building the traditional models
 4. NN.ipynb	                         --> containings the python code used for building our Neural Netowrk
 5. pretrained_model.ipynb		         --> containins all python code and the zero-shot prompts used for our pre trained models via watsonX
 6. agent.ipynb                        --> contains all python code for embedding few-shots, building the agent graph, and prompting. 

Data:
 * Data/Goemotions 1,2 & 3				      : Contains the 3 unfiltered datasets downloaded from the google-research API
 * Data/combined.csv                    : the 3 unfiltered datasets combined
 * Data/ekmann; test, train & Val	      : Contains all the preprocssed data from the original GoEmotions dataset split into train test and val sets. 
 * Data/definitions_ekman_emotions.csv	: definitions used to ground the LLM's
 * Data/reviews.csv	 		                :  file containing our self labelled data with customer reviews from different platforms.
 * Data/ekman_test_with_predictions     : shows how the LLM with no grounding is guessing many emotions

Other:
 * /src/LLM.py -             : our instructor used to recieve structured outputs from the LLM's
 * ekman_mapping.json				: containings the mapping of GoEmotions taxonomy to Ekmans taxonomy
