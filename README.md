# Roblox App Review Sentiment Analysis

This repository contains an end-to-end project for performing sentiment analysis on user reviews of the Roblox app. The project is divided into two main stages:

1.  **Data Scraping:** Collecting thousands of user reviews directly from the Google Play Store.
2.  **Model Training:** Building and training a deep learning model using TensorFlow and Keras to classify the sentiment of the scraped reviews as positive or negative.

The final model successfully classifies user sentiment with **88% - 92% accuracy**, creating a valuable tool for automatically processing and understanding user feedback at scale.

---

## Project Pipeline

### Part 1: Data Scraping

* **Tool:** The `google-play-scraper` Python library was used to programmatically collect reviews.
* **Scope:** Scraped a total of 15,000 user reviews for the Roblox app (`com.roblox.client`).
* **Data Collected:** The scraper gathered the review content and the corresponding user rating (score).
* **Output:** The raw data was saved to a `roblox_reviews.csv` file for use in the next stage.

### Part 2: Sentiment Analysis Model

* **Data Preprocessing:** The scraped reviews were cleaned. Ratings of 1-2 were labeled as `negative` (0), and ratings of 4-5 were labeled as `positive` (1). Reviews with a rating of 3 were discarded to create a clear binary classification problem.
* **Text Vectorization:** The text data was converted into numerical format using TensorFlow's `Tokenizer`, which transforms sentences into sequences of integers. Padding was applied to ensure all sequences had the same length.
* **Model Architecture:** A Sequential model was built with the following layers:
    * `Embedding` layer to learn vector representations of words.
    * `LSTM` (Long Short-Term Memory) layer to process sequential data.
    * `Dense` layers with `ReLU` and `Sigmoid` activation functions for classification.
* **Training & Callbacks:** The model was trained for 10 epochs. An `EarlyStopping` callback was used to prevent overfitting by monitoring the validation loss.

---

## Tech Stack

* **Data Scraping:** `google-play-scraper`, `pandas`
* **Data Analysis & Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Deep Learning:** `tensorflow`, `keras`
* **Machine Learning Utilities:** `scikit-learn`
* **Environment:** Jupyter Notebook, Google Colab

---

## Results

The trained deep learning model demonstrated strong performance in classifying review sentiment.

* **Final Validation Accuracy:** **92.21%**
* **Final Validation Loss:** 0.20

The training history shows a stable learning curve, with the model converging well without significant overfitting, thanks to the use of callbacks.

---

## How to Reproduce

To replicate this project, you need to run the two main components in order.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/Roblox-Sentiment-Analysis.git
    cd Roblox-Sentiment-Analysis
    ```

2.  **Part A: Scrape the Data**
    * Install the required library: `pip install google-play-scraper pandas`
    * Run the `Ifan_Hakim_Scrapping_from_Playstore_Roblox.ipynb` notebook.
    * This will generate the `roblox_reviews.csv` file.

3.  **Part B: Train the Model**
    * Ensure you have TensorFlow and other data science libraries installed: `pip install tensorflow scikit-learn matplotlib seaborn`
    * Run the `Ifan_Hakim_Proyek_Analisis_Sentimen_Belajar_Fundamental_Deep_Learning.ipynb` notebook.
    * This notebook will load the CSV file, preprocess the data, and train the sentiment analysis model.
