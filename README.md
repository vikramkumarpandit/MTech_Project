# Fake News Detection Project

## Overview
This project focuses on building a Machine Learning model to detect fake news articles using Natural Language Processing (NLP) techniques. The dataset used in this project contains labeled news articles categorized as real or fake. The objective is to preprocess the text data, extract relevant features, and train a classification model to accurately distinguish between real and fake news.

## Features
- **Text Preprocessing**: Includes removing stop words, stemming, and vectorization.
- **Machine Learning Models**: Logistic Regression, Naive Bayes, and Random Forest are used to classify the news articles.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score are calculated to evaluate model performance.

## Dataset
The dataset used in this project can be accessed from Kaggle: 
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

### Dataset Details:
- `Fake.csv`: Contains fake news articles with columns like `title`, `text`, `subject`, and `date`.
- `True.csv`: Contains real news articles with similar columns.

## Requirements
The project requires the following libraries:

```bash
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
FakeNewsDetection/
├── data/
│   ├── Fake.csv
│   ├── True.csv
├── notebooks/
│   ├── DataPreprocessing.ipynb
│   ├── ModelTraining.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
├── requirements.txt
├── README.md
```

## Preprocessing
- The `text` column is preprocessed by removing stop words, performing stemming, and converting text to lowercase.
- TF-IDF Vectorization is applied to convert text data into numerical features.

## Model Training
The following models are implemented:
1. **Logistic Regression**
2. **Naive Bayes**
3. **Random Forest**

### Key Steps:
1. Split the dataset into training and testing sets (80-20 split).
2. Train the models using the TF-IDF features.
3. Evaluate models using the testing set.

## Evaluation
The models are evaluated based on the following metrics:
- **Accuracy**: Percentage of correctly classified articles.
- **Precision**: Measure of true positive predictions.
- **Recall**: Measure of identifying relevant instances.
- **F1-Score**: Weighted average of Precision and Recall.

## Results
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 92%      | 91%       | 93%    | 92%      |
| Naive Bayes        | 89%      | 88%       | 90%    | 89%      |
| Random Forest      | 94%      | 93%       | 95%    | 94%      |

## Future Scope
1. Incorporate deep learning models like LSTM or Transformers for improved accuracy.
2. Explore additional datasets to enhance generalization.
3. Deploy the model using Flask or FastAPI for real-time predictions.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/FakeNewsDetection.git
```
2. Navigate to the project directory and install dependencies.
3. Run the preprocessing and model training scripts.
4. Evaluate the model and generate predictions.

## Contributors
- **Vikram Kumar**  
  [GitHub](https://github.com/vikramkumarpandit) | [Email](mailto:vikramkumar.pdt@gmail.com)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
