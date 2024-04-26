from metaflow import FlowSpec, step
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import data_processing as dp

class DataAnalysisFlow(FlowSpec):

    @step
    def start(self):
        self.df = dp.load_data('open_ave_data.xlsx')
        self.next(self.calculate_lengths)

    @step
    def calculate_lengths(self):
        self.df = dp.calculate_lengths(self.df)
        self.next(self.summary_statistics)

    @step
    def summary_statistics(self):
        self.stats = dp.summary_statistics(self.df)
        print(self.stats)
        self.next(self.process_text)

    @step
    def process_text(self):
        self.df = dp.process_text(self.df)
        self.next(self.generate_wordcloud_findings)

    @step
    def generate_wordcloud_findings(self):
        dp.generate_wordclouds(self.df, 'findings')
        self.next(self.generate_wordcloud_impression)

    @step
    def generate_wordcloud_impression(self):
        dp.generate_wordclouds(self.df, 'impression')
        self.next(self.tfidf_tsne_analysis)

    @step
    def tfidf_tsne_analysis(self):
        dp.perform_tfidf_tsne_analysis(self.df)
        self.next(self.prepare_text_for_classifier)

    @step
    def prepare_text_for_classifier(self):
        self.texts, self.field_types = dp.prepare_text_data(self.df)
        self.next(self.train_classifier)

    @step
    def train_classifier(self):
        self.tfidf_accuracy, self.tfidf_report, self.tfidf_f1 = dp.train_classifier(self.texts, self.field_types)
        print("TF-IDF Classifier Accuracy:", self.tfidf_accuracy)
        print("TF-IDF Classifier Report:\n", self.tfidf_report)
        self.next(self.prepare_text_for_word2vec)

    @step
    def prepare_text_for_word2vec(self):
        self.texts, self.field_types = dp.prepare_text_data(self.df)
        self.next(self.train_word2vec_model)

    @step
    def train_word2vec_model(self):
        self.model, self.avg_vectors = dp.train_and_get_word2vec(self.texts)
        self.next(self.train_word2vec_logistic_classifier)

    @step
    def train_word2vec_logistic_classifier(self):
        self.word2vec_accuracy, self.word2vec_report, self.word2vec_f1 = dp.train_word2vec_logistic_classifier(self.avg_vectors, self.field_types)
        print("Word2Vec Logistic Classifier Accuracy:", self.word2vec_accuracy)
        print("Word2Vec Logistic Classifier Report:\n", self.word2vec_report)
        self.next(self.compare_and_save_best_model)

    @step
    def compare_and_save_best_model(self):
        dp.compare_and_save_models(self.tfidf_f1, self.word2vec_f1, self.texts, self.field_types, self.model)
        self.next(self.end)

    @step
    def end(self):
        pass
if __name__ == '__main__':
    DataAnalysisFlow()