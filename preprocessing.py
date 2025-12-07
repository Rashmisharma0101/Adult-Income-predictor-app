from sklearn.base import BaseEstimator, TransformerMixin
import pandas  as pd

class AgeBinner(BaseEstimator, TransformerMixin):
    def __init__(self, bins=[0,25,35,50,65,100]):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['age_binned'] = pd.cut(X['age'], bins=self.bins, labels=False, include_lowest=True)
        return X.drop(columns=['age'])


class CountryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100):
        self.threshold = threshold
        self.common_countries = []

    def fit(self, X, y=None):
        counts = X['native-country'].value_counts()
        self.common_countries = counts[counts >= self.threshold].index.tolist()  # ensure list
        return self

    def transform(self, X):
        X = X.copy()
        # fallback: if somehow common_countries is empty, no error
        common = self.common_countries if self.common_countries else []
        X['native-country'] = X['native-country'].where(X['native-country'].isin(common), 'Other')
        return X


class CapitalPresence(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Capital-loss-indicator'] = (X['capital-loss'] > 0).astype(int)
        X['Capital-gain-indicator'] = (X['capital-gain'] > 0).astype(int)
        return X
    