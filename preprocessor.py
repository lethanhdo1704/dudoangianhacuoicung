"""
Real Estate Preprocessor Module
This module must be imported before loading any pickled models that use RealEstatePreprocessor
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


class RealEstatePreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for real estate data.
    Prevents data leakage by fitting only on training data.
    """
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Store encoders and scalers (FITTED ON TRAIN ONLY!)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encodings = {}
        self.fill_values = {}
        self.feature_columns = None  # CRITICAL: Store feature order
        self.scaled_columns = None   # CRITICAL: Store which columns were scaled
        
        # Features to drop (leakage)
        self.leakage_cols = ['Price_per_m2', 'Price_category', 'Area_category', 'Ward']
        
        # Location scores
        self.city_base_scores = {
            'Hồ Chí Minh': 50,
            'Hà Nội': 50,
        }
        
        self.district_scores = {
            'Hồ Chí Minh': {
                'Quận 1': 10, 'Quận 3': 9, 'Bình Thạnh': 8, 'Phú Nhuận': 8,
                'Quận 2': 7, 'Quận 7': 7, 'Quận 10': 7, 'Tân Bình': 7,
                'Quận 5': 6, 'Quận 6': 6, 'Gò Vấp': 6, 'Quận 8': 5,
                'Quận 9': 5, 'Thủ Đức': 5, 'Quận 12': 4, 'Tân Phú': 6,
                'Bình Tân': 4, 'Bình Chánh': 4, 'Hóc Môn': 3,
                'Củ Chi': 2, 'Nhà Bè': 3, 'Cần Giờ': 1
            },
            'Hà Nội': {
                'Hoàn Kiếm': 10, 'Ba Đình': 9, 'Đống Đa': 8, 'Hai Bà Trưng': 8,
                'Cầu Giấy': 7, 'Thanh Xuân': 7, 'Tây Hồ': 7,
                'Long Biên': 6, 'Hoàng Mai': 6, 'Nam Từ Liêm': 6, 'Bắc Từ Liêm': 6,
                'Hà Đông': 5, 'Đông Anh': 4, 'Gia Lâm': 4, 'Thanh Trì': 4,
                'Sóc Sơn': 3, 'Ba Vì': 2, 'Mỹ Đức': 2, 'Chương Mỹ': 2,
                'Thường Tín': 3, 'Mê Linh': 3, 'Hoài Đức': 3, 'Thạch Thất': 2
            }
        }
        
    def clean_strings(self, df):
        """Step 1: Normalize string columns"""
        string_cols = ['City', 'District']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()
                df[col] = df[col].str.replace(r'\.$', '', regex=True)
        return df
    
    def remove_leakage(self, df):
        """Step 2: Remove leakage columns"""
        cols_to_drop = [col for col in self.leakage_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df
    
    def handle_missing(self, df, is_train=True):
        """Step 3: Handle missing values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if is_train:
                    self.fill_values[col] = df[col].median()
                fill_val = self.fill_values.get(col, df[col].median())
                df[col].fillna(fill_val, inplace=True)
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isna().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def remove_bad_records(self, df):
        """Step 4: Remove invalid records"""
        initial_shape = df.shape[0]
        df = df[(df['Price'] > 0) & (df['Area'] > 0)]
        df = df.drop_duplicates()
        return df
    
    def handle_outliers(self, df, method='percentile'):
        """Step 5: Handle outliers"""
        cols = ['Price', 'Area', 'Frontage', 'Access Road']
        
        for col in cols:
            if col not in df.columns:
                continue
                
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
        
        return df
    
    def engineer_features(self, df):
        """Step 6: Feature engineering - NO LAMBDA (pickle-safe)"""
        # Basic features
        if 'Total_rooms' not in df.columns:
            df['Total_rooms'] = df['Bedrooms'] + df['Bathrooms']
        
        df['Bedroom_Bathroom_ratio'] = df['Bedrooms'] / df['Bathrooms'].replace(0, 1)
        df['Area_per_floor'] = df['Area'] / df['Floors'].replace(0, 1)
        df['Room_density'] = df['Total_rooms'] / df['Area']
        
        # Luxury score
        binary_features = ['Has_Frontage', 'Has_Access_Road', 
                          'Has_House_Direction', 'Has_Balcony_Direction']
        df['Luxury_score'] = sum(df[col] for col in binary_features if col in df.columns)
        
        # Location features
        df['City_Base_Score'] = df['City'].map(self.city_base_scores).fillna(40)
        
        # District score (no lambda - pickle-safe)
        district_scores = []
        for idx, row in df.iterrows():
            city = row['City']
            district = row['District']
            score = self.district_scores.get(city, {}).get(district, 3)
            district_scores.append(score)
        df['District_Score'] = district_scores
        
        df['Location_Score'] = df['City_Base_Score'] + df['District_Score']
        df['Location_Tier'] = pd.cut(
            df['Location_Score'],
            bins=[0, 50, 55, 60, 100],
            labels=['Suburban', 'Urban', 'Premium', 'Elite']
        )
        
        # Other features
        df['Is_Apartment'] = df['Address'].str.contains('Dự án|Project', case=False, na=False).astype(int)
        df['Full_Legal'] = (df['Legal status'] == 'Have Certificate').astype(int)
        df['Full_Furniture'] = (df['Furniture state'] == 'Full').astype(int)
        
        return df
    
    def target_encode_kfold(self, df, col, target='Price', is_train=True):
        """Target encoding with K-Fold (prevents leakage)"""
        if is_train:
            df[f'{col}_target_enc'] = 0.0
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in kf.split(df):
                train_data = df.iloc[train_idx]
                target_means = train_data.groupby(col)[target].mean()
                global_mean = train_data[target].mean()
                
                val_data = df.iloc[val_idx]
                encoded_values = val_data[col].map(target_means).fillna(global_mean)
                df.iloc[val_idx, df.columns.get_loc(f'{col}_target_enc')] = encoded_values.values
            
            self.target_encodings[col] = df.groupby(col)[target].mean().to_dict()
            self.target_encodings[f'{col}_global_mean'] = df[target].mean()
        else:
            global_mean = self.target_encodings.get(f'{col}_global_mean', 0)
            df[f'{col}_target_enc'] = df[col].map(self.target_encodings[col]).fillna(global_mean)
        
        return df
    
    def encode_categoricals(self, df, is_train=True):
        """Step 7: Encode categorical variables - NO LAMBDA (pickle-safe)"""
        df = self.target_encode_kfold(df, 'District', is_train=is_train)
        df = self.target_encode_kfold(df, 'City', is_train=is_train)
        
        label_cols = ['Legal status', 'Furniture state', 'Location_Tier']
        for col in label_cols:
            if col not in df.columns:
                continue
            
            if is_train:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                encoded_values = []
                for value in df[col].astype(str):
                    if value in self.label_encoders[col].classes_:
                        encoded_values.append(self.label_encoders[col].transform([value])[0])
                    else:
                        encoded_values.append(-1)
                df[f'{col}_encoded'] = encoded_values
        
        return df
    
    def scale_features(self, df, is_train=True):
        """Step 8: Scale numeric features"""
        exclude = ['Price', 'Address']
        
        if is_train:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude]
            
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.scaled_columns = numeric_cols
        else:
            if self.scaled_columns is None:
                raise ValueError("Scaler not fitted yet. Call fit_transform first.")
            
            cols_to_scale = [col for col in self.scaled_columns if col in df.columns]
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df
    
    def fit_transform(self, df):
        """Complete preprocessing for TRAINING data"""
        df = df.reset_index(drop=True)
        
        df = self.clean_strings(df)
        df = self.remove_leakage(df)
        df = self.handle_missing(df, is_train=True)
        df = self.remove_bad_records(df)
        df = self.handle_outliers(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df, is_train=True)
        
        y = df['Price'].copy()
        X = df.drop(columns=['Price', 'Address', 'City', 'District', 
                             'Legal status', 'Furniture state', 'Location_Tier',
                             'House direction', 'Balcony direction'], 
                    errors='ignore')
        
        X = self.scale_features(X, is_train=True)
        self.feature_columns = list(X.columns)
        
        return X, y
    
    def transform(self, df):
        """Apply preprocessing to TEST/NEW data"""
        df = df.reset_index(drop=True)
        
        df = self.clean_strings(df)
        df = self.remove_leakage(df)
        df = self.handle_missing(df, is_train=False)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df, is_train=False)
        
        X = df.drop(columns=['Price', 'Address', 'City', 'District', 
                             'Legal status', 'Furniture state', 'Location_Tier',
                             'House direction', 'Balcony direction'], 
                    errors='ignore')
        X = self.scale_features(X, is_train=False)
        
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(self.feature_columns)
            
            if missing_cols:
                for col in missing_cols:
                    X[col] = 0
            
            if extra_cols:
                X = X.drop(columns=list(extra_cols))
            
            X = X[self.feature_columns]
        
        return X