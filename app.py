from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import os
import gc
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT PREPROCESSOR FROM SEPARATE MODULE
# ============================================================================
from preprocessor import RealEstatePreprocessor

app = Flask(__name__)

# ============================================================================
# MEMORY-OPTIMIZED CONFIGURATION
# ============================================================================
MODEL_DIR = 'models'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
LOCATIONS_PATH = 'locations.json'
CHOICES_PATH = 'choices.json'

# LAZY LOADING: Models loaded on-demand
loaded_models = {}  # Cache for loaded models
model_info_list = []
metadata = {}
locations = {}
choices = {}
preprocessor = None

# Model file mapping
MODEL_FILES = {
    'Linear_Regression': 'Linear_Regression.joblib',
    'Ridge_Regression': 'Ridge_Regression.joblib',
    'Lasso_Regression': 'Lasso_Regression.joblib',
    'Random_Forest': 'Random_Forest.joblib',
    'Gradient_Boosting': 'Gradient_Boosting.joblib',
    'XGBoost': 'XGBoost.joblib',
    'Voting_Regressor': 'Voting_Regressor.joblib',
    'Stacking_Regressor': 'Stacking_Regressor.joblib'
}

# ============================================================================
# LAZY LOADING FUNCTIONS
# ============================================================================
def load_model(model_name):
    """Lazy load a model only when needed"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    model_file = MODEL_FILES.get(model_name)
    if not model_file:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_path = f'{MODEL_DIR}/{model_file}'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"⏳ Loading {model_name}...")
    model = joblib.load(model_path)
    
    # Cache only the most recently used model to save memory
    # Clear cache if we have more than 2 models loaded
    if len(loaded_models) >= 2:
        # Remove the oldest model (first item)
        oldest_model = next(iter(loaded_models))
        print(f"🗑️ Unloading {oldest_model} to free memory")
        del loaded_models[oldest_model]
        gc.collect()  # Force garbage collection
    
    loaded_models[model_name] = model
    print(f"✅ {model_name} loaded")
    return model


def initialize_app():
    """Initialize app data (lightweight - no models loaded)"""
    global model_info_list, metadata, locations, choices, preprocessor
    
    print("\n" + "="*80)
    print("🚀 REAL ESTATE PRICE PREDICTION API - MEMORY OPTIMIZED")
    print("="*80)
    
    try:
        # Load metadata (lightweight)
        with open(f'{MODEL_DIR}/best_model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load model comparison data (lightweight)
        comparison_df = pd.read_csv(f'{MODEL_DIR}/model_comparison.csv')
        
        # Build model info list (NO models loaded yet)
        for model_name, model_file in MODEL_FILES.items():
            model_path = f'{MODEL_DIR}/{model_file}'
            if os.path.exists(model_path):
                model_display_name = model_name.replace('_', ' ')
                model_stats = comparison_df[comparison_df['Model'] == model_display_name]
                
                if not model_stats.empty:
                    stats = model_stats.iloc[0]
                    model_info_list.append({
                        'id': model_name,
                        'name': model_display_name,
                        'test_r2': float(stats['Test_R²']),
                        'test_mae': float(stats['Test_MAE']),
                        'test_rmse': float(stats['Test_RMSE']),
                        'mape': float(stats['MAPE (%)']),
                        'tuned': stats['Tuned'] == '✅ Yes',
                        'is_best': model_name == metadata['best_model_file'].replace('.joblib', '')
                    })
        
        model_info_list.sort(key=lambda x: x['test_r2'], reverse=True)
        
        # Load preprocessor (necessary for all predictions)
        print("\n📦 Loading preprocessor...")
        
        # WORKAROUND: Make the class available in both preprocessor module and __main__
        import sys
        sys.modules['__main__'].RealEstatePreprocessor = RealEstatePreprocessor
        
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("✅ Preprocessor loaded")
        except AttributeError as e:
            print(f"⚠️ Import error, trying alternative method: {e}")
            # Fallback: Load with custom unpickler
            import pickle
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            print("✅ Preprocessor loaded (fallback method)")
        
        # Load locations and choices (lightweight JSON)
        with open(LOCATIONS_PATH, 'r', encoding='utf-8') as f:
            locations = json.load(f)
        
        with open(CHOICES_PATH, 'r', encoding='utf-8') as f:
            choices = json.load(f)
        
        print(f"\n✅ Models available: {len(model_info_list)}")
        print(f"✅ Best Model: {metadata['best_model_name']} (will load on first use)")
        print(f"✅ Locations: {len(locations)} provinces")
        print(f"✅ Memory mode: LAZY LOADING (models load on-demand)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


# Initialize on startup (lightweight)
initialize_app()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_currency(value):
    """Format currency value"""
    return f"{value:,.2f}"


def validate_input(data):
    """Validate input data"""
    errors = []
    
    required_fields = {
        'Area': 'Diện tích',
        'Bedrooms': 'Số phòng ngủ',
        'Bathrooms': 'Số phòng tắm',
        'Floors': 'Số tầng',
        'City': 'Tỉnh/Thành phố'
    }
    
    for field, display_name in required_fields.items():
        if field not in data or data[field] == '' or data[field] is None:
            errors.append(f"Vui lòng nhập {display_name}")
    
    numeric_validations = {
        'Area': (10, 10000, 'Diện tích'),
        'Bedrooms': (0, 20, 'Số phòng ngủ'),
        'Bathrooms': (0, 10, 'Số phòng tắm'),
        'Floors': (1, 20, 'Số tầng'),
        'Frontage': (0.1, 100, 'Mặt tiền'),
        'Access Road': (0.1, 100, 'Đường vào')
    }
    
    for field, (min_val, max_val, display_name) in numeric_validations.items():
        if field in data and data[field] != '' and data[field] is not None:
            try:
                value = float(data[field])
                if value < min_val or value > max_val:
                    errors.append(f"{display_name} phải từ {min_val} đến {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{display_name} phải là số")
    
    return errors


def prepare_input_dataframe(data):
    """Convert form data to DataFrame"""
    house_direction = data.get('House direction', '')
    balcony_direction = data.get('Balcony direction', '')
    frontage = data.get('Frontage', '')
    access_road = data.get('Access Road', '')
    
    input_data = {
        'Address': str(data.get('Address', 'N/A')),
        'Area': float(data.get('Area', 0)),
        'Frontage': float(frontage) if frontage and str(frontage).strip() != '' else np.nan,
        'Access Road': float(access_road) if access_road and str(access_road).strip() != '' else np.nan,
        'Floors': float(data.get('Floors', 1)),
        'Bedrooms': float(data.get('Bedrooms', 0)),
        'Bathrooms': float(data.get('Bathrooms', 0)),
        'Legal status': str(data.get('Legal status', 'N/A')),
        'Furniture state': str(data.get('Furniture state', 'N/A')),
        'Price': 0.0,
        'City': str(data.get('City', '')),
        'District': str(data.get('District', 'N/A')),
        'House direction': str(house_direction),
        'Balcony direction': str(balcony_direction),
        'Has_Frontage': 1,
        'Has_Access_Road': 1,
        'Has_House_Direction': int(1 if house_direction and str(house_direction).strip() != '' else 0),
        'Has_Balcony_Direction': int(1 if balcony_direction and str(balcony_direction).strip() != '' else 0),
    }
    
    return pd.DataFrame([input_data])


# Translation dictionaries
LEGAL_STATUS_VI = {
    'Have Certificate': 'Có sổ',
    'No Certificate': 'Chưa có sổ',
    'Waiting for Certificate': 'Đang chờ sổ',
    '': 'Không xác định',
    'N/A': 'Không xác định'
}

FURNITURE_STATE_VI = {
    'Full': 'Đầy đủ',
    'Basic': 'Cơ bản',
    'Empty': 'Trống',
    '': 'Không xác định',
    'N/A': 'Không xác định'
}


# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/locations')
def get_locations():
    return jsonify(locations)


@app.route('/api/choices')
def get_choices():
    return jsonify(choices)


@app.route('/api/models')
def get_models():
    return jsonify({
        'success': True,
        'models': model_info_list
    })


@app.route('/api/model-info')
def get_model_info():
    return jsonify({
        'model_name': metadata['best_model_name'],
        'test_r2': metadata['test_r2'],
        'test_mae': metadata['test_mae'],
        'test_rmse': metadata['test_rmse'],
        'mape': metadata['mape'],
        'training_date': metadata['training_date'],
        'n_features': metadata['n_features'],
        'n_train_samples': metadata['n_train_samples']
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_id = data.get('model_id', 'Stacking_Regressor')
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # LAZY LOAD: Load model only when needed
        try:
            selected_model = load_model(model_id)
        except Exception as e:
            return jsonify({
                'success': False,
                'errors': [f'Không thể tải model: {str(e)}']
            }), 400
        
        # Get model stats
        model_stats = next((m for m in model_info_list if m['id'] == model_id), None)
        if not model_stats:
            model_stats = {
                'name': model_id.replace('_', ' '),
                'test_r2': 0,
                'test_mae': 0,
                'test_rmse': 0,
                'mape': 0
            }
        
        # Prepare and transform input
        input_df = prepare_input_dataframe(data)
        X_input = preprocessor.transform(input_df)
        
        # Safety check
        if isinstance(X_input, tuple):
            X_input = X_input[0]
        
        # Convert to numpy array
        X_array = np.asarray(X_input, dtype=np.float64)
        
        # Predict
        prediction = selected_model.predict(X_array)[0]
        
        # Calculate confidence interval
        rmse = model_stats['test_rmse']
        lower_bound = max(0, prediction - 1.96 * rmse)
        upper_bound = prediction + 1.96 * rmse
        
        result = {
            'success': True,
            'prediction': {
                'value': float(prediction),
                'formatted': format_currency(prediction),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'lower_formatted': format_currency(lower_bound),
                'upper_formatted': format_currency(upper_bound)
            },
            'input_summary': {
                'Diện tích': f"{data['Area']} m²",
                'Phòng ngủ': data['Bedrooms'],
                'Phòng tắm': data['Bathrooms'],
                'Số tầng': data['Floors'],
                'Mặt tiền': f"{data.get('Frontage', 0)} m",
                'Đường vào': f"{data.get('Access Road', 0)} m",
                'Tỉnh/Thành': data['City'],
                'Quận/Huyện': data.get('District', 'N/A'),
                'Pháp lý': LEGAL_STATUS_VI.get(data.get('Legal status', ''), 'Không xác định'),
                'Nội thất': FURNITURE_STATE_VI.get(data.get('Furniture state', ''), 'Không xác định')
            },
            'model_info': {
                'id': model_id,
                'name': model_stats['name'],
                'r2_score': model_stats['test_r2'],
                'mae': model_stats['test_mae'],
                'rmse': model_stats['test_rmse'],
                'mape': model_stats.get('mape', 0),
                'is_tuned': model_stats.get('tuned', False),
                'is_best': model_stats.get('is_best', False)
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ Prediction: {format_currency(prediction)} using {model_stats['name']}")
        
        # Clean up
        gc.collect()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'errors': [f'Lỗi dự đoán: {str(e)}']
        }), 500


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    try:
        data = request.json
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Prepare input once
        input_df = prepare_input_dataframe(data)
        X_input = preprocessor.transform(input_df)
        
        # Safety check
        if isinstance(X_input, tuple):
            X_input = X_input[0]
        
        X_array = np.asarray(X_input, dtype=np.float64)
        
        predictions = []
        
        # Load and predict with each model one at a time
        for model_id in MODEL_FILES.keys():
            try:
                # Load model (lazy loading)
                model = load_model(model_id)
                
                # Predict
                pred = model.predict(X_array)[0]
                
                # Get stats
                model_stats = next((m for m in model_info_list if m['id'] == model_id), None)
                
                predictions.append({
                    'model_id': model_id,
                    'model_name': model_id.replace('_', ' '),
                    'prediction': float(pred),
                    'formatted': format_currency(pred),
                    'r2_score': model_stats['test_r2'] if model_stats else 0,
                    'mae': model_stats['test_mae'] if model_stats else 0,
                    'is_best': model_stats['is_best'] if model_stats else False
                })
                
                # Force cleanup after each prediction
                del model
                gc.collect()
                
            except Exception as e:
                print(f"⚠️ Error with {model_id}: {e}")
                continue
        
        # Sort by R² score
        predictions.sort(key=lambda x: x['r2_score'], reverse=True)
        
        # Calculate statistics
        pred_values = [p['prediction'] for p in predictions]
        avg_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        min_pred = min(pred_values)
        max_pred = max(pred_values)
        
        result = {
            'success': True,
            'predictions': predictions,
            'statistics': {
                'average': float(avg_pred),
                'average_formatted': format_currency(avg_pred),
                'std_dev': float(std_pred),
                'min': float(min_pred),
                'min_formatted': format_currency(min_pred),
                'max': float(max_pred),
                'max_formatted': format_currency(max_pred),
                'range': float(max_pred - min_pred),
                'range_formatted': format_currency(max_pred - min_pred)
            }
        }
        
        # Clean up
        gc.collect()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Compare models error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'errors': [str(e)]
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    import psutil
    import sys
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return jsonify({
            'status': 'healthy',
            'memory_mb': memory_info.rss / 1024 / 1024,
            'models_loaded': len(loaded_models),
            'python_version': sys.version
        })
    except:
        return jsonify({
            'status': 'healthy',
            'models_loaded': len(loaded_models)
        })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'errors': ['Không tìm thấy trang']}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'errors': ['Lỗi server']}), 500


if __name__ == '__main__':
    print("\n🌐 Starting Flask server (Memory Optimized)...")
    port = int(os.environ.get('PORT', 5000))  # Lấy PORT từ Render
    print(f"📍 Access at: http://0.0.0.0:{port}")
    print("💾 Models will load on-demand to save memory")
    print("Press CTRL+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=port)