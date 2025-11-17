


SEED = 42
SEQ_LENGTH = 168  # 7天*24小时
ID_COL = 'ID'
TARGET_COL = 'Hco2_h'


DATA_FILE = 'data.csv'
TEST_FILE = 'data_test.csv'
MODEL_PATH = 'best_model.pth'
FEATURE_COLS_PATH = 'feature_columns.pkl'
TRAIN_STATS_PATH = 'train_stats.pkl'


BATCH_SIZE = 32
HIDDEN_SIZE = 192
NUM_LAYERS = 6
EPOCHS = 200
PATIENCE = 15
VAL_SPLIT = 0.15


INPUT_COLS = [
    'soilT_heat',
    'soilT_cont',
    'lf_cont',
    'lf_heat'
]