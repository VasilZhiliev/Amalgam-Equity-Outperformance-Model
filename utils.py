import pandas as pd
import numpy as np
import copy
from typing import List, Tuple, Dict, Literal
import xgboost as xgb
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from scipy.stats import ttest_rel
import itertools
from sklearn.metrics import log_loss
import numba
numba.set_num_threads(8)

def split_year(data, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data by the year. The year is included in the second part.
    """
    return data[data['y'] < year].copy(), data[data['y'] >= year].copy()

def split_y_qtr(data, y_qtr: Tuple[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data by the year and quarter. The year and quarter are included in the second part.
    """
    y, qtr = y_qtr
    return data[(data['y'] < y) | ((data['y'] == y) & (data['qtr'] < qtr))].copy(), data[(data['y'] > y) | ((data['y'] == y) & (data['qtr'] >= qtr))].copy()

def get_decile(data: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Get the decile of the target column.
    """
    return data.groupby(['y', 'qtr'])[target_col].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='raise'))

def get_is_top_decile(data: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Get the indicator of whether the value is in the top decile.
    """
    quantile = data.groupby(['y', 'qtr'])[target_col].transform('quantile', 0.9)
    return data[target_col] >= quantile

def get_decile_return(data: pd.DataFrame, decile_col: str, decile: int) -> pd.Series:
    """
    Get the return of the decile on each quarter.
    """
    return data[data[decile_col] == decile].groupby(['y', 'qtr'])['RET'].mean()

def get_index_returns(data, index='vwretd') -> pd.Series:
    """
    Get the returns of an index.
    """
    return data.groupby(['y', 'qtr'])[index].mean()

def get_market_portfolio_returns(data) -> pd.Series:
    """
    Get the value weighted market portfolio returns.
    """
    data = data.copy()
    data['MC_sum'] = data.groupby(['y', 'qtr'])['MC'].transform('sum')
    data['w'] = data['MC'] / data['MC_sum']
    data['w_ret'] = data['w'] * data['RET']
    return data.groupby(['y', 'qtr'])['w_ret'].sum()

def y_qtr_to_abs_qtr(y_qtr: Tuple[int, int]) -> int:
    """
    Convert the year and quarter to the absolute quarter.
    """
    return y_qtr[0] * 4 + (y_qtr[1] - 1)

def abs_qtr_to_y_qtr(abs_qtr: int) -> Tuple[int, int]:
    """
    Convert the absolute quarter to the year and quarter.
    """
    y_qtr = divmod(abs_qtr, 4)
    return y_qtr[0], y_qtr[1] + 1

def y_qtr_delta(y_qtr: Tuple[int, int], delta: int) -> Tuple[int, int]:
    """
    Add the delta to the year and quarter.
    """
    abs_qtr = y_qtr_to_abs_qtr(y_qtr) + delta
    return abs_qtr_to_y_qtr(abs_qtr)

def y_qtr_diff(y_qtr1: Tuple[int, int], y_qtr2: Tuple[int, int]) -> int:
    """
    Get the difference between two year and quarter.
    """
    return y_qtr_to_abs_qtr(y_qtr1) - y_qtr_to_abs_qtr(y_qtr2)

# Regress on FF5 + Momentum
def regress_ff5mom(returns: pd.Series, ff: pd.DataFrame, leverage: int = 1, years: int = 14) -> pd.Series:
    tmp = ff.loc[:, ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']].copy()
    tmp = add_constant(tmp)

    aligned = pd.DataFrame({'ret': returns})
    aligned['rf'] = ff['rf']
    tmp = tmp.join(aligned, how='left')
    tmp['ret'] = tmp['ret'] - tmp['rf'] * leverage
    tmp.drop(columns='rf', inplace=True)

    tmp = tmp.dropna()
    if tmp.empty:
        raise ValueError("No data left after dropping NaNs. Check alignment of ff and returns.")

    model = OLS(tmp['ret'], tmp[['const', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']])
    model_fit = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})

    days_year_avg = tmp.shape[0] / years
    stars = model_fit.pvalues.apply(lambda x: '***' if x < 0.02 else '**' if x < 0.1 else '*' if x < 0.2 else '')
    vars = ['const', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']
    dct_result = {var: f'{model_fit.params[var]:.2f}\n({model_fit.tvalues[var]:.2f})' for var in vars}
    dct_result['R2'] = f'{model_fit.rsquared:.2f}'
    dct_result['const'] = f'{model_fit.params["const"] * days_year_avg:.2%}\n({model_fit.tvalues["const"]:.2f}{stars["const"]})'
    return pd.Series(dct_result)

def regress_capm(returns: pd.Series, ff: pd.DataFrame, leverage: int = 1):
    """
    Regress the returns on CAPM.
    """
    tmp = ff.loc[:, ['mktrf']]
    tmp = add_constant(tmp)
    tmp['ret'] = returns
    tmp['ret'] = tmp['ret'] - ff['rf'] * leverage
    tmp = tmp.dropna()
    model = OLS(tmp['ret'], tmp[['const', 'mktrf']])
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    return result

def ttest_risk_adj(returns1: pd.Series, returns2: pd.Series, rf: pd.Series) -> Tuple[float, float]:
    """
    Perform t-test on the risk adjusted returns.
    """
    returns1_adj = (returns1 - rf)/returns1.std()
    returns2_adj = (returns2 - rf)/returns2.std()
    returns = pd.concat([returns1_adj, returns2_adj], axis=1)
    returns = returns.dropna()
    return ttest_rel(returns.iloc[:, 0], returns.iloc[:, 1], alternative='greater')

def get_cum_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Get the cumulative returns.
    """
    return (returns + 1).cumprod()

def get_max_drawdown(returns: pd.Series) -> float:
    """
    Get the maximum drawdown.
    """
    cum = (returns + 1).cumprod()
    max_drawdown = 0
    max_cum = cum.iloc[0]
    for i in range(1, cum.shape[0]):
        max_cum = max(max_cum, cum.iloc[i])
        max_drawdown = max(max_drawdown, 1 - cum.iloc[i]/max_cum)
    return max_drawdown

def evaluate_returns(returns: pd.Series, rf: pd.Series, years: int = 14) -> pd.Series:
    days_year_avg = returns.shape[0] / years
    cum = (returns + 1).cumprod().iloc[-1]
    cagr = cum ** (1 / years) - 1

    excess = returns - rf
    sigma = excess.std() * (days_year_avg ** 0.5)
    sharpe = excess.mean() / excess.std() * (days_year_avg ** 0.5)

    max_drawdown = get_max_drawdown(returns)

    return pd.Series({
        'CAGR': f"{cagr:.2%}",
        'Standard deviation': f"{sigma:.4f}",
        'Sharpe ratio': f"{sharpe:.4f}",
        'Maximum drawdown': f"{max_drawdown:.2%}"
    })
    

class DataTrainValid:
    def __init__(self, data: pd.DataFrame, valid_y_qtr_cut: Tuple[int, int], features: List[str], label: str):
        trainset, validset = split_y_qtr(data, valid_y_qtr_cut)
        self.X_train = trainset[features]
        self.y_train = trainset[label]
        self.X_valid = validset[features]
        self.y_valid = validset[label]


class XgboostTuner:
    def __init__(self, data_tv: DataTrainValid) -> None:
        self.params = {
            "booster": "gbtree",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "n_estimators": 30,
            'eval_metric': 'logloss',
            'device': 'cuda',
            'objective': 'binary:logistic',
            'nthread': -1
        }
        self.tv = data_tv

    def evaluate_params(self, params: dict) -> float:
        if self.tv.y_train.nunique() < 2:
            print("Skipping fold: only one class in training set.")
            return float("inf")

        dtrain = xgb.DMatrix(self.tv.X_train, self.tv.y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(self.tv.X_valid, enable_categorical=True)
        params = copy.deepcopy(params)
        num_boost_round = params.pop('n_estimators')
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        preds = model.predict(dvalid)
        return log_loss(self.tv.y_valid, preds)

    def tune_params(self, tuning_params_dict: Dict[str, List]) -> None:
        print(f"Tuning parameters: {tuning_params_dict}")
        best_score = float("inf")
        best_tuning_params = {}
        for tuning_params in itertools.product(*tuning_params_dict.values()):
            params = copy.deepcopy(self.params)
            for i, param in enumerate(tuning_params_dict.keys()):
                params[param] = tuning_params[i]
            score = self.evaluate_params(params)
            if score < best_score:
                best_score = score
                best_tuning_params = params

        print(f"Best tuning parameters: {best_tuning_params}, score: {best_score}")
        self.params = best_tuning_params

    def tune_sequential(self) -> None:
        param_values = {
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 1, 10, 20],
        }
        self.tune_params(param_values)

        param_values = {
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
        self.tune_params(param_values)

        param_values = {
            'reg_alpha': [0, 0.1, 1, 5],
            'reg_lambda': [0, 1, 5],
        }
        self.tune_params(param_values)

        param_values = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 20, 30, 50, 75, 100],
        }
        self.tune_params(param_values)

    def tune(self, method: Literal['sequential', 'bayesian'], n_trials: int = 30) -> None:
        if method == 'sequential':
            self.tune_sequential()
        else:
            raise ValueError(f"Invalid method: {method}")

    def fit(self) -> xgb.XGBClassifier:
        params = copy.deepcopy(self.params)
        dtrain = xgb.DMatrix(self.tv.X_train, self.tv.y_train, enable_categorical=True)
        num_boost_round = params.pop('n_estimators')
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return model

    def tune_and_fit(self, method: Literal['sequential', 'bayesian', None], n_trials: int = 100) -> xgb.XGBClassifier:
        if method == 'sequential':
            self.tune_sequential()
        elif method is None:
            pass
        else:
            raise ValueError(f"Invalid method: {method}")
        return self.fit()


class XgboostRolling:
    def __init__(self, data: pd.DataFrame, features: List[str], label: str,
                 test_y_qtr_cut: Tuple[int, int], rolling_interval: int,
                 last_y_qtr: Tuple[int, int] = (2024, 4)) -> None:
        self.data = data
        self.features = features
        self.label = label
        self.test_y_qtr_cut = test_y_qtr_cut
        self.rolling_interval = rolling_interval
        last_abs_qtr = y_qtr_to_abs_qtr(last_y_qtr)
        lst_rolling_cut_abs = list(range(y_qtr_to_abs_qtr(test_y_qtr_cut), last_abs_qtr + 1, rolling_interval))
        self.lst_rolling_cut = [abs_qtr_to_y_qtr(abs_qtr) for abs_qtr in lst_rolling_cut_abs]
        self.models: List[xgb.Booster] = []
    
    def fit(self,
            train_window: int = 64,
            valid_window: int = 16,
            tuning_method: Literal['sequential', 'bayesian'] = 'bayesian',
            n_trials: int = 30) -> None:
    
        self.models = []
    
        anchored_start = (1961, 1)
    
        for cut in self.lst_rolling_cut:
            qtr_valid_start = y_qtr_delta(cut, -valid_window)
    
            # Ensure we have enough data to span from 1961-Q1 to validation start
            if y_qtr_to_abs_qtr(qtr_valid_start) - y_qtr_to_abs_qtr(anchored_start) < train_window:
                print(f"Skipping cut {cut}: not enough room for training + validation from 1961-Q1.")
                continue
    
            # Get data from 1961-Q1 to test cut
            data = self.data[(self.data['y'] > 1960) | ((self.data['y'] == 1961) & (self.data['qtr'] >= 1))]
            data, _ = split_y_qtr(data, cut)
    
            print(f"Train: {anchored_start[0]}-{anchored_start[1]} to {qtr_valid_start[0]}-{qtr_valid_start[1]}")
            print(f"Valid: {qtr_valid_start[0]}-{qtr_valid_start[1]} to {cut[0]}-{cut[1]}")
    
            data_tv = DataTrainValid(data, qtr_valid_start, self.features, self.label)
            tuner = XgboostTuner(data_tv)
            tuner.tune(tuning_method, n_trials)
    
            # Oversample most recent 5 years of data for robustness
            trainset1, trainset2 = split_y_qtr(data, y_qtr_delta(cut, -5 * 4))
            if trainset1.shape[0] == 0:
                print(f"Skipping cut {cut}: empty trainset1")
                continue
    
            repetition_count, remaining_samples = divmod(trainset2.shape[0], trainset1.shape[0])
            trainset2 = pd.concat([trainset2] * repetition_count + [trainset2.sample(remaining_samples, random_state=0)])
            trainset = pd.concat([trainset1, trainset2])
    
            if trainset[self.label].nunique() < 2:
                print(f"Skipping cut {cut}: only one class in training set.")
                continue
    
            params = copy.deepcopy(tuner.params)
            dtrain = xgb.DMatrix(trainset[self.features], trainset[self.label], enable_categorical=True)
            num_boost_round = params.pop('n_estimators')
            model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    
            self.models.append(model)

    
    def fit_last(self,
                 train_window: int = 64,
                 valid_window: int = 16,
                 tuning_method: Literal['sequential', 'bayesian'] = 'sequential',
                 n_trials: int = 30) -> xgb.Booster:
    
        cut = self.lst_rolling_cut[-1]
        qtr_valid_start = y_qtr_delta(cut, -valid_window)
        anchored_start = (1961, 1)
    
        if y_qtr_to_abs_qtr(qtr_valid_start) - y_qtr_to_abs_qtr(anchored_start) < train_window:
            print(f"Skipping final fit: not enough room for training + validation from 1961-Q1.")
            return None
    
        # Full anchored expanding window from 1961-Q1 to test cut
        data = self.data[(self.data['y'] > 1960) | ((self.data['y'] == 1961) & (self.data['qtr'] >= 1))]
        data, _ = split_y_qtr(data, cut)
    
        print(f"Train: {anchored_start[0]}-{anchored_start[1]} to {qtr_valid_start[0]}-{qtr_valid_start[1]}")
        print(f"Valid: {qtr_valid_start[0]}-{qtr_valid_start[1]} to {cut[0]}-{cut[1]}")
    
        data_tv = DataTrainValid(data, qtr_valid_start, self.features, self.label)
        tuner = XgboostTuner(data_tv)
        tuner.tune(tuning_method, n_trials)
    
        trainset1, trainset2 = split_y_qtr(data, y_qtr_delta(cut, -5 * 4))
        if trainset1.shape[0] == 0:
            print(f"Skipping final fit: empty trainset1")
            return None
    
        repetition_count, remaining_samples = divmod(trainset2.shape[0], trainset1.shape[0])
        trainset2 = pd.concat([trainset2] * repetition_count + [trainset2.sample(remaining_samples, random_state=0)])
        trainset = pd.concat([trainset1, trainset2])
    
        if trainset[self.label].nunique() < 2:
            print("Skipping final fit: only one class in training set.")
            return None
    
        params = copy.deepcopy(tuner.params)
        dtrain = xgb.DMatrix(trainset[self.features], trainset[self.label], enable_categorical=True)
        num_boost_round = params.pop('n_estimators')
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    
        return model

################################################################
    def get_testset_with_prob(self) -> pd.DataFrame:
        concated_testset = pd.DataFrame()
    
        for i, qtr_test_start in enumerate(self.lst_rolling_cut):
            try:
                model = self.models[i]
            except IndexError:
                print(f" No model for test cut {qtr_test_start}, skipping.")
                continue
    
            qtr_test_end = y_qtr_delta(qtr_test_start, self.rolling_interval)
    
            # Get testset window
            _, testset = split_y_qtr(self.data, qtr_test_start)
            testset, _ = split_y_qtr(testset, qtr_test_end)
    
            if testset.empty:
                print(f" Empty testset for {qtr_test_start} to {qtr_test_end}")
                continue
    
            # Columns to keep
            cols_to_keep = self.features + ['PERMNO', 'y', 'qtr']
            if 'date' in testset.columns:
                cols_to_keep += ['date']
            testset = testset[cols_to_keep].copy()
    
            dtrain = xgb.DMatrix(testset[self.features], enable_categorical=True)
            testset['prob'] = model.predict(dtrain)
    
            concated_testset = pd.concat([concated_testset, testset], ignore_index=True)
    
            print(f" Predicted: {qtr_test_start} to {qtr_test_end}, rows: {len(testset)}")
    
        if not concated_testset.empty:
            last_y, last_q = concated_testset[['y', 'qtr']].drop_duplicates().sort_values(['y', 'qtr']).iloc[-1]
            print(f" Finished predictions up to: {last_y}-Q{last_q}")
        else:
            print(" No predictions were made. Check model list or input data.")
    
        return concated_testset






