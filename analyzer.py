
"""
Usage:
    python analyzer.py --input ./case1 --out ./out/push_output.csv --meta ./out/meta_per_client.jsonl --summary ./out/summary.json

Output:
 - CSV with client_code,product,push_notification
 - meta jsonl with per-client diagnostics
 - summary JSON with distribution
"""

from __future__ import annotations
import os
import math
import json
import argparse
import logging
from typing import Dict, Any, List
import pandas as pd
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------- CONFIG / TARGETS ----------
CONFIG = {
    "push_max_len": 220,
    # target product distribution (approximate fractions; sum should be ~1.0)
    # tweak these to bias final allocation
    "target_distribution": {
        "Кредитная карта": 0.45,
        "Карта для путешествий": 0.15,
        "Премиальная карта": 0.08,
        "Обмен валют": 0.05,
        "Кредит наличными": 0.03,
        "Депозит Мультивалютный": 0.06,
        "Депозит Сберегательный": 0.06,
        "Депозит Накопительный": 0.06,
        "Инвестиции": 0.04,
        "Золотые слитки": 0.02
    },
    # scoring tuning
    "realism_w": 0.75,
    "diversity_w": 0.25,
    "big_oneoff_threshold": 200_000,
    "travel_month_threshold": 25_000,
    "fx_ops_threshold": 2,
    "creditcard_baseline": 2.0,  # small baseline so credit card not always dominates
}

PRODUCTS = [
    "Карта для путешествий",
    "Премиальная карта",
    "Кредитная карта",
    "Обмен валют",
    "Кредит наличными",
    "Депозит Мультивалютный",
    "Депозит Сберегательный",
    "Депозит Накопительный",
    "Инвестиции",
    "Золотые слитки",
]

TRAVEL_CATS = {"Путешествия", "Такси", "Отели", "Авиабилеты", "Поезда"}
RESTAURANTS_CATS = {"Кафе и рестораны", "Рестораны"}
JEWEL_CATS = {"Ювелирные украшения", "Косметика и Парфюмерия"}
ONLINE_CATS = {"Играем дома", "Смотрим дома", "Едим дома", "Кино", "Онлайн услуги", "Доставка", "Игры"}
KZT_VARIANTS = {"KZT", "₸", "тг", "тнг", "тг."}

# ---------- Utilities ----------
def safe_read_csv(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    if path_or_df is None:
        return pd.DataFrame()
    if isinstance(path_or_df, str) and os.path.exists(path_or_df):
        try:
            return pd.read_csv(path_or_df)
        except Exception:
            return pd.read_csv(path_or_df, encoding='utf-8', engine='python', on_bad_lines='skip')
    return pd.DataFrame()

def to_float(x, default=0.0):
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x)
        s = s.replace("\u00A0", " ").replace(",", ".").replace(" ", "")
        for sym in ["₸","тг","KZT","kzt","тг."]:
            s = s.replace(sym, "")
        return float(s) if s.strip() not in ["", "-", "nan", "None"] else float(default)
    except Exception:
        return float(default)

def fmt_money(x):
    try:
        v = float(x)
    except Exception:
        v = 0.0
    sign = "-" if v < 0 else ""
    v = abs(round(v, 2))
    integer = int(math.floor(v))
    frac = int(round((v - integer) * 100))
    int_str = f"{integer:,}".replace(",", " ")
    if frac == 0:
        return f"{sign}{int_str} ₸"
    else:
        return f"{sign}{int_str},{frac:02d} ₸"

def log_norm_ratio(numerator: float, denom: float) -> float:
    eps = 1e-9
    return (math.log1p(max(0.0, numerator)) / (math.log1p(max(eps, denom)) + eps)) * 100.0

def safe_col_candidates(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------- Feature extractor ----------
class FeatureExtractor:
    def __init__(self, tx: pd.DataFrame, tr: pd.DataFrame, client_code=None):
        self.tx = safe_read_csv(tx)
        self.tr = safe_read_csv(tr)
        self.client_code = client_code
        self._normalize()

    def _normalize(self):
        # normalize tx
        if 'amount' in self.tx.columns:
            self.tx['amount'] = pd.to_numeric(self.tx['amount'], errors='coerce').fillna(0).abs()
        else:
            self.tx['amount'] = 0.0
        # category candidates
        cat_col = safe_col_candidates(self.tx, ['category','Category','category_name','cat'])
        if cat_col and cat_col != 'category':
            self.tx['category'] = self.tx[cat_col].astype(str)
        elif 'category' not in self.tx.columns:
            self.tx['category'] = ''
        else:
            self.tx['category'] = self.tx['category'].astype(str)
        # date
        if 'date' in self.tx.columns:
            self.tx['date'] = pd.to_datetime(self.tx['date'], errors='coerce')

        # normalize tr
        if 'amount' in self.tr.columns:
            self.tr['amount'] = pd.to_numeric(self.tr['amount'], errors='coerce').fillna(0).abs()
        else:
            self.tr['amount'] = 0.0
        op_col = safe_col_candidates(self.tr, ['type','operation_type','operation','op_type','tx_type'])
        if op_col and op_col != 'type':
            self.tr['type'] = self.tr[op_col].astype(str)
        elif 'type' not in self.tr.columns:
            self.tr['type'] = self.tr['type'] if 'type' in self.tr.columns else ''
        else:
            self.tr['type'] = self.tr['type'].astype(str)
        # direction
        if 'direction' not in self.tr.columns:
            self.tr['direction'] = self.tr.get('direction','')
        # currency columns
        cur_col_tx = safe_col_candidates(self.tx, ['currency','Currency'])
        cur_col_tr = safe_col_candidates(self.tr, ['currency','Currency'])
        self.tx['currency'] = self.tx[cur_col_tx].astype(str) if cur_col_tx else self.tx.get('currency','')
        self.tr['currency'] = self.tr[cur_col_tr].astype(str) if cur_col_tr else self.tr.get('currency','')
        # filter by client_code if present
        if self.client_code is not None:
            if 'client_code' in self.tx.columns:
                try:
                    self.tx = self.tx[self.tx['client_code'].astype(str) == str(self.client_code)]
                except Exception:
                    pass
            if 'client_code' in self.tr.columns:
                try:
                    self.tr = self.tr[self.tr['client_code'].astype(str) == str(self.client_code)]
                except Exception:
                    pass

    def aggregates(self) -> Dict[str, Any]:
        tx = self.tx
        tr = self.tr

        total_spend_3m = float(tx['amount'].sum())
        total_spend_month = total_spend_3m / 3.0 if total_spend_3m > 0 else 0.0
        cat_sums_total = tx.groupby('category')['amount'].sum().sort_values(ascending=False)
        top3 = list(cat_sums_total.head(3).index.astype(str)) if not cat_sums_total.empty else []
        top3_sum_month = float((cat_sums_total.head(3).sum() if not cat_sums_total.empty else 0.0) / 3.0)

        travel_3m = float(tx[tx['category'].isin(TRAVEL_CATS)]['amount'].sum())
        travel_month = travel_3m / 3.0
        travel_count_3m = int(tx[tx['category'].isin(TRAVEL_CATS)].shape[0])

        rest_month = float(tx[tx['category'].isin(RESTAURANTS_CATS)]['amount'].sum() / 3.0)
        jew_month = float(tx[tx['category'].isin(JEWEL_CATS)]['amount'].sum() / 3.0)
        online_month = float(tx[tx['category'].isin(ONLINE_CATS)]['amount'].sum() / 3.0)
        online_count_3m = int(tx[tx['category'].isin(ONLINE_CATS)].shape[0])

        fx_ops_sum = float(tr[tr['type'].str.contains('fx', case=False, na=False)]['amount'].sum()) if not tr.empty else 0.0
        fx_count = int(tr[tr['type'].str.contains('fx', case=False, na=False)].shape[0]) if not tr.empty else 0
        currencies = set([str(x).upper() for x in list(tx.get('currency', [])) + list(tr.get('currency', [])) if pd.notna(x)])
        multi_currency = any([c not in KZT_VARIANTS for c in currencies]) if currencies else False

        # inflows/outflows
        inflows = 0.0
        outflows = 0.0
        if not tr.empty:
            dir_lower = tr['direction'].astype(str).str.lower() if 'direction' in tr else pd.Series([])
            if 'direction' in tr and dir_lower.isin(['in','incoming']).any():
                inflows = float(tr[dir_lower.isin(['in','incoming'])]['amount'].sum())
                outflows = float(tr[dir_lower.isin(['out','outgoing'])]['amount'].sum())
            else:
                inflows = float(tr[tr['type'].str.contains('_in', na=False)].get('amount', 0).sum()) if not tr.empty else 0.0
                outflows = float(tr[tr['type'].str.contains('_out', na=False)].get('amount', 0).sum()) if not tr.empty else 0.0

        has_installment = bool(tr['type'].str.contains('installment|installment_payment', case=False, na=False).any()) if not tr.empty else False
        has_loan_payments = bool(tr['type'].str.contains('loan_payment|cc_repayment', case=False, na=False).any()) if not tr.empty else False
        has_invest = bool(tr['type'].str.contains('invest', case=False, na=False).any()) if not tr.empty else False
        has_gold = bool(tr['type'].str.contains('gold', case=False, na=False).any()) if not tr.empty else False
        atm_withdrawals = int(tr[tr['type'].str.contains('atm_withdrawal', case=False, na=False)].shape[0]) if not tr.empty else 0
        max_tx = float(tx['amount'].max()) if not tx.empty else 0.0
        volatility = float(tx['amount'].std()) if tx.shape[0] > 1 else 0.0
        deposit_topups = int(tr[tr['type'].str.contains('deposit_topup|deposit_fx_topup', case=False, na=False)].shape[0]) if not tr.empty else 0

        monthly_income_est = inflows / 3.0 if inflows > 0 else 0.0

        return {
            "total_spend_3m": total_spend_3m,
            "total_spend_month": total_spend_month,
            "top3_categories": top3,
            "top3_sum_month": top3_sum_month,
            "travel_month": travel_month,
            "travel_count_3m": travel_count_3m,
            "rest_month": rest_month,
            "jew_month": jew_month,
            "online_month": online_month,
            "online_count_3m": online_count_3m,
            "fx_ops_sum": fx_ops_sum,
            "fx_count": fx_count,
            "currencies": list(currencies),
            "multi_currency": multi_currency,
            "inflows": inflows,
            "outflows": outflows,
            "has_installment": has_installment,
            "has_loan_payments": has_loan_payments,
            "has_invest": has_invest,
            "has_gold": has_gold,
            "atm_withdrawals": atm_withdrawals,
            "max_tx": max_tx,
            "volatility": volatility,
            "deposit_topups": deposit_topups,
            "monthly_income_est": monthly_income_est
        }

# ---------- Signal engine ----------
class SignalEngine:
    def __init__(self, A: Dict[str,Any], profile: Dict[str,Any]):
        self.A = A
        self.P = profile
        self.signals: Dict[str, Dict[str,float]] = {}
        self._compute()

    def _compute(self):
        total = to_float(self.A.get('total_spend_month', 0.0))
        bal = to_float(self.P.get('avg_monthly_balance_KZT', 0.0))

        # travel
        self.signals['Карта для путешествий'] = {
            'travel_share': min(1.0, to_float(self.A.get('travel_month',0.0)) / (total + 1.0)),
            'trips_per_month': min(1.0, (to_float(self.A.get('travel_count_3m',0))/3.0) / 6.0),
            'big_travel': 1.0 if to_float(self.A.get('travel_month',0.0)) >= CONFIG['travel_month_threshold'] else 0.0
        }
        # premium
        self.signals['Премиальная карта'] = {
            'balance_high': min(1.0, bal / 6_000_000.0),
            'luxury_share': min(1.0, (to_float(self.A.get('jew_month',0.0)) + to_float(self.A.get('rest_month',0.0))) / (total + 1.0)),
            'atm_activity': min(1.0, to_float(self.A.get('atm_withdrawals',0)) / 20.0)
        }
        # credit card
        self.signals['Кредитная карта'] = {
            'top3_share': min(1.0, to_float(self.A.get('top3_sum_month',0.0)) / (total + 1.0)),
            'online_share': min(1.0, to_float(self.A.get('online_month',0.0)) / (total + 1.0)),
            'has_installment': 1.0 if self.A.get('has_installment') else 0.0
        }
        # fx
        self.signals['Обмен валют'] = {
            'fx_volume': min(1.0, to_float(self.A.get('fx_ops_sum',0.0)) / 100_000.0),
            'fx_count': min(1.0, to_float(self.A.get('fx_count',0)) / 5.0),
            'multi_currency': 1.0 if self.A.get('multi_currency') else 0.0
        }
        # cash loan
        inflows = to_float(self.A.get('inflows',0.0))
        outflows = to_float(self.A.get('outflows',0.0))
        cash_gap = 0.0
        if inflows > 0:
            cash_gap = min(1.0, max(0.0, (outflows - inflows)) / (inflows + 1.0))
        self.signals['Кредит наличными'] = {
            'low_balance': 1.0 if bal < 100_000 else 0.0,
            'cash_gap': cash_gap,
            'big_oneoff': 1.0 if to_float(self.A.get('max_tx',0.0)) >= CONFIG['big_oneoff_threshold'] else 0.0,
            'loan_history': 1.0 if self.A.get('has_loan_payments') else 0.0
        }
        # deposits
        free_funds = max(0.0, bal - to_float(self.A.get('total_spend_month',0.0)))
        self.signals['Депозит Мультивалютный'] = {
            'free_funds': min(1.0, free_funds / 2_000_000.0),
            'multi_currency': 1.0 if self.A.get('multi_currency') else 0.0
        }
        self.signals['Депозит Сберегательный'] = {
            'free_funds': min(1.0, free_funds / 5_000_000.0),
            'low_volatility': 1.0 if to_float(self.A.get('volatility',0.0)) < max(1.0, to_float(self.A.get('total_spend_month',0.0))/10.0) else 0.0
        }
        self.signals['Депозит Накопительный'] = {
            'regular_topups': min(1.0, to_float(self.A.get('deposit_topups',0)) / 5.0),
            'free_funds': min(1.0, free_funds / 1_000_000.0)
        }
        # investments
        self.signals['Инвестиции'] = {
            'free_funds': min(1.0, free_funds / 1_000_000.0),
            'invest_history': 1.0 if self.A.get('has_invest') else 0.0,
            'suitable_age': 1.0 if isinstance(self.P.get('age',None),(int,float)) and to_float(self.P.get('age',0)) >= 18 else 0.0
        }
        # gold
        self.signals['Золотые слитки'] = {
            'gold_ops': 1.0 if self.A.get('has_gold') else 0.0,
            'large_free': min(1.0, free_funds / 5_000_000.0)
        }

    def get(self):
        return self.signals

# ---------- Scorer ----------
class Scorer:
    def __init__(self, signals: Dict[str,Dict[str,float]], A: Dict[str,Any], profile: Dict[str,Any]):
        self.signals = signals
        self.A = A
        self.P = profile
        self.benefits: Dict[str,float] = {}
        self.scores: Dict[str,float] = {}
        self._compute_benefits()
        self._compute_scores()

    def _compute_benefits(self):
        # monetary approximation (тг/мес)
        self.benefits['Карта для путешествий'] = to_float(self.A.get('travel_month',0.0)) * 0.04
        bal = to_float(self.P.get('avg_monthly_balance_KZT',0.0))
        if bal >= 6_000_000:
            tier = 0.04
        elif bal >= 1_000_000:
            tier = 0.03
        else:
            tier = 0.02
        luxury_month = to_float(self.A.get('jew_month',0.0)) + to_float(self.A.get('rest_month',0.0))
        saved_fees = to_float(self.A.get('atm_withdrawals',0)) * 500 + to_float(self.A.get('fx_ops_sum',0.0)) * 0.001
        self.benefits['Премиальная карта'] = min(100_000.0, luxury_month * 0.04 + max(0.0, to_float(self.A.get('total_spend_month',0.0)) - luxury_month) * tier + saved_fees)
        self.benefits['Кредитная карта'] = to_float(self.A.get('top3_sum_month',0.0)) * 0.10 + to_float(self.A.get('online_month',0.0)) * 0.10
        self.benefits['Обмен валют'] = to_float(self.A.get('fx_ops_sum',0.0)) * 0.005
        self.benefits['Кредит наличными'] = 0.0
        free_funds = max(0.0, bal - to_float(self.A.get('total_spend_month',0.0)))
        self.benefits['Депозит Мультивалютный'] = free_funds * 0.145 / 12.0
        self.benefits['Депозит Сберегательный'] = free_funds * 0.165 / 12.0
        self.benefits['Депозит Накопительный'] = free_funds * 0.155 / 12.0
        self.benefits['Инвестиции'] = free_funds * 0.01 + (1000.0 if self.A.get('has_invest') else 0.0)
        self.benefits['Золотые слитки'] = free_funds * 0.002

    def _compute_scores(self):
        product_weights = {
            'Карта для путешествий': {'travel_share':0.6,'trips_per_month':0.3,'big_travel':0.1},
            'Премиальная карта': {'balance_high':0.6,'luxury_share':0.3,'atm_activity':0.1},
            'Кредитная карта': {'top3_share':0.5,'online_share':0.35,'has_installment':0.15},
            'Обмен валют': {'fx_volume':0.6,'fx_count':0.25,'multi_currency':0.15},
            'Кредит наличными': {'low_balance':0.5,'cash_gap':0.35,'big_oneoff':0.15,'loan_history':0.2},
            'Депозит Мультивалютный': {'free_funds':0.6,'multi_currency':0.3},
            'Депозит Сберегательный': {'free_funds':0.7,'low_volatility':0.3},
            'Депозит Накопительный': {'regular_topups':0.6,'free_funds':0.4},
            'Инвестиции': {'free_funds':0.6,'invest_history':0.25,'suitable_age':0.15},
            'Золотые слитки': {'gold_ops':0.6,'large_free':0.4}
        }
        bal = to_float(self.P.get('avg_monthly_balance_KZT',0.0))
        for p in PRODUCTS:
            weights = product_weights.get(p, {})
            sig = self.signals.get(p, {})
            sig_sum = sum(weights.get(k,0.0) * float(sig.get(k,0.0)) for k in weights.keys())
            sig_score = sig_sum * 100.0
            monetary = float(self.benefits.get(p,0.0))
            monetary_norm = log_norm_ratio(monetary, max(1.0, bal))
            realism = CONFIG['realism_w'] * sig_score + (1.0 - CONFIG['realism_w']) * monetary_norm
            # diversity boost small deterministic
            diversity_boost = 0.05
            if p == 'Инвестиции':
                diversity_boost = (0.15 if self.signals['Инвестиции'].get('invest_history') else 0.03) + min(0.25, bal/1_000_000.0) * 0.25
            elif p == 'Депозит Накопительный':
                diversity_boost = min(1.0, bal/300_000.0) * 0.2
            elif p == 'Карта для путешествий':
                diversity_boost = min(0.2, (self.A.get('travel_count_3m',0)/3.0)/4.0) if self.A.get('travel_count_3m',0)>0 else 0.0

            diversity_score = diversity_boost * 100.0
            final = CONFIG['realism_w'] * realism + CONFIG['diversity_w'] * diversity_score
            # small heuristics by status/age
            if p == 'Премиальная карта' and 'Премиум' in str(self.P.get('status','')):
                final *= 1.06
            if p == 'Инвестиции' and isinstance(self.P.get('age',None),(int,float)) and to_float(self.P.get('age',0)) < 30:
                final *= 1.03
            # baseline tweak for creditcard so it won't always dominate but remains a fallback
            if p == 'Кредитная карта':
                final += CONFIG.get('creditcard_baseline', 0.0)
            self.scores[p] = float(max(0.0, final))

    def get(self):
        return self.scores, self.benefits

# ---------- Push generator ----------
class PushGenerator:
    def __init__(self, profile: Dict[str,Any], A: Dict[str,Any], benefits: Dict[str,float]):
        self.P = profile
        self.A = A
        self.benefits = benefits

    def generate(self, product: str) -> str:
        name = str(self.P.get('name') or self.P.get('first_name') or self.P.get('client_name') or "Клиент")
        if product == "Карта для путешествий":
            est = fmt_money(self.benefits.get(product,0.0))
            travel = fmt_money(self.A.get('travel_month',0.0))
            return f"{name}, в среднем вы тратите на поездки {travel}/мес. С тревел-картой вернулись бы ≈{est}. Открыть"
        if product == "Премиальная карта":
            bal_s = fmt_money(self.P.get('avg_monthly_balance_KZT',0))
            est = fmt_money(self.benefits.get(product,0.0))
            return f"{name}, ваш средний остаток {bal_s}. Премиальная карта даст повышенный кешбэк и бесплатные снятия — ≈{est}/мес. Подключите сейчас"
        if product == "Кредитная карта":
            top3 = self.A.get('top3_categories', [])
            cats = ', '.join(top3[:3]) if top3 else 'ваши категории'
            est = fmt_money(self.benefits.get(product,0.0))
            return f"{name}, ваши топ-категории — {cats}. Кредитная карта даёт до 10% в любимых категориях и онлайн-услугах — ≈{est}/мес. Оформить карту"
        if product == "Обмен валют":
            cur_list = [c for c in self.A.get('currencies',[]) if c not in KZT_VARIANTS]
            cur = ','.join(cur_list) if cur_list else 'валюте'
            return f"{name}, вы часто оперируете в {cur}. В приложении выгодный обмен и авто-покупка по целевому курсу. Настроить обмен"
        if product == "Кредит наличными":
            return f"{name}, если нужен запас на крупные траты — можно оформить кредит наличными онлайн без залога. Узнать лимит"
        if product in ("Депозит Мультивалютный","Депозит Сберегательный","Депозит Накопительный"):
            rate = 14.5
            if product == "Депозит Сберегательный": rate = 16.5
            if product == "Депозит Накопительный": rate = 15.5
            free = fmt_money(max(0.0, to_float(self.P.get('avg_monthly_balance_KZT',0.0)) - to_float(self.A.get('total_spend_month',0.0))))
            est = fmt_money(self.benefits.get(product,0.0))
            return f"{name}, у вас свободные средства ≈ {free}. Вклад {int(rate)}% даст ≈{est}/мес. Открыть вклад"
        if product == "Инвестиции":
            est = fmt_money(self.benefits.get(product,0.0))
            return f"{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт — потенциально ≈{est}/мес. Открыть счёт"
        if product == "Золотые слитки":
            return f"{name}, рассмотрите золотые слитки для диверсификации и сохранения стоимости. Посмотреть предложение"
        return f"{name}, рекомендуем {product}"

# ---------- Allocation balancer ----------
def allocate_products_greedy(per_client_scores: Dict[Any, Dict[str,float]], target_dist: Dict[str,float]) -> Dict[Any,str]:

    clients = list(per_client_scores.keys())
    n = len(clients)
    # compute target counts (at least 0)
    target_counts = {}
    remaining_products = set(target_dist.keys())
    total_assigned = 0
    for p, frac in target_dist.items():
        cnt = int(round(frac * n))
        target_counts[p] = max(0, cnt)
        total_assigned += target_counts[p]
    # adjust rounding so sum == n
    diff = n - total_assigned
    if diff != 0:
        # sort products by target fraction descending and adjust
        items = sorted(target_counts.items(), key=lambda x: target_dist.get(x[0],0), reverse=True)
        idx = 0
        step = 1 if diff > 0 else -1
        diff_abs = abs(diff)
        while diff_abs > 0:
            p = items[idx % len(items)][0]
            target_counts[p] = max(0, target_counts[p] + step)
            idx += 1
            diff_abs -= 1

    # prepare client ranking by their top score
    client_best = {}
    for cid, scores in per_client_scores.items():
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        client_best[cid] = sorted_scores

    # order clients by their highest score descending
    clients_sorted = sorted(clients, key=lambda c: client_best[c][0][1] if client_best[c] else 0.0, reverse=True)
    assignment = {}
    counts = {p:0 for p in target_counts.keys()}

    for cid in clients_sorted:
        placed = False
        for prod, sc in client_best[cid]:
            # skip zero-score products unless nothing else
            if sc <= 0 and not placed:
                continue
            if counts.get(prod,0) < target_counts.get(prod,0):
                assignment[cid] = prod
                counts[prod] += 1
                placed = True
                break
        if not placed:
            # try to assign to product with highest remaining capacity ratio (score relative)
            # prefer their top product regardless of quota if all quotas full
            top_prod = client_best[cid][0][0] if client_best[cid] else list(target_counts.keys())[0]
            assignment[cid] = top_prod
            counts[top_prod] = counts.get(top_prod,0) + 1

    return assignment

# ---------- Main pipeline: analyze_client (single pass) ----------
def compute_client_scores(profile: Dict[str,Any], tx_df_or_path, tr_df_or_path) -> Dict[str,Any]:
    """
    Returns dict:
      {
        'client_code': ...,
        'name': ...,
        'scores': {product: float},
        'benefits': {...},
        'aggregates': {...}
      }
    """
    profile = dict(profile) if not isinstance(profile, dict) else profile
    # normalize numeric profile
    profile['avg_monthly_balance_KZT'] = to_float(profile.get('avg_monthly_balance_KZT') or profile.get('avg_balance') or 0.0)
    try:
        profile['age'] = int(float(profile.get('age', 0))) if profile.get('age', None) not in [None, ''] else None
    except Exception:
        profile['age'] = None

    tx_df = safe_read_csv(tx_df_or_path)
    tr_df = safe_read_csv(tr_df_or_path)

    FE = FeatureExtractor(tx_df, tr_df, client_code=profile.get('client_code'))
    A = FE.aggregates()
    # keep balance in aggregates for convenience
    A['balance'] = profile['avg_monthly_balance_KZT']

    SE = SignalEngine(A, profile)
    signals = SE.get()

    SC = Scorer(signals, A, profile)
    scores, benefits = SC.get()

    return {
        "client_code": profile.get('client_code'),
        "name": profile.get('name') or profile.get('first_name') or profile.get('client_name'),
        "profile": profile,
        "aggregates": A,
        "signals": signals,
        "scores": scores,
        "benefits": benefits
    }

# ---------- Folder processing ----------
def process_case_folder(case_folder: str, out_csv: str, meta_jsonl: str, summary_json: str = None) -> Dict[str,Any]:
    # find clients.csv
    clients_path = None
    for root,_,files in os.walk(case_folder):
        for f in files:
            if f.lower() == 'clients.csv':
                clients_path = os.path.join(root,f)
                break
        if clients_path:
            break
    if not clients_path:
        raise FileNotFoundError("clients.csv not found in folder: %s" % case_folder)

    clients_df = pd.read_csv(clients_path, dtype=str)
    # ensure client_code column exists
    if 'client_code' not in clients_df.columns:
        raise ValueError("clients.csv must contain 'client_code' column")

    per_client = []
    per_client_scores = {}

    # iterate clients, compute scores (do NOT decide final product yet)
    for _, row in clients_df.iterrows():
        profile = row.to_dict()
        client_code = profile.get('client_code')
        # expected file names
        tx_path = os.path.join(case_folder, f"client_{client_code}_transactions_3m.csv")
        tr_path = os.path.join(case_folder, f"client_{client_code}_transfers_3m.csv")
        tx_df = safe_read_csv(tx_path)
        tr_df = safe_read_csv(tr_path)
        try:
            info = compute_client_scores(profile, tx_df, tr_df)
            per_client.append(info)
            per_client_scores[client_code] = info['scores']
        except Exception as ex:
            logging.exception("client %s processing failed: %s", client_code, ex)
            # fallback minimal scores - credit card baseline
            fallback_scores = {p: 0.0 for p in PRODUCTS}
            fallback_scores['Кредитная карта'] = CONFIG.get('creditcard_baseline', 1.0)
            per_client.append({
                "client_code": client_code, "name": profile.get('name'),
                "profile": profile, "aggregates": {}, "signals": {},
                "scores": fallback_scores, "benefits": {}
            })
            per_client_scores[client_code] = fallback_scores

    # Allocate products across all clients with greedy balancer
    assignment = allocate_products_greedy(per_client_scores, CONFIG['target_distribution'])

    results = []
    metas = []
    dist = Counter()

    for info in per_client:
        cid = info['client_code']
        assigned = assignment.get(cid, None)
        # if none (should not happen), choose max scoring product
        if assigned is None:
            assigned = max(info['scores'].items(), key=lambda x: x[1])[0]
        PG = PushGenerator(info['profile'], info['aggregates'], info['benefits'])
        push = PG.generate(assigned)
        if len(push) > CONFIG['push_max_len']:
            push = push[:CONFIG['push_max_len']-3].rstrip() + "..."
        # produce CSV row
        results.append({
            "client_code": int(cid) if str(cid).isdigit() else cid,
            "product": assigned,
            "push_notification": push
        })
        meta = {
            "client_code": cid,
            "name": info.get('name'),
            "profile": info.get('profile'),
            "aggregates": info.get('aggregates'),
            "signals": info.get('signals'),
            "scores": info.get('scores'),
            "benefits": info.get('benefits'),
            "assigned": assigned
        }
        metas.append(meta)
        dist[assigned] += 1

    # write outputs
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8')
    with open(meta_jsonl, "w", encoding='utf-8') as f:
        for rec in metas:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"n_clients": len(results), "distribution": dict(dist)}
    if summary_json:
        with open(summary_json, "w", encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Путь к папке case1 (с clients.csv и client_{id}_*.csv)")
    p.add_argument("--out", "-o", default="push_output.csv", help="Путь для CSV результата")
    p.add_argument("--meta", "-m", default="meta_per_client.jsonl", help="Путь для meta jsonl")
    p.add_argument("--summary", "-s", default="summary.json", help="Путь для summary json")
    args = p.parse_args()

    logging.info("Processing folder: %s", args.input)
    summary = process_case_folder(args.input, args.out, args.meta, args.summary)
    logging.info("Done. Clients processed: %d", summary.get('n_clients',0))
    logging.info("Product distribution: %s", summary.get('distribution'))

if __name__ == "__main__":
    main()
