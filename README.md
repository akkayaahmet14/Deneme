
!pip install git+https://github.com/rongardF/tvdatafeed matplotlib openpyxl ta
!pip install tradingview-screener==2.5.0

# Gerekli Kütüphaneler
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import get_all_symbols
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import linregress
from tqdm import tqdm

# Global Parametreler
commission = 0.001  # %0.1 komisyon
slippage = 0.002    # %0.2 kayma
features = ['close', 'ema', 'sma', 'adx', 'volatility', 'rsi',
            'macd', 'macd_signal', 'momentum', 'cci']

# === Zaman Dilimi Seçimi ===
def get_interval_choice():
    intervals = {
        '1': ('15 Dakika', Interval.in_15_minute),
        '2': ('30 Dakika', Interval.in_30_minute),
        '3': ('1 Saat',    Interval.in_1_hour),
        '4': ('2 Saat',    Interval.in_2_hour),
        '5': ('4 Saat',    Interval.in_4_hour),
        '6': ('Günlük',    Interval.in_daily),
        '7': ('Haftalık',  Interval.in_weekly),
        '8': ('Aylık',     Interval.in_monthly),
    }
    print("Zaman dilimini seçin:")
    for key, (name, _) in intervals.items():
        print(f"{key}: {name}")
    choice = input("Seçiminiz (6 varsayılan): ").strip()
    return intervals.get(choice, intervals['6'])[1]

# === BIST Sembol Listesi ===
def get_bist_symbols():
    try:
        symbols = get_all_symbols(market='turkey')
        return [s.replace('BIST:', '') for s in symbols]
    except Exception as e:
        print("Sembol hatası:", e)
        return []

# === Veri Toplama ve Göstergeler ===
def fetch_and_prepare_data(symbols, interval):
    tv = TvDatafeed()
    all_data = []
    for symbol in tqdm(symbols, desc="Veri Alınıyor", ncols=80):
        try:
            df = tv.get_hist(symbol=symbol, exchange='BIST', interval=interval, n_bars=300)
            if df is None or df.empty or len(df) < 50:
                continue
            df['symbol'] = symbol
            df['ema'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['sma'] = SMAIndicator(df['close'], window=20).sma_indicator()
            df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['momentum'] = df['close'].diff(4)
            df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
            df['volatility'] = df['close'].rolling(10).std()
            df['slope'] = df['close'].rolling(5).apply(lambda x: linregress(range(5), x)[0], raw=False)
            df.dropna(inplace=True)
            all_data.append(df)
        except Exception as e:
            print(f"{symbol} verisi alınamadı: {e}")
    return pd.concat(all_data) if all_data else pd.DataFrame()

# === Model Eğitimi (Train/Test Ayrımlı) ===
def train_model(df_all):
    X = df_all[features]
    y = np.where(df_all['slope'] > 0, 1, -1)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🧪 Test Doğruluk Oranı: {acc:.2%}")
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, digits=3))
    return model

# === Sinyal Üretimi ===
def generate_signals(df_all, model):
    results = []
    for symbol in df_all['symbol'].unique():
        df = df_all[df_all['symbol'] == symbol].copy()
        X = df[features]
        df['classifier_signal'] = model.predict(X)
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        signal_type = None
        score = 0
        if latest['classifier_signal'] == 1:
            signal_type = 'AL'
            if latest['rsi'] > 50 and latest['rsi'] > previous['rsi']:
                score += 1
            if latest['adx'] > 25 and latest['adx'] > previous['adx']:
                score += 2
            if latest['ema'] > previous['ema'] and latest['close'] > latest['ema']:
                score += 1
            if latest['ema'] > latest['sma']:
                score += 1
            if latest['macd'] > latest['macd_signal'] and latest['macd'] > previous['macd']:
                score += 2
            if latest['momentum'] > 0 and latest['momentum'] > previous['momentum']:
                score += 1
            if latest['cci'] > 100 and latest['cci'] > previous['cci']:
                score += 1
            if latest['volatility'] > df['volatility'].mean():
                score += 1
            if latest['slope'] > 0:
                score += 2
            results.append({'symbol': symbol, 'type': signal_type, 'score': score})
    return pd.DataFrame(results)

# === Backtest & Özet ===
def backtest_signals(df_all, model, al_symbols):
    df_all = df_all[df_all['symbol'].isin(al_symbols)]
    results = []
    for symbol in df_all['symbol'].unique():
        df = df_all[df_all['symbol'] == symbol].copy()
        X = df[features]
        df['classifier_signal'] = model.predict(X)
        df['next_close'] = df['close'].shift(-1)
        cost = df['close'] * (commission + slippage)
        df['net_pnl'] = np.where(
            df['classifier_signal'] == 1,
            (df['next_close'] - df['close']) - cost,
            np.where(
                df['classifier_signal'] == -1,
                (df['close'] - df['next_close']) - cost,
                0
            )
        )
        df['cumulative_return'] = df['net_pnl'].cumsum()
        df['correct'] = df['net_pnl'] > 0
        df['symbol'] = symbol
        results.append(df)
    backtest_df = pd.concat(results)
    backtest_df.dropna(inplace=True)
    summary = backtest_df.groupby('symbol').agg({
        'net_pnl': ['mean', 'sum'],
        'correct': 'mean',
        'cumulative_return': 'last'
    }).reset_index()
    summary.columns = ['symbol', 'avg_return', 'total_return', 'accuracy', 'cumulative_return']
    summary.to_excel("al_sinyalli_hisseler_ozet.xlsx", index=False)
    return backtest_df, summary

# === Ana Fonksiyon ===
def main():
    symbols = get_bist_symbols()
    if not symbols:
        print("Sembol bulunamadı. ")
        return
    interval = get_interval_choice()
    df_all = fetch_and_prepare_data(symbols, interval)
    if df_all.empty:
        print("Veri yok.")
        return
    print("Model eğitiliyor...")
    model = train_model(df_all)
    print("Sinyaller üretiliyor...")
    df_signals = generate_signals(df_all, model)
    if df_signals.empty:
        print("Hiç AL sinyali yok.")
        return
    df_signals.sort_values(by='score', ascending=False, inplace=True)
    df_signals.to_excel("tum_sinyaller.xlsx", index=False)
    al_df = df_signals[df_signals['type'] == 'AL']
    al_df.to_excel("al_sinyalleri.xlsx", index=False)
    print(f"{len(al_df)} adet AL sinyali bulundu.")
    print("Backtest başlatılıyor...")
    al_symbols = al_df['symbol'].tolist()
    _, summary = backtest_signals(df_all, model, al_symbols)
    print("Performans raporu kaydedildi: al_sinyalli_hisseler_ozet.xlsx")

if __name__ == "__main__":
    main()
