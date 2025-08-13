# utils/data.py

import requests
import pandas as pd
from datetime import datetime
import time

def get_gold_data(api_key, interval, symbol, outputsize=500):
    """
    Mengambil data dari Twelve Data.
    FINAL VERSION: Logika paginasi diperbaiki untuk mematuhi limit 5000 per permintaan.
    """
    all_data_list = []
    end_date = None
    remaining_size = outputsize

    while remaining_size > 0:
        # Pastikan ukuran permintaan saat ini tidak pernah melebihi 5000
        current_batch_size = min(remaining_size, 5000)
        
        api_url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={current_batch_size}&apikey={api_key}"
        
        if end_date:
            api_url += f"&end_date={end_date}"

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            if 'values' not in data:
                if data.get('code') == 429:
                    print("Batas API per menit tercapai. Menunggu 60 detik...")
                    time.sleep(60)
                    continue # Coba lagi permintaan yang sama setelah menunggu
                
                print("Respons API tidak mengandung 'values' atau ada error lain. Pesan dari server:")
                print(data)
                break

            df = pd.DataFrame(data['values'])
            
            if df.empty:
                break
            
            # Tambahkan batch data ke dalam list
            all_data_list.append(df)
            
            # Update end_date untuk iterasi berikutnya
            end_date = df['datetime'].min()
            
            # Kurangi sisa data yang perlu diambil
            remaining_size -= len(df)
            
            # Beri jeda 1 detik untuk API gratis
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Error koneksi saat mengambil data dari Twelve Data: {e}")
            return None
        except Exception as e:
            print(f"Error saat memproses data: {e}")
            return None
            
    if not all_data_list:
        return None

    # Gabungkan semua DataFrame dalam list menjadi satu
    all_data = pd.concat(all_data_list)
    
    # Hapus duplikat dan format DataFrame
    all_data.drop_duplicates(subset='datetime', keep='first', inplace=True)
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    all_data.set_index('datetime', inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close']
    if 'volume' not in all_data.columns:
        all_data['volume'] = 0  # fallback jika tidak tersedia
    numeric_cols.append('volume')
    for col in numeric_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
        
    all_data.sort_index(inplace=True)
    final_data = all_data.tail(outputsize)

    return final_data

# Price formatting utilities for data processing
def format_data_price(symbol, price, data_source="Twelve Data"):
    """
    Format price for data display with source-specific rules
    """
    try:
        price = float(price)
        symbol_upper = symbol.upper().replace('/', '')
        
        if data_source == "Binance":
            # Binance crypto-specific formatting
            if symbol_upper.endswith('USDT'):
                if price >= 100:
                    return f"${price:,.2f}"
                elif price >= 1:
                    return f"${price:.4f}"
                else:
                    return f"${price:.6f}"
            else:
                return f"${price:,.4f}"
        
        elif data_source == "Twelve Data":
            # Twelve Data multi-asset formatting
            if any(gold in symbol_upper for gold in ['XAU', 'GOLD']):
                return f"${price:,.2f}"
            elif any(fx in symbol_upper for fx in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']):
                if 'JPY' in symbol_upper:
                    return f"{price:.3f}"
                else:
                    return f"{price:.5f}"
            else:
                return f"${price:,.4f}"
        
        # Default formatting
        return f"${price:,.4f}"
        
    except (ValueError, TypeError):
        return str(price)

def get_price_precision(symbol):
    """
    Get appropriate precision for a symbol
    """
    symbol_upper = symbol.upper().replace('/', '')
    
    # High precision assets
    if any(fx in symbol_upper for fx in ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']):
        return 5
    
    # Medium precision
    elif any(jpy in symbol_upper for jpy in ['JPY']):
        return 3
    
    # Crypto precision based on price range
    elif symbol_upper.endswith('USDT'):
        return 4  # Most crypto pairs
    
    # Gold and commodities
    elif any(commodity in symbol_upper for commodity in ['XAU', 'GOLD']):
        return 2
    
    # Default
    return 4