# VNSTOCK API Documentation for StocketAI

This document provides comprehensive documentation of VNSTOCK API response formats, usage examples, and best practices for integration with StocketAI's extensible data acquisition framework.

## Generated Information
- **Test Environment**: Python with vnstock library
- **Test Data**: VCB (Vietcombank) stock symbol
- **Documentation Coverage**: All vnstock features for StocketAI's multi-provider data acquisition system

## 📝 Source Code Access Note

**Important**: This documentation can be updated anytime by accessing the actual vnstock source code located in the `vnstock/` directory. The source code contains the most current implementation details, method signatures, and response formats. For the most accurate and up-to-date information, always refer to the source files in:
- `vnstock/vnstock/` - Main package code
- `vnstock/explorer/` - Data provider implementations
- `vnstock/core/` - Core functionality
- `vnstock/tests/` - Test files showing expected behavior

## StocketAI Integration Context

This API documentation supports StocketAI's extensible data acquisition framework which:
- Supports multiple data providers (VCI, TCBS, MSN, FMARKET) with automatic fallback mechanisms
- Enables company universe definition and management (VN30, custom lists, or any configurable stock universe)
- Provides provider abstraction for adding new data sources beyond vnstock
- Ensures research flexibility, reproducibility, and quantitative finance methodologies

---

# Installation and Setup

## Basic Installation

```bash
pip install -U vnstock
```

## Development Installation

```bash
pip install git+https://github.com/thinh-vu/vnstock.git
```

## Requirements

- Python >= 3.10
- pandas
- requests
- beautifulsoup4
- seaborn
- openpyxl
- pydantic
- tenacity

---

# Quick Start Guide

## Basic Usage

```python
from vnstock import Vnstock

# Initialize with default settings
stock = Vnstock().stock(symbol='ACB', source='VCI')

# Get historical price data
history = stock.quote.history(start='2024-01-01', end='2024-12-31')
print(history)
```

## Multiple Data Sources

```python
from vnstock import Vnstock, Quote, Company, Finance

# Using different data sources
vci_stock = Vnstock().stock(symbol='ACB', source='VCI')
tcbs_stock = Vnstock().stock(symbol='ACB', source='TCBS')

# Direct API access
quote = Quote(symbol='ACB', source='VCI')
company = Company(symbol='ACB', source='TCBS')
finance = Finance(symbol='ACB', source='VCI')
```

## Basic Stock Analysis Example

```python
from vnstock import Vnstock

# Initialize Vnstock
stock = Vnstock().stock(symbol='ACB', source='VCI')

# Get historical price data
history = stock.quote.history(
    start='2024-01-01',
    end='2024-12-31',
    interval='1D'
)

print(f"Retrieved {len(history)} days of data for ACB")
print(f"Latest closing price: {history['close'].iloc[-1]:,.0f} VND")
print(f"Price range: {history['low'].min():,.0f} - {history['high'].max():,.0f} VND")
```

## Company Information Example

```python
from vnstock import Company, Finance

# Get company overview
company = Company(symbol='ACB', source='TCBS')
overview = company.overview()

print("Company Overview:")
print(f"Company Name: {overview.get('companyName', 'N/A')}")
print(f"Industry: {overview.get('industryName', 'N/A')}")

# Get financial ratios
finance = Finance(symbol='ACB', source='VCI')
ratios = finance.ratio(period='year', lang='vi')

if not ratios.empty:
    latest_ratios = ratios.iloc[-1]
    print("Key Financial Ratios:")
    print(f"P/E Ratio: {latest_ratios.get('pe', 'N/A')}")
    print(f"ROE: {latest_ratios.get('roe', 'N/A')}")
```

---

# Core API Classes and Methods

## Vnstock Class
Main entry point for the Vnstock library with unified interface.

### Constructor
```python
Vnstock(symbol: str = None, source: str = "VCI", show_log: bool = True)
```

**Parameters:**
- `symbol` (str, optional): Stock symbol for immediate analysis
- `source` (str, optional): Data source. Defaults to "VCI"
- `show_log` (bool, optional): Enable/disable logging. Defaults to True

**Supported Sources:**
- "VCI" - Vietstock
- "TCBS" - Techcom Securities
- "MSN" - MSN Money

### Methods

#### stock()
Initialize stock analysis with specified symbol and source.
```python
stock = Vnstock().stock(symbol='ACB', source='VCI')
```

#### fx()
Initialize forex analysis.
```python
fx = Vnstock().fx(symbol='EURUSD', source='MSN')
```

#### crypto()
Initialize cryptocurrency analysis.
```python
crypto = Vnstock().crypto(symbol='BTC', source='MSN')
```

#### world_index()
Initialize global index analysis.
```python
index = Vnstock().world_index(symbol='SPX', source='MSN')
```

#### fund()
Initialize mutual fund analysis.
```python
fund = Vnstock().fund(source='FMARKET')
```

## Quote Class
Price and trading data functionality.

### Constructor
```python
Quote(symbol: str, source: str = "VCI")
```

### Methods

#### history()
Get historical price data.
```python
history = quote.history(
    start='2024-01-01',
    end='2024-12-31',
    interval='1D'
)
```

**Parameters:**
- `start` (str): Start date in 'YYYY-MM-DD' format (required)
- `end` (str): End date in 'YYYY-MM-DD' format (required)
- `interval` (str, optional): Data interval. Defaults to '1D'
- `symbol` (str, optional): Override symbol for this call

#### intraday()
Get intraday tick data.
```python
intraday = quote.intraday(
    symbol='ACB',
    page_size=10000,
    show_log=False
)
```

**Parameters:**
- `symbol` (str, optional): Override symbol for this call
- `page_size` (int, optional): Number of records to retrieve. Defaults to 100
- `show_log` (bool, optional): Show progress log. Defaults to True

#### price_depth()
Get market depth data.
```python
depth = quote.price_depth(symbol='ACB')
```

## Company Class
Company information and corporate data.

### Constructor
```python
Company(symbol: str = 'ACB', source: str = "TCBS")
```

### Methods

#### overview()
Get company overview information.

#### profile()
Get detailed company profile.

#### shareholders()
Get major shareholders information.

#### officers()
Get company officers and management.

#### events()
Get corporate events and announcements.

#### news()
Get company news and press releases.

#### dividends()
Get dividend history.

## Finance Class
Financial statements and ratios.

### Constructor
```python
Finance(
    symbol: str,
    period: str = 'quarter',
    source: str = 'TCBS',
    get_all: bool = True
)
```

### Methods

#### balance_sheet()
Get balance sheet data.
```python
balance_sheet = finance.balance_sheet(
    period='year',
    lang='vi',
    dropna=True
)
```

#### income_statement()
Get income statement data.
```python
income_stmt = finance.income_statement(
    period='quarter',
    lang='en',
    dropna=True
)
```

#### cash_flow()
Get cash flow statement data.
```python
cash_flow = finance.cash_flow(
    period='year',
    dropna=True
)
```

#### ratio()
Get financial ratios data.
```python
ratios = finance.ratio(
    period='year',
    lang='vi',
    dropna=True
)
```

## Listing Class
Market listings and symbol information.

### Constructor
```python
Listing(source: str = "VCI")
```

### Methods

#### all_symbols()
Get all listed symbols.

#### symbols_by_industries()
Get symbols grouped by industry.

#### symbols_by_exchange()
Get symbols grouped by exchange.

#### symbols_by_group()
Get symbols by index group.
```python
vn30_symbols = listing.symbols_by_group(group='VN30')
```

## Trading Class
Real-time trading data and market information.

### Constructor
```python
Trading(symbol: str = 'VN30F1M', source: str = "VCI")
```

### Methods

#### price_board()
Get real-time price board for multiple symbols.
```python
price_board = trading.price_board(['VCB', 'ACB', 'TCB', 'BID'])
```

## Screener Class
Advanced stock screening capabilities.

### Constructor
```python
Screener(source: str = "TCBS")
```

### Methods

#### stock()
Screen stocks based on criteria.
```python
results = screener.stock(
    params={
        "exchangeName": "HOSE,HNX,UPCOM",
        "marketCap": "1000-50000",
        "pe": "5-20",
        "roe": "10-30"
    },
    limit=50
)
```

---

## 1. QUOTE API

### 1.1 Intraday Data
**Method**: `quote.intraday(symbol='VCB', page_size=5)`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['time', 'price', 'volume', 'match_type', 'id']`
- **Shape**: `(5, 5)` (depends on page_size parameter)
- **Sample Data Structure**:
```json
{
  "time": {
    "0": "2025-09-26 14:45:00+0700",
    "1": "2025-09-26 14:45:00+0700"
  },
  "price": {
    "0": 63.0,
    "1": 63.0
  },
  "volume": {
    "0": 1000,
    "1": 100
  },
  "match_type": {
    "0": "Buy",
    "1": "Buy"
  },
  "id": {
    "0": "371415082",
    "1": "371415081"
  }
}
```

**Column Details**:
- `time`: Timestamp with timezone (Asia/Ho_Chi_Minh)
- `price`: Transaction price (float)
- `volume`: Transaction volume (integer)
- `match_type`: Type of transaction ("Buy" or "Sell")
- `id`: Unique transaction identifier (string)

### 1.2 Historical Price Data
**Method**: `quote.history(symbol='VCB', start='2024-01-01', end='2024-01-05')`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['time', 'open', 'high', 'low', 'close', 'volume']`
- **Shape**: `(4, 6)` (number of trading days in range)
- **Sample Data Structure**:
```json
{
  "time": {
    "0": "2024-01-02 00:00:00",
    "1": "2024-01-03 00:00:00"
  },
  "open": {
    "0": 55.45,
    "1": 55.85
  },
  "high": {
    "0": 55.92,
    "1": 56.52
  },
  "low": {
    "0": 54.98,
    "1": 55.38
  },
  "close": {
    "0": 55.85,
    "1": 56.52
  },
  "volume": {
    "0": 1788420,
    "1": 1374590
  }
}
```

**Column Details**:
- `time`: Trading date (datetime without timezone)
- `open`: Opening price (float)
- `high`: Highest price of the day (float)
- `low`: Lowest price of the day (float)
- `close`: Closing price (float)
- `volume`: Total trading volume (integer)

---

## 2. COMPANY API

### 2.1 Company Overview
**Method**: `company.overview()`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['symbol', 'id', 'issue_share', 'history', 'company_profile', 'icb_name3', 'icb_name2', 'icb_name4', 'financial_ratio_issue_share', 'charter_capital']`
- **Shape**: `(1, 10)` (one row per company)
- **Sample Data Structure**:
```json
{
  "symbol": {"0": "VCB"},
  "id": {"0": "75836"},
  "issue_share": {"0": 8355675094},
  "history": {"0": " - Ngày 30/10/1962: Ngân hàng Ngoại thương Việt Nam..."},
  "company_profile": {"0": "Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam..."},
  "icb_name3": {"0": "Ngân hàng"},
  "icb_name2": {"0": "Ngân hàng"},
  "icb_name4": {"0": "Ngân hàng"},
  "financial_ratio_issue_share": {"0": 83555675094},
  "charter_capital": {"0": 83556750940000}
}
```

**Column Details**:
- `symbol`: Stock symbol (string)
- `id`: Company ID (string)
- `issue_share`: Number of issued shares (integer)
- `history`: Company history (long text)
- `company_profile`: Company profile description (text)
- `icb_name3`, `icb_name2`, `icb_name4`: Industry classification names
- `financial_ratio_issue_share`: Financial ratio data (integer)
- `charter_capital`: Charter capital (integer)

### 2.2 Company Shareholders
**Method**: `company.shareholders()`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['id', 'share_holder', 'quantity', 'share_own_percent', 'update_date']`
- **Shape**: `(48, 5)` (varies by company)
- **Sample Data Structure**:
```json
{
  "id": {"0": "89194301", "1": "89194247"},
  "share_holder": {"0": "Ngân Hàng Nhà Nước Việt Nam", "1": "Mizuho Bank Limited"},
  "quantity": {"0": 6250338579, "1": 1253366534},
  "share_own_percent": {"0": 0.748, "1": 0.15},
  "update_date": {"0": "2025-08-11", "1": "2025-08-11"}
}
```

**Column Details**:
- `id`: Shareholder record ID (string)
- `share_holder`: Name of the shareholder (string)
- `quantity`: Number of shares owned (integer)
- `share_own_percent`: Percentage ownership (float)
- `update_date`: Last update date (string, YYYY-MM-DD format)

---

## 3. LISTING API

### 3.1 All Symbols
**Method**: `listing.all_symbols()`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['symbol', 'organ_name']`
- **Shape**: `(1719, 2)` (total number of listed symbols)
- **Sample Data Structure**:
```json
{
  "symbol": {"0": "YTC", "1": "YEG"},
  "organ_name": {"0": "Công ty Cổ phần Xuất nhập khẩu Y tế Thành phố Hồ Chí Minh", "1": "Công ty Cổ phần Tập đoàn Yeah1"}
}
```

**Column Details**:
- `symbol`: Stock symbol (string)
- `organ_name`: Full company name (string)

### 3.2 Symbols by Group
**Method**: `listing.symbols_by_group('VN30')`

**Response Format**:
- **Type**: `pandas.core.series.Series`
- **Content**: List of stock symbols belonging to the specified group
- **Sample**: Returns VN30 constituent stocks like `['VCB', 'VNM', 'VIC', ...]`

**Notes**:
- Returns a pandas Series containing stock symbols
- Length varies based on the group (VN30 has 30 constituents)
- All symbols are returned as strings in uppercase

---

## 4. TRADING API

### 4.1 Price Board
**Method**: `trading.price_board(symbols_list=['VCB', 'VNM'])`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: Multi-level columns with 68 total columns including:
  - `('listing', 'symbol')`: Stock symbol
  - `('listing', 'ceiling')`: Ceiling price
  - `('listing', 'floor')`: Floor price
  - `('listing', 'ref_price')`: Reference price
  - `('match', 'match_price')`: Current match price
  - `('match', 'accumulated_volume')`: Total volume traded
  - `('match', 'highest')`: Highest price
  - `('match', 'lowest')`: Lowest price
  - `('bid_ask', 'bid_1_price')`: Best bid price
  - `('bid_ask', 'ask_1_price')`: Best ask price
- **Shape**: `(2, 68)` (number of symbols requested)
- **Sample Data Structure**:
```json
{
  "('listing', 'symbol')": {"0": "VCB", "1": "VNM"},
  "('listing', 'ceiling')": {"0": 67400, "1": 65800},
  "('listing', 'floor')": {"0": 58600, "1": 57200},
  "('listing', 'ref_price')": {"0": 63000, "1": 61500},
  "('match', 'match_price')": {"0": 63000, "1": 61100},
  "('match', 'accumulated_volume')": {"0": 3981900, "1": 2489900},
  "('match', 'highest')": {"0": 63400, "1": 61800},
  "('match', 'lowest')": {"0": 62900, "1": 61100},
  "('bid_ask', 'bid_1_price')": {"0": 62900, "1": 61100},
  "('bid_ask', 'ask_1_price')": {"0": 63000, "1": 61200}
}
```

**Key Data Categories**:
- **Listing Info**: Symbol, prices, exchange, trading status
- **Match Data**: Current trading statistics, volumes, prices
- **Bid/Ask Data**: Order book information with price levels

---

## 5. SCREENER API

### 5.1 Stock Screener
**Method**: `screener.stock(limit=5)`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: 98 columns including:
  - `ticker`: Stock symbol
  - `exchange`: Exchange (HSX, HNX, UPCOM)
  - `industry`: Industry classification
  - `market_cap`: Market capitalization
  - `roe`: Return on equity
  - `pe`: Price-to-earnings ratio
  - `pb`: Price-to-book ratio
  - `dividend_yield`: Dividend yield percentage
  - `price_growth_1w`: 1-week price growth
  - `foreign_vol_pct`: Foreign volume percentage
  - `rsi14`: RSI 14-day indicator
- **Shape**: `(5, 98)` (based on limit parameter)
- **Sample Data Structure**:
```json
{
  "ticker": {"0": "A32", "1": "AAA"},
  "exchange": {"0": "UPCOM", "1": "HSX"},
  "industry": {"0": "Hàng cá nhân & Gia dụng", "1": "Hóa chất"},
  "market_cap": {"0": null, "1": 3382.0},
  "roe": {"0": null, "1": 5.1},
  "pe": {"0": null, "1": 12.3},
  "pb": {"0": null, "1": 0.6},
  "dividend_yield": {"0": null, "1": 3.6},
  "price_growth_1w": {"0": null, "1": 5.01},
  "foreign_vol_pct": {"0": 0.0, "1": 2.48},
  "rsi14": {"0": null, "1": 63.1}
}
```

**Data Categories**:
- **Basic Info**: Ticker, exchange, industry, market cap
- **Valuation Ratios**: PE, PB, EV/EBITDA, dividend yield
- **Growth Metrics**: Revenue growth, EPS growth, price growth
- **Technical Indicators**: RSI, moving averages, volume ratios
- **Market Activity**: Foreign trading, trading value, volatility

---

## 6. FUND API

### 6.1 Fund Listing
**Method**: `fund.listing()`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['short_name', 'name', 'fund_type', 'fund_owner_name', 'management_fee', 'inception_date', 'nav', 'nav_change_previous', 'nav_change_last_year', 'nav_change_inception', 'nav_change_1m', 'nav_change_3m', 'nav_change_6m', 'nav_change_12m', 'nav_change_24m', 'nav_change_36m', 'nav_change_36m_annualized', 'nav_update_at', 'fund_id_fmarket', 'fund_code', 'vsd_fee_id']`
- **Shape**: `(58, 21)` (total number of available funds)
- **Sample Data Structure**:
```json
{
  "short_name": {"0": "DCDS", "1": "SSISCA"},
  "name": {"0": "QUỸ ĐẦU TƯ CHỨNG KHOÁN NĂNG ĐỘNG DC", "1": "QUỸ ĐẦU TƯ LỢI THẾ CẠNH TRANH BỀN VỮNG SSI"},
  "fund_type": {"0": "Quỹ cổ phiếu", "1": "Quỹ cổ phiếu"},
  "fund_owner_name": {"0": "CÔNG TY CỔ PHẦN QUẢN LÝ QUỸ DRAGON CAPITAL VIỆT NAM", "1": "CÔNG TY TNHH QUẢN LÝ QUỸ SSI"},
  "management_fee": {"0": 1.95, "1": 1.75},
  "nav": {"0": 107819.3, "1": 468337.82},
  "nav_change_1m": {"0": -0.13, "1": 1.22},
  "nav_change_1y": {"0": 32.1, "1": 14.84},
  "nav_change_3y_annualized": {"0": 23.07, "1": 18.54}
}
```

**Column Details**:
- `short_name`: Fund identifier (string)
- `name`: Full fund name (string)
- `fund_type`: Type of fund (string)
- `fund_owner_name`: Management company (string)
- `management_fee`: Management fee percentage (float)
- `inception_date`: Fund launch date (string)
- `nav`: Net Asset Value (float)
- `nav_change_*`: NAV performance over different periods (float)
- `fund_id_fmarket`: Fmarket fund ID (integer)

### 6.2 Fund Top Holdings
**Method**: `fund.details.top_holding(fund_symbol)`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: `['stock_code', 'industry', 'net_asset_percent', 'type_asset', 'update_at', 'fundId', 'short_name']`
- **Shape**: `(10, 7)` (top 10 holdings)
- **Sample Data Structure**:
```json
{
  "stock_code": {"0": "VPB", "1": "MWG"},
  "industry": {"0": "Ngân hàng", "1": "Bán lẻ"},
  "net_asset_percent": {"0": 7.58, "1": 5.65},
  "type_asset": {"0": "STOCK", "1": "STOCK"},
  "update_at": {"0": "2025-09-10", "1": "2025-09-10"},
  "fundId": {"0": 28, "1": 28},
  "short_name": {"0": "DCDS", "1": "DCDS"}
}
```

**Column Details**:
- `stock_code`: Stock symbol held by the fund (string)
- `industry`: Industry of the holding (string)
- `net_asset_percent`: Percentage of fund assets (float)
- `type_asset`: Asset type ("STOCK" or "BOND")
- `update_at`: Last update date (string)
- `fundId`: Fund identifier (integer)
- `short_name`: Fund symbol (string)

---

## 7. FINANCE API

### 7.1 Balance Sheet
**Method**: `finance.balance_sheet(symbol='VCB', period='year', lang='vi')`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: Vietnamese financial statement columns including:
  - `CP`: Company symbol
  - `Năm`: Year
  - Various balance sheet items in Vietnamese (Tài sản, Nợ, Vốn chủ sở hữu, etc.)
- **Shape**: Variable based on available data periods
- **Sample Data Structure**:
```json
{
  "CP": {"0": "VCB", "1": "VCB"},
  "Năm": {"0": 2024, "1": 2023},
  "Tài sản": {"0": 123456789, "1": 987654321},
  "Nợ": {"0": 234567890, "1": 198765432},
  "Vốn chủ sở hữu": {"0": 345678901, "1": 287654321}
}
```

**Parameters**:
- `period`: 'year' or 'quarter' (default: 'year')
- `lang`: 'vi' or 'en' (default: 'vi')
- `dropna`: Boolean to drop columns with all zeros (default: True)

### 7.2 Income Statement
**Method**: `finance.income_statement(symbol='VCB', period='year', lang='vi')`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: Vietnamese income statement columns including:
  - `CP`: Company symbol
  - `Năm`: Year
  - Various income statement items in Vietnamese (Doanh thu, Lợi nhuận, Chi phí, etc.)
- **Shape**: Variable based on available data periods
- **Sample Data Structure**:
```json
{
  "CP": {"0": "VCB", "1": "VCB"},
  "Năm": {"0": 2024, "1": 2023},
  "Doanh thu": {"0": 123456789, "1": 987654321},
  "Lợi nhuận gộp": {"0": 234567890, "1": 198765432},
  "Lợi nhuận thuần": {"0": 345678901, "1": 287654321}
}
```

**Parameters**:
- `period`: 'year' or 'quarter' (default: 'year')
- `lang`: 'vi' or 'en' (default: 'vi')
- `dropna`: Boolean to drop columns with all zeros (default: True)

### 7.3 Cash Flow Statement
**Method**: `finance.cash_flow(symbol='VCB', period='year', lang='vi')`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: Vietnamese cash flow statement columns including:
  - `CP`: Company symbol
  - `Năm`: Year
  - Various cash flow items in Vietnamese (Dòng tiền từ hoạt động kinh doanh, đầu tư, tài chính)
- **Shape**: Variable based on available data periods
- **Sample Data Structure**:
```json
{
  "CP": {"0": "VCB", "1": "VCB"},
  "Năm": {"0": 2024, "1": 2023},
  "Dòng tiền từ HĐKD": {"0": 123456789, "1": 987654321},
  "Dòng tiền từ HĐĐT": {"0": -234567890, "1": -198765432},
  "Dòng tiền từ HĐTC": {"0": 345678901, "1": 287654321}
}
```

**Parameters**:
- `period`: 'year' or 'quarter' (default: 'year')
- `lang`: 'vi' or 'en' (default: 'vi')
- `dropna`: Boolean to drop columns with all zeros (default: True)

### 7.4 Financial Ratios
**Method**: `finance.ratio(symbol='VCB', period='year', lang='vi')`

**Response Format**:
- **Type**: `pandas.core.frame.DataFrame`
- **Columns**: Multi-level columns with financial ratios including:
  - Top level: Report categories (Chỉ tiêu cân đối kế toán, Chỉ tiêu lưu chuyển tiền tệ, etc.)
  - Bottom level: Specific ratio names in Vietnamese
- **Shape**: Variable based on available ratios and periods
- **Sample Data Structure**:
```json
{
  "Meta": {
    "CP": {"0": "VCB", "1": "VCB"},
    "Năm": {"0": 2024, "1": 2023}
  },
  "Chỉ tiêu cân đối kế toán": {
    "ROE": {"0": 18.5, "1": 16.2},
    "ROA": {"0": 2.1, "1": 1.8}
  },
  "Chỉ tiêu kết quả kinh doanh": {
    "NIM": {"0": 3.2, "1": 3.5},
    "CAR": {"0": 12.5, "1": 11.8}
  }
}
```

**Parameters**:
- `period`: 'year' or 'quarter' (default: 'year')
- `lang`: 'vi' or 'en' (default: 'vi')
- `dropna`: Boolean to drop columns with all zeros (default: True)
- `flatten_columns`: Boolean to flatten multi-level columns (default: False)
- `separator`: String separator for flattened columns (default: "_")
- `drop_levels`: Levels to drop when flattening (default: None)

**Notes**:
- All Finance API methods require both `source` and `symbol` parameters in initialization
- Data is sourced from VCI (Vietcap) provider
- Vietnamese language returns column names in Vietnamese
- English language returns column names in English
- Methods may fail due to network issues or API limitations (RetryError)

---

## 8. COMPANY API

### 8.1 Company Profile
**Method**: `company.profile()`

### 8.2 Company News
**Method**: `company.news(limit=10, start_date='2024-01-01')`

### 8.3 Company Events
**Method**: `company.events(event_type='all', limit=10)`

### 8.4 Company Officers
**Method**: `company.officers()`

---

## 12. API INITIALIZATION PATTERNS

Based on the testing, here are the correct initialization patterns for each API:

### 7.1 Quote API
```python
quote = Quote(symbol='VCB')
# No source parameter required (defaults to VCI)
```

### 7.2 Company API
```python
company = Company(source='vci', symbol='VCB')
# Requires both source and symbol parameters
```

### 7.3 Finance API
```python
finance = Finance(source='vci', symbol='VCB')
# Requires both source and symbol parameters
```

### 7.4 Listing API
```python
listing = Listing()
# No parameters required
```

### 7.5 Trading API
```python
trading = Trading(source='vci', symbol='VCB')
# Requires both source and symbol parameters
```

### 7.6 Screener API
```python
screener = Screener()
# No parameters required (defaults to TCBS source)
```

### 7.7 Fund API
```python
fund = Fund()
# No parameters required
```

---

## 4. DATA TYPES AND PATTERNS

### 4.1 Common Data Types
- **Price Data**: `float` (e.g., 63.0, 55.45)
- **Volume Data**: `integer` (e.g., 1000, 1788420)
- **Time Data**: `pandas.Timestamp` with/without timezone
- **String Data**: `str` (uppercase for symbols, mixed case for other text)
- **Match Types**: `str` ("Buy", "Sell")

### 4.2 DataFrame Characteristics
- All successful API calls return pandas DataFrames or Series
- DataFrames include proper column names and data types
- Timezone information is preserved where relevant
- Numeric data is properly typed (float/int) rather than strings

### 4.3 Error Patterns
- **Missing Methods**: Some expected methods don't exist (e.g., `quote.price()`, `company.profile()`)
- **Parameter Errors**: Incorrect parameters cause TypeErrors
- **Network/API Errors**: Wrapped in RetryError exceptions
- **Validation Errors**: Symbol validation requires 3-10 character strings

---

## 5. USAGE NOTES

### 5.1 Symbol Requirements
- Symbols must be 3-10 characters long
- Case-insensitive but normalized to uppercase
- Must be valid Vietnamese stock symbols

### 5.2 Date Format Requirements
- Use ISO format: `YYYY-MM-DD`
- Historical data respects start/end date ranges
- Intraday data uses current date

### 5.3 Pagination
- `page_size` parameter controls number of records returned
- Different APIs may use different pagination parameter names
- Some APIs may not support pagination

### 5.4 Data Sources
- Most APIs support multiple sources: 'vci', 'tcbs', 'msn'
- Default source varies by API
- Some APIs are restricted to specific sources (e.g., Screener only supports 'tcbs')

---

## 9. RECOMMENDATIONS

### 9.1 For Developers
1. Always check API method availability before calling
2. Handle both network errors and API-specific errors
3. Validate symbols before making API calls
4. Be aware of data type conversions (especially for time and numeric data)

### 9.2 For API Maintenance
1. Standardize method naming across APIs
2. Improve error messages for missing methods
3. Add comprehensive method documentation
4. Consider adding method discovery/introspection capabilities

### 9.3 For Testing
1. Use valid stock symbols (3-10 characters)
2. Test with small date ranges first
3. Verify both successful and error response formats
4. Check data type consistency across different symbols

---

# Advanced Usage Examples

## Portfolio Analysis Example

```python
from vnstock import Vnstock, Screener

# Screen for high-quality stocks
screener = Screener(source='TCBS')

# Multi-criteria screening for portfolio candidates
candidates = screener.stock(
    params={
        "exchangeName": "HOSE,HNX",
        "marketCap": "1000-50000",      # Mid to large cap
        "pe": "8-20",                   # Reasonable P/E
        "pb": "0.8-2",                  # Reasonable P/B
        "roe": ">12",                   # Strong profitability
        "dividendYield": ">2",          # Dividend paying
        "debtToEquity": "<0.8"          # Conservative debt
    },
    limit=20
)

print(f"Found {len(candidates)} portfolio candidates")

# Build diversified portfolio
portfolio = candidates.groupby('industryName').head(3)  # Max 3 per industry

# Calculate portfolio weights based on market cap
total_market_cap = portfolio['marketCap'].sum()
portfolio['weight'] = portfolio['marketCap'] / total_market_cap

print("Portfolio Composition:")
print(portfolio[['symbol', 'companyName', 'industryName', 'weight', 'pe', 'roe']].to_string())
```

## Technical Analysis Example

```python
from vnstock import Vnstock
import pandas as pd

# Get historical data for technical analysis
stock = Vnstock().stock(symbol='ACB', source='VCI')
data = stock.quote.history(start='2024-01-01', end='2024-12-31', interval='1D')

# Calculate moving averages
data['SMA_20'] = data['close'].rolling(window=20).mean()
data['SMA_50'] = data['close'].rolling(window=50).mean()
data['EMA_12'] = data['close'].ewm(span=12).mean()
data['EMA_26'] = data['close'].ewm(span=26).mean()

# Calculate RSI
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Generate signals
data['Signal'] = 0
data.loc[data['SMA_20'] > data['SMA_50'], 'Signal'] = 1  # Golden cross
data.loc[data['SMA_20'] < data['SMA_50'], 'Signal'] = -1  # Death cross

print(f"Latest Technical Indicators for ACB:")
print(f"SMA 20: {data['SMA_20'].iloc[-1]:.2f}")
print(f"SMA 50: {data['SMA_50'].iloc[-1]:.2f}")
print(f"RSI: {data['RSI'].iloc[-1]:.2f}")
```

## Risk Analysis Example

```python
from vnstock import Vnstock
import numpy as np

# Get historical data for multiple stocks
symbols = ['ACB', 'VCB', 'TCB', 'BID']
risk_data = {}

for symbol in symbols:
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        data = stock.quote.history(start='2024-01-01', end='2024-12-31', interval='1D')

        # Calculate daily returns
        data['daily_return'] = data['close'].pct_change()

        # Calculate risk metrics
        daily_returns = data['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        var_95 = daily_returns.quantile(0.05)  # 95% Value at Risk
        max_drawdown = (data['close'] / data['close'].cummax() - 1).min()

        risk_data[symbol] = {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'avg_return': daily_returns.mean() * 252  # Annualized return
        }

        print(f"{symbol}:")
        print(f"  Annual Volatility: {volatility:.4f}")
        print(f"  95% VaR: {var_95:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.4f}")

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
```

## Correlation Analysis Example

```python
from vnstock import Vnstock

# Get data for multiple stocks
symbols = ['ACB', 'VCB', 'TCB', 'BID', 'CTG']
stock_data = {}

for symbol in symbols:
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        data = stock.quote.history(start='2024-01-01', end='2024-12-31', interval='1D')
        stock_data[symbol] = data['close']
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")

# Create correlation matrix
if stock_data:
    prices_df = pd.DataFrame(stock_data)
    returns_df = prices_df.pct_change().dropna()

    correlation_matrix = returns_df.corr()

    print("Stock Correlation Matrix:")
    print(correlation_matrix.round(3))
```

---

# Error Handling and Troubleshooting

## Common Exceptions

### ValueError
Raised when invalid parameters are provided.

```python
try:
    stock = Vnstock().stock(symbol='INVALID', source='VCI')
except ValueError as e:
    print(f"Invalid symbol: {e}")
```

### ConnectionError
Raised when network connection fails.

```python
try:
    data = stock.quote.history(start='2024-01-01', end='2024-12-31')
except ConnectionError as e:
    print(f"Connection failed: {e}")
```

### TimeoutError
Raised when request times out.

```python
try:
    data = stock.quote.history(start='2024-01-01', end='2024-12-31')
except TimeoutError as e:
    print(f"Request timed out: {e}")
```

## Error Handling Best Practices

### Try-Catch Blocks

```python
def safe_data_retrieval(symbol, source='VCI'):
    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
        data = stock.quote.history(start='2024-01-01', end='2024-12-31')
        return data
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None
```

### Fallback Sources

```python
def get_data_with_fallback(symbol, start_date, end_date):
    sources = ['VCI', 'TCBS']

    for source in sources:
        try:
            stock = Vnstock().stock(symbol=symbol, source=source)
            data = stock.quote.history(start=start_date, end=end_date)
            print(f"Successfully retrieved data from {source}")
            return data
        except Exception as e:
            print(f"Failed with {source}: {e}")
            continue

    print("All sources failed")
    return None
```

---

# Configuration and Environment Setup

## Environment Variables

```bash
# Set default source
export VNSTOCK_DEFAULT_SOURCE=VCI

# Set timeout
export VNSTOCK_TIMEOUT=30

# Set log level
export VNSTOCK_LOG_LEVEL=INFO

# Set cache settings
export VNSTOCK_CACHE_ENABLED=true
export VNSTOCK_CACHE_SIZE=128
```

## Configuration File

Create `~/.vnstock/config.json`:

```json
{
  "default_source": "VCI",
  "timeout": 30,
  "max_retries": 3,
  "log_level": "INFO",
  "cache_enabled": true,
  "cache_size": 128,
  "user_agent": "vnstock/3.2.7"
}
```

---

# Rate Limiting and Performance

## Understanding Limits

- **VCI**: 100 requests per minute
- **TCBS**: 50 requests per minute
- **MSN**: 200 requests per minute
- **FMARKET**: 30 requests per minute

## Implementing Rate Limiting

```python
import time
from ratelimit import limits, sleep_and_retry

class RateLimitedVnstock:
    def __init__(self, calls_per_minute=50):
        self.calls_per_minute = calls_per_minute
        self.last_call_time = 0

    @sleep_and_retry
    @limits(calls=50, period=60)
    def get_data(self, symbol, start_date, end_date):
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        return stock.quote.history(start=start_date, end=end_date)

# Usage
rate_limited = RateLimitedVnstock(calls_per_minute=30)
data = rate_limited.get_data('ACB', '2024-01-01', '2024-12-31')
```

## Performance Optimization

```python
# Process data in chunks for large datasets
def process_large_dataset(symbol, start_date, end_date, chunk_size=1000):
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    all_data = []

    # Get data in chunks
    current_start = start_date
    while True:
        try:
            chunk = stock.quote.history(
                start=current_start,
                end=end_date,
                interval='1D'
            )

            if chunk.empty:
                break

            all_data.append(chunk)
            current_start = chunk.index[-1] + pd.Timedelta(days=1)

            if len(all_data) * len(chunk) > chunk_size:
                break

        except Exception as e:
            print(f"Error processing chunk: {e}")
            break

    return pd.concat(all_data) if all_data else pd.DataFrame()
```

---

# Data Sources

## VCI (Vietstock)

**Specialization:** Comprehensive Vietnamese stock data

**Supported Classes:**
- Quote
- Company
- Finance
- Listing
- Trading

**Key Features:**
- Historical price data
- Financial statements
- Company information
- Market listings
- Trading data

## TCBS (Techcom Securities)

**Specialization:** Advanced analytics and screening

**Supported Classes:**
- Quote
- Company
- Finance
- Screener
- Trading

**Key Features:**
- Stock screening
- Technical analysis
- Market insights
- Company data
- Trading information

## MSN (MSN Money)

**Specialization:** International markets

**Supported Classes:**
- Quote
- Listing

**Key Features:**
- Forex data
- Cryptocurrency data
- Global indices
- International stocks
- Real-time updates

## FMARKET

**Specialization:** Mutual funds

**Supported Classes:**
- Fund

**Key Features:**
- Fund listings
- Performance data
- Fund analysis
- NAV data

---

*This documentation is based on actual API testing with vnstock library version in the current environment and includes comprehensive information from the docs/vnstock directory.*
