"""
Data Validation Module for VN30 Stock Price Prediction System.

This module provides comprehensive data quality validation, health monitoring,
and integrity checks for financial data. It includes:

- Data quality assessment and scoring
- Automated health monitoring
- Data integrity validation
- Quality reporting and alerting
- Historical quality tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataQualityDimension(Enum):
    """Dimensions of data quality."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class AlertSeverity(Enum):
    """Severity levels for data quality alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityThresholds:
    """Thresholds for data quality dimensions."""
    completeness_threshold: float = 0.95  # 95% completeness required
    accuracy_threshold: float = 0.90      # 90% accuracy required
    consistency_threshold: float = 0.85    # 85% consistency required
    timeliness_threshold: float = 0.80     # 80% timeliness required
    validity_threshold: float = 0.95       # 95% validity required
    uniqueness_threshold: float = 0.99     # 99% uniqueness required


@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    name: str
    dimension: DataQualityDimension
    description: str
    validator_function: Callable
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True


@dataclass
class ValidationConfig:
    """Configuration for data validation operations."""
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    quality_report_path: Optional[str] = None
    historical_tracking: bool = True
    max_history_days: int = 90


class DataValidator:
    """
    Comprehensive data validation and quality monitoring system.

    Provides automated data quality assessment, health monitoring,
    and alerting for financial datasets.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the DataValidator.

        Args:
            config: ValidationConfig object with validation parameters
        """
        self.config = config or ValidationConfig()
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.quality_history: List[Dict] = []
        self.alerts: List[Dict] = []

        # Initialize default validation rules
        self._initialize_default_rules()

    def validate_constituents_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate constituents data quality.

        Args:
            df: Constituents DataFrame to validate

        Returns:
            Dictionary with validation results and quality scores
        """
        logger.info("Starting constituents data validation")

        validation_results = {
            'timestamp': datetime.now(),
            'data_shape': df.shape,
            'quality_scores': {},
            'rule_results': {},
            'alerts': [],
            'overall_score': 0.0
        }

        # Run all applicable validation rules
        for rule_name, rule in self.validation_rules.items():
            if not rule.enabled:
                continue

            try:
                result = rule.validator_function(df)
                validation_results['rule_results'][rule_name] = result

                # Calculate quality score for this dimension
                score = self._calculate_dimension_score(rule.dimension, result)
                validation_results['quality_scores'][rule.dimension.value] = score

            except Exception as e:
                logger.error(f"Error running validation rule {rule_name}: {e}")
                validation_results['rule_results'][rule_name] = {
                    'error': str(e),
                    'passed': False
                }

        # Calculate overall quality score
        validation_results['overall_score'] = self._calculate_overall_score(
            validation_results['quality_scores']
        )

        # Generate alerts if needed
        if self.config.enable_alerting:
            alerts = self._generate_alerts(validation_results)
            validation_results['alerts'] = alerts
            self.alerts.extend(alerts)

        # Store in history if enabled
        if self.config.historical_tracking:
            self._store_quality_history(validation_results)

        logger.info(f"Constituents validation completed. Overall score: {validation_results['overall_score']:.2f}")
        return validation_results

    def validate_price_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate price data quality.

        Args:
            df: Price DataFrame to validate

        Returns:
            Dictionary with validation results and quality scores
        """
        logger.info("Starting price data validation")

        validation_results = {
            'timestamp': datetime.now(),
            'data_shape': df.shape,
            'quality_scores': {},
            'rule_results': {},
            'alerts': [],
            'overall_score': 0.0
        }

        # Run price-specific validation rules
        for rule_name, rule in self.validation_rules.items():
            if not rule.enabled or not self._is_price_rule(rule_name):
                continue

            try:
                result = rule.validator_function(df)
                validation_results['rule_results'][rule_name] = result

                # Calculate quality score for this dimension
                score = self._calculate_dimension_score(rule.dimension, result)
                validation_results['quality_scores'][rule.dimension.value] = score

            except Exception as e:
                logger.error(f"Error running validation rule {rule_name}: {e}")
                validation_results['rule_results'][rule_name] = {
                    'error': str(e),
                    'passed': False
                }

        # Calculate overall quality score
        validation_results['overall_score'] = self._calculate_overall_score(
            validation_results['quality_scores']
        )

        # Generate alerts if needed
        if self.config.enable_alerting:
            alerts = self._generate_alerts(validation_results)
            validation_results['alerts'] = alerts
            self.alerts.extend(alerts)

        # Store in history if enabled
        if self.config.historical_tracking:
            self._store_quality_history(validation_results)

        logger.info(f"Price validation completed. Overall score: {validation_results['overall_score']:.2f}")
        return validation_results

    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules."""

        # Completeness rules
        self.validation_rules['completeness_check'] = ValidationRule(
            name='completeness_check',
            dimension=DataQualityDimension.COMPLETENESS,
            description='Check for missing values in key columns',
            validator_function=self._check_completeness,
            severity=AlertSeverity.HIGH
        )

        # Accuracy rules
        self.validation_rules['price_relationships'] = ValidationRule(
            name='price_relationships',
            dimension=DataQualityDimension.ACCURACY,
            description='Validate OHLC price relationships',
            validator_function=self._check_price_relationships,
            severity=AlertSeverity.HIGH
        )

        self.validation_rules['non_negative_prices'] = ValidationRule(
            name='non_negative_prices',
            dimension=DataQualityDimension.ACCURACY,
            description='Check for negative or zero prices',
            validator_function=self._check_non_negative_prices,
            severity=AlertSeverity.CRITICAL
        )

        # Consistency rules
        self.validation_rules['data_types'] = ValidationRule(
            name='data_types',
            dimension=DataQualityDimension.CONSISTENCY,
            description='Validate data types consistency',
            validator_function=self._check_data_types,
            severity=AlertSeverity.MEDIUM
        )

        self.validation_rules['column_naming'] = ValidationRule(
            name='column_naming',
            dimension=DataQualityDimension.CONSISTENCY,
            description='Check column naming conventions',
            validator_function=self._check_column_naming,
            severity=AlertSeverity.LOW
        )

        # Timeliness rules
        self.validation_rules['data_freshness'] = ValidationRule(
            name='data_freshness',
            dimension=DataQualityDimension.TIMELINESS,
            description='Check data freshness and update frequency',
            validator_function=self._check_data_freshness,
            severity=AlertSeverity.MEDIUM
        )

        # Validity rules
        self.validation_rules['symbol_format'] = ValidationRule(
            name='symbol_format',
            dimension=DataQualityDimension.VALIDITY,
            description='Validate stock symbol format',
            validator_function=self._check_symbol_format,
            severity=AlertSeverity.HIGH
        )

        self.validation_rules['date_format'] = ValidationRule(
            name='date_format',
            dimension=DataQualityDimension.VALIDITY,
            description='Validate date formats and ranges',
            validator_function=self._check_date_format,
            severity=AlertSeverity.MEDIUM
        )

        # Uniqueness rules
        self.validation_rules['duplicate_symbols'] = ValidationRule(
            name='duplicate_symbols',
            dimension=DataQualityDimension.UNIQUENESS,
            description='Check for duplicate symbols',
            validator_function=self._check_duplicate_symbols,
            severity=AlertSeverity.HIGH
        )

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness."""
        results = {}

        # Check overall completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

        results['overall_completeness'] = completeness_ratio
        results['total_cells'] = total_cells
        results['missing_cells'] = missing_cells

        # Check column-wise completeness
        results['column_completeness'] = {}
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            col_total = len(df)
            col_completeness = 1 - (col_missing / col_total) if col_total > 0 else 0
            results['column_completeness'][col] = {
                'completeness_ratio': col_completeness,
                'missing_count': col_missing,
                'total_count': col_total
            }

        results['passed'] = completeness_ratio >= self.config.quality_thresholds.completeness_threshold
        return results

    def _check_price_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check OHLC price relationship validity."""
        results = {}

        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            results['error'] = 'Required OHLC columns not found'
            results['passed'] = False
            return results

        # Check high >= max(open, close)
        high_valid = (df['high'] >= df[['open', 'close']].max(axis=1)).all()

        # Check low <= min(open, close)
        low_valid = (df['low'] <= df[['open', 'close']].min(axis=1)).all()

        results['high_relationship_valid'] = high_valid
        results['low_relationship_valid'] = low_valid
        results['overall_relationship_valid'] = high_valid and low_valid
        results['passed'] = results['overall_relationship_valid']

        return results

    def _check_non_negative_prices(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for negative or zero prices."""
        results = {}

        price_columns = ['open', 'high', 'low', 'close']
        invalid_prices = {}

        for col in price_columns:
            if col in df.columns:
                invalid_count = ((df[col] <= 0) | (df[col].isnull())).sum()
                invalid_prices[col] = invalid_count

        results['invalid_prices'] = invalid_prices
        results['total_invalid'] = sum(invalid_prices.values())
        results['passed'] = results['total_invalid'] == 0

        return results

    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data type consistency."""
        results = {}

        expected_types = {
            'symbol': ['object', 'string'],
            'open': ['float64', 'int64'],
            'high': ['float64', 'int64'],
            'low': ['float64', 'int64'],
            'close': ['float64', 'int64'],
            'volume': ['float64', 'int64'],
            'weight': ['float64'],
            'shares_outstanding': ['float64', 'int64'],
            'market_cap': ['float64', 'int64']
        }

        type_issues = {}

        for col, expected_type_list in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in expected_type_list:
                    type_issues[col] = {
                        'expected': expected_type_list,
                        'actual': actual_type
                    }

        results['type_issues'] = type_issues
        results['passed'] = len(type_issues) == 0

        return results

    def _check_column_naming(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check column naming conventions."""
        results = {}

        # Check for lowercase and underscores
        invalid_columns = []
        for col in df.columns:
            if not (col.islower() or '_' in col):
                invalid_columns.append(col)

        results['invalid_columns'] = invalid_columns
        results['passed'] = len(invalid_columns) == 0

        return results

    def _check_data_freshness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data freshness."""
        results = {}

        if hasattr(df.index, 'max') and hasattr(df.index, 'min'):
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    max_date = df.index.max()
                    min_date = df.index.min()
                    days_span = (max_date - min_date).days

                    # Check if data is recent (within last 7 days)
                    days_since_update = (datetime.now() - max_date).days
                    is_recent = days_since_update <= 7

                    results['max_date'] = max_date.isoformat()
                    results['min_date'] = min_date.isoformat()
                    results['days_span'] = days_span
                    results['days_since_update'] = days_since_update
                    results['is_recent'] = is_recent
                    results['passed'] = is_recent

                else:
                    results['error'] = 'Index is not datetime'
                    results['passed'] = False

            except Exception as e:
                results['error'] = str(e)
                results['passed'] = False
        else:
            results['error'] = 'Cannot determine data freshness'
            results['passed'] = False

        return results

    def _check_symbol_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check stock symbol format validity."""
        results = {}

        if 'symbol' not in df.columns:
            results['error'] = 'Symbol column not found'
            results['passed'] = False
            return results

        # Basic symbol format validation (Vietnamese stock symbols)
        symbol_pattern = r'^[A-Z]{3,4}$'
        valid_symbols = df['symbol'].astype(str).str.match(symbol_pattern)

        invalid_count = (~valid_symbols).sum()
        results['invalid_count'] = invalid_count
        results['total_count'] = len(df)
        results['validity_ratio'] = (valid_symbols.sum() / len(df)) if len(df) > 0 else 0
        results['passed'] = invalid_count == 0

        return results

    def _check_date_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check date format validity."""
        results = {}

        try:
            if isinstance(df.index, pd.DatetimeIndex):
                # Check for reasonable date range
                min_date = df.index.min()
                max_date = df.index.max()

                # Check if dates are reasonable (not in future, not too old)
                future_dates = (df.index > datetime.now()).sum()
                very_old_dates = (df.index < datetime(2000, 1, 1)).sum()

                results['min_date'] = min_date.isoformat()
                results['max_date'] = max_date.isoformat()
                results['future_dates'] = future_dates
                results['very_old_dates'] = very_old_dates
                results['passed'] = future_dates == 0 and very_old_dates == 0

            else:
                results['error'] = 'Index is not DatetimeIndex'
                results['passed'] = False

        except Exception as e:
            results['error'] = str(e)
            results['passed'] = False

        return results

    def _check_duplicate_symbols(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate symbols."""
        results = {}

        if 'symbol' in df.columns:
            duplicate_count = df['symbol'].duplicated().sum()
            results['duplicate_count'] = duplicate_count
            results['total_count'] = len(df)
            results['passed'] = duplicate_count == 0
        else:
            results['error'] = 'Symbol column not found'
            results['passed'] = False

        return results

    def _calculate_dimension_score(self, dimension: DataQualityDimension, rule_result: Dict) -> float:
        """Calculate quality score for a specific dimension."""
        try:
            if not rule_result.get('passed', False):
                return 0.0

            # Different scoring logic based on dimension
            if dimension == DataQualityDimension.COMPLETENESS:
                return rule_result.get('overall_completeness', 0.0)
            elif dimension == DataQualityDimension.ACCURACY:
                return 1.0 if rule_result.get('overall_relationship_valid', False) else 0.0
            elif dimension == DataQualityDimension.CONSISTENCY:
                issues = rule_result.get('type_issues', {})
                return 1.0 if len(issues) == 0 else 0.5
            elif dimension == DataQualityDimension.TIMELINESS:
                return 1.0 if rule_result.get('is_recent', False) else 0.3
            elif dimension == DataQualityDimension.VALIDITY:
                return rule_result.get('validity_ratio', 0.0)
            elif dimension == DataQualityDimension.UNIQUENESS:
                return 1.0 if rule_result.get('duplicate_count', 1) == 0 else 0.0
            else:
                return 0.5  # Default score

        except Exception as e:
            logger.error(f"Error calculating dimension score: {e}")
            return 0.0

    def _calculate_overall_score(self, quality_scores: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        if not quality_scores:
            return 0.0

        # Weighted average of dimension scores
        weights = {
            DataQualityDimension.COMPLETENESS.value: 0.25,
            DataQualityDimension.ACCURACY.value: 0.25,
            DataQualityDimension.CONSISTENCY.value: 0.15,
            DataQualityDimension.TIMELINESS.value: 0.15,
            DataQualityDimension.VALIDITY.value: 0.15,
            DataQualityDimension.UNIQUENESS.value: 0.05
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, score in quality_scores.items():
            weight = weights.get(dimension, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_alerts(self, validation_results: Dict[str, Any]) -> List[Dict]:
        """Generate alerts based on validation results."""
        alerts = []

        overall_score = validation_results.get('overall_score', 0.0)

        # Overall score alerts
        if overall_score < 0.5:
            alerts.append({
                'timestamp': datetime.now().isoformat(),
                'severity': AlertSeverity.CRITICAL.value,
                'message': f'Overall data quality score {overall_score:.2f} is critically low',
                'dimension': 'overall'
            })
        elif overall_score < 0.7:
            alerts.append({
                'timestamp': datetime.now().isoformat(),
                'severity': AlertSeverity.HIGH.value,
                'message': f'Overall data quality score {overall_score:.2f} is below acceptable threshold',
                'dimension': 'overall'
            })

        # Dimension-specific alerts
        for dimension, score in validation_results.get('quality_scores', {}).items():
            threshold = getattr(self.config.quality_thresholds,
                              f'{dimension}_threshold', 0.8)

            if score < threshold:
                severity = AlertSeverity.HIGH.value if score < 0.5 else AlertSeverity.MEDIUM.value
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'severity': severity,
                    'message': f'{dimension} quality score {score:.2f} below threshold {threshold:.2f}',
                    'dimension': dimension
                })

        return alerts

    def _store_quality_history(self, validation_results: Dict[str, Any]) -> None:
        """Store validation results in history."""
        history_entry = {
            'timestamp': validation_results['timestamp'].isoformat(),
            'overall_score': validation_results['overall_score'],
            'quality_scores': validation_results['quality_scores'],
            'data_shape': validation_results['data_shape']
        }

        self.quality_history.append(history_entry)

        # Keep only recent history
        max_age = datetime.now() - timedelta(days=self.config.max_history_days)
        self.quality_history = [
            entry for entry in self.quality_history
            if datetime.fromisoformat(entry['timestamp']) > max_age
        ]

    def _is_price_rule(self, rule_name: str) -> bool:
        """Check if a rule is specific to price data."""
        price_specific_rules = {
            'price_relationships',
            'non_negative_prices',
            'data_freshness'
        }
        return rule_name in price_specific_rules

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_alerts': len(self.alerts),
                'quality_history_length': len(self.quality_history),
                'active_rules': len([r for r in self.validation_rules.values() if r.enabled])
            },
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'quality_trends': self._calculate_quality_trends(),
            'rule_status': {
                name: {
                    'enabled': rule.enabled,
                    'dimension': rule.dimension.value,
                    'severity': rule.severity.value
                }
                for name, rule in self.validation_rules.items()
            }
        }

        # Save report if path configured
        if self.config.quality_report_path:
            self._save_quality_report(report)

        return report

    def _calculate_quality_trends(self) -> Dict[str, Any]:
        """Calculate quality trends over time."""
        if len(self.quality_history) < 2:
            return {'error': 'Insufficient history for trend analysis'}

        try:
            # Convert to DataFrame for analysis
            history_df = pd.DataFrame(self.quality_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

            # Calculate trends
            trends = {}
            for dimension in DataQualityDimension:
                dim_col = dimension.value
                if dim_col in history_df.columns:
                    scores = history_df[dim_col].dropna()
                    if len(scores) >= 2:
                        # Simple trend: compare recent vs older scores
                        recent_scores = scores.tail(max(1, len(scores) // 3))
                        older_scores = scores.head(max(1, len(scores) // 3))

                        recent_avg = recent_scores.mean()
                        older_avg = older_scores.mean()

                        trends[dim_col] = {
                            'current_score': recent_avg,
                            'previous_score': older_avg,
                            'trend': 'improving' if recent_avg > older_avg else 'declining',
                            'change': recent_avg - older_avg
                        }

            return trends

        except Exception as e:
            return {'error': f'Error calculating trends: {e}'}

    def _save_quality_report(self, report: Dict[str, Any]) -> None:
        """Save quality report to file."""
        try:
            report_path = Path(self.config.quality_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error saving quality report: {e}")

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            'config': self.config.__dict__,
            'rules_count': len(self.validation_rules),
            'enabled_rules': len([r for r in self.validation_rules.values() if r.enabled]),
            'alerts_count': len(self.alerts),
            'history_length': len(self.quality_history),
            'timestamp': datetime.now().isoformat()
        }
