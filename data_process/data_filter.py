import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import anthropic
import openai
import os
from collections import Counter


class MDDDataPreprocessor:
    """
    MDD Data Preprocessing Class
    Implements feature selection and data filtering pipeline
    """

    def __init__(self, data_path: str, api_key: str = None):
        """
        Initialize preprocessor

        Args:
            data_path: Path to UK Biobank data
            api_key: Anthropic API key (if not provided, read from environment variable)
        """
        self.data = pd.read_csv(data_path)
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Define 22 selected features
        self.selected_features = [
            # Demographics & Socioeconomic
            'age', 'sex', 'education_level', 'income',
            'employment_status', 'working_hours',

            # Lifestyle & Psychosocial
            'bmi', 'sleep_duration', 'alcohol_consumption',
            'self_harm_history', 'suicidal_behavior',
            'perceived_happiness', 'work_satisfaction',
            'health_satisfaction', 'family_satisfaction',
            'financial_satisfaction',

            # Clinical
            'long_standing_illness',

            # Biochemical
            'hdl', 'ldl', 'total_cholesterol', 'triglycerides',

            # Label
            'mdd_diagnosis'
        ]

    def feature_selection(self,
                          algorithm_features: List[str] = None,
                          clinical_features: List[str] = None) -> pd.DataFrame:
        """
        Feature selection phase: integrate algorithm-driven and clinical knowledge-driven features

        Args:
            algorithm_features: Feature list from algorithmic analysis
            clinical_features: Feature list from clinical knowledge

        Returns:
            DataFrame with selected features
        """
        print("=" * 60)
        print("Stage 1: Feature Selection")
        print("=" * 60)

        # If no feature lists provided, use predefined 22 features
        if algorithm_features is None and clinical_features is None:
            features_to_keep = self.selected_features
        else:
            # Integrate features from both methods
            algo_set = set(algorithm_features) if algorithm_features else set()
            clinical_set = set(clinical_features) if clinical_features else set()
            features_to_keep = list(algo_set.union(clinical_set))
            features_to_keep.append('mdd_diagnosis')  # Ensure label is included

        # Check if features exist
        available_features = [f for f in features_to_keep if f in self.data.columns]
        missing_features = set(features_to_keep) - set(available_features)

        if missing_features:
            print(f"Warning: The following features are not in the data: {missing_features}")

        print(f"Selected {len(available_features)} features")
        print(f"Feature list: {available_features}")

        return self.data[available_features].copy()

    def filter_missing_values(self,
                              df: pd.DataFrame,
                              threshold: float = 0.3) -> pd.DataFrame:
        """
        Filter samples with missing values exceeding threshold

        Args:
            df: Input DataFrame
            threshold: Missing value ratio threshold (default 30%)

        Returns:
            Filtered DataFrame
        """
        print("\n" + "=" * 60)
        print("Stage 2: Data Filtering - Missing Value Check")
        print("=" * 60)

        initial_count = len(df)

        # Calculate missing value ratio for each row
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)

        # Retain samples with missing ratio <= threshold
        df_filtered = df[missing_ratio <= threshold].copy()

        excluded_count = initial_count - len(df_filtered)
        print(f"Initial sample count: {initial_count}")
        print(f"Excluded samples: {excluded_count} ({excluded_count / initial_count * 100:.2f}%)")
        print(f"Retained samples: {len(df_filtered)}")

        return df_filtered

    def create_clinical_summary(self, row: pd.Series) -> str:
        """
        Create structured clinical description for a single sample

        Args:
            row: Sample data row

        Returns:
            Structured text description
        """
        summary_parts = []

        # Demographics
        summary_parts.append(
            f"Patient is a {row.get('age', 'unknown')} year-old {row.get('sex', 'unknown')} individual.")

        # Socioeconomic
        if 'education_level' in row:
            summary_parts.append(f"Education level: {row['education_level']}.")
        if 'employment_status' in row:
            summary_parts.append(f"Employment status: {row['employment_status']}.")

        # Lifestyle
        if 'bmi' in row and pd.notna(row['bmi']):
            summary_parts.append(f"BMI: {row['bmi']:.1f}.")
        if 'sleep_duration' in row and pd.notna(row['sleep_duration']):
            summary_parts.append(f"Average sleep duration: {row['sleep_duration']} hours.")

        # Psychosocial
        if 'perceived_happiness' in row:
            summary_parts.append(f"Perceived happiness: {row['perceived_happiness']}.")

        # Clinical
        if 'long_standing_illness' in row:
            summary_parts.append(f"Long-standing illness: {row['long_standing_illness']}.")

        # Biochemical
        biochem = []
        if 'hdl' in row and pd.notna(row['hdl']):
            biochem.append(f"HDL: {row['hdl']:.2f}")
        if 'ldl' in row and pd.notna(row['ldl']):
            biochem.append(f"LDL: {row['ldl']:.2f}")
        if biochem:
            summary_parts.append(f"Biochemical markers - {', '.join(biochem)}.")

        return " ".join(summary_parts)

    def llm_diagnostic_inference(self, clinical_summary: str, model_name: str) -> str:
        """
        Use LLM for diagnostic inference

        Args:
            clinical_summary: Clinical description text
            model_name: Model identifier (for logging)

        Returns:
            Diagnostic inference result ('MDD' or 'Non-MDD')
        """
        prompt = f"""Based on the following clinical information, provide a diagnostic inference for Major Depressive Disorder (MDD).

Clinical Summary:
{clinical_summary}

Please respond with ONLY one of these two options:
- MDD (if the patient likely has Major Depressive Disorder)
- Non-MDD (if the patient likely does not have Major Depressive Disorder)

Response:"""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response = message.content[0].text.strip()

            # Standardize output
            if "MDD" in response and "Non-MDD" not in response:
                return "MDD"
            else:
                return "Non-MDD"

        except Exception as e:
            print(f"Error ({model_name}): {e}")
            return "Error"

    def triple_llm_screening(self,
                             df: pd.DataFrame,
                             sample_size: int = None) -> pd.DataFrame:
        """
        Use three LLMs for consistency screening

        Args:
            df: Input DataFrame
            sample_size: Number of samples to process (None means all, can set small number for testing)

        Returns:
            Screened DataFrame
        """
        print("\n" + "=" * 60)
        print("Stage 3: Triple-LLM Consistency Screening")
        print("=" * 60)

        if sample_size:
            df = df.head(sample_size)
            print(f"Note: Processing only first {sample_size} samples for demonstration")

        initial_count = len(df)
        retained_indices = []

        for idx, row in df.iterrows():
            # Create clinical description
            clinical_summary = self.create_clinical_summary(row)

            # Three model inferences (simplified here as calling same API three times,
            # in practice should call different models)
            inference_1 = self.llm_diagnostic_inference(clinical_summary, "Model-1")
            inference_2 = self.llm_diagnostic_inference(clinical_summary, "Model-2")
            inference_3 = self.llm_diagnostic_inference(clinical_summary, "Model-3")

            # Ground truth label
            ground_truth = "MDD" if row['mdd_diagnosis'] == 1 else "Non-MDD"

            # Check if three models are consistent and match ground truth
            inferences = [inference_1, inference_2, inference_3]

            if all(inf == ground_truth for inf in inferences):
                retained_indices.append(idx)
                status = "✓ Retained"
            else:
                status = "✗ Excluded"

            print(f"Sample {idx}: {status} (Ground truth: {ground_truth}, Inferences: {inferences})")

        df_screened = df.loc[retained_indices].copy()

        excluded_count = initial_count - len(df_screened)
        print(f"\nInitial sample count: {initial_count}")
        print(f"Excluded samples: {excluded_count} ({excluded_count / initial_count * 100:.2f}%)")
        print(f"Retained samples: {len(df_screened)}")

        return df_screened

    def bias_assessment(self,
                        df_original: pd.DataFrame,
                        df_filtered: pd.DataFrame,
                        key_variables: List[str] = None) -> Dict:
        """
        Assess whether filtering process introduces bias

        Args:
            df_original: Original DataFrame
            df_filtered: Filtered DataFrame
            key_variables: Key variables to compare

        Returns:
            Bias assessment results dictionary
        """
        print("\n" + "=" * 60)
        print("Stage 4: Bias Assessment")
        print("=" * 60)

        if key_variables is None:
            key_variables = ['age', 'sex', 'education_level', 'mdd_diagnosis']

        # Only keep variables present in both DataFrames
        key_variables = [v for v in key_variables if v in df_original.columns and v in df_filtered.columns]

        assessment_results = {}

        for var in key_variables:
            print(f"\nVariable: {var}")

            # Numeric variables: compare means
            if df_original[var].dtype in ['int64', 'float64']:
                mean_original = df_original[var].mean()
                mean_filtered = df_filtered[var].mean()
                diff_pct = abs(mean_filtered - mean_original) / mean_original * 100

                print(f"  Original mean: {mean_original:.2f}")
                print(f"  Filtered mean: {mean_filtered:.2f}")
                print(f"  Difference: {diff_pct:.2f}%")

                assessment_results[var] = {
                    'type': 'numeric',
                    'original_mean': mean_original,
                    'filtered_mean': mean_filtered,
                    'diff_percent': diff_pct,
                    'substantial_shift': diff_pct > 10  # 10% as threshold
                }

            # Categorical variables: compare distributions
            else:
                dist_original = df_original[var].value_counts(normalize=True)
                dist_filtered = df_filtered[var].value_counts(normalize=True)

                print(f"Original distribution:\n{dist_original}")
                print(f"Filtered distribution:\n{dist_filtered}")

                assessment_results[var] = {
                    'type': 'categorical',
                    'original_dist': dist_original.to_dict(),
                    'filtered_dist': dist_filtered.to_dict()
                }

        # Overall assessment
        numeric_vars_shifted = [v for v, r in assessment_results.items()
                                if r['type'] == 'numeric' and r['substantial_shift']]

        if numeric_vars_shifted:
            print(f"Warning: Substantial shift detected in: {numeric_vars_shifted}")
        else:
            print(f"No substantial systematic bias detected")

        return assessment_results

    def run_full_pipeline(self,
                          missing_threshold: float = 0.3,
                          llm_sample_size: int = None) -> pd.DataFrame:
        """
        Run complete data preprocessing pipeline

        Args:
            missing_threshold: Missing value threshold
            llm_sample_size: Number of samples for LLM screening (None means all)

        Returns:
            Final preprocessed DataFrame
        """
        print("\n" + "=" * 60)
        print("MDD Data Preprocessing Complete Pipeline")
        print("=" * 60)

        # Stage 1: Feature Selection
        df_selected = self.feature_selection()

        # Stage 2: Missing Value Filtering
        df_filtered = self.filter_missing_values(df_selected, missing_threshold)

        # Stage 3: Triple-LLM Screening
        df_screened = self.triple_llm_screening(df_filtered, llm_sample_size)

        # Stage 4: Bias Assessment
        bias_results = self.bias_assessment(df_selected, df_screened)

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Final dataset size: {len(df_screened)} samples")
        print(f"Number of features: {len(df_screened.columns)}")

        return df_screened


# Usage example
if __name__ == "__main__":
    # load ukb dataset
    # ukb_data = pd.read_csv("ukb_data.csv")

    # Run preprocessing
    # Note: Need to set ANTHROPIC_API_KEY or Other LLMs Keys environment variable
    preprocessor = MDDDataPreprocessor('ukb_data.csv')

    # Run full pipeline (process only 5 samples for demonstration to avoid excessive API calls)
    final_data = preprocessor.run_full_pipeline(
        missing_threshold=0.3,
        llm_sample_size=5  # Process only 5 samples for demonstration
    )

    # Save final data
    final_data.to_csv('preprocessed_mdd_data.csv', index=False)
    print("\nPreprocessed data saved to: preprocessed_mdd_data.csv")
