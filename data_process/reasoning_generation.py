import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import anthropic
import openai
import os
import json
from tqdm import tqdm


class ReasoningPathGenerator:
    """
    Reasoning Path Generation Class
    Implements CoT-based reasoning path construction for MDD diagnosis
    """

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize reasoning path generator

        Args:
            api_key: Anthropic API key (if not provided, read from environment variable)
            model: LLM model to use for generation
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

        # CoT Prompt Template
        self.cot_prompt = """You are a clinical expert specializing in Major Depressive Disorder (MDD) diagnosis. 

Your task is to analyze the following clinical information and provide a structured diagnostic reasoning process.

Clinical Information:
{clinical_summary}

Please follow these steps in your reasoning:
1. Comprehend and summarize the key clinical features
2. Identify relevant risk factors and protective factors for MDD
3. Evaluate the presence of depressive symptoms or indicators
4. Consider differential diagnoses if applicable
5. Provide a final diagnostic conclusion

Format your response as follows:
REASONING:
[Your step-by-step reasoning process]

DIAGNOSIS:
[MDD or Non-MDD]"""

        self.refinement_prompt = """Review and refine the following diagnostic reasoning prompt to make it more effective for multi-step clinical reasoning.

Original Prompt:
{original_prompt}

Original Clinical Summary:
{clinical_summary}

Original Reasoning:
{original_reasoning}

Please provide an enhanced version of the prompt that:
1. Provides clearer guidance for structured reasoning
2. Emphasizes important clinical considerations
3. Improves the quality of step-by-step analysis

Return ONLY the refined prompt text, without any additional explanation."""

    def textualize_clinical_data(self, row: pd.Series) -> str:
        """
        Convert tabular clinical data to structured text description

        Args:
            row: Sample data row from preprocessed dataset

        Returns:
            Structured natural language clinical description
        """
        parts = []

        # Demographics
        age = row.get('age', 'unknown')
        sex = row.get('sex', 'unknown')
        parts.append(f"Patient Demographics: {age} years old, {sex}")

        # Socioeconomic Status
        socioeconomic = []
        if 'education_level' in row and pd.notna(row['education_level']):
            socioeconomic.append(f"education level: {row['education_level']}")
        if 'income' in row and pd.notna(row['income']):
            socioeconomic.append(f"annual income: ${row['income']}")
        if 'employment_status' in row and pd.notna(row['employment_status']):
            socioeconomic.append(f"employment status: {row['employment_status']}")
        if socioeconomic:
            parts.append(f"Socioeconomic Status: {', '.join(socioeconomic)}")

        # Lifestyle Factors
        lifestyle = []
        if 'bmi' in row and pd.notna(row['bmi']):
            lifestyle.append(f"BMI: {row['bmi']:.1f}")
        if 'sleep_duration' in row and pd.notna(row['sleep_duration']):
            lifestyle.append(f"sleep duration: {row['sleep_duration']:.1f} hours")
        if 'alcohol_consumption' in row and pd.notna(row['alcohol_consumption']):
            lifestyle.append(f"alcohol use: {row['alcohol_consumption']}")
        if lifestyle:
            parts.append(f"Lifestyle Factors: {', '.join(lifestyle)}")

        # Psychosocial Indicators
        psychosocial = []
        if 'self_harm_history' in row and pd.notna(row['self_harm_history']):
            psychosocial.append(f"self-harm history: {row['self_harm_history']}")
        if 'suicidal_behavior' in row and pd.notna(row['suicidal_behavior']):
            psychosocial.append(f"suicidal behavior: {row['suicidal_behavior']}")
        if 'perceived_happiness' in row and pd.notna(row['perceived_happiness']):
            psychosocial.append(f"happiness score: {row['perceived_happiness']}/10")
        if 'work_satisfaction' in row and pd.notna(row['work_satisfaction']):
            psychosocial.append(f"work satisfaction: {row['work_satisfaction']}/10")
        if psychosocial:
            parts.append(f"Psychosocial Indicators: {', '.join(psychosocial)}")

        # Clinical History
        if 'long_standing_illness' in row and pd.notna(row['long_standing_illness']):
            parts.append(f"Clinical History: long-standing illness: {row['long_standing_illness']}")

        # Biochemical Markers
        biochemical = []
        if 'hdl' in row and pd.notna(row['hdl']):
            biochemical.append(f"HDL: {row['hdl']:.2f} mg/dL")
        if 'ldl' in row and pd.notna(row['ldl']):
            biochemical.append(f"LDL: {row['ldl']:.2f} mg/dL")
        if 'total_cholesterol' in row and pd.notna(row['total_cholesterol']):
            biochemical.append(f"total cholesterol: {row['total_cholesterol']:.2f} mg/dL")
        if 'triglycerides' in row and pd.notna(row['triglycerides']):
            biochemical.append(f"triglycerides: {row['triglycerides']:.2f} mg/dL")
        if biochemical:
            parts.append(f"Biochemical Markers: {', '.join(biochemical)}")

        return "\n".join(parts)

    def generate_reasoning_path(self,
                                clinical_summary: str,
                                prompt: str,
                                ground_truth: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Generate reasoning path using LLM

        Args:
            clinical_summary: Textualized clinical description
            prompt: CoT prompt template
            ground_truth: Ground truth diagnosis ('MDD' or 'Non-MDD')

        Returns:
            Tuple of (reasoning_path, prediction, is_valid)
        """
        try:
            full_prompt = prompt.format(clinical_summary=clinical_summary)

            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            response = message.content[0].text.strip()

            # Parse reasoning and diagnosis
            reasoning_path = None
            prediction = None

            if "REASONING:" in response and "DIAGNOSIS:" in response:
                parts = response.split("DIAGNOSIS:")
                reasoning_path = parts[0].replace("REASONING:", "").strip()
                diagnosis_text = parts[1].strip()

                # Extract diagnosis
                if "Non-MDD" in diagnosis_text:
                    prediction = "Non-MDD"
                elif "MDD" in diagnosis_text:
                    prediction = "MDD"

            # Validate
            is_valid = (prediction == ground_truth) if prediction else False

            return reasoning_path, prediction, is_valid

        except Exception as e:
            print(f"Error in reasoning generation: {e}")
            return None, None, False

    def initial_reasoning_generation(self,
                                     df: pd.DataFrame,
                                     max_attempts: int = 3) -> pd.DataFrame:
        """
        Initial reasoning path generation with validation

        Args:
            df: Preprocessed DataFrame
            max_attempts: Maximum number of attempts (T in the paper)

        Returns:
            DataFrame with valid reasoning paths
        """
        print("=" * 60)
        print("Stage 1: Initial Reasoning Generation")
        print("=" * 60)

        results = []
        discarded_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating reasoning paths"):
            # Convert to text
            clinical_summary = self.textualize_clinical_data(row)
            ground_truth = "MDD" if row['mdd_diagnosis'] == 1 else "Non-MDD"

            # Try up to max_attempts times
            success = False
            for attempt in range(max_attempts):
                reasoning_path, prediction, is_valid = self.generate_reasoning_path(
                    clinical_summary=clinical_summary,
                    prompt=self.cot_prompt,
                    ground_truth=ground_truth
                )

                if is_valid:
                    results.append({
                        'index': idx,
                        'clinical_summary': clinical_summary,
                        'reasoning_path': reasoning_path,
                        'prediction': prediction,
                        'ground_truth': ground_truth,
                        'prompt': self.cot_prompt,
                        'attempts_used': attempt + 1
                    })
                    success = True
                    print(f"  Sample {idx}: Valid (attempt {attempt + 1}/{max_attempts})")
                    break

            if not success:
                discarded_count += 1
                print(f"  Sample {idx}: Discarded after {max_attempts} attempts")

        print(f"\nInitial Generation Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Valid samples: {len(results)}")
        print(f"Discarded samples: {discarded_count}")

        return pd.DataFrame(results)

    def refine_prompt(self,
                      original_prompt: str,
                      clinical_summary: str,
                      original_reasoning: str) -> Optional[str]:
        """
        Refine the CoT prompt for better reasoning guidance

        Args:
            original_prompt: Original CoT prompt
            clinical_summary: Clinical description
            original_reasoning: Original reasoning path

        Returns:
            Enhanced prompt or None if refinement fails
        """
        try:
            refinement_request = self.refinement_prompt.format(
                original_prompt=original_prompt,
                clinical_summary=clinical_summary,
                original_reasoning=original_reasoning
            )

            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": refinement_request}
                ]
            )

            enhanced_prompt = message.content[0].text.strip()
            return enhanced_prompt

        except Exception as e:
            print(f"Error in prompt refinement: {e}")
            return None

    def refinement_step(self,
                        df_initial: pd.DataFrame,
                        max_refinement_attempts: int = 3) -> pd.DataFrame:
        """
        Refinement step to enhance reasoning quality

        Args:
            df_initial: DataFrame from initial generation
            max_refinement_attempts: Maximum refinement attempts (N in the paper)

        Returns:
            DataFrame with refined reasoning paths
        """
        print("\n" + "=" * 60)
        print("Stage 2: Refinement Step")
        print("=" * 60)

        refined_results = []
        reverted_count = 0

        for idx, row in tqdm(df_initial.iterrows(), total=len(df_initial), desc="Refining reasoning paths"):
            # Get enhanced prompt
            enhanced_prompt = self.refine_prompt(
                original_prompt=row['prompt'],
                clinical_summary=row['clinical_summary'],
                original_reasoning=row['reasoning_path']
            )

            if not enhanced_prompt:
                # Revert to original if refinement fails
                refined_results.append(row.to_dict())
                reverted_count += 1
                print(f"Sample {row['index']}: Reverted (refinement failed)")
                continue

            # Try refinement up to max_refinement_attempts times
            success = False
            for attempt in range(max_refinement_attempts):
                refined_reasoning, refined_prediction, is_valid = self.generate_reasoning_path(
                    clinical_summary=row['clinical_summary'],
                    prompt=enhanced_prompt,
                    ground_truth=row['ground_truth']
                )

                if is_valid:
                    refined_results.append({
                        'index': row['index'],
                        'clinical_summary': row['clinical_summary'],
                        'reasoning_path': refined_reasoning,
                        'prediction': refined_prediction,
                        'ground_truth': row['ground_truth'],
                        'prompt': enhanced_prompt,
                        'initial_attempts': row['attempts_used'],
                        'refinement_attempts': attempt + 1,
                        'refined': True
                    })
                    success = True
                    print(
                        f"  Sample {row['index']}: Refined successfully (attempt {attempt + 1}/{max_refinement_attempts})")
                    break

            if not success:
                # Revert to original
                result = row.to_dict()
                result['refined'] = False
                result['refinement_attempts'] = max_refinement_attempts
                refined_results.append(result)
                reverted_count += 1
                print(f"  Sample {row['index']}: Reverted to original after {max_refinement_attempts} attempts")

        print(f"\nRefinement Summary:")
        print(f"Total samples: {len(df_initial)}")
        print(f"Successfully refined: {len(refined_results) - reverted_count}")
        print(f"Reverted to original: {reverted_count}")

        return pd.DataFrame(refined_results)

    def run_full_pipeline(self,
                          df_preprocessed: pd.DataFrame,
                          max_initial_attempts: int = 3,
                          max_refinement_attempts: int = 3,
                          sample_limit: int = None) -> pd.DataFrame:
        """
        Run complete reasoning path generation pipeline

        Args:
            df_preprocessed: Preprocessed DataFrame from preprocessing stage
            max_initial_attempts: Maximum attempts for initial generation (T)
            max_refinement_attempts: Maximum attempts for refinement (N)
            sample_limit: Limit number of samples to process (for testing)

        Returns:
            Final reasoning dataset
        """
        print("\n" + "=" * 60)
        print("Reasoning Path Generation Complete Pipeline")
        print("=" * 60)

        # Limit samples if specified
        if sample_limit:
            df_preprocessed = df_preprocessed.head(sample_limit)
            print(f"Processing {sample_limit} samples for demonstration\n")

        # Stage 1: Initial Generation
        df_initial = self.initial_reasoning_generation(
            df=df_preprocessed,
            max_attempts=max_initial_attempts
        )

        # Stage 2: Refinement
        df_final = self.refinement_step(
            df_initial=df_initial,
            max_refinement_attempts=max_refinement_attempts
        )

        print("\n" + "=" * 60)
        print("Reasoning Generation Complete!")
        print("=" * 60)
        print(f"Final dataset size: {len(df_final)} samples")
        print(f"Refined samples: {df_final['refined'].sum() if 'refined' in df_final.columns else 'N/A'}")

        return df_final

    def save_reasoning_dataset(self,
                               df: pd.DataFrame,
                               output_path: str,
                               format: str = 'json'):
        """
        Save reasoning dataset in specified format

        Args:
            df: Reasoning dataset DataFrame
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        if format == 'json':
            # Convert to JSON format suitable for training
            training_data = []
            for _, row in df.iterrows():
                training_data.append({
                    'instruction': row['prompt'],
                    'input': row['clinical_summary'],
                    'reasoning': row['reasoning_path'],
                    'output': row['prediction'],
                    'ground_truth': row['ground_truth']
                })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

        elif format == 'csv':
            df.to_csv(output_path, index=False)

        print(f"Reasoning dataset saved to: {output_path}")


# Usage example
if __name__ == "__main__":
    # Load preprocessed data from previous stage
    df_preprocessed = pd.read_csv('preprocessed_mdd_data.csv')
    print(f"Loaded {len(df_preprocessed)} preprocessed samples")

    # Initialize generator
    # Note: Requires ANTHROPIC_API_KEY environment variable
    generator = ReasoningPathGenerator()

    # Run full pipeline (process only 3 samples for demonstration)
    reasoning_dataset = generator.run_full_pipeline(
        df_preprocessed=df_preprocessed,
        max_initial_attempts=3,
        max_refinement_attempts=2,
        sample_limit=3  # Process only 3 samples for demonstration
    )

    # Save in both formats
    generator.save_reasoning_dataset(
        df=reasoning_dataset,
        output_path='reasoning_dataset.json',
        format='json'
    )

    generator.save_reasoning_dataset(
        df=reasoning_dataset,
        output_path='reasoning_dataset.csv',
        format='csv'
    )

    print("\n" + "=" * 60)
    print("Example of generated reasoning path:")
    print("=" * 60)
    if len(reasoning_dataset) > 0:
        example = reasoning_dataset.iloc[0]
        print(f"\nClinical Summary:\n{example['clinical_summary']}")
        print(f"\nReasoning Path:\n{example['reasoning_path']}")
        print(f"\nPrediction: {example['prediction']}")
        print(f"Ground Truth: {example['ground_truth']}")