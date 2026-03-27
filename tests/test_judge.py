"""
Unit tests for augmentation judge module.
"""

import unittest
import pandas as pd
import os
import tempfile
from src.augmentation.judge import (
    judge_batch,
    judge_augmented_dataframe,
    get_judge_prompt,
)


class TestJudgePrompt(unittest.TestCase):
    """Test judge prompt generation."""
    
    def test_judge_prompt_st1(self):
        """Test judge prompt generation for ST1."""
        texts = ["Hello world", "This is a test"]
        labels = [0, 1]
        prompt = get_judge_prompt(texts, labels, task="st1")
        
        self.assertIn("ROLE", prompt)
        self.assertIn("Label 0", prompt)
        self.assertIn("Label 1", prompt)
        self.assertIn("Hello world", prompt)
        self.assertIn("OUTPUT FORMAT", prompt)
    
    def test_judge_prompt_st2(self):
        """Test judge prompt generation for ST2."""
        texts = ["Sample text"]
        labels = ["10000"]  # One label combo
        prompt = get_judge_prompt(texts, labels, task="st2")
        
        self.assertIn("ROLE", prompt)
        self.assertIn("polarization types", prompt)
        self.assertIn("Sample text", prompt)
        self.assertIn("OUTPUT FORMAT", prompt)


class TestJudgeBatch(unittest.TestCase):
    """Test batch judging (requires API calls - skipped in CI)."""
    
    def test_judge_batch_returns_list(self):
        """Test that judge_batch returns proper structure."""
        # Mock test: just check that function accepts inputs and returns list structure
        texts = ["Test text 1"]
        labels = [0]
        
        # This will fail if no API key, but structure should be list
        result = judge_batch(texts, labels, task="st1")
        self.assertIsInstance(result, list)
    
    def test_judge_batch_empty_input(self):
        """Test judge_batch with empty input."""
        result = judge_batch([], [], task="st1")
        self.assertEqual(len(result), 0)


class TestJudgeDataFrame(unittest.TestCase):
    """Test DataFrame judging."""
    
    def test_judge_dataframe_st1(self):
        """Test judging a ST1 DataFrame."""
        df = pd.DataFrame({
            'text': ['Text 1', 'Text 2'],
            'polarization': [0, 1]
        })
        
        filtered_df, scores_df = judge_augmented_dataframe(df, task="st1", threshold=0.0)
        
        # Should return at least some results
        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertIsInstance(scores_df, pd.DataFrame)
    
    def test_judge_dataframe_st2(self):
        """Test judging a ST2 DataFrame."""
        df = pd.DataFrame({
            'text': ['Text 1', 'Text 2'],
            'political': [1, 0],
            'racial/ethnic': [0, 1],
            'religious': [0, 0],
            'gender/sexual': [0, 0],
            'other': [0, 0]
        })
        
        filtered_df, scores_df = judge_augmented_dataframe(df, task="st2", threshold=0.0)
        
        self.assertIsInstance(filtered_df, pd.DataFrame)
        self.assertIsInstance(scores_df, pd.DataFrame)
    
    def test_judge_empty_dataframe(self):
        """Test judging empty DataFrame."""
        df = pd.DataFrame({'text': [], 'polarization': []})
        filtered_df, scores_df = judge_augmented_dataframe(df, task="st1")
        
        self.assertEqual(len(filtered_df), 0)
        self.assertEqual(len(scores_df), 0)


if __name__ == '__main__':
    unittest.main()
